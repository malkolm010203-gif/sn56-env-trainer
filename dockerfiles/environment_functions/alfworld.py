def alfworld_rollout(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests

    if not getattr(alfworld_rollout, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]
        base_url = server_list[rank % len(server_list)] if server_list else ""
        alfworld_rollout.base_url = base_url
        try:
            create_res = requests.post(f"{base_url}/create", timeout=300)
            create_res.raise_for_status()
            alfworld_rollout.env_id = create_res.json()["id"]
            alfworld_rollout.initialized = True
        except Exception as e:
            raise e

    env_id = alfworld_rollout.env_id
    env_endpoint = alfworld_rollout.base_url

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_rewards = []

    tokenizer = trainer.processing_class
    DATA_LEN = 2500
    TIMEOUT = 2400

    conversation_start = [
        {"from": "human", "value": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.'},
        {"from": "gpt", "value": "OK. I'll follow your instructions and try my best to solve the task."}
    ]

    game_id = random.randint(0, DATA_LEN - 1)

    for i, prompt in enumerate(prompts):
        first_prompt_ids = []
        first_completion_ids = []
        first_logprobs = []
        invalid_count = 0
        valid_count = 0
        done = False
        solved = False
        turn = 0

        payload = {"id": env_id, "game": game_id, "world_type": "Text"}
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            obs = reset_data["observation"]
            actions = reset_data["available_actions"]
            fmt_obs = f"{obs}\nAVAILABLE ACTIONS: {','.join(actions)}"
        except:
            continue

        messages = []
        for m in conversation_start:
            role = "user" if m["from"] == "human" else "assistant"
            messages.append({"role": role, "content": m["value"]})
        messages.append({"role": "user", "content": fmt_obs})

        while not done and turn < max_turns:
            out = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            p_ids = out.get("prompt_ids", [])
            c_ids = out.get("completion_ids", [])
            lp = out.get("logprobs", [])
            txt = tokenizer.decode(c_ids, skip_special_tokens=True).strip()

            if turn == 0:
                first_prompt_ids = p_ids
                first_completion_ids = c_ids
                first_logprobs = lp

            messages.append({"role": "assistant", "content": txt})

            action = txt
            if action.endswith("</s>"):
                action = action[:-5]
            if "Action:" in action:
                action = action.split("Action:")[-1].strip()

            try:
                step_res = requests.post(f"{env_endpoint}/step", json={"id": env_id, "action": action}, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                state = step_data["observation"]
                reward = step_data["reward"]
                done = step_data["done"]
                actions = step_data["available_actions"]
                fmt_obs = f"{state}\nAVAILABLE ACTIONS: {','.join(actions)}"
            except:
                fmt_obs = "Invalid Action.\n\n" + fmt_obs
                done = False
                state = "Invalid"
                reward = 0.0

            if done and reward > 0:
                solved = True

            if "Nothing happens" in state or "Invalid" in state:
                invalid_count += 1
            else:
                valid_count += 1

            if not done:
                messages.append({"role": "user", "content": fmt_obs})
            turn += 1

        r = 1.0 if solved else 0.0
        r += 0.15 * (1.0 - turn / max_turns) if solved else 0.0
        r += 0.02 * valid_count if not solved else 0.0
        r -= 0.03 * invalid_count
        r = max(-0.5, min(1.5, r))

        all_prompt_ids.append(first_prompt_ids)
        all_completion_ids.append(first_completion_ids)
        all_logprobs.append(first_logprobs)
        all_rewards.append(r)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_rewards": all_rewards
    }


def alfworld_rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
