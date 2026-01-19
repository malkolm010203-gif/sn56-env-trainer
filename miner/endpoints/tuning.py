from fastapi import Depends, HTTPException
from fastapi.routing import APIRouter
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import verify_get_request

from core.models.payload_models import TrainingRepoResponse
from core.models.tournament_models import TournamentType


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    # Only participate in TEXT and ENVIRONMENT tournaments
    if task_type == TournamentType.IMAGE:
        raise HTTPException(status_code=404, detail="Not participating in IMAGE tournament")
    
    # For ENVIRONMENT - use base G.O.D repo with alfworld example
    if task_type == TournamentType.ENVIRONMENT:
        return TrainingRepoResponse(
            github_repo="https://github.com/rayonlabs/G.O.D",
            commit_hash="b3468779"
        )
    
    # For TEXT - use our DoRA-enabled repo
    return TrainingRepoResponse(
        github_repo="https://github.com/malkolm010203-gif/sn56-text-trainer",
        commit_hash="c7d52fcc334a98b4547748cff662ee5b59785110"
    )


def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/training_repo/{task_type}",
        get_training_repo,
        tags=["Subnet"],
        methods=["GET"],
        response_model=TrainingRepoResponse,
        summary="Get Training Repo",
        description="Retrieve the training repository and commit hash for the tournament.",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    return router
