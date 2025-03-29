from .audio import router as audio_router
from fastapi import APIRouter

router = APIRouter()

router.include_router(audio_router)

__all__ = ["router"]
