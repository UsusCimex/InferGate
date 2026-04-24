from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_provider_manager

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/memory/status")
async def memory_status(manager=Depends(get_provider_manager)):
    return manager.status_snapshot()


@router.get("/memory/preview-load/{model_id}")
async def preview_load(model_id: str, manager=Depends(get_provider_manager)):
    return manager.preview_load(model_id)
