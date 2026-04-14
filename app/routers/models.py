from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_provider_manager
from app.services.provider_manager import ModelNotFoundError

router = APIRouter()


@router.get("/v1/models")
async def list_models(manager=Depends(get_provider_manager)):
    models = manager.list_models()
    return {"object": "list", "data": models}


@router.post("/v1/models/{model_id}/load")
async def load_model(model_id: str, manager=Depends(get_provider_manager)):
    try:
        await manager.load_model(model_id)
        return {"status": "ok", "model": model_id, "loaded": True}
    except ModelNotFoundError:
        return JSONResponse(
            {"error": {"message": f"Model '{model_id}' not found"}},
            status_code=404,
        )


@router.post("/v1/models/{model_id}/unload")
async def unload_model(model_id: str, manager=Depends(get_provider_manager)):
    try:
        await manager.unload_model(model_id)
        return {"status": "ok", "model": model_id, "loaded": False}
    except ModelNotFoundError:
        return JSONResponse(
            {"error": {"message": f"Model '{model_id}' not found"}},
            status_code=404,
        )
