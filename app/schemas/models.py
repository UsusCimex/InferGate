from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "infergate"
    display_name: str = ""
    category: str = ""
    loaded: bool = False
    enabled: bool = True
    metadata: dict = {}


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
