from __future__ import annotations

from pydantic import BaseModel


class CacheStatsResponse(BaseModel):
    global_stats: dict = {}
    per_model: dict = {}


class CacheInvalidateResponse(BaseModel):
    deleted: int
    message: str
