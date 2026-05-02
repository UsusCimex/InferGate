from app.services.cache_backends.base import CacheBackend
from app.services.cache_backends.factory import make_cache_backend
from app.services.cache_backends.local import LocalCacheBackend

__all__ = ["CacheBackend", "LocalCacheBackend", "make_cache_backend"]


# RedisCacheBackend is exposed lazily to keep `redis` import optional.
def __getattr__(name: str):
    if name == "RedisCacheBackend":
        from app.services.cache_backends.redis import RedisCacheBackend
        return RedisCacheBackend
    raise AttributeError(f"module 'app.services.cache_backends' has no attribute {name!r}")
