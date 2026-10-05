---
paths:
  - "app/providers/**"
  - "app/worker.py"
  - "app/services/provider_manager.py"
---

# Providers, remote proxies and the worker

- A provider subclasses an ABC from `app/providers/base.py`, carries `@register_provider` and lives in `app/providers/{image,text,tts,stt,upscale,embedding}/` (the registry imports only those subpackages).
- Each category exists three times: the local provider, its gateway-side proxy in `app/providers/remote.py` (`CATEGORY_REGISTRY`, endpoint specs in `_remote_protocol.py`) and the matching endpoint in `app/worker.py`. A new parameter or method must be threaded through all three.
- Raise `ValueError` for bad input: the worker turns it into 400 `invalid_request`; anything else becomes a 500 that the gateway forwards.
- Worker inference runs under `reload_lock`, one request at a time; blocking model calls go through `asyncio.to_thread` or the provider's executor.
- Heavy dependencies belong only in `deploy/workers/<id>/requirements.txt`, never in `requirements/base.txt` (the gateway image has no torch). `compel` and `peft` are installed only for sdxl-base and sd35-medium.
- `tests/conftest.py` fakes and `tests/test_remote_e2e.py` (fake worker over ASGITransport) must keep covering the gateway side of any change.
