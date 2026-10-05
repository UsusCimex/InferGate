---
paths:
  - "app/routers/**"
  - "app/schemas/**"
  - "app/main.py"
  - "app/middleware/**"
---

# Public API

- Request schemas use `extra="forbid"` with `Field` constraints; an unknown field must stay a 422 that names it.
- Errors use the envelope `{"error": {"message", "type", ...}}` with a `type` from the table in `docs/api.md`; give new errors a type too.
- Keep OpenAI compatibility for the OpenAI-shaped routes; extensions get their own fields or routes.
- Cached endpoints follow the same sequence: `X-InferGate-No-Cache` → `should_cache` → get/HIT → miss metrics → `scheduler.submit` inside `manager.active_request` → histogram → put → `X-InferGate-*` headers. Cache keys must include every parameter that changes the output.
- Capability checks (`vision`, `voices`, `voice_clone_only`) run before `ensure_loaded`, so a bad request never loads a model.
- PictoLex reads `X-InferGate-Cache`, `-Model`, `-Generation-Ms`, `-Queue-Position`, `-Load-Ms`, `-Inference-Ms` and sends `X-InferGate-No-Cache`; keep their names and formats.
- Document new routes, fields and headers in `docs/api.md`.
