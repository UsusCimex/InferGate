---
paths:
  - "config/**"
  - "deploy/**"
  - "requirements/**"
---

# Model configs and deployment

- A model that reuses a provider needs five edits: `config/models/<id>.yaml`, `deploy/workers/<id>/requirements.txt`, a row in `deploy/docker-bake.hcl`, a `worker-<id>` service (dots → dashes) with profile `<id>` in `deploy/docker-compose.yml`, and `WORKER_URL_<ID>` on the `gateway` service. Missing the last one makes the gateway load the model in-process, where torch is absent.
- Hardware-dependent values are `${oc.env:<ID>_<FIELD>,default}` / `${oc.decode:${oc.env:...}}` (ID upper-cased, every non-alphanumeric character → `_`); keep the defaults safe for a 12 GB card.
- No comments in model YAMLs or `pyproject.toml`; capabilities go in the `capabilities` block, client-facing tags (`voice-clone`) in `metadata.tags`.
- `deploy/.env` holds `HF_TOKEN`: never print or commit it. After changing it, recreate the affected worker.
- Python under `app/` is baked into the images: rebuild the worker image (`docker buildx bake -f deploy/docker-bake.hcl worker-<id>`) after code changes. YAMLs and weights are bind-mounted.
- Keep `docs/models.md` (catalog and profiles) and `docs/configuration.md` in step with these files.
