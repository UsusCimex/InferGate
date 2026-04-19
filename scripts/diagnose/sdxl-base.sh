#!/usr/bin/env bash
# Launch + smoke-test sdxl-base — 6.6GB UNet, no offload/quant.
# Run from project root: bash scripts/diagnose/sdxl-base.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-sdxl-base"
MODEL_ID="sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
OUTPUT="sdxl_diag.png"
READY_TIMEOUT=600

# shellcheck source=_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Starting $SERVICE …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

log "POST /v1/images/generations (prompt=cyberpunk cat, 1024×1024) …"
fire_image_request "$MODEL_ID" "a cyberpunk cat" "$OUTPUT"
report_result "$SERVICE" "$OUTPUT"

if [[ "$HTTP_CODE" == "200" ]]; then
    echo "  → Subsequent requests should be ~15-20s (30 steps, UNet)."
fi
