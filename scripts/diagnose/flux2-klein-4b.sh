#!/usr/bin/env bash
# Launch + smoke-test flux2-klein-4b. Note: text encoder is large (Qwen2.5-
# VL-class), so even with a 4B transformer this model needs ~22GB without
# quantization. On 12GB cards expect heavy spillover to system RAM → slow.
# Run from project root: bash scripts/diagnose/flux2-klein-4b.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-flux2-klein-4b"
MODEL_ID="flux2-klein-4b"
HF_CACHE="models/models--black-forest-labs--FLUX.2-klein-4B"
OUTPUT="flux2_diag.png"
READY_TIMEOUT=900

# shellcheck source=_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
update_env "$ENV_FILE" FLUX2_KLEIN_4B_CPU_OFFLOAD false
update_env "$ENV_FILE" FLUX2_KLEIN_4B_SEQUENTIAL_OFFLOAD false
update_env "$ENV_FILE" FLUX2_KLEIN_4B_WARMUP false
ok "Env flags set"

log "Starting $SERVICE …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

log "POST /v1/images/generations (prompt=cyberpunk cat, 1024×1024) …"
fire_image_request "$MODEL_ID" "a cyberpunk cat" "$OUTPUT"
report_result "$SERVICE" "$OUTPUT"

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "  → If OOM or very slow: enable nf4 quantization via"
    echo "      FLUX2_KLEIN_4B_QUANTIZATION=nf4"
    echo "    and add bitsandbytes + build-essential to the worker."
fi
