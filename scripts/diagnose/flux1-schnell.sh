#!/usr/bin/env bash
# Launch + smoke-test flux1-schnell with nf4 quantization (no offload).
# Run from project root: bash scripts/diagnose/flux1-schnell.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-flux1-schnell"
MODEL_ID="flux1-schnell"
HF_CACHE="models/models--black-forest-labs--FLUX.1-schnell"
OUTPUT="flux_diag.png"
READY_TIMEOUT=900

# shellcheck source=_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
update_env "$ENV_FILE" FLUX1_SCHNELL_QUANTIZATION nf4
update_env "$ENV_FILE" FLUX1_SCHNELL_CPU_OFFLOAD false
update_env "$ENV_FILE" FLUX1_SCHNELL_SEQUENTIAL_OFFLOAD false
update_env "$ENV_FILE" FLUX1_SCHNELL_WARMUP false
ok "Env flags set (nf4 + no offload)"

log "Starting $SERVICE …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

log "POST /v1/images/generations (prompt=cyberpunk cat, 512×512) …"
fire_image_request "$MODEL_ID" "a cyberpunk cat" "$OUTPUT" '"size":"512x512"'
report_result "$SERVICE" "$OUTPUT"

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "  → If you see sigmas[sigma_idx+1] IndexError, it's a FLUX+scheduler"
    echo "    bug independent of our stack. nf4 path bypasses scheduler offload."
fi
