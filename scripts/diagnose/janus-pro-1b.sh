#!/usr/bin/env bash
# Launch + smoke-test Janus-Pro-1B — DeepSeek's autoregressive T2I model.
# Native 384×384, ~3-4GB VRAM at bf16, MIT. First non-diffusion provider.
# Run from project root: bash scripts/diagnose/janus-pro-1b.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-janus-pro-1b"
MODEL_ID="janus-pro-1b"
HF_CACHE="models/models--deepseek-ai--Janus-Pro-1B"
OUTPUT="janus_diag.png"
READY_TIMEOUT=900

# shellcheck source=_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Starting $SERVICE (first build pulls janus from git — expect 5-10 min) …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

log "POST /v1/images/generations (prompt=cyberpunk cat) — native 384×384 …"
fire_image_request "$MODEL_ID" "a cyberpunk cat" "$OUTPUT"
report_result "$SERVICE" "$OUTPUT"

if [[ "$HTTP_CODE" == "200" ]]; then
    echo "  → 384×384 native (no upscaler). AR samples 576 tokens sequentially,"
    echo "    so latency is fixed regardless of step count."
fi
