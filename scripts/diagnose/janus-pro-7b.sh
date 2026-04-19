#!/usr/bin/env bash
# Launch + smoke-test Janus-Pro-7B with NF4 quantization.
# 7B bf16 weights = ~14GB → won't fit 12GB. NF4 brings weights to ~5GB,
# total runtime footprint ~7-8GB. Much better quality than 1B at same
# 384×384 resolution.
# Run from project root: bash scripts/diagnose/janus-pro-7b.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-janus-pro-7b"
MODEL_ID="janus-pro-7b"
HF_CACHE="models/models--deepseek-ai--Janus-Pro-7B"
OUTPUT="janus7b_diag.png"
READY_TIMEOUT=1800  # first run downloads 14GB of bf16 weights

# shellcheck source=_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
update_env "$ENV_FILE" JANUS_PRO_7B_QUANTIZATION nf4
ok "Env flags set (NF4 quantization on)"

log "Starting $SERVICE (first build + 14GB weights download — expect 15-30 min) …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT" || {
    echo "  → If OOM: lower JANUS_PRO_7B_MAX_CONCURRENT or stay on janus-pro-1b."
    echo "  → If bnb error: confirm build-essential installed in worker image."
    exit 1
}

log "POST /v1/images/generations (prompt=cyberpunk cat) — native 384×384 …"
fire_image_request "$MODEL_ID" "a cyberpunk cat" "$OUTPUT"
report_result "$SERVICE" "$OUTPUT"

if [[ "$HTTP_CODE" == "200" ]]; then
    echo "  → Compare $OUTPUT with janus_diag.png (1B) — 7B should be"
    echo "    noticeably better at prompt fidelity + anatomy."
    echo "  → NF4 costs ~5-10% quality vs bf16 but saves ~9GB VRAM."
fi
