#!/usr/bin/env bash
# Launch + smoke-test Meissonic — masked non-autoregressive T2I, 1B params,
# 1024×1024, 64 steps, Apache-2.0. Pipeline code is vendored at build time
# (git clone of github.com/viiika/Meissonic → /app/_meissonic).
# Run from project root: bash scripts/diagnose/meissonic.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-meissonic"
MODEL_ID="meissonic"
HF_CACHE="models/models--MeissonFlow--Meissonic"
OUTPUT="meissonic_diag.png"
READY_TIMEOUT=900

# shellcheck source=_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Starting $SERVICE (first build clones viiika/Meissonic + downloads ~9GB weights) …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT" || {
    echo "  → If 'Meissonic source not found': POST_INSTALL git clone failed."
    echo "    Check worker build logs for 'git clone' errors."
    echo "  → If OOM: Meissonic needs ~10GB at fp16 for 1024². Close other"
    echo "    GPU users or set MEISSONIC_WIDTH/HEIGHT to 512."
    exit 1
}

log "POST /v1/images/generations (prompt=cyberpunk cat, 1024×1024, 64 steps) …"
fire_image_request "$MODEL_ID" "a cyberpunk cat in neon Tokyo" "$OUTPUT"
report_result "$SERVICE" "$OUTPUT"

if [[ "$HTTP_CODE" == "200" ]]; then
    echo "  → Meissonic is masked-token (not diffusion, not AR) — 64 steps"
    echo "    unmask image tokens in parallel batches. Compare against SD35/SDXL"
    echo "    for a qualitatively different failure mode on hard prompts."
fi
