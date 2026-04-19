#!/usr/bin/env bash
# Launch + smoke-test z-image-turbo — 6B DiT, 8-step distilled, Apache-2.0.
# Run from project root: bash scripts/diagnose/z-image-turbo.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-z-image-turbo"
MODEL_ID="z-image-turbo"
HF_CACHE="models/models--Tongyi-MAI--Z-Image-Turbo"
OUTPUT="z_image_diag.png"
READY_TIMEOUT=1200

# shellcheck source=_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
update_env "$ENV_FILE" Z_IMAGE_TURBO_CPU_OFFLOAD false
update_env "$ENV_FILE" Z_IMAGE_TURBO_SEQUENTIAL_OFFLOAD false
update_env "$ENV_FILE" Z_IMAGE_TURBO_WARMUP false
if ! grep -q '^Z_IMAGE_TURBO_QUANTIZATION=' "$ENV_FILE"; then
    update_env "$ENV_FILE" Z_IMAGE_TURBO_QUANTIZATION null
fi
ok "Env flags set"

log "Starting $SERVICE (first build pulls diffusers from git — expect 5-10 min) …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT" || {
    echo "  → If OOM: set Z_IMAGE_TURBO_QUANTIZATION=nf4 and re-run."
    echo "  → If ZImagePipeline not found: diffusers git pin may have drifted."
    exit 1
}

log "POST /v1/images/generations (prompt=cyberpunk cat, 1024×1024) …"
fire_image_request "$MODEL_ID" "a cyberpunk cat" "$OUTPUT"
report_result "$SERVICE" "$OUTPUT"

if [[ "$HTTP_CODE" == "200" ]]; then
    echo "  → Subsequent requests should be ~3-8s (8-step distilled)."
fi
