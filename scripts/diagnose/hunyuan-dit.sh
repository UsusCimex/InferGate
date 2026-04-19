#!/usr/bin/env bash
# Launch + bilingual smoke-test Hunyuan-DiT v1.2 — Tencent's 1.5B DiT with
# native Chinese + English prompt support. ~7GB VRAM, no offload/quant.
# Gated: accept terms at https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers
# Run from project root: bash scripts/diagnose/hunyuan-dit.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-hunyuan-dit"
MODEL_ID="hunyuan-dit"
HF_CACHE="models/models--Tencent-Hunyuan--HunyuanDiT-v1.2-Diffusers"
OUTPUT_EN="hunyuan_diag_en.png"
OUTPUT_ZH="hunyuan_diag_zh.png"
READY_TIMEOUT=600

# shellcheck source=_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
update_env "$ENV_FILE" HUNYUAN_DIT_CPU_OFFLOAD false
update_env "$ENV_FILE" HUNYUAN_DIT_SEQUENTIAL_OFFLOAD false
update_env "$ENV_FILE" HUNYUAN_DIT_WARMUP false
ok "Env flags set"

log "Starting $SERVICE …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT" || {
    echo "  → If GatedRepoError: accept terms at"
    echo "    https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
    exit 1
}

echo
echo "─── Bilingual test ────────────────────────────────────────"
log "[EN] POST /v1/images/generations …"
fire_image_request "$MODEL_ID" "a cyberpunk cat in neon Tokyo" "$OUTPUT_EN"
echo "  English: HTTP $HTTP_CODE, ${ELAPSED}s → $OUTPUT_EN ($OUTPUT_SIZE bytes)"

log "[ZH] POST /v1/images/generations …"
fire_image_request "$MODEL_ID" "赛博朋克风格的猫，霓虹灯下的东京" "$OUTPUT_ZH"
echo "  Chinese: HTTP $HTTP_CODE, ${ELAPSED}s → $OUTPUT_ZH ($OUTPUT_SIZE bytes)"
echo

ok "Done. Hunyuan-DiT's key differentiator is native Chinese handling —"
echo "  compare $OUTPUT_EN vs $OUTPUT_ZH for semantic accuracy, not just shape."
