#!/usr/bin/env bash
# Feature test: per-request tunables (negative_prompt, num_inference_steps,
# guidance_scale) reach the provider and affect generation.
#
# Approach: fire two requests against sdxl-base with the same seed — one
# using YAML defaults, one with aggressive overrides (10 steps, cfg=3.0,
# negative prompt). If the PNGs come back byte-identical, the overrides
# never reached the pipeline → feature broken.
#
# Run from project root: bash scripts/feature/per-request-tunables.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sdxl-base"
SERVICE="worker-sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
READY_TIMEOUT=600
BASELINE="feature_tunables_baseline.png"
OVERRIDE="feature_tunables_override.png"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Rebuilding gateway (schema + router changed) and (re)starting $SERVICE …"
"${COMPOSE[@]}" build gateway
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

log "[A] baseline — YAML defaults (30 steps, cfg=7.0), seed=42"
fire_image_request "$MODEL_ID" "a cyberpunk cat" "$BASELINE" '"seed":42'
A_CODE=$HTTP_CODE
A_TIME=$ELAPSED
echo "  HTTP $A_CODE, ${A_TIME}s → $BASELINE ($OUTPUT_SIZE bytes)"

log "[B] override — 10 steps, cfg=3.0, negative_prompt, seed=42"
EXTRA='"seed":42,"negative_prompt":"ugly, distorted, low quality, blurry","num_inference_steps":10,"guidance_scale":3.0'
fire_image_request "$MODEL_ID" "a cyberpunk cat" "$OVERRIDE" "$EXTRA"
B_CODE=$HTTP_CODE
B_TIME=$ELAPSED
echo "  HTTP $B_CODE, ${B_TIME}s → $OVERRIDE ($OUTPUT_SIZE bytes)"

echo
echo "─── Feature applied? ─────────────────────────────────────"
if [[ "$A_CODE" != "200" || "$B_CODE" != "200" ]]; then
    err "One of the requests failed (HTTP $A_CODE, $B_CODE). Aborting verdict."
    exit 1
fi

if cmp -s "$BASELINE" "$OVERRIDE"; then
    err "FAIL — $BASELINE and $OVERRIDE are byte-identical."
    echo "  → Override params did not reach the provider. Check:"
    echo "      • Gateway was rebuilt with the new schema (it should have been)"
    echo "      • app/routers/images.py passes new fields into params dict"
    echo "      • DiffusersImageProvider.generate merges params over YAML defaults"
    exit 1
fi

ok "PASS — images differ. Per-request overrides reached the provider."

# Heuristic speed check: override is 10 steps vs baseline 30 steps, should
# be ~3× faster (ignoring fixed overhead of text encode + VAE decode).
if (( B_TIME > 0 && A_TIME > B_TIME * 3 / 2 )); then
    ok "Speed check: override (${B_TIME}s) meaningfully faster than baseline (${A_TIME}s)."
else
    log "Speed hint: override was ${B_TIME}s vs baseline ${A_TIME}s."
    log "  If similar, num_inference_steps may not have applied even though images differ."
fi

echo
echo "Open side-by-side to verify visually:"
echo "  $BASELINE — full 30-step generation, standard CFG"
echo "  $OVERRIDE — 10-step, low CFG, with negative prompt"
