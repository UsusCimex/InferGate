#!/usr/bin/env bash
# Feature test: per-request scheduler swap via `scheduler` kwarg.
#
# Same seed, same prompt, same steps — three different schedulers.
# If two images come back byte-identical, the swap is not applying.
# (Note: euler is sdxl-base's own default, so the baseline [unset] and
# "euler" variants should be byte-identical — that's expected and useful
# as a cross-check.)
#
# Run from project root: bash scripts/feature/scheduler-swap.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sdxl-base"
SERVICE="worker-sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
READY_TIMEOUT=600

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Rebuilding gateway + worker (provider code changed) …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

# Use seed=42 + modest steps=20 for all three so timing differences reflect
# scheduler choice, not step count.
COMMON='"seed":42,"num_inference_steps":20'
PROMPT="a cyberpunk cat"

run_variant() {
    local label="$1" extra="$2" out="$3"
    log "[$label] $extra"
    fire_image_request "$MODEL_ID" "$PROMPT" "$out" "${COMMON},${extra}"
    echo "  HTTP $HTTP_CODE, ${ELAPSED}s → $out ($OUTPUT_SIZE bytes)"
}

run_variant "euler_a"  '"scheduler":"euler_a"'           feature_sched_euler_a.png
run_variant "dpm++_2m" '"scheduler":"dpm++_2m"'          feature_sched_dpm.png
run_variant "ddim"     '"scheduler":"ddim"'              feature_sched_ddim.png

echo
echo "─── Feature applied? ─────────────────────────────────────"
FAIL=0
pair_differs() {
    local a="$1" b="$2"
    if cmp -s "$a" "$b"; then
        err "IDENTICAL — $a ≡ $b. Swap didn't take effect for one or both."
        FAIL=1
    else
        ok "differ — $a ≠ $b"
    fi
}

pair_differs feature_sched_euler_a.png feature_sched_dpm.png
pair_differs feature_sched_euler_a.png feature_sched_ddim.png
pair_differs feature_sched_dpm.png    feature_sched_ddim.png

echo
if (( FAIL )); then
    err "FAIL — at least one scheduler swap did not apply."
    echo "  → Check _maybe_swap_scheduler in app/providers/image/diffusers_provider.py"
    echo "  → Check body.scheduler is forwarded in app/routers/images.py"
    exit 1
fi

ok "PASS — all three schedulers produce distinct PNGs."
echo "Open feature_sched_*.png side-by-side to see how sampler choice"
echo "affects the same-seed same-prompt generation."

# Sanity: unknown name → gateway returns 500 (worker raises ValueError;
# proper end-to-end error forwarding is a separate task, see README TODO).
log "Error path: unknown scheduler 'nonsense' should produce HTTP 500 …"
RESP=$(mktemp --suffix=.json)
CODE=$(curl -s -o "$RESP" -w '%{http_code}' \
    -X POST http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"x\",\"scheduler\":\"nonsense\"}" || echo 000)
if [[ "$CODE" == "500" ]]; then
    ok "unknown scheduler → HTTP 500 (worker log should contain ValueError)"
else
    err "unknown scheduler → unexpected HTTP $CODE, body: $(cat "$RESP")"
fi
rm -f "$RESP"
