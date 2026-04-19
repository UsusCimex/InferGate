#!/usr/bin/env bash
# Feature test: A1111-style prompt weighting via compel.
#
# Three requests, same seed + steps + scheduler. The only variable is
# how the word "cyberpunk" is weighted in the prompt.
#
#   A) "cyberpunk cat"           — baseline, raw tokenizer path (compel skipped)
#   B) "(cyberpunk:0.3) cat"     — weight way down, should look less cyberpunk
#   C) "(cyberpunk:1.8) cat"     — weight way up, should look MORE cyberpunk
#
# Because A contains no weight syntax, the provider uses the raw tokenizer
# route → A's output is the pre-compel baseline. B and C go through compel
# and should both differ from A *and* from each other.
#
# Run from project root: bash scripts/feature/token-weighting.sh
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

log "Rebuilding gateway + worker (provider + requirements changed) …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

COMMON='"seed":7,"num_inference_steps":20,"scheduler":"dpm++_2m"'
OUT_A="feature_weight_A_baseline.png"
OUT_B="feature_weight_B_low.png"
OUT_C="feature_weight_C_high.png"

run_variant() {
    local label="$1" prompt="$2" out="$3"
    log "[$label] prompt=$prompt"
    # fire_image_request escapes the prompt into the JSON template itself;
    # we need the literal parens/colons, so use a curl invocation directly.
    local resp
    resp=$(mktemp --suffix=.json)
    local t0=$SECONDS
    local code
    code=$(curl -s -o "$resp" -w '%{http_code}' \
        -X POST http://localhost:8000/v1/images/generations \
        -H 'Content-Type: application/json' \
        -H 'X-InferGate-No-Cache: true' \
        -d "$(printf '{"model":"%s","prompt":"%s",%s}' "$MODEL_ID" "$prompt" "$COMMON")" \
        || echo 000)
    local elapsed=$(( SECONDS - t0 ))

    if [[ "$code" == "200" ]]; then
        decode_b64_png "$resp" "$out" || cp "$resp" "$out"
    else
        cp "$resp" "$out"
    fi
    rm -f "$resp"

    local sz
    sz=$(wc -c < "$out" 2>/dev/null || echo 0)
    echo "  HTTP $code, ${elapsed}s → $out ($sz bytes)"

    # Export these so the caller can check them
    LAST_CODE=$code
}

run_variant A "cyberpunk cat"         "$OUT_A"; CODE_A=$LAST_CODE
run_variant B "(cyberpunk:0.3) cat"   "$OUT_B"; CODE_B=$LAST_CODE
run_variant C "(cyberpunk:1.8) cat"   "$OUT_C"; CODE_C=$LAST_CODE

echo
echo "─── Feature applied? ─────────────────────────────────────"
if [[ "$CODE_A" != "200" || "$CODE_B" != "200" || "$CODE_C" != "200" ]]; then
    err "One of the requests failed (HTTP A=$CODE_A B=$CODE_B C=$CODE_C)."
    exit 1
fi

fail=0
check_diff() {
    if cmp -s "$1" "$2"; then
        err "IDENTICAL — $1 ≡ $2. Weighting not applied."
        fail=1
    else
        ok "differ — $1 ≠ $2"
    fi
}

check_diff "$OUT_A" "$OUT_B"
check_diff "$OUT_A" "$OUT_C"
check_diff "$OUT_B" "$OUT_C"

echo
if (( fail )); then
    err "FAIL — compel weighting did not change the output for at least one variant."
    echo "  → Verify compel installed in worker image:"
    echo "      docker compose -f deploy/docker-compose.yml exec $SERVICE pip show compel"
    echo "  → Check worker logs for 'Compel initialised for $MODEL_ID' line"
    exit 1
fi

ok "PASS — weight 0.3 / 1.8 produce distinct images from the baseline."
echo
echo "Open all three side-by-side:"
echo "  $OUT_A — 'cyberpunk cat' (plain, baseline, tokenizer route)"
echo "  $OUT_B — '(cyberpunk:0.3) cat' (compel, cyberpunk attenuated)"
echo "  $OUT_C — '(cyberpunk:1.8) cat' (compel, cyberpunk amplified)"
echo
echo "Visual cue: C should look more aggressively cyberpunk than A; B less."
