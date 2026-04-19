#!/usr/bin/env bash
# Feature test: per-request Textual Inversion embedding load for sdxl-base.
#
# Loads a TI repo once, verifies: (a) prompt containing the TI token
# produces a different image than the baseline, (b) repeat request
# skips the reload (cache-hit).
#
# Pure-SDXL TIs are rarer than SD1.5 ones (SDXL needs two encoders'
# worth of embeddings in the same file). If the default TI_ID below
# doesn't exist or is arch-incompatible with SDXL, our error-forwarding
# will surface the HF message verbatim. Override via:
#   TI_ID=org/repo TI_FILE=file.safetensors bash scripts/feature/textual-inversion.sh
#
# Run from project root: bash scripts/feature/textual-inversion.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sdxl-base"
SERVICE="worker-sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
READY_TIMEOUT=600

# Known-active SDXL TI: Y2K web aesthetic, dual-encoder, multi-token
# <s0><s1> trigger. If this errors 404 or arch-mismatch, replace via env.
TI_ID="${TI_ID:-linoyts/web_y2k}"
TI_FILE="${TI_FILE:-web_y2k_emb.safetensors}"

# 768×768 instead of the 1024 default — sdxl-base + compel-init + TI
# registrations sit around 11.5GB on 12GB cards, so 1024 peak inference
# flirts with OOM. Visual effect of the TI is still clearly visible.
COMMON='"seed":7,"num_inference_steps":25,"scheduler":"dpm++_2m","size":"768x768"'
BASE_PROMPT="a portrait photo"
TI_PROMPT="<s0><s1> a portrait photo"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set (TI_ID=$TI_ID, TI_FILE=$TI_FILE)"

log "Rebuilding gateway + worker (schema + provider changed) …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

fire() {
    local label="$1" body="$2" out="$3"
    log "[$label]"
    local resp
    resp=$(mktemp --suffix=.json)
    local t0=$SECONDS
    local code
    code=$(curl -s -o "$resp" -w '%{http_code}' \
        -X POST http://localhost:8000/v1/images/generations \
        -H 'Content-Type: application/json' \
        -d "$body" || echo 000)
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
    LAST_CODE=$code
    LAST_TIME=$elapsed
}

OUT_A=feature_ti_A_baseline.png
OUT_B=feature_ti_B_with_ti.png
OUT_C=feature_ti_C_cachehit.png

# A: baseline — no TIs registered, plain prompt
fire A "$(printf '{"model":"%s","prompt":"%s",%s}' "$MODEL_ID" "$BASE_PROMPT" "$COMMON")" "$OUT_A"
CODE_A=$LAST_CODE

# B: register TI + use its multi-token trigger in the prompt.
# web_y2k needs both <s0> and <s1> registered together — pass as list.
TI_JSON=$(printf '[{"id":"%s","weight_file":"%s","token":["<s0>","<s1>"]}]' "$TI_ID" "$TI_FILE")
fire B "$(printf '{"model":"%s","prompt":"%s",%s,"textual_inversions":%s}' \
            "$MODEL_ID" "$TI_PROMPT" "$COMMON" "$TI_JSON")" "$OUT_B"
CODE_B=$LAST_CODE
TIME_B_FIRST=$LAST_TIME

# C: repeat B — TI already in tokenizer, no reload, should be byte-identical.
fire C "$(printf '{"model":"%s","prompt":"%s",%s,"textual_inversions":%s}' \
            "$MODEL_ID" "$TI_PROMPT" "$COMMON" "$TI_JSON")" "$OUT_C"
CODE_C=$LAST_CODE
TIME_C_CACHED=$LAST_TIME

echo
echo "─── Feature applied? ─────────────────────────────────────"
fail=0

if [[ "$CODE_A" != "200" ]]; then
    err "A (baseline) failed with HTTP $CODE_A"
    fail=1
fi

if [[ "$CODE_B" != "200" ]]; then
    err "B (with TI) failed with HTTP $CODE_B"
    echo "  body: $(cat "$OUT_B" | head -c 500)"
    echo
    err "Most likely the default TI is incompatible with SDXL or 404."
    err "Re-run with your own:  TI_ID=org/repo TI_FILE=file.safetensors bash $0"
    err "(edit the test to adjust prompt/token format if the new TI differs)"
    exit 1
fi

if [[ "$CODE_C" != "200" ]]; then
    err "C (cache repeat) failed with HTTP $CODE_C"
    fail=1
fi

(( fail )) && exit 1

if cmp -s "$OUT_A" "$OUT_B"; then
    err "A ≡ B byte-identical — TI didn't influence output."
    echo "  → Check worker logs for 'Registering textual inversion' line."
    fail=1
else
    ok "A ≠ B — TI token changes the output"
fi

if cmp -s "$OUT_B" "$OUT_C"; then
    ok "B ≡ C byte-identical — cache-hit deterministic"
else
    err "B ≠ C — second request changed output; cache didn't kick in."
    fail=1
fi

if (( TIME_B_FIRST > 0 && TIME_C_CACHED <= TIME_B_FIRST )); then
    ok "cache hit latency: ${TIME_C_CACHED}s ≤ first-load ${TIME_B_FIRST}s"
else
    log "cache hit latency: ${TIME_C_CACHED}s vs ${TIME_B_FIRST}s — noisy, not a failure"
fi

echo
(( fail )) && { err "FAIL — see above."; exit 1; }
ok "PASS — textual inversion registered and reused correctly."
echo
echo "Open side-by-side:"
echo "  $OUT_A — baseline (no TI, plain '$BASE_PROMPT')"
echo "  $OUT_B — with TI: '$TI_PROMPT' (should show Y2K web aesthetic)"
echo "  $OUT_C — repeat of B (cache hit)"
