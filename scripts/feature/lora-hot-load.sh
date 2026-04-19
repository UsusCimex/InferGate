#!/usr/bin/env bash
# Feature test: per-request LoRA hot-load for sdxl-base.
#
# Scenario covers the four cases that matter:
#   A) baseline           — no loras, standard output
#   B) single lora 1.0    — one adapter at full strength
#   C) single lora 0.0    — same adapter at zero → should roughly match A
#   D) multi lora         — two adapters chained (tests set_adapters list path)
#   E) cache hit          — repeat B; should NOT re-download (watch logs)
#   F) error path         — bogus repo id → HTTP 400 with clear message
#
# LoRA used: ostris/crayon_style_lora_sdxl — ~200MB, visibly transforms SDXL
# outputs into crayon drawings. Obvious effect makes tests decisive.
# Second LoRA for multi: artificialguybr/PixelArtRedmond-V2 — pixel-art style.
#
# Run from project root: bash scripts/feature/lora-hot-load.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sdxl-base"
SERVICE="worker-sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
READY_TIMEOUT=600

LORA_1="ostris/crayon_style_lora_sdxl"
LORA_2="nerijs/pixel-art-xl"

COMMON='"seed":99,"num_inference_steps":20,"scheduler":"dpm++_2m"'
PROMPT="a cyberpunk cat"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Rebuilding gateway + worker (schema + provider + requirements changed) …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

# Helper: fire a request with a raw JSON body (for LoRA arrays).
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
        -H 'X-InferGate-No-Cache: true' \
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

OUT_A=feature_lora_A_baseline.png
OUT_B=feature_lora_B_crayon.png
OUT_C=feature_lora_C_zero.png
OUT_D=feature_lora_D_multi.png
OUT_E=feature_lora_E_cache.png

fire A "$(printf '{"model":"%s","prompt":"%s",%s}' "$MODEL_ID" "$PROMPT" "$COMMON")"          "$OUT_A"
CODE_A=$LAST_CODE

fire B "$(printf '{"model":"%s","prompt":"%s",%s,"loras":[{"id":"%s","weight":1.0}]}' \
            "$MODEL_ID" "$PROMPT" "$COMMON" "$LORA_1")"                                       "$OUT_B"
CODE_B=$LAST_CODE; TIME_B_FIRST=$LAST_TIME

fire C "$(printf '{"model":"%s","prompt":"%s",%s,"loras":[{"id":"%s","weight":0.0}]}' \
            "$MODEL_ID" "$PROMPT" "$COMMON" "$LORA_1")"                                       "$OUT_C"
CODE_C=$LAST_CODE

fire D "$(printf '{"model":"%s","prompt":"%s",%s,"loras":[{"id":"%s","weight":0.6},{"id":"%s","weight":0.4}]}' \
            "$MODEL_ID" "$PROMPT" "$COMMON" "$LORA_1" "$LORA_2")"                             "$OUT_D"
CODE_D=$LAST_CODE

fire E "$(printf '{"model":"%s","prompt":"%s",%s,"loras":[{"id":"%s","weight":1.0}]}' \
            "$MODEL_ID" "$PROMPT" "$COMMON" "$LORA_1")"                                       "$OUT_E"
CODE_E=$LAST_CODE; TIME_E_CACHED=$LAST_TIME

echo
echo "─── Feature applied? ─────────────────────────────────────"
fail=0
check_status() {
    local label="$1" code="$2"
    if [[ "$code" != "200" ]]; then
        err "$label returned HTTP $code (expected 200)"
        fail=1
    fi
}
check_status A "$CODE_A"
check_status B "$CODE_B"
check_status C "$CODE_C"
check_status D "$CODE_D"
check_status E "$CODE_E"
(( fail )) && exit 1

check_diff() {
    local a="$1" b="$2" rel="$3"
    if cmp -s "$a" "$b"; then
        if [[ "$rel" == "expect_same" ]]; then
            ok "same bytes — $a ≡ $b (weight=0 effectively a no-op)"
        else
            err "UNEXPECTED same bytes — $a ≡ $b ($rel)"
            fail=1
        fi
    else
        if [[ "$rel" == "expect_diff" ]]; then
            ok "differ — $a ≠ $b"
        else
            err "UNEXPECTED differ — $a ≠ $b ($rel)"
            fail=1
        fi
    fi
}

# B: LoRA at weight 1.0 must change output substantially vs baseline
check_diff "$OUT_A" "$OUT_B" expect_diff
# D: multi-LoRA must differ from both baseline and single-LoRA
check_diff "$OUT_A" "$OUT_D" expect_diff
check_diff "$OUT_B" "$OUT_D" expect_diff
# E: cache hit should reproduce exactly the same bytes as B (same seed, same
# adapter loaded, same weight, scheduler already warmed)
check_diff "$OUT_B" "$OUT_E" expect_same

echo
if (( TIME_B_FIRST > 0 && TIME_E_CACHED <= TIME_B_FIRST )); then
    ok "cache hit latency: ${TIME_E_CACHED}s ≤ first-load ${TIME_B_FIRST}s (as expected)"
else
    log "cache hit latency: ${TIME_E_CACHED}s vs first-load ${TIME_B_FIRST}s — noisy on tiny LoRAs, not a failure"
fi

log "[F] Error path: bogus LoRA repo → HTTP 400 + clean message"
RESP=$(mktemp --suffix=.json)
CODE=$(curl -s -o "$RESP" -w '%{http_code}' \
    -X POST http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"%s","prompt":"x","loras":[{"id":"this-org/does-not-exist","weight":1.0}]}' "$MODEL_ID")" \
    || echo 000)
echo "  HTTP $CODE body: $(cat "$RESP" | head -c 300)"
if [[ "$CODE" == "400" ]] && grep -q "Failed to load LoRA" "$RESP"; then
    ok "bogus LoRA → 400 with 'Failed to load LoRA' message"
else
    err "bogus LoRA → HTTP $CODE (expected 400 with structured message)"
    fail=1
fi
rm -f "$RESP"

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }

ok "PASS — LoRA hot-load works end-to-end."
echo
echo "Open side-by-side to inspect:"
echo "  $OUT_A — baseline (no LoRA)"
echo "  $OUT_B — crayon LoRA @ 1.0  (should look drawn in crayon)"
echo "  $OUT_C — crayon LoRA @ 0.0  (LoRA loaded but inactive)"
echo "  $OUT_D — crayon @ 0.6 + pixel-art @ 0.4 (blended style)"
echo "  $OUT_E — repeat of B (cache hit — same bytes as B)"
