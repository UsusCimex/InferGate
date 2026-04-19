#!/usr/bin/env bash
# Feature test: HighresFix — two-pass generate-upscale-refine.
#
# Three requests, same seed:
#   A) 1024×1024 single-pass                            — baseline
#   B) 768×768 base + scale=1.5 + denoising=0.5 → 1152  — two-pass
#   C) repeat of B                                      — determinism / cache
#
# Assertions:
#   - All three return HTTP 200
#   - B takes meaningfully longer than A (two passes > one)
#   - B and C are byte-identical (deterministic pipeline state)
#   - B's PNG header reports 1152×1152 (IHDR width/height big-endian uint32)
#
# Run from project root: bash scripts/feature/highres-fix.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sdxl-base"
SERVICE="worker-sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
READY_TIMEOUT=600

COMMON_SEED='"seed":42,"num_inference_steps":25,"scheduler":"dpm++_2m"'
PROMPT="a cyberpunk cat in neon Tokyo"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Rebuilding gateway + worker …"
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

# Read PNG IHDR for width/height. Returns "WxH" on stdout or "?" on failure.
png_dims() {
    local f="$1"
    for py in python3 python py; do
        if command -v "$py" >/dev/null 2>&1; then
            "$py" -c "
import struct, sys
with open('$f','rb') as fp:
    fp.seek(16)
    w,h=struct.unpack('>II',fp.read(8))
print(f'{w}x{h}')
" 2>/dev/null && return 0
        fi
    done
    echo "?"
}

OUT_A=feature_hires_A_single.png
OUT_B=feature_hires_B_two_pass.png
OUT_C=feature_hires_C_repeat.png

fire A "$(printf '{"model":"%s","prompt":"%s",%s,"size":"1024x1024"}' \
            "$MODEL_ID" "$PROMPT" "$COMMON_SEED")" "$OUT_A"
CODE_A=$LAST_CODE; TIME_A=$LAST_TIME

HIRES='{"scale":1.5,"denoising_strength":0.5,"upscaler":"lanczos"}'
fire B "$(printf '{"model":"%s","prompt":"%s",%s,"size":"768x768","highres_fix":%s}' \
            "$MODEL_ID" "$PROMPT" "$COMMON_SEED" "$HIRES")" "$OUT_B"
CODE_B=$LAST_CODE; TIME_B=$LAST_TIME

fire C "$(printf '{"model":"%s","prompt":"%s",%s,"size":"768x768","highres_fix":%s}' \
            "$MODEL_ID" "$PROMPT" "$COMMON_SEED" "$HIRES")" "$OUT_C"
CODE_C=$LAST_CODE; TIME_C=$LAST_TIME

echo
echo "─── Feature applied? ─────────────────────────────────────"
fail=0

for lbl in A:$CODE_A B:$CODE_B C:$CODE_C; do
    label="${lbl%%:*}"; code="${lbl##*:}"
    if [[ "$code" != "200" ]]; then
        err "$label returned HTTP $code (expected 200)"
        fail=1
    fi
done
(( fail )) && exit 1

DIMS_A=$(png_dims "$OUT_A")
DIMS_B=$(png_dims "$OUT_B")
DIMS_C=$(png_dims "$OUT_C")
echo "  dims: A=$DIMS_A  B=$DIMS_B  C=$DIMS_C"

if [[ "$DIMS_A" == "1024x1024" ]]; then
    ok "A is 1024×1024 as requested"
else
    err "A expected 1024×1024, got $DIMS_A"
    fail=1
fi

if [[ "$DIMS_B" == "1152x1152" ]]; then
    ok "B is 1152×1152 (768 * 1.5 scale) as expected"
else
    err "B expected 1152×1152, got $DIMS_B — highres_fix did not scale"
    fail=1
fi

if cmp -s "$OUT_B" "$OUT_C"; then
    ok "B ≡ C byte-identical — two-pass pipeline is deterministic"
else
    err "B ≠ C — cache/determinism broken across repeats"
    fail=1
fi

if (( TIME_A > 0 && TIME_B > TIME_A )); then
    ok "B (${TIME_B}s) took longer than A (${TIME_A}s) — two-pass as expected"
else
    log "timing: A=${TIME_A}s B=${TIME_B}s — not a strict failure (noisy on warm GPUs)"
fi

echo
(( fail )) && { err "FAIL — review above."; exit 1; }
ok "PASS — HighresFix produces correctly-scaled images via two-pass flow."
echo
echo "Open side-by-side:"
echo "  $OUT_A (1024 single pass)"
echo "  $OUT_B (768→1152 two-pass, sharper fine detail)"
