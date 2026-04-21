#!/usr/bin/env bash
# Feature test: img2img + inpainting via base64 on sdxl-base.
#
# What this verifies end-to-end (with a real UNet, not a fake provider):
#   A) baseline text2img          — standard 512×512 generation
#   B) img2img (denoising=0.5)    — uses A as the `image` input
#   C) inpaint with centred mask  — uses A + a programmatic mask
#   D) repeat of B (cache hit)    — same seed, same input → byte-equal
#   E) error path                 — mask without image → HTTP 422
#
# Scheduler (dpm++_2m) + seed are pinned across runs so B ≡ D is
# deterministic — this doubles as a regression check for the seed-fix
# (generator now actually drives reproducibility) and the img2img pipe
# cache (AutoPipelineForImage2Image.from_pipe reuse).
#
# Mask construction: a 256×256 grayscale PNG with a white circle, built
# client-side (pure PIL). Using `sdxl-base` 512×512 avoids the 1024 bucket
# stress on 12GB cards; both A and the masks are resized internally if
# they don't match.
#
# Run from project root: bash scripts/feature/img2img.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sdxl-base"
SERVICE="worker-sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
READY_TIMEOUT=600

COMMON='"seed":17,"num_inference_steps":20,"scheduler":"dpm++_2m","size":"512x512"'
PROMPT_BASE="a cyberpunk cat in neon Tokyo"
PROMPT_EDIT="the same cat, now in a sunflower field at noon"
PROMPT_INPAINT="a tiny red balloon"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

command -v curl >/dev/null 2>&1 || { err "curl required"; exit 1; }

PY=""
for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[[ -n "$PY" ]] || { err "python required (for base64-encoding the input image and building the mask)"; exit 1; }

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Rebuilding gateway + worker (schema/provider changed) …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

# ── Helper: POST /v1/images/generations with body in a file ───────────
# (request bodies get long once base64 images land in there — a file is
# less fragile than stitching with printf or shell escaping.)
fire_with_body_file() {
    local label="$1" body_file="$2" out="$3"
    log "[$label]"
    local resp
    resp=$(mktemp --suffix=.json)
    local t0=$SECONDS
    local code
    code=$(curl -s -o "$resp" -w '%{http_code}' \
        -X POST http://localhost:8000/v1/images/generations \
        -H 'Content-Type: application/json' \
        -H 'X-InferGate-No-Cache: true' \
        --data-binary "@${body_file}" \
        || echo 000)
    local elapsed=$(( SECONDS - t0 ))

    if [[ "$code" == "200" ]]; then
        decode_b64_png "$resp" "$out" || cp "$resp" "$out"
    else
        cp "$resp" "$out"
    fi

    local sz
    sz=$(wc -c < "$out" 2>/dev/null || echo 0)
    echo "  HTTP $code, ${elapsed}s → $out ($sz bytes)"
    if [[ "$code" != "200" ]]; then
        echo "  ── Response body ─────────────────────────────────────"
        sed 's/^/  /' "$resp"
        echo
    fi
    rm -f "$resp"

    LAST_CODE=$code
    LAST_TIME=$elapsed
}

OUT_A=feature_img2img_A_baseline.png
OUT_B=feature_img2img_B_img2img.png
OUT_C=feature_img2img_C_inpaint.png
OUT_D=feature_img2img_D_cache.png

# ── (A) Baseline text2img ────────────────────────────────────────────
BODY_A=$(mktemp --suffix=.json)
printf '{"model":"%s","prompt":"%s",%s}' "$MODEL_ID" "$PROMPT_BASE" "$COMMON" > "$BODY_A"
fire_with_body_file A "$BODY_A" "$OUT_A"
rm -f "$BODY_A"
CODE_A=$LAST_CODE
[[ "$CODE_A" == "200" ]] || { err "baseline failed (HTTP $CODE_A); inspect $OUT_A"; exit 1; }

# ── Build base64 payload of A + a centred-circle mask ────────────────
# Image: pure stdlib base64 on the host (no Pillow needed — we're just
# wrapping an existing PNG file's bytes).
# Mask: generated inside the worker container via `docker compose exec`
# because Pillow is a transitive dep of diffusers there but may not be
# installed on the host. Keeps the script host-Pillow-free on Windows/
# macOS devboxes where `pip install Pillow` on system Python is awkward.
log "Encoding baseline image (host stdlib) and building mask (inside worker) …"
IMAGE_B64=$("$PY" -c "
import base64, sys
sys.stdout.write(base64.b64encode(open(sys.argv[1], 'rb').read()).decode())
" "$OUT_A")

MASK_B64=$("${COMPOSE[@]}" exec -T "$SERVICE" python -c "
import base64, io, sys
from PIL import Image, ImageDraw
m = Image.new('L', (512, 512), 0)
ImageDraw.Draw(m).ellipse((180, 180, 332, 332), fill=255)  # centred circle
buf = io.BytesIO(); m.save(buf, format='PNG')
sys.stdout.write(base64.b64encode(buf.getvalue()).decode())
" 2>/dev/null | tr -d '\r\n')

if [[ -z "$MASK_B64" ]]; then
    err "failed to build mask inside ${SERVICE}. Fallback: install Pillow on host and retry."
    err "  Pillow install:  $PY -m pip install Pillow"
    exit 1
fi
ok "payloads built (image=${#IMAGE_B64}B base64, mask=${#MASK_B64}B base64)"

# Persist base64 payloads to tempfiles so mkbody can reference them by
# path rather than embedding them in a python -c argv. A 400KB IMAGE_B64
# inlined as `-c '… """$IMAGE_B64""" …'` blows past ARG_MAX (~128KB on
# Linux; similar on Windows/Git-bash). File I/O sidesteps that entirely.
IMAGE_PATH=$(mktemp --suffix=.b64)
MASK_PATH=$(mktemp --suffix=.b64)
printf '%s' "$IMAGE_B64" > "$IMAGE_PATH"
printf '%s' "$MASK_B64"  > "$MASK_PATH"
# Note: we intentionally keep IMAGE_B64/MASK_B64 in-scope after writing
# to disk — the error-path check (E) still references MASK_B64 inline,
# and a 400KB shell var has no measurable overhead for a script that
# already spawned docker exec. Earlier versions unset these and broke
# step E with `set -u` on unbound variable.

cleanup_payloads() { rm -f "$IMAGE_PATH" "$MASK_PATH"; }
trap cleanup_payloads EXIT

# Helper to materialise a JSON request body. Paths to the base64
# payloads are passed as short argv (a few dozen bytes), and Python
# reads the actual content from disk — this keeps argv well under ARG_MAX.
mkbody() {
    local out="$1" prompt="$2" mode="$3"
    MODEL_ID="$MODEL_ID" PROMPT="$prompt" MODE="$mode" \
    IMAGE_PATH="$IMAGE_PATH" MASK_PATH="$MASK_PATH" OUT="$out" \
    "$PY" <<'PYEOF'
import json, os
with open(os.environ['IMAGE_PATH']) as f:
    image_b64 = f.read()
body = {
    'model': os.environ['MODEL_ID'],
    'prompt': os.environ['PROMPT'],
    'seed': 17,
    'num_inference_steps': 20,
    'scheduler': 'dpm++_2m',
    'size': '512x512',
    'denoising_strength': 0.5,
    'image': image_b64,
}
if os.environ['MODE'] == 'inpaint':
    with open(os.environ['MASK_PATH']) as f:
        body['mask'] = f.read()
    body['denoising_strength'] = 0.9
with open(os.environ['OUT'], 'w') as f:
    json.dump(body, f)
PYEOF
}

# ── (B) img2img ──────────────────────────────────────────────────────
BODY_B=$(mktemp --suffix=.json); mkbody "$BODY_B" "$PROMPT_EDIT" img2img
fire_with_body_file B "$BODY_B" "$OUT_B"; rm -f "$BODY_B"
CODE_B=$LAST_CODE; TIME_B_FIRST=$LAST_TIME

# ── (C) inpaint ──────────────────────────────────────────────────────
BODY_C=$(mktemp --suffix=.json); mkbody "$BODY_C" "$PROMPT_INPAINT" inpaint
fire_with_body_file C "$BODY_C" "$OUT_C"; rm -f "$BODY_C"
CODE_C=$LAST_CODE

# ── (D) img2img repeat (cache hit / determinism) ─────────────────────
BODY_D=$(mktemp --suffix=.json); mkbody "$BODY_D" "$PROMPT_EDIT" img2img
fire_with_body_file D "$BODY_D" "$OUT_D"; rm -f "$BODY_D"
CODE_D=$LAST_CODE; TIME_D_SECOND=$LAST_TIME

echo
echo "─── Feature applied? ─────────────────────────────────────"
fail=0

for L in B C D; do
    declare -n code="CODE_$L"
    if [[ "$code" != "200" ]]; then
        err "$L failed with HTTP $code"
        "${COMPOSE[@]}" logs --no-color --tail 30 "$SERVICE" | sed 's/^/  /'
        fail=1
    fi
done
(( fail )) && exit 1

# A ≠ B: img2img with a different prompt + denoising=0.5 must modify A
if cmp -s "$OUT_A" "$OUT_B"; then
    err "A ≡ B byte-identical — img2img path didn't engage"
    fail=1
else
    ok "A ≠ B — img2img modified the baseline"
fi

# A ≠ C: inpaint under a centred mask must differ from A
if cmp -s "$OUT_A" "$OUT_C"; then
    err "A ≡ C byte-identical — inpaint path didn't engage"
    fail=1
else
    ok "A ≠ C — inpaint modified the masked region"
fi

# B ≠ C: different prompt + mask → must differ too
if cmp -s "$OUT_B" "$OUT_C"; then
    err "B ≡ C byte-identical — pipeline selection is indistinguishable"
    fail=1
else
    ok "B ≠ C — img2img and inpaint produce different results"
fi

# B ≡ D: same seed, same inputs, same adapter state → byte-equal
if cmp -s "$OUT_B" "$OUT_D"; then
    ok "B ≡ D — img2img is deterministic with fixed seed"
else
    err "B ≠ D — seed did not drive reproducibility (regression on seed-fix)"
    fail=1
fi

if (( TIME_B_FIRST > 0 && TIME_D_SECOND <= TIME_B_FIRST * 2 )); then
    ok "repeat latency: ${TIME_D_SECOND}s ≈ first-load ${TIME_B_FIRST}s (img2img pipe cached)"
else
    log "repeat latency: ${TIME_D_SECOND}s vs ${TIME_B_FIRST}s — noisy, not a failure"
fi

# ── (E) Error path: mask without image → 422 ─────────────────────────
log "[E] Error path: mask without image → HTTP 422 (schema guard)"
RESP=$(mktemp --suffix=.json)
CODE=$(curl -s -o "$RESP" -w '%{http_code}' \
    -X POST http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"%s","prompt":"x","mask":"%s"}' "$MODEL_ID" "$MASK_B64")" \
    || echo 000)
echo "  HTTP $CODE body: $(head -c 300 "$RESP")"
if [[ "$CODE" == "422" ]] && grep -q "mask" "$RESP" && grep -q "image" "$RESP"; then
    ok "mask-without-image rejected at validation time (HTTP 422, structured message)"
else
    err "Expected 422 + 'mask requires image'-style message; got HTTP $CODE"
    fail=1
fi
rm -f "$RESP"

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }

ok "PASS — img2img + inpaint + determinism + schema-guard all green."
echo
echo "Open side-by-side to inspect:"
echo "  $OUT_A — baseline (text2img)"
echo "  $OUT_B — img2img with prompt '$PROMPT_EDIT' (denoising=0.5)"
echo "  $OUT_C — inpaint with centred circle mask, prompt '$PROMPT_INPAINT'"
echo "  $OUT_D — repeat of B (byte-equal to B; determinism check)"
