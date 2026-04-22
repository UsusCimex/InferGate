#!/usr/bin/env bash
# Feature test: /v1/images/upscale end-to-end on realesrgan-x4.
#
# Round-trip:
#   1. Generate a 128×128 gradient PNG inside the worker container (PIL
#      is available there via spandrel/torch's transitive deps).
#   2. POST it to the gateway, receive back an upscaled PNG.
#   3. Parse the IHDR chunk to assert dimensions are exactly input × 4.
#   4. Round-trip a second identical request and assert cache HIT.
#   5. Test the b64_json envelope on a third call.
#   6. Reject an invalid image (random bytes) with 400.
#
# Content-aware (colour / gradient) assertions are skipped — Real-ESRGAN
# on a tiny synthetic gradient produces something plausible but not
# byte-deterministic across fp16 CUDA contexts. Shape + dimensions +
# status codes are what this script guards against regressions.
#
# Run from project root: bash scripts/feature/upscale.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="realesrgan-x4"
SERVICE="worker-realesrgan-x4"
HF_CACHE="models/models--ai-forever--Real-ESRGAN"
READY_TIMEOUT=900

GATEWAY_URL="http://localhost:8000"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

command -v curl >/dev/null 2>&1 || { err "curl required"; exit 1; }
PY=""
for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[[ -n "$PY" ]] || { err "python required (for PNG IHDR parsing)"; exit 1; }

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Building + starting gateway + $SERVICE …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"
ok "compose up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

# ── Build a 128×128 gradient PNG inside the worker container ──────
INPUT_PNG=$(mktemp --suffix=.png)
OUTPUT_PNG=$(mktemp --suffix=.png)
RESP_FILE=$(mktemp --suffix=.txt)
cleanup() { rm -f "$INPUT_PNG" "$OUTPUT_PNG" "$RESP_FILE" "${RESP_FILE}.headers" 2>/dev/null || true; }
trap cleanup EXIT

log "Generating 128×128 gradient PNG (timestamp-seeded) inside $SERVICE …"
# Seed the gradient on time so each run produces a distinct SHA-256 —
# otherwise rerunning the script starts with the previous run's cache
# warm and the MISS→HIT assertion reads as HIT→HIT.
SEED=$(( $(date +%s) % 256 ))
"${COMPOSE[@]}" exec -T -e "SEED=$SEED" "$SERVICE" python <<'PYEOF' > "$INPUT_PNG"
import io, os, sys
from PIL import Image
seed = int(os.environ.get("SEED", "0"))
img = Image.new("RGB", (128, 128))
for y in range(128):
    for x in range(128):
        img.putpixel((x, y), ((x * 2 + seed) % 256, y * 2, (x + y + seed) % 256))
buf = io.BytesIO(); img.save(buf, format="PNG")
sys.stdout.buffer.write(buf.getvalue())
PYEOF
SZ=$(wc -c < "$INPUT_PNG")
[[ "$SZ" -gt 100 ]] || { err "PNG generation failed (size=$SZ)"; exit 1; }
ok "input PNG: $INPUT_PNG ($SZ bytes)"

png_dims() {
    # Read IHDR width/height (big-endian u32 each, at byte offset 16 + 4).
    "$PY" -c "
import struct, sys
with open(sys.argv[1], 'rb') as f:
    head = f.read(24)
# PNG signature is 8 bytes; IHDR chunk type at offset 8+4=12, data at 16+0 = 16.
w, h = struct.unpack('>II', head[16:24])
print(f'{w}x{h}')
" "$1"
}

BEFORE_DIMS=$(png_dims "$INPUT_PNG")
[[ "$BEFORE_DIMS" == "128x128" ]] || { err "unexpected input dims: $BEFORE_DIMS"; exit 1; }

http_post() {
    local out="$1"; shift
    local t0=$SECONDS
    local code
    code=$(curl -s -o "$out" -w '%{http_code}' \
        -X POST "${GATEWAY_URL}/v1/images/upscale" \
        -D "${out}.headers" \
        "$@" || echo 000)
    local elapsed=$(( SECONDS - t0 ))
    echo "  HTTP $code, ${elapsed}s → $(wc -c < "$out") bytes"
    LAST_CODE=$code
}

fail=0

# ── (1) raw PNG response, assert dims = 128×4 = 512 ───────────────
log "[1] response_format=png, assert dims scale 4×"
http_post "$OUTPUT_PNG" \
    -F "file=@${INPUT_PNG}" -F "model=${MODEL_ID}" -F "response_format=png" \
    -H "X-InferGate-No-Cache: true"
if [[ "$LAST_CODE" != "200" ]]; then
    err "expected 200, got $LAST_CODE (body: $(head -c 300 "$OUTPUT_PNG"))"
    fail=1
else
    AFTER_DIMS=$(png_dims "$OUTPUT_PNG")
    if [[ "$AFTER_DIMS" == "512x512" ]]; then
        ok "upscaled PNG dims 128×128 → $AFTER_DIMS (4× scale applied)"
    else
        err "wrong dims: got $AFTER_DIMS, expected 512x512"
        fail=1
    fi
fi

# ── (2) cache MISS + HIT on identical input ──────────────────────
log "[2] cache MISS → HIT on repeat"
http_post "$RESP_FILE" -F "file=@${INPUT_PNG}" -F "model=${MODEL_ID}" -F "response_format=png"
FIRST_CACHE=$(grep -i '^x-infergate-cache' "${RESP_FILE}.headers" | head -1 | tr -d '\r' | awk '{print $NF}')
http_post "$RESP_FILE" -F "file=@${INPUT_PNG}" -F "model=${MODEL_ID}" -F "response_format=png"
SECOND_CACHE=$(grep -i '^x-infergate-cache' "${RESP_FILE}.headers" | head -1 | tr -d '\r' | awk '{print $NF}')
if [[ "$FIRST_CACHE" == "MISS" && "$SECOND_CACHE" == "HIT" ]]; then
    ok "cache headers: MISS then HIT"
else
    err "unexpected cache headers: first=$FIRST_CACHE second=$SECOND_CACHE"
    fail=1
fi

# ── (3) b64_json envelope (OpenAI-style) ─────────────────────────
log "[3] b64_json default envelope"
http_post "$RESP_FILE" -F "file=@${INPUT_PNG}" -F "model=${MODEL_ID}" \
    -H "X-InferGate-No-Cache: true"
if [[ "$LAST_CODE" == "200" ]]; then
    # Write the check script to a tempfile rather than `python -c '...'` —
    # heredoc-in-command-substitution quoting was silently eating the
    # output on earlier attempts.
    CHECK_PY=$(mktemp --suffix=.py)
    cat > "$CHECK_PY" <<'PYEOF'
import json, base64, struct, sys
d = json.load(open(sys.argv[1]))
assert "created" in d and isinstance(d.get("data"), list), f"envelope missing: keys={sorted(d.keys())}"
b64 = d["data"][0]["b64_json"]
png = base64.b64decode(b64)
assert png.startswith(b"\x89PNG"), "payload is not a PNG"
w, h = struct.unpack(">II", png[16:24])
assert (w, h) == (512, 512), f"dims {w}x{h} != 512x512"
print("OK")
PYEOF
    ENVELOPE_OK=$("$PY" "$CHECK_PY" "$RESP_FILE" 2>&1 || true)
    rm -f "$CHECK_PY"
    if [[ "$ENVELOPE_OK" == "OK" ]]; then
        ok "b64_json envelope correct, decoded PNG is 512×512"
    else
        err "envelope check failed: $ENVELOPE_OK"
        fail=1
    fi
else
    err "b64_json request: HTTP $LAST_CODE"
    fail=1
fi

# ── (4) invalid image (random bytes) → 400 ────────────────────────
log "[4] random bytes as 'image' → 400 from spandrel's decode path"
BOGUS=$(mktemp --suffix=.png)
head -c 512 /dev/urandom > "$BOGUS"
http_post "$RESP_FILE" -F "file=@${BOGUS}" -F "model=${MODEL_ID}" -F "response_format=png"
rm -f "$BOGUS"
if [[ "$LAST_CODE" == "400" ]]; then
    ok "bogus image → 400"
else
    err "expected 400, got $LAST_CODE (body: $(head -c 200 "$RESP_FILE"))"
    fail=1
fi

# ── (5) missing file → 422 ────────────────────────────────────────
log "[5] missing file → 422"
http_post "$RESP_FILE" -F "model=${MODEL_ID}"
if [[ "$LAST_CODE" == "422" ]]; then
    ok "missing file → 422"
else
    err "expected 422, got $LAST_CODE"
    fail=1
fi

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }
ok "PASS — /v1/images/upscale works end-to-end on realesrgan-x4."
