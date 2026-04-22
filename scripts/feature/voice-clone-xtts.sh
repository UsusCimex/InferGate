#!/usr/bin/env bash
# Feature test: /v1/audio/speech/voice-clone end-to-end on xtts-v2.
#
# Same shape as the earlier voice-clone.sh (openaudio-s1-mini) but
# targets the XTTS-v2 worker — that's the one we actually ship as the
# voice-cloning default since openaudio remains blocked on fish-speech
# upstream. 4 assertions:
#
#   1. multipart upload → HTTP 200, audio/* Content-Type, non-empty body
#   2. cache MISS → HIT on an identical (text + reference + language) request
#   3. empty reference_audio → HTTP 400 with a clear message
#   4. missing reference_audio field → HTTP 422 (multipart validator)
#
# We don't assert cloned-voice timbre — it's subjective and not
# byte-deterministic across fp16 CUDA kernel choices. This script
# guards against wire-format / caching / validation regressions.
#
# Reference audio: override with REFERENCE_AUDIO=<path/to/clip.wav>
# for a realistic voice. Default is a 2-second 440Hz sine tone built
# from Python stdlib — XTTS accepts any valid WAV; cloning quality
# on a sine tone is low, but the pipeline runs and returns audio.
#
# Run from project root: bash scripts/feature/voice-clone-xtts.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="xtts-v2"
SERVICE="worker-xtts-v2"
HF_CACHE="models/tts/tts_models--multilingual--multi-dataset--xtts_v2"
READY_TIMEOUT=900

GATEWAY_URL="http://localhost:8000"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

command -v curl >/dev/null 2>&1 || { err "curl required"; exit 1; }
PY=""
for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[[ -n "$PY" ]] || { err "python required (for synthetic WAV generation)"; exit 1; }

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
update_env "$ENV_FILE" XTTS_V2_ENABLED true
ok "Env flags set (xtts-v2 enabled)"

log "Building + starting gateway + $SERVICE …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

# ── Reference audio ──────────────────────────────────────────────
REF_WAV=${REFERENCE_AUDIO:-}
CLEANUP_REF=0
if [[ -z "$REF_WAV" ]]; then
    REF_WAV=$(mktemp --suffix=.wav)
    CLEANUP_REF=1
    log "No REFERENCE_AUDIO set — generating 2s 440Hz sine as placeholder …"
    # Generate inside gateway container — Windows-native Python on a
    # Git-bash host can't open a /tmp/ path (different filesystem
    # namespaces). gateway is python:3.12-slim so `wave` stdlib is there.
    "${COMPOSE[@]}" exec -T gateway python <<'PYEOF' > "$REF_WAV"
import wave, struct, math, io, sys
buf = io.BytesIO()
with wave.open(buf, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
    for i in range(44100):
        s = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * i / 22050))
        w.writeframes(struct.pack('<h', s))
sys.stdout.buffer.write(buf.getvalue())
PYEOF
fi
[[ -f "$REF_WAV" ]] || { err "reference WAV missing: $REF_WAV"; exit 1; }
ok "reference: $REF_WAV ($(wc -c < "$REF_WAV") bytes)"

RESP_FILE=$(mktemp --suffix=.bin)
cleanup() {
    rm -f "$RESP_FILE" "${RESP_FILE}.headers" 2>/dev/null || true
    (( CLEANUP_REF )) && rm -f "$REF_WAV" 2>/dev/null || true
}
trap cleanup EXIT

http_post() {
    local out="$1"; shift
    local t0=$SECONDS
    local code
    code=$(curl -s -o "$out" -w '%{http_code}' \
        -X POST "${GATEWAY_URL}/v1/audio/speech/voice-clone" \
        -D "${out}.headers" \
        "$@" || echo 000)
    local elapsed=$(( SECONDS - t0 ))
    echo "  HTTP $code, ${elapsed}s → $(wc -c < "$out") bytes"
    LAST_CODE=$code
}

fail=0

# ── (1) Basic voice-clone ────────────────────────────────────────
log "[1] basic voice-clone: text + reference → audio"
http_post "$RESP_FILE" \
    -F "reference_audio=@${REF_WAV}" \
    -F "input=Hello world, this is a voice cloning test." \
    -F "model=${MODEL_ID}" \
    -F "response_format=wav" \
    -H "X-InferGate-No-Cache: true"
if [[ "$LAST_CODE" == "200" ]]; then
    CT=$(grep -i '^content-type' "${RESP_FILE}.headers" | head -1 | tr -d '\r')
    SZ=$(wc -c < "$RESP_FILE")
    if grep -q 'audio/' <<<"$CT" && [[ "$SZ" -gt 1000 ]]; then
        ok "cloned audio returned: $CT, $SZ bytes"
    else
        err "unexpected response — CT='$CT' size=$SZ"
        fail=1
    fi
else
    err "expected 200, got $LAST_CODE"
    echo "  body: $(head -c 500 "$RESP_FILE")"
    "${COMPOSE[@]}" logs --no-color --tail 40 "$SERVICE" | sed 's/^/  /'
    fail=1
fi

# ── (2) Cache MISS → HIT on identical request ───────────────────
# Nonce makes this idempotent — re-running the script shouldn't hit
# stale cache from a previous invocation.
PROBE="Cache probe $(date +%s%N)"
log "[2] cache MISS → HIT on identical request"
http_post "$RESP_FILE" \
    -F "reference_audio=@${REF_WAV}" -F "input=${PROBE}" \
    -F "model=${MODEL_ID}" -F "response_format=wav"
FIRST_CACHE=$(grep -i '^x-infergate-cache' "${RESP_FILE}.headers" | head -1 | tr -d '\r' | awk '{print $NF}')
http_post "$RESP_FILE" \
    -F "reference_audio=@${REF_WAV}" -F "input=${PROBE}" \
    -F "model=${MODEL_ID}" -F "response_format=wav"
SECOND_CACHE=$(grep -i '^x-infergate-cache' "${RESP_FILE}.headers" | head -1 | tr -d '\r' | awk '{print $NF}')
if [[ "$FIRST_CACHE" == "MISS" && "$SECOND_CACHE" == "HIT" ]]; then
    ok "cache MISS then HIT"
else
    err "cache headers: first=$FIRST_CACHE second=$SECOND_CACHE (expected MISS then HIT)"
    fail=1
fi

# ── (3) Empty reference → 400 ───────────────────────────────────
log "[3] empty reference_audio → 400"
EMPTY=$(mktemp --suffix=.wav)
http_post "$RESP_FILE" \
    -F "reference_audio=@${EMPTY}" \
    -F "input=short text" -F "model=${MODEL_ID}"
rm -f "$EMPTY"
if [[ "$LAST_CODE" == "400" ]]; then
    ok "empty reference → 400"
else
    err "expected 400, got $LAST_CODE"
    fail=1
fi

# ── (4) Missing reference field → 422 ───────────────────────────
log "[4] missing reference_audio field → 422 (multipart validation)"
http_post "$RESP_FILE" -F "input=short text" -F "model=${MODEL_ID}"
if [[ "$LAST_CODE" == "422" ]]; then
    ok "missing reference field → 422"
else
    err "expected 422, got $LAST_CODE"
    fail=1
fi

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }
ok "PASS — /v1/audio/speech/voice-clone works end-to-end on $MODEL_ID."
