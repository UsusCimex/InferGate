#!/usr/bin/env bash
# Feature test: /v1/audio/speech/voice-clone end-to-end on qwen3-tts-06b.
#
# Same shape as voice-clone-xtts.sh but targets the Qwen3-TTS 0.6B-Base
# worker. The critical difference: Qwen3-TTS REQUIRES both the
# reference_audio AND its transcription (reference_text). Skipping
# reference_text → provider raises ValueError → worker converts to
# HTTP 400 + `{"error": {"type": "invalid_request"}}` → gateway mirrors
# it back. We assert on the 400 + body message so a future refactor
# can't silently swallow the contract.
#
# Assertions:
#   1. multipart with reference_audio + reference_text → HTTP 200, audio/*
#   2. cache MISS → HIT on an identical (text + reference + ref_text) request
#   3. missing reference_text → HTTP 400 + "reference_text" in body (Qwen3-specific)
#   4. empty reference_audio → HTTP 400 (router-level validation)
#   5. missing reference_audio field → HTTP 422 (multipart validator)
#
# Reference audio: override with REFERENCE_AUDIO=<path/to/clip.wav>
# and REFERENCE_TEXT=<transcript> for a realistic voice. Default is a
# 2-second 440Hz sine with a dummy transcript — cloning quality will
# be poor, but the pipeline runs end-to-end.
#
# Run from project root: bash scripts/feature/voice-clone-qwen3-tts.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="qwen3-tts-06b"
SERVICE="worker-qwen3-tts-06b"
HF_CACHE="models/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"
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
ok "Env flags set (qwen3-tts-06b uses YAML default enabled=true)"

log "Building + starting gateway + $SERVICE …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

# ── Reference audio + transcript ─────────────────────────────────
REF_WAV=${REFERENCE_AUDIO:-}
REF_TEXT=${REFERENCE_TEXT:-"This is a reference speech sample for voice cloning."}
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
ok "reference: $REF_WAV ($(wc -c < "$REF_WAV") bytes), text: \"$REF_TEXT\""

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

# ── (1) Basic voice-clone with ref_text ──────────────────────────
log "[1] basic voice-clone: text + reference + ref_text → audio"
http_post "$RESP_FILE" \
    -F "reference_audio=@${REF_WAV}" \
    -F "reference_text=${REF_TEXT}" \
    -F "input=Hello world, this is a Qwen3 voice cloning test." \
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
    -F "reference_audio=@${REF_WAV}" -F "reference_text=${REF_TEXT}" \
    -F "input=${PROBE}" \
    -F "model=${MODEL_ID}" -F "response_format=wav"
FIRST_CACHE=$(grep -i '^x-infergate-cache' "${RESP_FILE}.headers" | head -1 | tr -d '\r' | awk '{print $NF}')
http_post "$RESP_FILE" \
    -F "reference_audio=@${REF_WAV}" -F "reference_text=${REF_TEXT}" \
    -F "input=${PROBE}" \
    -F "model=${MODEL_ID}" -F "response_format=wav"
SECOND_CACHE=$(grep -i '^x-infergate-cache' "${RESP_FILE}.headers" | head -1 | tr -d '\r' | awk '{print $NF}')
if [[ "$FIRST_CACHE" == "MISS" && "$SECOND_CACHE" == "HIT" ]]; then
    ok "cache MISS then HIT"
else
    err "cache headers: first=$FIRST_CACHE second=$SECOND_CACHE (expected MISS then HIT)"
    fail=1
fi

# ── (3) Missing reference_text → 400 (Qwen3-TTS-specific) ──────
# Provider raises ValueError → worker converts to HTTP 400 with a
# JSON body naming the missing field. We assert both the status and
# that "reference_text" appears in the body so callers get a useful
# message and silent conversion to 200 can't slip in.
log "[3] missing reference_text → 400 + 'reference_text' in body"
http_post "$RESP_FILE" \
    -F "reference_audio=@${REF_WAV}" \
    -F "input=Should fail without ref_text" \
    -F "model=${MODEL_ID}" \
    -F "response_format=wav"
if [[ "$LAST_CODE" == "400" ]] && grep -q "reference_text" "$RESP_FILE"; then
    ok "missing ref_text → 400 with explanatory body"
else
    err "expected 400 + 'reference_text' in body, got $LAST_CODE"
    echo "  body: $(head -c 300 "$RESP_FILE")"
    fail=1
fi

# ── (4) Empty reference → 400 ───────────────────────────────────
log "[4] empty reference_audio → 400"
EMPTY=$(mktemp --suffix=.wav)
http_post "$RESP_FILE" \
    -F "reference_audio=@${EMPTY}" \
    -F "reference_text=${REF_TEXT}" \
    -F "input=short text" -F "model=${MODEL_ID}"
rm -f "$EMPTY"
if [[ "$LAST_CODE" == "400" ]]; then
    ok "empty reference → 400"
else
    err "expected 400, got $LAST_CODE"
    fail=1
fi

# ── (5) Missing reference field → 422 ───────────────────────────
log "[5] missing reference_audio field → 422 (multipart validation)"
http_post "$RESP_FILE" -F "reference_text=${REF_TEXT}" \
    -F "input=short text" -F "model=${MODEL_ID}"
if [[ "$LAST_CODE" == "422" ]]; then
    ok "missing reference field → 422"
else
    err "expected 422, got $LAST_CODE"
    fail=1
fi

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }
ok "PASS — /v1/audio/speech/voice-clone works end-to-end on $MODEL_ID."
