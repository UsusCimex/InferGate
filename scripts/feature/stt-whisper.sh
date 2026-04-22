#!/usr/bin/env bash
# Feature test: /v1/audio/transcriptions end-to-end via whisper-base worker.
#
# Builds a synthetic WAV on the host (pure stdlib — `wave` + `struct`),
# posts it to the gateway in three shapes, asserts the response envelope
# in each and that the cache round-trips correctly.
#
# We do NOT assert the transcribed text content — synthetic audio isn't
# speech, and Whisper's output on a 1-second sine tone is non-deterministic
# (often empty, sometimes a short hallucination). What we verify:
#   * HTTP 200 on default json, text, and verbose_json
#   * Content-Type switches text/plain ↔ application/json correctly
#   * verbose_json returns language + duration + segments[] keys
#   * Second identical request is a cache HIT
#   * Bogus response_format → 400
#   * Missing file → 422 (FastAPI multipart validation)
#
# A full content-aware test (TTS → STT round-trip) is viable but needs
# two workers running simultaneously; deferred until the kokoro + whisper
# pair is a common setup.
#
# Run from project root: bash scripts/feature/stt-whisper.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="whisper-base"
SERVICE="worker-whisper-base"
HF_CACHE="models/models--Systran--faster-whisper-base"
READY_TIMEOUT=600

GATEWAY_URL="http://localhost:8000"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

command -v curl >/dev/null 2>&1 || { err "curl required"; exit 1; }
PY=""
for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[[ -n "$PY" ]] || { err "python required (for WAV generation + JSON parsing)"; exit 1; }

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Building + starting gateway + $SERVICE …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"
ok "compose up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

# ── Build a 1-second 440Hz sine-wave WAV (pure stdlib on the host) ──
AUDIO_WAV=$(mktemp --suffix=.wav)
cleanup() { rm -f "$AUDIO_WAV" "$RESP_FILE" 2>/dev/null || true; }
trap cleanup EXIT
"$PY" <<PYEOF
import wave, struct, math, sys
path = r"$AUDIO_WAV"
sr, freq, dur = 16000, 440, 1
with wave.open(path, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    for i in range(sr * dur):
        s = int(0.1 * 32767 * math.sin(2 * math.pi * freq * i / sr))
        w.writeframes(struct.pack('<h', s))
PYEOF
SZ=$(wc -c < "$AUDIO_WAV")
ok "synthetic WAV built: $AUDIO_WAV ($SZ bytes)"

RESP_FILE=$(mktemp --suffix=.txt)

http_post() {
    local label="$1" out="$2"; shift 2
    local t0=$SECONDS
    local code
    code=$(curl -s -o "$out" -w '%{http_code}' \
        -X POST "${GATEWAY_URL}/v1/audio/transcriptions" \
        -D "${out}.headers" \
        "$@" || echo 000)
    local elapsed=$(( SECONDS - t0 ))
    echo "  [$label] HTTP $code, ${elapsed}s → $(wc -c < "$out") bytes"
    LAST_CODE=$code
    LAST_TIME=$elapsed
}

fail=0
expect_200() { [[ "$1" == "200" ]] || { err "$2: HTTP $1 (body: $(head -c 300 "$RESP_FILE"))"; fail=1; }; }

# ── (1) Default json format ─────────────────────────────────────────
log "[1] default json format"
http_post json "$RESP_FILE" -F "file=@${AUDIO_WAV}" -F "model=${MODEL_ID}" \
                            -H "X-InferGate-No-Cache: true"
expect_200 "$LAST_CODE" "[1]"
if [[ "$LAST_CODE" == "200" ]]; then
    KEYS=$("$PY" -c "import json,sys; print(','.join(sorted(json.load(open(r'$RESP_FILE')).keys())))")
    if [[ "$KEYS" == "text" ]]; then
        ok "json returned strict {text: ...} shape"
    else
        err "json returned extra keys (got '$KEYS', expected 'text')"
        fail=1
    fi
fi

# ── (2) Text plain format ───────────────────────────────────────────
log "[2] response_format=text"
http_post text "$RESP_FILE" -F "file=@${AUDIO_WAV}" -F "model=${MODEL_ID}" \
                            -F "response_format=text" -H "X-InferGate-No-Cache: true"
expect_200 "$LAST_CODE" "[2]"
if [[ "$LAST_CODE" == "200" ]]; then
    CT=$(grep -i '^content-type' "${RESP_FILE}.headers" | head -1 | tr -d '\r')
    if grep -q 'text/plain' <<<"$CT"; then
        ok "text/plain Content-Type"
    else
        err "unexpected Content-Type: $CT"
        fail=1
    fi
fi

# ── (3) verbose_json with language hint ─────────────────────────────
log "[3] response_format=verbose_json, language=en"
http_post vjson "$RESP_FILE" -F "file=@${AUDIO_WAV}" -F "model=${MODEL_ID}" \
                             -F "response_format=verbose_json" -F "language=en" \
                             -H "X-InferGate-No-Cache: true"
expect_200 "$LAST_CODE" "[3]"
if [[ "$LAST_CODE" == "200" ]]; then
    CHECK=$("$PY" -c "
import json
d = json.load(open(r'$RESP_FILE'))
need = {'text', 'language', 'duration', 'segments'}
missing = need - set(d.keys())
print('OK' if not missing and isinstance(d.get('segments'), list) else 'MISSING:' + ','.join(sorted(missing)))
")
    if [[ "$CHECK" == "OK" ]]; then
        ok "verbose_json has text + language + duration + segments[]"
    else
        err "verbose_json shape incomplete — $CHECK"
        fail=1
    fi
fi

# ── (4) Cache MISS + HIT on identical request ───────────────────────
log "[4] cache MISS → HIT on repeat"
http_post c-miss "$RESP_FILE" -F "file=@${AUDIO_WAV}" -F "model=${MODEL_ID}"
MISS_STATE=$(grep -i '^x-infergate-cache' "${RESP_FILE}.headers" | head -1 | tr -d '\r' | awk '{print $NF}')
http_post c-hit  "$RESP_FILE" -F "file=@${AUDIO_WAV}" -F "model=${MODEL_ID}"
HIT_STATE=$(grep -i '^x-infergate-cache' "${RESP_FILE}.headers" | head -1 | tr -d '\r' | awk '{print $NF}')
if [[ "$MISS_STATE" == "MISS" && "$HIT_STATE" == "HIT" ]]; then
    ok "cache MISS then HIT as expected"
else
    err "cache headers unexpected: first='$MISS_STATE' second='$HIT_STATE'"
    fail=1
fi

# ── (5) Bogus response_format → 400 ─────────────────────────────────
log "[5] response_format=srt → 400 (not supported in this iteration)"
http_post bad-fmt "$RESP_FILE" -F "file=@${AUDIO_WAV}" -F "model=${MODEL_ID}" \
                               -F "response_format=srt"
if [[ "$LAST_CODE" == "400" ]] && grep -q "response_format" "$RESP_FILE"; then
    ok "srt rejected with 400 + structured message"
else
    err "expected 400, got $LAST_CODE (body: $(head -c 200 "$RESP_FILE"))"
    fail=1
fi

# ── (6) Missing file → 422 ──────────────────────────────────────────
log "[6] missing file field → 422 (multipart validation)"
http_post no-file "$RESP_FILE" -F "model=${MODEL_ID}"
if [[ "$LAST_CODE" == "422" ]]; then
    ok "missing file → 422"
else
    err "expected 422, got $LAST_CODE"
    fail=1
fi

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }
ok "PASS — /v1/audio/transcriptions works end-to-end on whisper-base."
