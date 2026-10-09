#!/usr/bin/env bash
# Feature test: /v1/audio/speech on voxcpm2 (preset narrators, seeding, formats, voice clone).
#
# Assertions:
#   1. every preset voice: HTTP 200, audio/mpeg
#   2. cache MISS, then HIT on an identical request, same bytes
#   3. cache bypassed: the same seed gives identical bytes, another seed different bytes
#   4. response_format=wav: RIFF/WAVE at 48 kHz
#   5. unknown voice: HTTP 400 invalid_request with the preset ids in error.voices
#   6. voice-clone with a preset clip as the upload + its transcript: HTTP 200, audio/mpeg
#
# Run from project root: bash scripts/feature/tts-voxcpm2.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

COMPOSE=(docker compose -f deploy/docker-compose.yml --profile voxcpm2)
MODEL_ID="voxcpm2"
SERVICE="worker-voxcpm2"
HF_CACHE="models/models--openbmb--VoxCPM2"
READY_TIMEOUT=900
GATEWAY_URL="http://localhost:8000"
VOICES=(vox_clara vox_arthur vox_lily vox_daniel)
VOICES_DIR="app/providers/tts/voxcpm2_voices"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

command -v curl >/dev/null 2>&1 || { err "curl required"; exit 1; }
[[ -f deploy/.env ]] || { err "deploy/.env not found; copy it from deploy/.env.example"; exit 1; }

log "Building + starting gateway + $SERVICE..."
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID..."
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

speech() {
    local out="$1" body="$2"; shift 2
    local t0=$SECONDS
    LAST_CODE=$(curl -s -o "$out" -D "${out}.headers" -w '%{http_code}' \
        -X POST "${GATEWAY_URL}/v1/audio/speech" \
        -H "Content-Type: application/json" "$@" -d "$body" || echo 000)
    echo "  HTTP $LAST_CODE, $(( SECONDS - t0 ))s: $(wc -c < "$out") bytes"
}

header() {
    awk -v key="$2:" 'tolower($1) == key { sub(/\r$/, "", $2); print $2; exit }' "$1.headers"
}

seeded() {
    echo "{\"model\":\"$MODEL_ID\",\"voice\":\"vox_lily\",\"input\":\"The sun is warm today.\",\"seed\":$1}"
}

fail=0
NO_CACHE=(-H "X-InferGate-No-Cache: true")

log "[1] every preset voice: expect 200 audio/mpeg"
for voice in "${VOICES[@]}"; do
    speech "$WORK/$voice.mp3" \
        "{\"model\":\"$MODEL_ID\",\"voice\":\"$voice\",\"input\":\"The little fox found a shiny red box.\"}" \
        "${NO_CACHE[@]}"
    if [[ "$LAST_CODE" == "200" && "$(header "$WORK/$voice.mp3" content-type)" == audio/mpeg* ]]; then
        ok "$voice: $(wc -c < "$WORK/$voice.mp3") bytes of mp3"
    else
        err "$voice: HTTP $LAST_CODE, $(head -c 300 "$WORK/$voice.mp3")"
        fail=1
    fi
done

log "[2] cache MISS, then HIT on an identical request"
PROBE="{\"model\":\"$MODEL_ID\",\"voice\":\"vox_daniel\",\"input\":\"Ben brushes his teeth.\",\"seed\":$(date +%s)}"
speech "$WORK/probe1.mp3" "$PROBE"
speech "$WORK/probe2.mp3" "$PROBE"
FIRST=$(header "$WORK/probe1.mp3" x-infergate-cache)
SECOND=$(header "$WORK/probe2.mp3" x-infergate-cache)
if [[ "$FIRST" == "MISS" && "$SECOND" == "HIT" ]] && cmp -s "$WORK/probe1.mp3" "$WORK/probe2.mp3"; then
    ok "cache MISS then HIT with the same bytes"
else
    err "cache headers: first=$FIRST second=$SECOND (expected MISS then HIT)"
    fail=1
fi

log "[3] cache bypassed: seed 7 twice, then seed 8"
speech "$WORK/seed7a.mp3" "$(seeded 7)" "${NO_CACHE[@]}"
speech "$WORK/seed7b.mp3" "$(seeded 7)" "${NO_CACHE[@]}"
speech "$WORK/seed8.mp3" "$(seeded 8)" "${NO_CACHE[@]}"
if cmp -s "$WORK/seed7a.mp3" "$WORK/seed7b.mp3" && ! cmp -s "$WORK/seed7a.mp3" "$WORK/seed8.mp3"; then
    ok "same seed gives identical audio, another seed different audio"
else
    err "seeding is not deterministic (sizes: $(wc -c < "$WORK/seed7a.mp3") / $(wc -c < "$WORK/seed7b.mp3") / $(wc -c < "$WORK/seed8.mp3"))"
    fail=1
fi

log "[4] response_format=wav: expect RIFF/WAVE at 48 kHz"
speech "$WORK/clara.wav" \
    "{\"model\":\"$MODEL_ID\",\"voice\":\"vox_clara\",\"input\":\"Good night, little fox.\",\"response_format\":\"wav\"}" \
    "${NO_CACHE[@]}"
MAGIC="$(dd if="$WORK/clara.wav" bs=1 count=4 2>/dev/null)$(dd if="$WORK/clara.wav" bs=1 skip=8 count=4 2>/dev/null)"
RATE=$(od -An -tu4 -j24 -N4 "$WORK/clara.wav" | tr -d ' ')
if [[ "$LAST_CODE" == "200" && "$MAGIC" == "RIFFWAVE" && "$RATE" == "48000" ]]; then
    ok "wav: $(header "$WORK/clara.wav" content-type), ${RATE} Hz"
else
    err "expected a 48 kHz RIFF/WAVE body, got HTTP $LAST_CODE magic='$MAGIC' rate='$RATE'"
    fail=1
fi

log "[5] unknown voice: expect 400 with the preset list"
speech "$WORK/unknown.json" "{\"model\":\"$MODEL_ID\",\"voice\":\"af_heart\",\"input\":\"Hello.\"}"
if [[ "$LAST_CODE" == "400" ]] \
    && grep -qF '"type":"invalid_request"' "$WORK/unknown.json" \
    && grep -qF '"voices":["vox_clara","vox_arthur","vox_lily","vox_daniel"]' "$WORK/unknown.json"; then
    ok "unknown voice gives 400: $(head -c 200 "$WORK/unknown.json")"
else
    err "expected 400 invalid_request with error.voices, got $LAST_CODE: $(head -c 300 "$WORK/unknown.json")"
    fail=1
fi

log "[6] voice-clone with vox_lily's clip and its transcript: expect 200 audio/mpeg"
REF_TEXT=$(tr -d '\r' < "$VOICES_DIR/vox_lily.txt")
t0=$SECONDS
LAST_CODE=$(curl -s -o "$WORK/clone.mp3" -D "$WORK/clone.mp3.headers" -w '%{http_code}' \
    -X POST "${GATEWAY_URL}/v1/audio/speech/voice-clone" "${NO_CACHE[@]}" \
    -F "reference_audio=@${VOICES_DIR}/vox_lily.wav" \
    -F "reference_text=${REF_TEXT}" \
    -F "input=Ben brushes his teeth and goes to school." \
    -F "model=${MODEL_ID}" || echo 000)
echo "  HTTP $LAST_CODE, $(( SECONDS - t0 ))s: $(wc -c < "$WORK/clone.mp3") bytes"
if [[ "$LAST_CODE" == "200" && "$(header "$WORK/clone.mp3" content-type)" == audio/mpeg* ]]; then
    ok "cloned audio returned"
else
    err "expected 200 audio/mpeg, got $LAST_CODE: $(head -c 300 "$WORK/clone.mp3")"
    "${COMPOSE[@]}" logs --no-color --tail 40 "$SERVICE" | sed 's/^/  /'
    fail=1
fi

echo
(( fail )) && { err "FAIL: review output above."; exit 1; }
ok "PASS: preset voices, seeding, formats and voice cloning work end-to-end on $MODEL_ID."
