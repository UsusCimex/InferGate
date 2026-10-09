#!/usr/bin/env bash
# Feature test: speech round trip, kokoro-82m speaks and Parakeet TDT v2 and v3 write it down.
#
#   1. Kokoro says a known sentence as mp3; each Parakeet model returns its words.
#   2. verbose_json carries text, duration and one segment.
#   3. The repeat comes from the cache.
#   4. Bytes that are not audio give 400.
#
# Both models run on the CPU; the first load downloads the int8 export (~0.7 GB each).
#
# Run from project root: bash scripts/feature/stt-parakeet.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODELS=("parakeet-tdt-0.6b-v2" "parakeet-tdt-0.6b-v3")
READY_TIMEOUT=900
GATEWAY_URL="http://localhost:8000"
SENTENCE="The little fox jumps over the red box and runs to the green garden."
PY="$(command -v python3 || command -v python)"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found; copy it from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "kokoro-82m,${MODELS[0]},${MODELS[1]}"
ok "Env flags set"

SERVICES=(worker-kokoro-82m worker-parakeet-tdt-0-6b-v2 worker-parakeet-tdt-0-6b-v3)
log "Building + starting gateway, kokoro and both Parakeet workers..."
"${COMPOSE[@]}" build gateway "${SERVICES[@]}"
"${COMPOSE[@]}" up -d gateway "${SERVICES[@]}"

wait_for_worker worker-kokoro-82m "models/models--hexgrad--Kokoro-82M" kokoro-82m "$READY_TIMEOUT"
for model in "${MODELS[@]}"; do
    wait_for_worker "worker-${model//./-}" "models/models--istupakov--${model}-onnx" "$model" "$READY_TIMEOUT"
done

SPEECH=$(mktemp --suffix=.mp3)
RESP=$(mktemp --suffix=.json)
trap 'rm -f "$SPEECH" "$RESP" "$RESP.headers"' EXIT

code=$(curl -s -o "$SPEECH" -w '%{http_code}' -X POST "$GATEWAY_URL/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"kokoro-82m\", \"input\": \"$SENTENCE\", \"voice\": \"af_heart\", \"response_format\": \"mp3\"}")
[[ "$code" == "200" ]] || { err "kokoro: HTTP $code"; exit 1; }
ok "kokoro spoke the sentence: $(wc -c < "$SPEECH") bytes of mp3"

transcribe() {
    curl -s -o "$RESP" -D "$RESP.headers" -w '%{http_code}' -X POST "$GATEWAY_URL/v1/audio/transcriptions" "$@"
}

fail=0
for model in "${MODELS[@]}"; do
    log "[$model] the sentence back as text (the first call loads the model)"
    code=$(transcribe -F "file=@$SPEECH" -F "model=$model" -F "response_format=verbose_json")
    if [[ "$code" != "200" ]]; then
        err "$model: HTTP $code: $(head -c 300 "$RESP")"
        fail=1
        continue
    fi
    verdict=$("$PY" - "$RESP" <<'PYEOF'
import json, re, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
words = set(re.findall(r"[a-z]+", d["text"].lower()))
missing = {"fox", "jumps", "red", "box", "green", "garden"} - words
shape = isinstance(d.get("segments"), list) and len(d["segments"]) == 1 and d.get("duration", 0) > 1
print(("OK " if not missing and shape else "BAD ") + json.dumps(d["text"]) + (f" missing {sorted(missing)}" if missing else ""))
PYEOF
)
    if [[ "$verdict" == OK* ]]; then ok "$model: ${verdict#OK }"; else err "$model: ${verdict#BAD }"; fail=1; fi

    transcribe -F "file=@$SPEECH" -F "model=$model" -F "response_format=verbose_json" > /dev/null
    if grep -qi '^x-infergate-cache: HIT' "$RESP.headers"; then ok "$model: repeat from the cache"; else err "$model: repeat missed the cache"; fail=1; fi

    code=$(transcribe -F "file=@$ENV_FILE.example;filename=speech.mp3" -F "model=$model" -H "X-InferGate-No-Cache: true")
    if [[ "$code" == "400" ]]; then ok "$model: not audio gives 400"; else err "$model: not audio gave HTTP $code"; fail=1; fi
done

echo
(( fail )) && { err "FAIL: review output above."; exit 1; }
ok "PASS: kokoro-82m speech comes back as text from both Parakeet models."
