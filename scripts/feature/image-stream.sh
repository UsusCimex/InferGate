#!/usr/bin/env bash
# Feature test: streamed image generation on sdxl-base.
#
#   1. stream: true with partial_images: 2 sends two image_generation.partial_image events
#      (VAE previews of the running steps) and one image_generation.completed event.
#   2. The repeat with the same seed comes from the cache as one completed event.
#   3. response_format: png answers with the PNG itself.
#
# Run from project root: bash scripts/feature/image-stream.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sdxl-base"
SERVICE="worker-sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
READY_TIMEOUT=600
PY="$(command -v python3 || command -v python)"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found; copy it from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Rebuilding gateway + worker..."
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID..."
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

SEED=$RANDOM
STREAM=$(mktemp --suffix=.sse)
HEADERS=$(mktemp)
fail=0

stream_request() {
    curl -sN -D "$HEADERS" -o "$STREAM" \
        -X POST http://localhost:8000/v1/images/generations \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"a lighthouse at dusk\",\"seed\":$SEED,\"num_inference_steps\":20,\"stream\":true,\"partial_images\":2}"
}

# Saves the image of event N as feature_stream_N.png and prints the event names.
split_events() {
    "$PY" - "$STREAM" <<'PYEOF'
import base64, json, sys
names = []
for i, block in enumerate(open(sys.argv[1]).read().strip().split("\n\n")):
    fields = dict(line.split(": ", 1) for line in block.splitlines())
    data = json.loads(fields["data"])
    if "b64_json" in data:
        open(f"feature_stream_{i}.png", "wb").write(base64.b64decode(data["b64_json"]))
    names.append(fields["event"])
print(" ".join(names))
PYEOF
}

log "[1] stream with two previews"
stream_request
EVENTS=$(split_events)
EXPECTED="image_generation.partial_image image_generation.partial_image image_generation.completed"
if [[ "$EVENTS" == "$EXPECTED" ]]; then
    ok "two previews and the final image"
else
    err "events: $EVENTS"
    fail=1
fi
if [[ -f feature_stream_0.png && -f feature_stream_2.png ]] && ! cmp -s feature_stream_0.png feature_stream_2.png; then
    ok "the first preview differs from the final image"
else
    err "no preview that differs from the final image"
    fail=1
fi

log "[2] the same seed again comes from the cache"
stream_request
EVENTS=$(split_events)
if grep -qi '^x-infergate-cache: HIT' "$HEADERS" && [[ "$EVENTS" == "image_generation.completed" ]]; then
    ok "cache HIT as one completed event"
else
    err "repeat: events '$EVENTS', headers: $(tr -d '\r' < "$HEADERS" | grep -i '^x-infergate-cache' || true)"
    fail=1
fi

log "[3] response_format png"
CODE=$(curl -s -o feature_stream_plain.png -w '%{http_code}' \
    -X POST http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -H 'X-InferGate-No-Cache: true' \
    -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"a lighthouse at dusk\",\"num_inference_steps\":20,\"response_format\":\"png\"}" || echo 000)
if [[ "$CODE" == "200" ]] && head -c 8 feature_stream_plain.png | od -An -tx1 | grep -q '89 50 4e 47'; then
    ok "PNG in the response body"
else
    err "response_format png: HTTP $CODE"
    fail=1
fi

rm -f "$STREAM" "$HEADERS"
echo
(( fail )) && { err "FAIL: review output above."; exit 1; }
ok "PASS: streamed and plain PNG answers work on $MODEL_ID."
echo "Open feature_stream_0.png, feature_stream_1.png and feature_stream_2.png to see the previews turn into the image."
