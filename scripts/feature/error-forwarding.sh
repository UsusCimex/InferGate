#!/usr/bin/env bash
# Feature test: ValueError from worker provider → HTTP 400 + JSON body
# forwarded to client by gateway.
#
# Pre: worker converts ValueError to 400 with `{"error": {"message": ..., "type": "invalid_request"}}`.
# Gateway httpx.HTTPStatusError handler reads worker's body + status, mirrors them.
#
# Uses sdxl-base + an unknown scheduler name to trigger the error path.
# Run from project root: bash scripts/feature/error-forwarding.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sdxl-base"
SERVICE="worker-sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
READY_TIMEOUT=600

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
ok "Env flags set"

log "Rebuilding gateway + worker (main.py + worker.py changed) …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

fail=0

log "[A] Unknown scheduler → expect HTTP 400 + 'Unknown scheduler' in body"
RESP=$(mktemp --suffix=.json)
CODE=$(curl -s -o "$RESP" -w '%{http_code}' \
    -X POST http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"x\",\"scheduler\":\"nonsense\"}" || echo 000)
body=$(cat "$RESP")
echo "  HTTP $CODE"
echo "  body: $body"
if [[ "$CODE" == "400" ]] && grep -q "Unknown scheduler" <<<"$body"; then
    ok "forwarded: HTTP 400 with worker's 'Unknown scheduler' message"
else
    err "expected 400 + 'Unknown scheduler', got HTTP $CODE"
    fail=1
fi
rm -f "$RESP"

log "[B] Valid request should still return HTTP 200 PNG (regression check)"
fire_image_request "$MODEL_ID" "a red apple" /tmp/err_test_ok.png '"seed":1,"num_inference_steps":5'
if [[ "$HTTP_CODE" == "200" ]]; then
    ok "HTTP 200, ${ELAPSED}s — normal path unaffected"
else
    err "regression: valid request returned HTTP $HTTP_CODE"
    fail=1
fi
rm -f /tmp/err_test_ok.png

log "[C] Ill-formed prompt (empty) → pydantic schema 422"
RESP=$(mktemp --suffix=.json)
CODE=$(curl -s -o "$RESP" -w '%{http_code}' \
    -X POST http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"\"}" || echo 000)
echo "  HTTP $CODE"
if [[ "$CODE" == "422" ]]; then
    ok "schema-level 422 works (independent of upstream forwarding)"
else
    log "  (non-critical) expected 422 for empty prompt, got $CODE"
fi
rm -f "$RESP"

echo
if (( fail )); then
    err "FAIL — error-forwarding is incomplete."
    echo "  → Check:"
    echo "    - app/worker.py converts ValueError → JSONResponse(status_code=400)"
    echo "    - app/main.py has @app.exception_handler(httpx.HTTPStatusError) that"
    echo "      mirrors response.status_code + response.json()"
    exit 1
fi
ok "PASS — upstream worker errors are forwarded with correct status + body."
