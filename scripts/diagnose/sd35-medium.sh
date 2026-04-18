#!/usr/bin/env bash
# Launch + smoke-test sd35-medium — 5GB, no offload, no quantization.
# On 12GB cards this fits entirely in VRAM → fast inference, no Blackwell
# accelerate-hook deadlocks.
#
# Run from project root:
#   bash scripts/diagnose_sd35_medium.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-sd35-medium"
OUTPUT="sd35_diag.png"
READY_TIMEOUT=600

log() { printf '\033[1;36m[diag]\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[err ]\033[0m %s\n' "$*"; }

if [[ ! -f "$ENV_FILE" ]]; then
    err "$ENV_FILE not found — create from deploy/.env.example first."
    exit 1
fi

update_env() {
    local key="$1" val="$2"
    if grep -q "^${key}=" "$ENV_FILE"; then
        log "Updating ${key}=${val}"
        local tmp
        tmp=$(mktemp)
        sed "s|^${key}=.*|${key}=${val}|" "$ENV_FILE" > "$tmp"
        mv "$tmp" "$ENV_FILE"
    else
        log "Appending ${key}=${val}"
        printf '\n%s=%s\n' "$key" "$val" >> "$ENV_FILE"
    fi
}

# sd35-medium fits in 12GB with drop_t5=true → no offload, no quantization.
# Sequential/model offload both hang on Blackwell/WSL2; force off.
update_env COMPOSE_PROFILES sd35-medium
update_env SD35_MEDIUM_CPU_OFFLOAD false
update_env SD35_MEDIUM_SEQUENTIAL_OFFLOAD false
update_env SD35_MEDIUM_DROP_T5 true      # saves ~9.5GB; comfortable on 12GB
update_env SD35_MEDIUM_WARMUP false       # keep off for first test
ok "Env flags set (no offload, no quant, drop_t5)"

log "Starting $SERVICE (builds if Dockerfile args changed) …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for gateway to mark sd35-medium available …"
start=$SECONDS
while true; do
    worker_logs=$("${COMPOSE[@]}" logs --no-color "$SERVICE" 2>&1 || true)
    if grep -qE 'Application startup failed|CUDA driver error|CRITICAL' <<<"$worker_logs"; then
        err "Worker startup failure detected. Last 50 log lines:"
        "${COMPOSE[@]}" logs --no-color --tail 50 "$SERVICE"
        exit 1
    fi

    gateway_logs=$("${COMPOSE[@]}" logs --no-color gateway 2>&1 || true)
    if grep -q 'Worker ready: sd35-medium.*model is now available' <<<"$gateway_logs"; then
        ok "Gateway marks sd35-medium as available"
        break
    fi

    if (( SECONDS - start > READY_TIMEOUT )); then
        err "Timed out. Last worker logs:"
        "${COMPOSE[@]}" logs --no-color --tail 40 "$SERVICE"
        echo
        err "Last gateway logs:"
        "${COMPOSE[@]}" logs --no-color --tail 20 gateway
        exit 1
    fi
    sleep 5
done

log "POST /v1/images/generations (prompt=cyberpunk cat, 1024×1024) …"
RESP_JSON=$(mktemp --suffix=.json)
t0=$SECONDS
http_code=$(curl -s -o "$RESP_JSON" -w '%{http_code}' \
    -X POST http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d '{"model":"sd35-medium","prompt":"a cyberpunk cat"}' || echo 000)
elapsed=$(( SECONDS - t0 ))

# OpenAI-compatible response is JSON with b64-encoded PNG in data[0].b64_json.
# Decode portably: try python variants, then sed+base64.
decode_b64_png() {
    local resp="$1" out="$2"
    for py in python3 python py; do
        if command -v "$py" >/dev/null 2>&1; then
            "$py" -c "
import json, base64
with open('$resp') as f:
    d = json.load(f)
with open('$out', 'wb') as f:
    f.write(base64.b64decode(d['data'][0]['b64_json']))
" && return 0
        fi
    done
    # Fallback: POSIX sed + base64 (always present in Git Bash/WSL)
    if command -v base64 >/dev/null 2>&1; then
        sed -n 's/.*"b64_json":"\([^"]*\)".*/\1/p' "$resp" | base64 -d > "$out" && return 0
    fi
    return 1
}

if [[ "$http_code" == "200" ]]; then
    if decode_b64_png "$RESP_JSON" "$OUTPUT"; then
        :
    else
        err "Failed to decode base64 payload; saving raw JSON to $OUTPUT"
        cp "$RESP_JSON" "$OUTPUT"
    fi
else
    cp "$RESP_JSON" "$OUTPUT"
fi
rm -f "$RESP_JSON"

echo
echo "─── Result ────────────────────────────────────────────────"
echo "  HTTP status : $http_code"
echo "  Elapsed     : ${elapsed}s"
echo "  Output      : $OUTPUT ($(wc -c < "$OUTPUT" 2>/dev/null || echo 0) bytes)"
echo

case "$http_code" in
    200)
        ok "SUCCESS — image generated in ${elapsed}s."
        echo "  → Subsequent requests should be ~5-15s (kernels cached)."
        echo "  → Test speed: curl -w '%{time_total}\\n' -o out2.png \\"
        echo "       -X POST http://localhost:8000/v1/images/generations \\"
        echo "       -H 'Content-Type: application/json' \\"
        echo "       -d '{\"model\":\"sd35-medium\",\"prompt\":\"a robot\"}'"
        ;;
    500|504)
        err "FAILED — $http_code."
        echo "  Response body:"
        cat "$OUTPUT"; echo
        echo "  Last 40 worker log lines:"
        "${COMPOSE[@]}" logs --no-color --tail 40 "$SERVICE"
        ;;
    *)
        err "Unexpected HTTP $http_code."
        echo "  Response body:"
        cat "$OUTPUT"; echo
        ;;
esac
