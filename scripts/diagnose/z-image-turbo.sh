#!/usr/bin/env bash
# Launch + smoke-test z-image-turbo — 6B DiT, 8 steps, Apache-2.0.
# First attempt: bf16 (no quant). If OOM, re-run with Z_IMAGE_TURBO_QUANTIZATION=nf4.
#
# Run from project root:
#   bash scripts/diagnose_z_image_turbo.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-z-image-turbo"
OUTPUT="z_image_diag.png"
READY_TIMEOUT=900   # diffusers from git means longer first build

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

update_env COMPOSE_PROFILES z-image-turbo
update_env Z_IMAGE_TURBO_CPU_OFFLOAD false
update_env Z_IMAGE_TURBO_SEQUENTIAL_OFFLOAD false
update_env Z_IMAGE_TURBO_WARMUP false
# Start with no quant; if OOM, user can re-run with Z_IMAGE_TURBO_QUANTIZATION=nf4.
if ! grep -q '^Z_IMAGE_TURBO_QUANTIZATION=' "$ENV_FILE"; then
    update_env Z_IMAGE_TURBO_QUANTIZATION null
fi
ok "Env flags set"

log "Starting $SERVICE (first build pulls diffusers from git — expect 5-10 min) …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for gateway to mark z-image-turbo available …"
start=$SECONDS
while true; do
    worker_logs=$("${COMPOSE[@]}" logs --no-color "$SERVICE" 2>&1 || true)
    if grep -qE 'Application startup failed|CUDA driver error|CRITICAL' <<<"$worker_logs"; then
        err "Worker startup failure detected."
        # Dump the last 60 lines — OOM and ZImagePipeline-not-found surface here.
        "${COMPOSE[@]}" logs --no-color --tail 60 "$SERVICE"
        echo
        if grep -q 'out of memory\|OutOfMemoryError' <<<"$worker_logs"; then
            err "Root cause: OOM. Re-run with nf4:"
            echo "    echo 'Z_IMAGE_TURBO_QUANTIZATION=nf4' >> $ENV_FILE"
            echo "    bash scripts/diagnose_z_image_turbo.sh"
        elif grep -q 'ZImagePipeline' <<<"$worker_logs"; then
            err "Root cause: ZImagePipeline not found in this diffusers build."
            echo "    Check the worker's diffusers version — should be >0.37 (git main)."
        fi
        exit 1
    fi

    gateway_logs=$("${COMPOSE[@]}" logs --no-color gateway 2>&1 || true)
    if grep -q 'Worker ready: z-image-turbo.*model is now available' <<<"$gateway_logs"; then
        ok "Gateway marks z-image-turbo as available"
        break
    fi

    if (( SECONDS - start > READY_TIMEOUT )); then
        err "Timed out. Last worker logs:"
        "${COMPOSE[@]}" logs --no-color --tail 60 "$SERVICE"
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
    -d '{"model":"z-image-turbo","prompt":"a cyberpunk cat"}' || echo 000)
elapsed=$(( SECONDS - t0 ))

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
    if command -v base64 >/dev/null 2>&1; then
        sed -n 's/.*"b64_json":"\([^"]*\)".*/\1/p' "$resp" | base64 -d > "$out" && return 0
    fi
    return 1
}

if [[ "$http_code" == "200" ]]; then
    if decode_b64_png "$RESP_JSON" "$OUTPUT"; then :; else
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
        echo "  → Open $OUTPUT and compare with sd35_diag.png / sdxl_diag.png."
        echo "  → Subsequent requests should be ~3-8s (8-step distilled)."
        ;;
    500|504)
        err "FAILED — $http_code."
        echo "  Response body:"
        cat "$OUTPUT"; echo
        echo "  Last 40 worker log lines:"
        "${COMPOSE[@]}" logs --no-color --tail 40 "$SERVICE"
        echo
        echo "  → If OutOfMemory during generation: text encoder too big for bf16."
        echo "    Set Z_IMAGE_TURBO_QUANTIZATION=nf4 in deploy/.env and re-run."
        ;;
    *)
        err "Unexpected HTTP $http_code."
        echo "  Response body:"
        cat "$OUTPUT"; echo
        ;;
esac
