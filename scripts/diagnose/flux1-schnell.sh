#!/usr/bin/env bash
# Diagnostic script: does the FlowMatch sigma IndexError come from our warmup,
# or is it an underlying diffusers + sequential_cpu_offload bug on Blackwell?
#
# Steps:
#   1. Ensure FLUX1_SCHNELL_WARMUP=false + COMPOSE_PROFILES includes flux1-schnell
#   2. `docker compose up -d` (creates container if missing, starts if stopped)
#   3. Wait for "Worker ready" in compose logs
#   4. Fire a single generation request at 512×512
#   5. Report outcome
#
# Run from project root:
#   bash scripts/diagnose_flux1_schnell.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-flux1-schnell"
OUTPUT="flux_diag.png"
READY_TIMEOUT=600   # seconds to wait for "Worker ready"

log() { printf '\033[1;36m[diag]\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[err ]\033[0m %s\n' "$*"; }

# ─── 1. Ensure env flag ────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    err "$ENV_FILE not found — create it from deploy/.env.example first."
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

update_env COMPOSE_PROFILES flux1-schnell
# NF4 quantization + no offload: ~9GB in VRAM, no accelerate hooks
# (sequential/model offload deadlocks on Blackwell).
update_env FLUX1_SCHNELL_QUANTIZATION nf4
update_env FLUX1_SCHNELL_CPU_OFFLOAD false
update_env FLUX1_SCHNELL_SEQUENTIAL_OFFLOAD false
update_env FLUX1_SCHNELL_WARMUP false  # keep off for first test
ok "Env flags set (nf4 + no offload)"

# ─── 2. Bring up the worker (idempotent, rebuilds if args changed) ─────
log "Starting $SERVICE (builds if Dockerfile args changed) …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

# ─── 3. Wait for truly-available ───────────────────────────────────────
# Three stages:
#   (a) worker process loads model
#   (b) gateway registers worker (model appears in /v1/models)
#   (c) gateway completes worker health-check → "model is now available"
# Only after (c) can we actually serve requests. (c) is marked in gateway
# logs by "model is now available".
log "Waiting up to ${READY_TIMEOUT}s for gateway to mark flux1-schnell available …"
start=$SECONDS
while true; do
    worker_logs=$("${COMPOSE[@]}" logs --no-color "$SERVICE" 2>&1 || true)
    if grep -qE 'Application startup failed|CUDA driver error|CRITICAL' <<<"$worker_logs"; then
        err "Worker startup failure detected. Last 50 log lines:"
        "${COMPOSE[@]}" logs --no-color --tail 50 "$SERVICE"
        exit 1
    fi

    gateway_logs=$("${COMPOSE[@]}" logs --no-color gateway 2>&1 || true)
    if grep -q 'Worker ready: flux1-schnell.*model is now available' <<<"$gateway_logs"; then
        ok "Gateway marks flux1-schnell as available"
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

# ─── 4. Fire request ───────────────────────────────────────────────────
log "POST /v1/images/generations (prompt=cyberpunk cat, 512×512) …"
RESP_JSON=$(mktemp --suffix=.json)
t0=$SECONDS
http_code=$(curl -s -o "$RESP_JSON" -w '%{http_code}' \
    -X POST http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d '{"model":"flux1-schnell","prompt":"a cyberpunk cat","size":"512x512"}' || echo 000)
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

# ─── 5. Report ─────────────────────────────────────────────────────────
echo
echo "─── Result ────────────────────────────────────────────────"
echo "  HTTP status : $http_code"
echo "  Elapsed     : ${elapsed}s"
echo "  Output      : $OUTPUT ($(wc -c < "$OUTPUT" 2>/dev/null || echo 0) bytes)"
echo

case "$http_code" in
    200)
        ok "SUCCESS — image generated. Warmup was the culprit (or the bug resolved itself)."
        echo "  → Next: re-enable warmup with safer defaults (drop num_inference_steps=1 override)."
        ;;
    500|504)
        err "FAILED — worker returned $http_code."
        echo "  Response body:"
        cat "$OUTPUT"; echo
        echo "  Last 40 worker log lines:"
        "${COMPOSE[@]}" logs --no-color --tail 40 "$SERVICE"
        echo
        echo "  → If you see the same 'IndexError sigmas[sigma_idx + 1]', the bug is"
        echo "    in FLUX + sequential_cpu_offload (diffusers), NOT our warmup."
        echo "    Next step: drop sequential_offload, enable nf4 quantization so 12B"
        echo "    transformer fits in 12GB without offload."
        ;;
    *)
        err "Unexpected HTTP $http_code."
        echo "  Response body:"
        cat "$OUTPUT"; echo
        echo "  Check: ${COMPOSE[*]} ps"
        ;;
esac
