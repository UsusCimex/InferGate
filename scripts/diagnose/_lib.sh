# Shared helpers for InferGate diagnostic scripts.
#
# Source at top of each script AFTER setting:
#   - PROJECT_ROOT (cd'd to)
#   - COMPOSE      (array — e.g. `COMPOSE=(docker compose -f deploy/docker-compose.yml)`)
#
# Convention: log/ok/err colourise stderr for humans; the rest of the script
# keeps stdout for machine-readable output (HTTP code, elapsed, etc.).

# ── pretty prints ────────────────────────────────────────────────────────
log() { printf '\033[1;36m[diag]\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[err ]\033[0m %s\n' "$*"; }

# ── idempotent env-file update ───────────────────────────────────────────
# Usage: update_env <env-file> <key> <value>
update_env() {
    local file="$1" key="$2" val="$3"
    if grep -q "^${key}=" "$file" 2>/dev/null; then
        log "Updating ${key}=${val}"
        local tmp
        tmp=$(mktemp)
        sed "s|^${key}=.*|${key}=${val}|" "$file" > "$tmp"
        mv "$tmp" "$file"
    else
        log "Appending ${key}=${val}"
        printf '\n%s=%s\n' "$key" "$val" >> "$file"
    fi
}

# ── model-cache progress ─────────────────────────────────────────────────
# Usage: show_download_progress <cache-dir-path>
show_download_progress() {
    local cache_dir="$1"
    if [[ -d "$cache_dir" ]]; then
        local sz
        sz=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)
        log "cache: $cache_dir → $sz"
    fi
}

# ── wait for worker ready ────────────────────────────────────────────────
# Polls gateway logs every 5s for "Worker ready: <model_id> ... model is now
# available". Every 30s also prints the HF cache dir size so long downloads
# are observable. Fails fast on known fatal patterns in worker logs.
#
# Requires: $COMPOSE array set in caller scope.
# Usage:
#   wait_for_worker <service> <cache-dir> <model-id> <timeout-seconds>
wait_for_worker() {
    local service="$1" cache_dir="$2" model_id="$3" timeout="$4"
    local fatal='Application startup failed|CUDA driver error|CRITICAL|ValueError'

    local start=$SECONDS
    local progress_tick=$start
    while true; do
        local worker_logs
        worker_logs=$("${COMPOSE[@]}" logs --no-color "$service" 2>&1 || true)
        if grep -qE "$fatal" <<<"$worker_logs"; then
            err "Worker startup failure detected. Last 60 log lines:"
            "${COMPOSE[@]}" logs --no-color --tail 60 "$service"
            return 1
        fi

        local gateway_logs
        gateway_logs=$("${COMPOSE[@]}" logs --no-color gateway 2>&1 || true)
        if grep -q "Worker ready: ${model_id}.*model is now available" <<<"$gateway_logs"; then
            ok "Gateway marks $model_id as available"
            return 0
        fi

        local now=$SECONDS
        if (( now - progress_tick >= 30 )); then
            show_download_progress "$cache_dir"
            progress_tick=$now
        fi

        if (( now - start > timeout )); then
            err "Timed out after ${timeout}s. Last 60 worker log lines:"
            "${COMPOSE[@]}" logs --no-color --tail 60 "$service"
            echo
            err "Last 20 gateway log lines:"
            "${COMPOSE[@]}" logs --no-color --tail 20 gateway
            return 1
        fi
        sleep 5
    done
}

# ── decode OpenAI-compat JSON image response → PNG file ──────────────────
# Tries python3/python/py for robust decoding, falls back to sed+base64.
# Usage: decode_b64_png <response-json> <output-png>
decode_b64_png() {
    local resp="$1" out="$2"
    for py in python3 python py; do
        if command -v "$py" >/dev/null 2>&1; then
            "$py" -c "
import json, base64
with open('$resp') as f: d = json.load(f)
with open('$out', 'wb') as f: f.write(base64.b64decode(d['data'][0]['b64_json']))
" && return 0
        fi
    done
    if command -v base64 >/dev/null 2>&1; then
        sed -n 's/.*"b64_json":"\([^"]*\)".*/\1/p' "$resp" | base64 -d > "$out" && return 0
    fi
    return 1
}

# ── fire a single image-gen request and decode ──────────────────────────
# Sets the global vars HTTP_CODE, ELAPSED, OUTPUT_SIZE.
# Usage: fire_image_request <model-id> <prompt> <output-path> [json-extra]
fire_image_request() {
    local model="$1" prompt="$2" out="$3" extra="${4:-}"
    local resp
    resp=$(mktemp --suffix=.json)

    local body
    if [[ -n "$extra" ]]; then
        body=$(printf '{"model":"%s","prompt":"%s",%s}' "$model" "$prompt" "$extra")
    else
        body=$(printf '{"model":"%s","prompt":"%s"}' "$model" "$prompt")
    fi

    local t0=$SECONDS
    HTTP_CODE=$(curl -s -o "$resp" -w '%{http_code}' \
        -X POST http://localhost:8000/v1/images/generations \
        -H 'Content-Type: application/json' \
        -d "$body" || echo 000)
    ELAPSED=$(( SECONDS - t0 ))

    if [[ "$HTTP_CODE" == "200" ]]; then
        if decode_b64_png "$resp" "$out"; then :; else
            err "Failed to decode base64 payload; saving raw JSON to $out"
            cp "$resp" "$out"
        fi
    else
        cp "$resp" "$out"
    fi
    rm -f "$resp"

    OUTPUT_SIZE=$(wc -c < "$out" 2>/dev/null || echo 0)
}

# ── print final result banner ────────────────────────────────────────────
# Reads HTTP_CODE / ELAPSED / OUTPUT_SIZE set by fire_image_request.
# Usage: report_result <service> <output-path>
report_result() {
    local service="$1" out="$2"
    echo
    echo "─── Result ────────────────────────────────────────────────"
    echo "  HTTP status : $HTTP_CODE"
    echo "  Elapsed     : ${ELAPSED}s"
    echo "  Output      : $out ($OUTPUT_SIZE bytes)"
    echo

    case "$HTTP_CODE" in
        200)
            ok "SUCCESS — image generated in ${ELAPSED}s."
            ;;
        500|504)
            err "FAILED — HTTP $HTTP_CODE."
            echo "  Response body:"
            cat "$out"; echo
            echo "  Last 40 worker log lines:"
            "${COMPOSE[@]}" logs --no-color --tail 40 "$service"
            ;;
        *)
            err "Unexpected HTTP $HTTP_CODE."
            echo "  Response body:"
            cat "$out"; echo
            ;;
    esac
}
