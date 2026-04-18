#!/usr/bin/env bash
# Launch + smoke-test Hunyuan-DiT v1.2 — 1.5B DiT by Tencent, bilingual
# (Chinese + English). ~7GB VRAM at fp16, no offload, no quant.
#
# The model is gated with a non-commercial community license — you may need
# to visit https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers
# and accept terms before download works. HF_TOKEN must be set.
#
# Run from project root:
#   bash scripts/diagnose_hunyuan_dit.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
SERVICE="worker-hunyuan-dit"
OUTPUT="hunyuan_diag.png"
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

update_env COMPOSE_PROFILES hunyuan-dit
update_env HUNYUAN_DIT_CPU_OFFLOAD false
update_env HUNYUAN_DIT_SEQUENTIAL_OFFLOAD false
update_env HUNYUAN_DIT_WARMUP false
ok "Env flags set"

log "Starting $SERVICE (builds if needed) …"
"${COMPOSE[@]}" up -d --build "$SERVICE" gateway
ok "Service up issued"

log "Waiting up to ${READY_TIMEOUT}s for gateway to mark hunyuan-dit available …"
start=$SECONDS
while true; do
    worker_logs=$("${COMPOSE[@]}" logs --no-color "$SERVICE" 2>&1 || true)
    if grep -qE 'Application startup failed|CUDA driver error|CRITICAL' <<<"$worker_logs"; then
        err "Worker startup failure detected."
        "${COMPOSE[@]}" logs --no-color --tail 60 "$SERVICE"
        echo
        if grep -q 'GatedRepoError\|401\|access to model' <<<"$worker_logs"; then
            err "Root cause: model access denied."
            echo "    → Visit https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
            echo "      and click 'Agree and access repository'. Then re-run."
        elif grep -q 'out of memory\|OutOfMemoryError' <<<"$worker_logs"; then
            err "Root cause: OOM. Set HUNYUAN_DIT_QUANTIZATION=nf4 in $ENV_FILE,"
            echo "    add bitsandbytes + build-essential to the worker, re-run."
        fi
        exit 1
    fi

    gateway_logs=$("${COMPOSE[@]}" logs --no-color gateway 2>&1 || true)
    if grep -q 'Worker ready: hunyuan-dit.*model is now available' <<<"$gateway_logs"; then
        ok "Gateway marks hunyuan-dit as available"
        break
    fi

    if (( SECONDS - start > READY_TIMEOUT )); then
        err "Timed out. Last worker logs:"
        "${COMPOSE[@]}" logs --no-color --tail 60 "$SERVICE"
        exit 1
    fi
    sleep 5
done

# Bilingual test: one English, one Chinese.
run_test() {
    local label="$1" prompt="$2" outfile="$3"
    log "POST /v1/images/generations ($label) …"
    local resp
    resp=$(mktemp --suffix=.json)
    local t0=$SECONDS
    local code
    code=$(curl -s -o "$resp" -w '%{http_code}' \
        -X POST http://localhost:8000/v1/images/generations \
        -H 'Content-Type: application/json' \
        --data-binary "$(printf '{"model":"hunyuan-dit","prompt":"%s"}' "$prompt")" || echo 000)
    local elapsed=$(( SECONDS - t0 ))

    if [[ "$code" == "200" ]]; then
        for py in python3 python py; do
            if command -v "$py" >/dev/null 2>&1; then
                "$py" -c "
import json, base64
with open('$resp') as f: d = json.load(f)
with open('$outfile', 'wb') as f: f.write(base64.b64decode(d['data'][0]['b64_json']))
" && break
            fi
        done
        if [[ ! -s "$outfile" ]]; then
            sed -n 's/.*"b64_json":"\([^"]*\)".*/\1/p' "$resp" | base64 -d > "$outfile" || true
        fi
    else
        cp "$resp" "$outfile"
    fi
    rm -f "$resp"
    echo "  $label: HTTP $code, ${elapsed}s → $outfile ($(wc -c < "$outfile" 2>/dev/null || echo 0) bytes)"
    return 0
}

echo
echo "─── Bilingual test ────────────────────────────────────────"
run_test "English" "a cyberpunk cat in neon Tokyo" "${OUTPUT%.png}_en.png"
run_test "中文"    "赛博朋克风格的猫，霓虹灯下的东京"   "${OUTPUT%.png}_zh.png"
echo

ok "Done. Compare ${OUTPUT%.png}_en.png and ${OUTPUT%.png}_zh.png."
echo "  → Hunyuan-DiT's key differentiator is native Chinese prompt handling —"
echo "    the Chinese prompt should produce a semantically accurate image,"
echo "    not just a transliteration."
