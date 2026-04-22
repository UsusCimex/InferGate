#!/usr/bin/env bash
# Feature test: worker-side hot-reload via gateway ConfigWatcher →
# ProviderManager.reload_model() → RemoteProvider.reload() → worker
# POST /reload.
#
# Two reload classes are exercised end-to-end:
#
#   1. metadata-only edit (metadata.description)
#      Expected gateway log: "Reloaded model kokoro-82m via worker
#      /reload (action=metadata)"
#      Expected worker log:  "Reloaded kokoro-82m (metadata only)"
#
#   2. full-reload edit (model.default_params.speed)
#      Expected gateway log: "Reloaded model kokoro-82m via worker
#      /reload (action=full_reload)"
#      Expected worker log:  "Reloaded kokoro-82m (full reload complete)"
#
# After each edit we also fire a /v1/audio/speech request to prove
# the worker stayed serviceable through the swap.
#
# YAML is snapshot-restored on trap EXIT regardless of outcome.
#
# Runs on kokoro-82m because:
#   * CPU TTS → no GPU pressure if another test left models loaded
#   * a "speed" parameter exists in default_params so mutating it is
#     a meaningful full-reload trigger
#   * read-only bind mount of config/models/kokoro-82m.yaml into the
#     worker means the host edit is immediately visible to the worker
#     too (though worker itself only re-reads via POST /reload body,
#     not the mounted file)
#
# Run from project root: bash scripts/feature/worker-reload.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_YAML="config/models/kokoro-82m.yaml"
MODEL_ID="kokoro-82m"
SERVICE="worker-kokoro-82m"
HF_CACHE="models/models--hexgrad--Kokoro-82M"
READY_TIMEOUT=600
GATEWAY_URL="http://localhost:8000"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

command -v curl >/dev/null 2>&1 || { err "curl required"; exit 1; }
PY=""
for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[[ -n "$PY" ]] || { err "python required"; exit 1; }

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }
[[ -f "$MODEL_YAML" ]] || { err "$MODEL_YAML missing — bad working dir?"; exit 1; }

# Snapshot YAML for trap-restore.
ORIGINAL=$(mktemp --suffix=.yaml)
cp "$MODEL_YAML" "$ORIGINAL"
restore_yaml() {
    cp "$ORIGINAL" "$MODEL_YAML"
    rm -f "$ORIGINAL"
    log "restored $MODEL_YAML"
}
trap restore_yaml EXIT

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"

log "Rebuilding gateway + kokoro worker (new /reload endpoint code) …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"
ok "compose up issued"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

# Small grace period after "Worker ready" so the worker monitor's next
# probe lands inside is_loaded() state before we start mutating YAML.
sleep 2

# Helper: read a field from /v1/models → kokoro-82m entry.
read_field() {
    local field="$1"
    curl -sS "${GATEWAY_URL}/v1/models" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
for m in d.get('data', []):
    if m.get('id') == '$MODEL_ID':
        if '$field' == 'description':
            print(m.get('metadata', {}).get('description', ''))
        else:
            print(m.get('$field', ''))
        break
"
}

# Helper: post a trivial TTS request and assert 200.
smoke_tts() {
    local label="$1"
    local resp; resp=$(mktemp --suffix=.wav)
    local code
    code=$(curl -s -o "$resp" -w '%{http_code}' \
        -X POST "${GATEWAY_URL}/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -H 'X-InferGate-No-Cache: true' \
        -d "$(printf '{"model":"%s","input":"hot reload smoke %s","voice":"af_heart"}' "$MODEL_ID" "$label")" \
        || echo 000)
    local sz; sz=$(wc -c < "$resp" 2>/dev/null || echo 0)
    rm -f "$resp"
    if [[ "$code" != "200" ]]; then
        err "smoke_tts [$label]: HTTP $code ($sz bytes)"
        return 1
    fi
    ok "smoke_tts [$label]: HTTP 200 ($sz bytes)"
}

# ── Phase 1: metadata-only edit ─────────────────────────────────────
log "[phase 1] editing metadata.description (expect action=metadata) …"
BEFORE_DESC=$(read_field description)
NEW_DESC="Hot-reloaded metadata at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" <<PYEOF
import re
path = "$MODEL_YAML"
with open(path) as f:
    text = f.read()
text = re.sub(r'^  description:.*\$', '  description: "$NEW_DESC"', text, count=1, flags=re.MULTILINE)
with open(path, "w") as f:
    f.write(text)
PYEOF

# Wait for watcher cycle (interval 2s + debounce 0.5s) + HTTP roundtrip slack
log "waiting 6s for ConfigWatcher + worker /reload roundtrip …"
sleep 6

AFTER_DESC=$(read_field description)
fail=0
if [[ "$AFTER_DESC" == "$NEW_DESC" ]]; then
    ok "[phase 1] gateway /v1/models now shows new description"
else
    err "[phase 1] description still '$AFTER_DESC' (expected '$NEW_DESC')"
    fail=1
fi

# Gateway log should mention the reload path and action=metadata
if "${COMPOSE[@]}" logs --no-color --tail 200 gateway 2>&1 | grep -qE "Reloaded model ${MODEL_ID} via worker /reload \(action=metadata"; then
    ok "[phase 1] gateway log: worker /reload action=metadata"
else
    err "[phase 1] no 'Reloaded model kokoro-82m via worker /reload (action=metadata' in gateway log"
    fail=1
fi

# Worker log should mention metadata-only reload
if "${COMPOSE[@]}" logs --no-color --tail 200 "$SERVICE" 2>&1 | grep -qE "Reloaded ${MODEL_ID} \(metadata only\)"; then
    ok "[phase 1] worker log: Reloaded kokoro-82m (metadata only)"
else
    err "[phase 1] no 'Reloaded kokoro-82m (metadata only)' in worker log"
    fail=1
fi

smoke_tts "phase1" || fail=1

# ── Phase 2: full-reload edit (model.default_params.speed) ──────────
log "[phase 2] editing model.default_params.speed (expect action=full_reload) …"
# kokoro YAML has speed via ${oc.env:KOKORO_82M_SPEED,1.0}. We replace the
# whole line with a concrete new value so the change is unambiguous on disk.
NEW_SPEED="1.25"
"$PY" <<PYEOF
import re
path = "$MODEL_YAML"
with open(path) as f:
    text = f.read()
text = re.sub(
    r'^    speed:.*\$',
    f'    speed: $NEW_SPEED',
    text, count=1, flags=re.MULTILINE,
)
with open(path, "w") as f:
    f.write(text)
PYEOF

log "waiting 10s for full-reload (new provider load() on kokoro takes ~seconds) …"
sleep 10

# Gateway log: full_reload action
if "${COMPOSE[@]}" logs --no-color --tail 200 gateway 2>&1 | grep -qE "Reloaded model ${MODEL_ID} via worker /reload \(action=full_reload"; then
    ok "[phase 2] gateway log: worker /reload action=full_reload"
else
    err "[phase 2] no 'action=full_reload' in gateway log"
    "${COMPOSE[@]}" logs --no-color --tail 40 gateway | sed 's/^/  /'
    fail=1
fi

# Worker log: full reload complete
if "${COMPOSE[@]}" logs --no-color --tail 200 "$SERVICE" 2>&1 | grep -qE "Reloaded ${MODEL_ID} \(full reload complete\)"; then
    ok "[phase 2] worker log: Reloaded kokoro-82m (full reload complete)"
else
    err "[phase 2] no 'full reload complete' in worker log"
    "${COMPOSE[@]}" logs --no-color --tail 40 "$SERVICE" | sed 's/^/  /'
    fail=1
fi

smoke_tts "phase2" || fail=1

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }
ok "PASS — worker-side hot-reload works end-to-end for both metadata and full-reload paths."
