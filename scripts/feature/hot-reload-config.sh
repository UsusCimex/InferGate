#!/usr/bin/env bash
# Feature test: ConfigWatcher picks up per-model YAML edits at runtime.
#
# Scenario:
#   1. Start gateway only (no workers needed — we're testing the
#      gateway-side registry refresh, not worker-side model reload).
#   2. Snapshot GET /v1/models → remember kokoro-82m's display_name.
#   3. Rewrite the `display_name:` line in config/models/kokoro-82m.yaml.
#   4. Wait ~5s (watcher interval 2s + debounce 0.5s + slack).
#   5. GET /v1/models again — expect the new display_name.
#   6. Also assert metadata.description change propagates, to prove it
#      isn't just `display_name` that comes through.
#   7. Trap-based cleanup restores the YAML even on test failure, so
#      the repo stays clean.
#
# Why kokoro-82m:
#   * TTS category, CPU-only, tiny config — no VRAM impact from a
#     gateway-only run.
#   * Even though its worker (if launched) would ignore gateway-side
#     reload, THIS test only exercises the gateway's registry view,
#     which is what the watcher controls.
#
# This does not require the kokoro worker to be running — the gateway
# registers the model purely from YAML. `/v1/models` reflects the
# registry without ever touching the worker.
#
# Run from project root: bash scripts/feature/hot-reload-config.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

COMPOSE=(docker compose -f deploy/docker-compose.yml)
GATEWAY_URL="http://localhost:8000"
MODEL_YAML="config/models/kokoro-82m.yaml"
MODEL_ID="kokoro-82m"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

command -v curl >/dev/null 2>&1 || { err "curl required"; exit 1; }
PY=""
for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[[ -n "$PY" ]] || { err "python required"; exit 1; }

[[ -f "$MODEL_YAML" ]] || { err "$MODEL_YAML missing — bad working dir?"; exit 1; }

# Snapshot YAML so we can restore on any exit path.
ORIGINAL=$(mktemp --suffix=.yaml)
cp "$MODEL_YAML" "$ORIGINAL"
restore_yaml() {
    cp "$ORIGINAL" "$MODEL_YAML"
    rm -f "$ORIGINAL"
    log "restored $MODEL_YAML to its original contents"
}
trap restore_yaml EXIT

log "Rebuilding + starting gateway (ConfigWatcher is new code) …"
"${COMPOSE[@]}" build gateway
"${COMPOSE[@]}" up -d gateway
ok "compose up issued"

# Wait for /health before any /v1/models reads.
start=$SECONDS
until curl -sS -o /dev/null -w '%{http_code}\n' "${GATEWAY_URL}/health" | grep -q 200; do
    (( SECONDS - start > 90 )) && { err "gateway did not become ready in 90s"; exit 1; }
    sleep 1
done
ok "gateway /health responds 200"

# ── (1) Baseline: read current display_name + description ────────────
read_model_field() {
    local field="$1"
    curl -sS "${GATEWAY_URL}/v1/models" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
for m in d.get('data', []):
    if m.get('id') == '$MODEL_ID':
        if '$field' == 'description':
            print(m.get('metadata', {}).get('description', ''))
        else:
            print(m.get(\"$field\", ''))
        break
"
}

BEFORE_NAME=$(read_model_field display_name)
BEFORE_DESC=$(read_model_field description)
log "baseline: display_name='$BEFORE_NAME', description='$BEFORE_DESC'"

[[ -n "$BEFORE_NAME" ]] || { err "could not read baseline display_name — model $MODEL_ID not in /v1/models"; exit 1; }

# ── (2) Mutate the YAML — new name tagged with a timestamp so we can
# tell it apart from anything stale the gateway might have cached. ────
NEW_NAME="Kokoro 82M [reloaded-$(date +%s)]"
NEW_DESC="Hot-reloaded at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PY" <<PYEOF
import re, sys
path = "$MODEL_YAML"
with open(path) as f:
    text = f.read()
text = re.sub(
    r'^display_name:.*\$',
    'display_name: "$NEW_NAME"',
    text, count=1, flags=re.MULTILINE,
)
text = re.sub(
    r'^  description:.*\$',
    '  description: "$NEW_DESC"',
    text, count=1, flags=re.MULTILINE,
)
with open(path, "w") as f:
    f.write(text)
PYEOF
ok "YAML mutated on disk:"
echo "    display_name → $NEW_NAME"
echo "    description  → $NEW_DESC"

# ── (3) Wait for watcher cycle to pick it up ────────────────────────
log "waiting 6s for ConfigWatcher (interval=2s + debounce=0.5s + slack) …"
sleep 6

# ── (4) Assert new values show up in /v1/models ─────────────────────
AFTER_NAME=$(read_model_field display_name)
AFTER_DESC=$(read_model_field description)
log "after: display_name='$AFTER_NAME', description='$AFTER_DESC'"

fail=0
if [[ "$AFTER_NAME" == "$NEW_NAME" ]]; then
    ok "display_name hot-reloaded: '$AFTER_NAME'"
else
    err "display_name NOT reloaded (got '$AFTER_NAME', expected '$NEW_NAME')"
    err "gateway log tail:"
    "${COMPOSE[@]}" logs --no-color --tail 30 gateway | sed 's/^/  /'
    fail=1
fi

if [[ "$AFTER_DESC" == "$NEW_DESC" ]]; then
    ok "metadata.description hot-reloaded: '$AFTER_DESC'"
else
    err "metadata.description NOT reloaded (got '$AFTER_DESC', expected '$NEW_DESC')"
    fail=1
fi

# ── (5) Gateway log should mention the reload so operators know it
# happened through the hot-reload path (not a coincidental refresh). ─
if "${COMPOSE[@]}" logs --no-color --tail 120 gateway 2>&1 | grep -qE "Reloaded model ${MODEL_ID}|Config changed: kokoro-82m\.yaml"; then
    ok "gateway log shows the reload event"
else
    log "note: couldn't find 'Reloaded model' in gateway logs — non-fatal if metadata updated"
fi

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }
ok "PASS — hot-reload of per-model YAML works end-to-end."
