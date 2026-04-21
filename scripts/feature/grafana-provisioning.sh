#!/usr/bin/env bash
# Feature test: Grafana auto-provisioning (datasource + dashboard).
#
# What this verifies end-to-end:
#   1. Gateway exposes infergate_* Prometheus metrics
#   2. Prometheus successfully scrapes the gateway target (state="up")
#   3. Prometheus returns non-empty data for a real InferGate metric
#   4. Grafana auto-provisioned the Prometheus datasource with the
#      expected uid (infergate-prometheus) — no UI click-through
#   5. Grafana auto-loaded the InferGate dashboard from
#      /var/lib/grafana/dashboards (uid=infergate-main) into the "InferGate" folder
#   6. Dashboard references a resolvable datasource uid (sanity on the JSON)
#
# This runs gateway-only (no workers) — cheap & fast. Metrics populated
# here are those emitted for gateway-level requests (HTTP, models_loaded,
# queue depth). Inference/cache metrics stay at zero, which is expected
# and doesn't invalidate the test: we're checking provisioning wiring,
# not inference behaviour.
#
# Run from project root: bash scripts/feature/grafana-provisioning.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

COMPOSE=(
    docker compose
    -f deploy/docker-compose.yml
    -f deploy/monitoring/docker-compose.monitoring.yml
)

GATEWAY_URL="http://localhost:8000"
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"
GRAFANA_AUTH="admin:admin"
DS_UID="infergate-prometheus"
DASHBOARD_UID="infergate-main"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

command -v curl >/dev/null 2>&1 || { err "curl required"; exit 1; }

# Pick a python interpreter — tolerate python3 / python / py (Windows bash).
PY=""
for candidate in python3 python py; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
[[ -n "$PY" ]] || { err "python interpreter required (python3 / python / py)"; exit 1; }

# jq-free JSON query helper. Uses Python to pick a value out of JSON passed
# on stdin via a dotted path with bracket indices, e.g. `.data.result[0].value[1]`
# or `.status`. Missing keys return empty string (matches jq `// ""`).
# Supports the three specific shapes this script needs — it's not a jq clone.
jq_get() {
    local path="$1"
    "$PY" -c "
import json, sys, re
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
parts = re.findall(r'\.([A-Za-z_][\w]*)|\[(\d+)\]', '''$path''')
for key, idx in parts:
    if key:
        if isinstance(d, dict):
            d = d.get(key, '')
        else:
            d = ''
    elif idx:
        try:
            d = d[int(idx)]
        except Exception:
            d = ''
    if d == '':
        break
if d is None:
    d = ''
print(d if not isinstance(d, (dict, list)) else json.dumps(d))
"
}

# For target-health we need to filter by label job='infergate' first — give
# it a dedicated helper rather than extending jq_get into a full query lang.
prometheus_target_health() {
    "$PY" -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for t in d.get('data', {}).get('activeTargets', []):
    if t.get('labels', {}).get('job') == 'infergate':
        print(t.get('health', ''))
        break
"
}

# Collect unique panel-level datasource uids from a Grafana dashboard JSON.
dashboard_panel_uids() {
    "$PY" -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
uids = set()
for p in d.get('dashboard', {}).get('panels', []):
    ds = p.get('datasource')
    if isinstance(ds, dict) and ds.get('uid'):
        uids.add(ds['uid'])
print(','.join(sorted(uids)))
"
}

# ── Bring up gateway + monitoring stack ───────────────────────────────
# --build forces the gateway image to be rebuilt — required when base.txt
# changes (e.g. prometheus-client was promoted from optional to core). No-op
# on unchanged layers thanks to BuildKit cache.
log "Starting gateway + prometheus + grafana (rebuilding gateway if needed) …"
"${COMPOSE[@]}" up -d --build gateway prometheus grafana
ok "compose up issued"

wait_http() {
    local label="$1" url="$2" timeout="$3"
    local start=$SECONDS
    while true; do
        local code
        code=$(curl -s -o /dev/null -w '%{http_code}' "$url" || echo 000)
        if [[ "$code" == "200" || "$code" == "302" ]]; then
            ok "$label ready ($url → $code)"
            return 0
        fi
        if (( SECONDS - start > timeout )); then
            err "$label not ready after ${timeout}s (last $code)"
            return 1
        fi
        sleep 2
    done
}

wait_http gateway    "${GATEWAY_URL}/health"         60
wait_http prometheus "${PROMETHEUS_URL}/-/ready"     60
wait_http grafana    "${GRAFANA_URL}/api/health"     60

# ── Generate traffic so metrics aren't all zero ───────────────────────
log "Generating request traffic to populate request metrics …"
for _ in $(seq 1 5); do
    curl -sS "${GATEWAY_URL}/health"    >/dev/null || true
    curl -sS "${GATEWAY_URL}/v1/models" >/dev/null || true
done
ok "10 requests issued"

# Prometheus scrape interval = 15s — give it one full cycle.
log "Waiting 20s for Prometheus to scrape …"
sleep 20

fail=0

# ── (1) Gateway emits infergate_* metrics ─────────────────────────────
log "[1] Gateway /metrics/prometheus contains infergate_* series"
METRICS_BODY=$(curl -sS "${GATEWAY_URL}/metrics/prometheus" || true)
if grep -q '^infergate_requests_total' <<<"$METRICS_BODY" \
   && grep -q '^infergate_models_loaded' <<<"$METRICS_BODY"; then
    ok "gateway emits infergate_requests_total + infergate_models_loaded"
else
    err "gateway metrics endpoint missing expected series"
    echo "$METRICS_BODY" | head -30
    fail=1
fi

# ── (2) Prometheus scraped gateway successfully ───────────────────────
log "[2] Prometheus target gateway:8000 is UP"
TARGETS_JSON=$(curl -sS "${PROMETHEUS_URL}/api/v1/targets" || echo '{}')
TARGET_HEALTH=$(prometheus_target_health <<<"$TARGETS_JSON")
if [[ "$TARGET_HEALTH" == "up" ]]; then
    ok "infergate target health=up"
else
    err "infergate target health='$TARGET_HEALTH' (expected 'up')"
    echo "$TARGETS_JSON"
    fail=1
fi

# ── (3) Prometheus returns data for infergate_requests_total ──────────
log "[3] Prometheus has non-zero infergate_requests_total"
QUERY_JSON=$(curl -sS --data-urlencode 'query=sum(infergate_requests_total)' \
    "${PROMETHEUS_URL}/api/v1/query" || echo '{}')
QUERY_STATUS=$(jq_get '.status' <<<"$QUERY_JSON")
QUERY_VALUE=$(jq_get '.data.result[0].value[1]' <<<"$QUERY_JSON")
[[ -z "$QUERY_VALUE" ]] && QUERY_VALUE=0
if [[ "$QUERY_STATUS" == "success" ]] && (( $(printf '%.0f' "$QUERY_VALUE") > 0 )); then
    ok "sum(infergate_requests_total) = ${QUERY_VALUE}"
else
    err "query status='$QUERY_STATUS', value='$QUERY_VALUE'"
    echo "$QUERY_JSON"
    fail=1
fi

# ── (4) Grafana datasource auto-provisioned ───────────────────────────
log "[4] Grafana datasource uid=${DS_UID} present"
DS_JSON=$(curl -sS -u "$GRAFANA_AUTH" "${GRAFANA_URL}/api/datasources/uid/${DS_UID}" || echo '{}')
DS_TYPE=$(jq_get '.type' <<<"$DS_JSON")
DS_URL=$(jq_get '.url' <<<"$DS_JSON")
if [[ "$DS_TYPE" == "prometheus" ]]; then
    ok "datasource provisioned: type=prometheus url=${DS_URL}"
else
    err "datasource uid=${DS_UID} not provisioned (type='$DS_TYPE')"
    echo "$DS_JSON"
    fail=1
fi

# ── (5) Grafana dashboard auto-loaded ─────────────────────────────────
log "[5] Grafana dashboard uid=${DASHBOARD_UID} present"
# Dashboard provisioner runs every 30s; we've already slept 20s. Give it one more cycle.
DASH_JSON=""
DASH_TITLE=""
for attempt in 1 2 3; do
    DASH_JSON=$(curl -sS -u "$GRAFANA_AUTH" "${GRAFANA_URL}/api/dashboards/uid/${DASHBOARD_UID}" || echo '{}')
    DASH_TITLE=$(jq_get '.dashboard.title' <<<"$DASH_JSON")
    if [[ -n "$DASH_TITLE" ]]; then
        break
    fi
    log "  attempt ${attempt}/3 — dashboard not yet loaded, waiting 15s …"
    sleep 15
done
DASH_FOLDER=$(jq_get '.meta.folderTitle' <<<"$DASH_JSON")
if [[ "$DASH_TITLE" == "InferGate — Gateway & Inference" ]]; then
    ok "dashboard loaded: title='${DASH_TITLE}' folder='${DASH_FOLDER}'"
else
    err "dashboard uid=${DASHBOARD_UID} not found or title mismatch (got '$DASH_TITLE')"
    err "provisioning log tail:"
    "${COMPOSE[@]}" logs --no-color --tail 40 grafana | grep -iE 'provision|dashboard|datasource' || true
    fail=1
fi

# ── (6) Dashboard panels reference the resolvable datasource uid ──────
log "[6] Dashboard panels reference datasource uid=${DS_UID}"
PANEL_UIDS=$(dashboard_panel_uids <<<"$DASH_JSON")
if grep -q "$DS_UID" <<<"$PANEL_UIDS"; then
    ok "panels datasource uids: ${PANEL_UIDS}"
else
    err "no panel targets datasource uid=${DS_UID} (got: '${PANEL_UIDS}')"
    err "this means the JSON ships wrong uids and panels will render 'Datasource not found'"
    fail=1
fi

echo
(( fail )) && {
    err "FAIL — see above."
    err "Quick diagnostics:"
    err "  grafana provisioning log:"
    "${COMPOSE[@]}" logs --no-color --tail 20 grafana | grep -iE 'provision|dashboard|datasource' || true
    exit 1
}

ok "PASS — Grafana provisioning wired end-to-end."
echo
echo "Open in browser to inspect visually:"
echo "  Grafana dashboard : ${GRAFANA_URL}/d/${DASHBOARD_UID}  (admin / admin)"
echo "  Prometheus targets: ${PROMETHEUS_URL}/targets"
echo "  Gateway metrics   : ${GATEWAY_URL}/metrics/prometheus"
