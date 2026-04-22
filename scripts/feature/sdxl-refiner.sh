#!/usr/bin/env bash
# Feature test: SDXL Refiner ensemble on sdxl-base worker.
#
# Four scenarios, same seed across each group so we can tell what the
# refiner actually changed:
#
#   A) baseline            — sdxl-base alone, no refiner
#   B) refiner at 0.8      — base does 80% steps, refiner polishes 20%
#   C) repeat of B         — determinism check (same seed → same bytes)
#   D) refiner at 0.9      — refiner gets a smaller share; output differs
#                            from B (diff refiner_switch_at → diff image)
#   E) bounds              — refiner_switch_at=1.5 → 422
#
# We don't compare against a golden image — fp16 CUDA kernel autotune
# makes exact bytes non-portable. We do assert B ≡ C byte-for-byte on
# the same machine and same run (determinism) and A ≠ B ≠ D (refiner
# actually changed the output).
#
# VRAM note: fp16 base (~7GB) + fp16 refiner (~6GB) = ~13GB. RTX 4090/3090
# fit bare. On 12GB cards (3060/3080/4070/5070) enable
#   SDXL_BASE_SEQUENTIAL_OFFLOAD=true
# in deploy/.env before running.
#
# Run from project root: bash scripts/feature/sdxl-refiner.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sdxl-base"
SERVICE="worker-sdxl-base"
HF_CACHE="models/models--stabilityai--stable-diffusion-xl-base-1.0"
REFINER_CACHE="models/models--stabilityai--stable-diffusion-xl-refiner-1.0"
READY_TIMEOUT=1800  # refiner is ~6GB, first-load can be 20+ min on slow links

GATEWAY_URL="http://localhost:8000"
REFINER_HUB="stabilityai/stable-diffusion-xl-refiner-1.0"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

command -v curl >/dev/null 2>&1 || { err "curl required"; exit 1; }

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
update_env "$ENV_FILE" SDXL_BASE_REFINER_HUB_ID "$REFINER_HUB"
# 12 GB cards can't hold base + refiner in VRAM — sequential_cpu_offload
# streams layers on demand (slower but safe).
if [[ "${SDXL_BASE_SEQUENTIAL_OFFLOAD:-}" != "false" ]]; then
    update_env "$ENV_FILE" SDXL_BASE_SEQUENTIAL_OFFLOAD true
fi
# Offloaded inference takes minutes per image; raise queue timeout so
# the request doesn't 504 before finishing.
update_env "$ENV_FILE" SDXL_BASE_TIMEOUT 1200
ok "Env flags set (refiner_hub_id=$REFINER_HUB, sequential_offload=true, timeout=1200s)"

log "Rebuilding gateway + $SERVICE …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID (loads base + refiner ~6 GB) …"
wait_for_worker "$SERVICE" "$REFINER_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

COMMON='"seed":42,"num_inference_steps":25,"size":"1024x1024","scheduler":"dpm++_2m"'
PROMPT="an ornate oil painting of a steampunk clockwork cat"

OUT_A=feature_refiner_A_base.png
OUT_B=feature_refiner_B_sw08.png
OUT_C=feature_refiner_C_repeat.png
OUT_D=feature_refiner_D_sw09.png

fire() {
    local label="$1" body="$2" out="$3"
    log "[$label]"
    local resp
    resp=$(mktemp --suffix=.json)
    local t0=$SECONDS
    local code
    # Matches upstream SDXL_BASE_TIMEOUT + slack so curl doesn't 504
    # while the worker is still producing.
    code=$(curl -s --max-time 1500 -o "$resp" -w '%{http_code}' \
        -X POST "${GATEWAY_URL}/v1/images/generations" \
        -H 'Content-Type: application/json' \
        -H 'X-InferGate-No-Cache: true' \
        -d "$body" || echo 000)
    local elapsed=$(( SECONDS - t0 ))
    if [[ "$code" == "200" ]]; then
        decode_b64_png "$resp" "$out" || cp "$resp" "$out"
    else
        cp "$resp" "$out"
    fi
    local sz; sz=$(wc -c < "$out" 2>/dev/null || echo 0)
    echo "  HTTP $code, ${elapsed}s → $out ($sz bytes)"
    LAST_CODE=$code
}

# ── (A) baseline without refiner ──────────────────────────────────
fire A "$(printf '{"model":"%s","prompt":"%s",%s}' "$MODEL_ID" "$PROMPT" "$COMMON")" "$OUT_A"
CODE_A=$LAST_CODE

# ── (B) refiner at 0.8 ────────────────────────────────────────────
fire B "$(printf '{"model":"%s","prompt":"%s",%s,"refiner_switch_at":0.8}' \
          "$MODEL_ID" "$PROMPT" "$COMMON")" "$OUT_B"
CODE_B=$LAST_CODE

# ── (C) refiner at 0.8 repeat (determinism) ───────────────────────
fire C "$(printf '{"model":"%s","prompt":"%s",%s,"refiner_switch_at":0.8}' \
          "$MODEL_ID" "$PROMPT" "$COMMON")" "$OUT_C"
CODE_C=$LAST_CODE

# ── (D) refiner at 0.9 (different share) ──────────────────────────
fire D "$(printf '{"model":"%s","prompt":"%s",%s,"refiner_switch_at":0.9}' \
          "$MODEL_ID" "$PROMPT" "$COMMON")" "$OUT_D"
CODE_D=$LAST_CODE

echo
echo "─── Feature applied? ─────────────────────────────────────"
fail=0
for L in A B C D; do
    declare -n code="CODE_$L"
    if [[ "$code" != "200" ]]; then
        err "$L returned HTTP $code"
        "${COMPOSE[@]}" logs --no-color --tail 40 "$SERVICE" | sed 's/^/  /'
        fail=1
    fi
done
(( fail )) && exit 1

check_diff() {
    local a="$1" b="$2" expectation="$3"
    if cmp -s "$a" "$b"; then
        if [[ "$expectation" == "same" ]]; then
            ok "$a ≡ $b (byte-identical, as expected)"
        else
            err "$a ≡ $b (UNEXPECTED — refiner didn't change output)"
            fail=1
        fi
    else
        if [[ "$expectation" == "diff" ]]; then
            ok "$a ≠ $b"
        else
            err "$a ≠ $b (UNEXPECTED — should be byte-identical)"
            fail=1
        fi
    fi
}

check_diff "$OUT_A" "$OUT_B" diff  # refiner changes output vs base alone
check_diff "$OUT_B" "$OUT_C" same  # same seed + same switch_at → identical
check_diff "$OUT_B" "$OUT_D" diff  # different switch_at → different polish

# ── (E) bounds validation ─────────────────────────────────────────
log "[E] refiner_switch_at=1.5 → 422"
CODE=$(curl -s -o /dev/null -w '%{http_code}' \
    -X POST "${GATEWAY_URL}/v1/images/generations" \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"%s","prompt":"x","refiner_switch_at":1.5}' "$MODEL_ID")" \
    || echo 000)
if [[ "$CODE" == "422" ]]; then
    ok "out-of-range refiner_switch_at → 422"
else
    err "expected 422, got $CODE"
    fail=1
fi

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }
ok "PASS — SDXL Refiner ensemble works end-to-end."
echo
echo "Open side-by-side:"
echo "  $OUT_A — base alone (no refiner)"
echo "  $OUT_B — base → refiner @ switch_at=0.8 (classic 80/20 recipe)"
echo "  $OUT_C — repeat of B (byte-equal to B; determinism)"
echo "  $OUT_D — base → refiner @ switch_at=0.9 (refiner gets smaller share)"
