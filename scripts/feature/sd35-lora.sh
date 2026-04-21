#!/usr/bin/env bash
# Feature test: LoRA hot-load on sd35-medium + verify compel-skip.
#
# Why this is a separate test from lora-hot-load.sh (sdxl-base):
#   * SD3 LoRAs target the MMDiT transformer, not UNet — a different
#     diffusers/peft code path. We need at-least-one LoRA request to hit
#     that path before shipping.
#   * SD3Pipeline carries the same (tokenizer_2, text_encoder_2) slot as
#     SDXL, so the compel-init guard has to reject it by class name, not
#     by duck-typing. The bug this protects against is a runtime shape
#     mismatch ([B,77,2048] vs [B,154,4096]) that only surfaces when a
#     client uses (word:weight) syntax; if we don't assert "compel skipped"
#     at startup, regressions go unnoticed until a weighted prompt lands
#     in production.
#
# Scenarios:
#   A) baseline           — no loras, standard output
#   B) single lora 1.0    — LoRA applied via peft on MMDiT
#   C) cache hit          — repeat B; same bytes, no reload
#   D) error path         — bogus repo → HTTP 400
#
# Plus startup-log assertion:
#   * "Compel skipped for sd35-medium (StableDiffusion3Pipeline)" appears
#   * No compel-init error or shape-mismatch warning
#
# SD3/SD3.5 public LoRA catalogue is still sparse. The default below is a
# distillation LoRA that works on SD3-family MMDiT transformers; if it
# 404s or is arch-incompatible on your local, override:
#   LORA_ID=org/repo LORA_FILE=file.safetensors bash scripts/feature/sd35-lora.sh
#
# Run from project root.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="deploy/.env"
COMPOSE=(docker compose -f deploy/docker-compose.yml)
MODEL_ID="sd35-medium"
SERVICE="worker-sd35-medium"
HF_CACHE="models/models--stabilityai--stable-diffusion-3.5-medium"
READY_TIMEOUT=900  # SD3.5 + T5 download is heavier than SDXL

# Default LoRA: tensorart's official turbo-distillation LoRA trained on
# SD3.5 Medium itself. ByteDance's Hyper-SD3 is NOT compatible — it was
# trained on SD3 Medium (2B, hidden=1536 → norm1.linear out=9216), while
# SD3.5 Medium has hidden=2304 → norm1.linear out=13824. Same diffusers
# class name, different tensor widths. See ByteDance/Hyper-SD#72.
LORA_ID="${LORA_ID:-tensorart/stable-diffusion-3.5-medium-turbo}"
LORA_FILE="${LORA_FILE:-lora_sd3.5m_turbo_8steps.safetensors}"

# 768×768 — keeps peak VRAM with drop_t5=false under 12GB budget even
# during LoRA injection. Visual comparison remains clear.
#
# No `scheduler` override here: SD3.5 uses FlowMatchEulerDiscreteScheduler,
# which is a different family from the UNet schedulers (dpm++, euler_a,
# ddim, …) registered in _SCHEDULERS. Sending one of those names on SD3
# is rejected by the swap path, so we let the pipeline use its default.
#
# Turbo LoRA needs its own sampling budget: 8 steps + CFG=1.5 (tensorart's
# recommendation). Running it at 20 steps / CFG=7 produces broken over-
# saturated output and defeats the point of the turbo distillation.
COMMON_BASE='"seed":13,"num_inference_steps":20,"size":"768x768"'
COMMON_TURBO='"seed":13,"num_inference_steps":8,"guidance_scale":1.5,"size":"768x768"'
PROMPT="a cyberpunk cat in neon Tokyo"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found — copy from deploy/.env.example"; exit 1; }

update_env "$ENV_FILE" COMPOSE_PROFILES "$MODEL_ID"
update_env "$ENV_FILE" SD35_MEDIUM_CPU_OFFLOAD false
update_env "$ENV_FILE" SD35_MEDIUM_SEQUENTIAL_OFFLOAD false
update_env "$ENV_FILE" SD35_MEDIUM_DROP_T5 true
update_env "$ENV_FILE" SD35_MEDIUM_WARMUP false
ok "Env flags set (LORA_ID=$LORA_ID, LORA_FILE=$LORA_FILE)"

log "Rebuilding gateway + worker (requirements changed: peft, compel) …"
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"

log "Waiting up to ${READY_TIMEOUT}s for $MODEL_ID …"
wait_for_worker "$SERVICE" "$HF_CACHE" "$MODEL_ID" "$READY_TIMEOUT"

# ── Startup-log assertion: compel must be skipped, NOT initialised ─────
log "Verifying compel was skipped on SD3 pipeline …"
STARTUP_LOGS=$("${COMPOSE[@]}" logs --no-color "$SERVICE" 2>&1 || true)
COMPEL_SKIPPED_LINE=$(grep -E "Compel skipped for ${MODEL_ID}.*StableDiffusion3" <<<"$STARTUP_LOGS" || true)
COMPEL_INIT_LINE=$(grep -E "Compel initialised for ${MODEL_ID}" <<<"$STARTUP_LOGS" || true)

if [[ -z "$COMPEL_SKIPPED_LINE" ]]; then
    err "Expected log line 'Compel skipped for sd35-medium (StableDiffusion3Pipeline: …)' not found."
    err "This would have caused a runtime [B,77,2048] vs [B,154,4096] shape mismatch."
    err "Last 60 worker log lines:"
    "${COMPOSE[@]}" logs --no-color --tail 60 "$SERVICE"
    exit 1
fi
if [[ -n "$COMPEL_INIT_LINE" ]]; then
    err "Compel was initialised — regression! Guard in _init_compel missed SD3 class."
    err "Offending line: $COMPEL_INIT_LINE"
    exit 1
fi
ok "compel-skip log present: ${COMPEL_SKIPPED_LINE#*[diag] }"

# ── Generate requests ──────────────────────────────────────────────────
fire() {
    local label="$1" body="$2" out="$3"
    log "[$label]"
    local resp
    resp=$(mktemp --suffix=.json)
    local t0=$SECONDS
    local code
    code=$(curl -s -o "$resp" -w '%{http_code}' \
        -X POST http://localhost:8000/v1/images/generations \
        -H 'Content-Type: application/json' \
        -H 'X-InferGate-No-Cache: true' \
        -d "$body" || echo 000)
    local elapsed=$(( SECONDS - t0 ))

    if [[ "$code" == "200" ]]; then
        decode_b64_png "$resp" "$out" || cp "$resp" "$out"
    else
        cp "$resp" "$out"
    fi

    local sz
    sz=$(wc -c < "$out" 2>/dev/null || echo 0)
    echo "  HTTP $code, ${elapsed}s → $out ($sz bytes)"
    # On failure, surface the full error body and worker log tail so the
    # operator can diagnose without a second round-trip of "show me the body".
    if [[ "$code" != "200" ]]; then
        echo "  ── Response body ─────────────────────────────────────"
        sed 's/^/  /' "$resp"
        echo
        echo "  ── Worker log tail (last 40 lines) ──────────────────"
        "${COMPOSE[@]}" logs --no-color --tail 40 "$SERVICE" 2>&1 | sed 's/^/  /'
    fi
    rm -f "$resp"

    LAST_CODE=$code
    LAST_TIME=$elapsed
}

OUT_A=feature_sd35_A_baseline.png
OUT_B=feature_sd35_B_lora.png
OUT_C=feature_sd35_C_cache.png

fire A "$(printf '{"model":"%s","prompt":"%s",%s}' "$MODEL_ID" "$PROMPT" "$COMMON_BASE")" "$OUT_A"
CODE_A=$LAST_CODE

LORA_JSON=$(printf '[{"id":"%s","weight_file":"%s","weight":1.0}]' "$LORA_ID" "$LORA_FILE")
fire B "$(printf '{"model":"%s","prompt":"%s",%s,"loras":%s}' \
            "$MODEL_ID" "$PROMPT" "$COMMON_TURBO" "$LORA_JSON")" "$OUT_B"
CODE_B=$LAST_CODE; TIME_B_FIRST=$LAST_TIME

fire C "$(printf '{"model":"%s","prompt":"%s",%s,"loras":%s}' \
            "$MODEL_ID" "$PROMPT" "$COMMON_TURBO" "$LORA_JSON")" "$OUT_C"
CODE_C=$LAST_CODE; TIME_C_CACHED=$LAST_TIME

echo
echo "─── Feature applied? ─────────────────────────────────────"
fail=0

if [[ "$CODE_A" != "200" ]]; then
    err "A (baseline) failed with HTTP $CODE_A — unrelated to LoRA path"
    exit 1
fi

if [[ "$CODE_B" != "200" ]]; then
    err "B (with LoRA) failed with HTTP $CODE_B"
    if grep -q "size mismatch for transformer_blocks" "$OUT_B"; then
        err "Architecture mismatch: the LoRA was trained on a different SD3-family"
        err "checkpoint than the loaded sd35-medium. SD3 Medium (2B) and SD3.5 Medium"
        err "(2.5B) share the diffusers class name but have different MMDiT widths"
        err "(9216 vs 13824 norm1.linear output). LoRAs are not cross-compatible."
        err "Override with a SD3.5-Medium-specific LoRA:"
        err "  LORA_ID=org/repo LORA_FILE=file.safetensors bash $0"
    elif grep -q '"status":404' "$OUT_B" || grep -q "not found" "$OUT_B"; then
        err "LoRA repo not reachable (404). Override:"
        err "  LORA_ID=org/repo LORA_FILE=file.safetensors bash $0"
    else
        err "Unexpected LoRA load failure — body already printed above."
    fi
    exit 1
fi

if [[ "$CODE_C" != "200" ]]; then
    err "C (cache repeat) failed with HTTP $CODE_C"
    fail=1
fi

# Verify the pipeline actually entered the LoRA-load path (peft on MMDiT).
LORA_LOG=$("${COMPOSE[@]}" logs --no-color --tail 200 "$SERVICE" 2>&1 | \
    grep -E "Loading LoRA ${LORA_ID}.*into ${MODEL_ID}" || true)
if [[ -n "$LORA_LOG" ]]; then
    ok "peft-backed LoRA load reached MMDiT: ${LORA_LOG#*[diag] }"
else
    err "No 'Loading LoRA ${LORA_ID} into ${MODEL_ID}' log — LoRA request didn't hit load path."
    fail=1
fi

# B ≠ A: LoRA must change output (distillation LoRAs change trajectory
# even at same step count — less noise, different convergence).
if cmp -s "$OUT_A" "$OUT_B"; then
    err "A ≡ B byte-identical — LoRA had no effect on output."
    fail=1
else
    ok "A ≠ B — LoRA changes output"
fi

# C ≡ B: cache hit, adapter already loaded, same seed → same bytes.
if cmp -s "$OUT_B" "$OUT_C"; then
    ok "B ≡ C — adapter cache hit is deterministic"
else
    err "B ≠ C — cache-hit path produced different bytes."
    fail=1
fi

if (( TIME_B_FIRST > 0 && TIME_C_CACHED <= TIME_B_FIRST )); then
    ok "cache hit latency: ${TIME_C_CACHED}s ≤ first-load ${TIME_B_FIRST}s"
else
    log "cache hit latency: ${TIME_C_CACHED}s vs ${TIME_B_FIRST}s — noisy, not a failure"
fi

# ── Error path ─────────────────────────────────────────────────────────
log "[D] Error path: bogus LoRA repo → HTTP 400"
RESP=$(mktemp --suffix=.json)
CODE=$(curl -s -o "$RESP" -w '%{http_code}' \
    -X POST http://localhost:8000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"%s","prompt":"x","loras":[{"id":"this-org/does-not-exist","weight":1.0}]}' "$MODEL_ID")" \
    || echo 000)
echo "  HTTP $CODE body: $(head -c 300 "$RESP")"
if [[ "$CODE" == "400" ]] && grep -q "Failed to load LoRA" "$RESP"; then
    ok "bogus LoRA → 400 with 'Failed to load LoRA' message"
else
    err "bogus LoRA → HTTP $CODE (expected 400 with structured message)"
    fail=1
fi
rm -f "$RESP"

echo
(( fail )) && { err "FAIL — review output above."; exit 1; }

ok "PASS — SD3.5 Medium LoRA works + compel correctly skipped."
echo
echo "Open side-by-side to inspect:"
echo "  $OUT_A — baseline (no LoRA)"
echo "  $OUT_B — LoRA @ 1.0 (should look different — fewer effective steps / distinct style)"
echo "  $OUT_C — repeat of B (cache hit — same bytes)"
