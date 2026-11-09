#!/usr/bin/env bash
# Feature test: JSON answers and reasoning of qwen3.5-4b.
#
#   1. json_object with thinking on and max_tokens 300: reasoning in reasoning_content, JSON in content.
#   2. reasoning_effort none: no reasoning, JSON in content.
#   3. reasoning_effort low: reasoning capped by the thinking budget.
#   4. json_schema strict: content has exactly the schema keys.
#   5. Stream: reasoning_content deltas come before content deltas.
#
# Leaves deploy/.env alone. The first request loads the model (minutes over the Windows bind mount).
#
# Run from project root: bash scripts/feature/chat-structured.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="qwen3.5-4b"
SERVICE="worker-qwen3-5-4b"
COMPOSE=(docker compose -f deploy/docker-compose.yml --profile "$MODEL")
READY_TIMEOUT=900
GATEWAY_URL="http://localhost:8000"
PY="$(command -v python3 || command -v python)"

# shellcheck source=../diagnose/_lib.sh
source scripts/diagnose/_lib.sh

log "Building + starting gateway and $SERVICE..."
"${COMPOSE[@]}" build gateway "$SERVICE"
"${COMPOSE[@]}" up -d gateway "$SERVICE"
wait_for_worker "$SERVICE" "models/models--cyankiwi--Qwen3.5-4B-AWQ-4bit" "$MODEL" "$READY_TIMEOUT"

status=0
PYTHONIOENCODING=utf-8 "$PY" - "$GATEWAY_URL/v1/chat/completions" "$MODEL" <<'PYEOF' || status=1
import json
import sys
import time
import urllib.request

url, model = sys.argv[1], sys.argv[2]
messages = [{"role": "user", "content": "Note: buy milk, bread and eggs tomorrow morning. Give it a title and tags."}]
schema = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["title", "tags"],
    "additionalProperties": False,
}


def post(**fields):
    body = json.dumps({"model": model, "messages": messages, **fields}).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=1800)


def answer(**fields):
    started = time.monotonic()
    with post(**fields) as response:
        result = json.load(response)
    return result, time.monotonic() - started


def report(name, passed, detail):
    print(("\033[1;32m[ ok ]\033[0m " if passed else "\033[1;31m[err ]\033[0m ") + f"{name}: {detail}")
    return passed


def json_content(message):
    try:
        return json.loads(message["content"])
    except (TypeError, ValueError):
        return None


results = []

result, seconds = answer(response_format={"type": "json_object"}, max_tokens=300)
message = result["choices"][0]["message"]
results.append(report(
    "json_object with thinking",
    bool(message.get("reasoning_content")) and isinstance(json_content(message), dict),
    f"{seconds:.0f} s, {len(message.get('reasoning_content') or '')} chars of reasoning, content {message['content']!r:.80}",
))

result, seconds = answer(response_format={"type": "json_object"}, reasoning_effort="none")
message = result["choices"][0]["message"]
results.append(report(
    "reasoning_effort none",
    "reasoning_content" not in message and isinstance(json_content(message), dict),
    f"{seconds:.1f} s, content {message['content']!r:.80}",
))

result, seconds = answer(response_format={"type": "json_object"}, reasoning_effort="low")
message = result["choices"][0]["message"]
tokens = result["usage"]["completion_tokens"]
results.append(report(
    "reasoning_effort low",
    bool(message.get("reasoning_content")) and isinstance(json_content(message), dict) and tokens < 1000,
    f"{seconds:.0f} s, {tokens} tokens, content {message['content']!r:.80}",
))

result, seconds = answer(
    response_format={"type": "json_schema", "json_schema": {"name": "note", "strict": True, "schema": schema}},
    thinking=False,
)
content = json_content(result["choices"][0]["message"])
results.append(report(
    "json_schema strict",
    isinstance(content, dict) and set(content) == {"title", "tags"} and isinstance(content["tags"], list),
    f"{seconds:.1f} s, {content}",
))

order = []
with post(stream=True, reasoning_effort="low") as response:
    for line in response:
        line = line.decode().strip()
        if line.startswith("data: {"):
            for key in json.loads(line[6:])["choices"][0]["delta"]:
                if not order or order[-1] != key:
                    order.append(key)
results.append(report("stream", order == ["reasoning_content", "content"], f"delta order {order}"))

sys.exit(0 if all(results) else 1)
PYEOF

echo
(( status )) && { err "FAIL: review output above."; exit 1; }
ok "PASS: $MODEL keeps reasoning apart and answers JSON, budgeted and to a schema."
