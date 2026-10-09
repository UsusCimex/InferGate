from __future__ import annotations

import logging
import sys
import types

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.monitoring import RequestIdMiddleware, tracing

_ENDPOINT = "http://collector:4318"


def test_tracing_is_off_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert tracing.configure_tracing(FastAPI(), "test") is False


def test_sdk_disabled_overrides_the_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert tracing.tracing_requested() is False


def test_tracing_without_the_extra_only_warns(monkeypatch, caplog):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    with caplog.at_level(logging.WARNING, logger="app.monitoring.tracing"):
        assert tracing.configure_tracing(FastAPI(), "test") is False
    assert any("not installed" in r.message for r in caplog.records)


class _BrokenExporter:
    def __init__(self):
        raise TypeError("Descriptors cannot be created directly")


@pytest.mark.parametrize(("exporter", "message"), [
    (None, "not installed"),
    (_BrokenExporter, "did not load"),
])
def test_a_missing_or_broken_exporter_only_warns(monkeypatch, caplog, exporter, message):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _ENDPOINT)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    stand_ins = {
        "opentelemetry": {},
        "opentelemetry.trace": {},
        "opentelemetry.instrumentation": {},
        "opentelemetry.instrumentation.fastapi": {"FastAPIInstrumentor": object},
        "opentelemetry.instrumentation.httpx": {"HTTPXClientInstrumentor": object},
        "opentelemetry.sdk": {},
        "opentelemetry.sdk.resources": {"Resource": object},
        "opentelemetry.sdk.trace": {"TracerProvider": object},
        "opentelemetry.sdk.trace.export": {"BatchSpanProcessor": lambda exporter: exporter},
        "opentelemetry.exporter": {},
        "opentelemetry.exporter.otlp": {},
        "opentelemetry.exporter.otlp.proto": {},
        "opentelemetry.exporter.otlp.proto.http": {},
    }
    for name, attributes in stand_ins.items():
        module = types.ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["opentelemetry"].trace = sys.modules["opentelemetry.trace"]
    trace_exporter = None
    if exporter is not None:
        trace_exporter = types.ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")
        trace_exporter.OTLPSpanExporter = exporter
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", trace_exporter)

    with caplog.at_level(logging.WARNING, logger="app.monitoring.tracing"):
        assert tracing.configure_tracing(FastAPI(), "test-worker") is False
    assert message in caplog.text


def _worker_client() -> httpx.AsyncClient:
    """httpx client whose sockets answer every request with a canned 200."""
    import httpcore

    transport = httpx.AsyncHTTPTransport()
    transport._pool = httpcore.AsyncConnectionPool(network_backend=httpcore.AsyncMockBackend(
        [b"HTTP/1.1 200 OK\r\n", b"Content-Length: 2\r\n", b"\r\n", b"ok"] * 2
    ))
    return httpx.AsyncClient(transport=transport, base_url="http://worker")


@pytest.fixture
def span_exporter(monkeypatch):
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.instrumentation.fastapi")
    httpx_instrumentation = pytest.importorskip("opentelemetry.instrumentation.httpx")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.delenv("OTEL_TRACES_SAMPLER", raising=False)
    monkeypatch.delenv("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", raising=False)
    yield InMemorySpanExporter()
    httpx_instrumentation.HTTPXClientInstrumentor().uninstrument()


async def test_a_request_and_its_worker_call_form_one_trace(span_exporter):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.trace import SpanKind

    app = FastAPI()
    worker = _worker_client()

    @app.get("/v1/ping")
    async def ping():
        return {"worker": (await worker.get("/generate")).text}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.add_middleware(RequestIdMiddleware)
    assert tracing.configure_tracing(app, "test-gateway", SimpleSpanProcessor(span_exporter))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.get("/health")
        assert (await client.get("/v1/ping", headers={"X-Request-ID": "trace-me"})).json() == {
            "worker": "ok"
        }
    await worker.get("/generate")
    await worker.aclose()

    spans = span_exporter.get_finished_spans()
    server = [s for s in spans if s.kind == SpanKind.SERVER]
    client_spans = [s for s in spans if s.kind == SpanKind.CLIENT]
    assert len(server) == 1
    assert server[0].attributes["infergate.request_id"] == "trace-me"
    assert len(client_spans) == 1
    assert client_spans[0].parent.span_id == server[0].context.span_id
    assert len(spans) == 2
