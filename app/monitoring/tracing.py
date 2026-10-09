"""OpenTelemetry spans of HTTP requests and outgoing httpx calls, exported over OTLP.

Off unless OTEL_EXPORTER_OTLP_ENDPOINT (or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT) is set and the
`tracing` extra is installed; the exporter reads the other standard OTEL_* variables itself.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_ENDPOINT_VARS = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")
# Probes and scrapes would bury the requests that matter.
_EXCLUDED_URLS = "/health,/metrics"

_enabled = False


def tracing_requested() -> bool:
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return False
    return any(os.environ.get(name) for name in _ENDPOINT_VARS)


def configure_tracing(app: FastAPI, service_name: str, span_processor: Any = None) -> bool:
    """Trace `app` and outgoing httpx calls; returns whether tracing is on.

    The service name is fixed: the gateway and the workers share deploy/.env, so
    OTEL_SERVICE_NAME would name them all alike. `span_processor` replaces the OTLP
    exporter and turns tracing on without the environment.
    """
    global _enabled
    if span_processor is None and not tracing_requested():
        return False
    # The worker calls this at import time: a missing or broken package must not stop it.
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if span_processor is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            span_processor = BatchSpanProcessor(OTLPSpanExporter())
    except ImportError as e:
        logger.warning("OTLP endpoint is set but OpenTelemetry is not installed (extra 'tracing'): %s", e)
        return False
    except Exception as e:
        logger.warning("OTLP endpoint is set but OpenTelemetry did not load, tracing is off: %s", e)
        return False

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}),
        sampler=None if os.environ.get("OTEL_TRACES_SAMPLER") else _request_sampler(),
    )
    provider.add_span_processor(span_processor)
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls=os.environ.get("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", _EXCLUDED_URLS),
        exclude_spans=["receive", "send"],
    )
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    _enabled = True
    logger.info("Tracing on: spans of %s go to the OTLP endpoint", service_name)
    return True


def tag_request_id(request_id: str) -> None:
    """Put the request id on the current span, so a trace leads to its log lines."""
    if _enabled:
        from opentelemetry import trace

        trace.get_current_span().set_attribute("infergate.request_id", request_id)


def _request_sampler() -> Any:
    """Sample every request and what it calls; drop traces that start elsewhere (worker probes, polls)."""
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON, ParentBased, Sampler
    from opentelemetry.trace import SpanKind

    class _ServerRoots(Sampler):
        def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None,
                          links=None, trace_state=None):
            sampler = ALWAYS_ON if kind == SpanKind.SERVER else ALWAYS_OFF
            return sampler.should_sample(
                parent_context, trace_id, name, kind, attributes, links, trace_state
            )

        def get_description(self) -> str:
            return "ServerRoots"

    return ParentBased(root=_ServerRoots())
