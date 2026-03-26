"""OpenTelemetry bootstrap — call ``setup_telemetry(app)`` once at startup.

Gracefully degrades to a no-op when OTel packages are not installed.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_telemetry(app: object) -> None:
    """Wire up tracing for the FastAPI application.

    Configures a :class:`TracerProvider` with:
    - OTLP exporter if ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set
    - Console exporter if ``OTEL_LOG_SPANS=1`` is set
    - No-op otherwise (zero overhead when neither is configured)

    Also instruments FastAPI so every request gets an automatic span.
    Silently skips if OpenTelemetry is not installed.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:
        logger.debug("OpenTelemetry not installed — skipping instrumentation")
        return

    resource = Resource.create(
        {"service.name": "cleareye", "service.version": "0.1.0"}
    )
    provider = TracerProvider(resource=resource)

    # OTLP exporter (e.g. Jaeger, Grafana Tempo, etc.)
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
            logger.info("OTLP trace exporter → %s", otlp_endpoint)
        except Exception as e:
            logger.warning("Failed to initialise OTLP exporter: %s", e)

    # Console exporter for local debugging
    if os.environ.get("OTEL_LOG_SPANS", "").strip() in ("1", "true"):
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("Console span exporter enabled")

    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry instrumentation active")
    except ImportError:
        logger.warning("FastAPI OTel instrumentor not installed — skipping")
