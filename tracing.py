# tracing.py — framework-agnostic version

"""
Phoenix Tracing Configuration
Works with Phoenix Cloud (Arize) - PRODUCTION READY, framework-agnostic
"""

import os
from functools import wraps

try:
    from phoenix.otel import register
    from opentelemetry import trace
    PHOENIX_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Phoenix import failed: {e}")
    print("⚠️ Tracing will be disabled for this session")
    PHOENIX_AVAILABLE = False


def get_secret(key: str, default=None):
    """Get config value from environment variables (Render dashboard, or local .env)"""
    return os.getenv(key, default)


class TracerWrapper:
    """Adds Phoenix-style .tool() decorator and openinference_span_kind handling."""

    def __init__(self, tracer):
        self._tracer = tracer

    def start_as_current_span(self, name, openinference_span_kind=None, **kwargs):
        attributes = kwargs.get('attributes', {})
        if openinference_span_kind:
            attributes['openinference.span.kind'] = openinference_span_kind
        kwargs['attributes'] = attributes
        return self._tracer.start_as_current_span(name, **kwargs)

    def tool(self, name: str = None, description: str = None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                span_name = name or func.__name__
                with self._tracer.start_as_current_span(
                    span_name,
                    attributes={
                        "openinference.span.kind": "tool",
                        "tool.name": span_name,
                        "tool.description": description or "",
                    }
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("tool.success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("tool.success", False)
                        span.set_attribute("tool.error", str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator


def setup_phoenix_tracing():
    if not PHOENIX_AVAILABLE:
        print("⚠️ Phoenix not available, skipping tracing setup")
        return None

    project_name = get_secret("PHOENIX_PROJECT_NAME", "restaurant-rag-production")
    phoenix_api_key = get_secret("PHOENIX_API_KEY")
    if not phoenix_api_key:
        print("⚠️ WARNING: PHOENIX_API_KEY not configured")
        return None

    print(f"📁 Phoenix Project: {project_name}")
    print(f"🔐 API Key configured: {phoenix_api_key[:15]}...")

    try:
        tracer_provider = register(protocol="http/protobuf", project_name=project_name)
        print("✅ Phoenix tracing initialized successfully")
        base_tracer = tracer_provider.get_tracer(__name__)
        return TracerWrapper(base_tracer)
    except Exception as e:
        print(f"❌ Failed to initialize Phoenix tracing: {e}")
        import traceback
        traceback.print_exc()
        return None


tracer = setup_phoenix_tracing()

if tracer is None:
    print("⚠️ Using dummy tracer (tracing disabled)")

    class DummyTracer:
        def start_as_current_span(self, *args, **kwargs):
            from contextlib import contextmanager
            @contextmanager
            def dummy_span(*a, **kw):
                class DummySpan:
                    def set_attribute(self, *a, **kw): pass
                    def record_exception(self, *a, **kw): pass
                    def set_status(self, *a, **kw): pass
                yield DummySpan()
            return dummy_span()

        def tool(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    tracer = DummyTracer()

__all__ = ['tracer']