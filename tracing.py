"""
Phoenix Tracing Configuration
Works with Phoenix Cloud (Arize) - PRODUCTION READY
Based on working legacy version with TracerWrapper added
"""

import os
import streamlit as st
from functools import wraps
from phoenix.otel import register
from opentelemetry import trace


def get_secret(key: str, default=None):
    """Get secret from Streamlit Cloud or environment variable"""
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)


class TracerWrapper:
    """
    Wrapper around OpenTelemetry Tracer that adds Phoenix-style .tool() decorator
    and handles openinference_span_kind
    """
    
    def __init__(self, tracer):
        self._tracer = tracer
    
    def start_as_current_span(self, name, openinference_span_kind=None, **kwargs):
        """Handle openinference_span_kind parameter"""
        attributes = kwargs.get('attributes', {})
        
        if openinference_span_kind:
            attributes['openinference.span.kind'] = openinference_span_kind
        
        kwargs['attributes'] = attributes
        return self._tracer.start_as_current_span(name, **kwargs)
    
    def tool(self, name: str = None, description: str = None):
        """Decorator to trace tool/function calls"""
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
    """
    Setup Phoenix Cloud tracing - SIMPLIFIED VERSION LIKE OLD CODE
    Uses environment variables automatically picked up by register()
    """
    
    # Get project name
    project_name = get_secret("PHOENIX_PROJECT_NAME", "restaurant-rag-production")
    
    # Validate that API key is set (register() will use it from env)
    phoenix_api_key = get_secret("PHOENIX_API_KEY")
    if not phoenix_api_key:
        print("⚠️ WARNING: PHOENIX_API_KEY not configured")
        return None
    
    print(f"📁 Phoenix Project: {project_name}")
    print(f"🔐 API Key configured: {phoenix_api_key[:15]}...")
    
    try:
        # ✅ Use register() like the old working version
        # It automatically picks up PHOENIX_API_KEY and PHOENIX_COLLECTOR_ENDPOINT from env
        tracer_provider = register(
            protocol="http/protobuf",  # ← Important from old version!
            project_name=project_name,
        )
        
        print("✅ Phoenix tracing initialized successfully")
        print(f"✅ Project: {project_name}")
        
        # Get tracer from provider and wrap it
        base_tracer = tracer_provider.get_tracer(__name__)
        return TracerWrapper(base_tracer)
    
    except Exception as e:
        print(f"❌ Failed to initialize Phoenix tracing: {e}")
        import traceback
        traceback.print_exc()
        return None


# Initialize Tracer
tracer = setup_phoenix_tracing()

# Fallback to dummy tracer if initialization failed
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