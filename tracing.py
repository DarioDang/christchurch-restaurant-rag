# tracing.py
import os
from phoenix.otel import register  # type: ignore

def _init_tracer():
    project_name = os.getenv("PHOENIX_PROJECT_NAME", "restaurant-rag-project")
    tp = register(protocol="http/protobuf", project_name=project_name)
    return tp.get_tracer("course-ta")

tracer = _init_tracer()

