# api/main.py

"""
FastAPI entry point. Run from project root with:
    uvicorn api.main:app --reload
"""

import json 

import random 
from contextlib import asynccontextmanager
from dataclasses import dataclass 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from openai import OpenAI 
from opentelemetry.trace import Status, StatusCode
from config import OPENAI_CHAT_MODEL
from prompt import DEVELOPER_PROMPT, EXAMPLE_QUERIES
from tracing import tracer 
from core import init_qdrant, RestaurantSearchTools
from core.history import sanitize_client_history, cleanup_chat_history
from api.tools import Tools, smart_search_schema
from models.schemas import ChatRequest, ChatResponse 
from utils.analytics import get_analytics
from fastapi.responses import StreamingResponse
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
client = OpenAI()
MAX_TOOL_ITERATIONS = 5 # safely cap - see note below 
import re

def _check_distance_hallucination(reply: str, tool_output_json: str) -> bool:
    """
    Returns True if a hallucination is detected: the reply mentions a
    distance (e.g. "1.2km away") but none of the tool's results actually
    contained a distance_km field. This is a deterministic safety net —
    the model has repeatedly invented plausible-looking distances when
    none exist in the retrieved data, despite explicit prompt instructions
    not to. Logged to Phoenix rather than silently trusted.
    """
    try:
        parsed = json.loads(tool_output_json)
        results = parsed.get("results", [])
        any_real_distance = any("distance_km" in r for r in results)
    except Exception:
        return False

    if any_real_distance:
        return False  # legitimate distances exist, nothing to flag

    mentions_distance = bool(re.search(r'\d+(\.\d+)?\s*km', reply, re.IGNORECASE))
    return mentions_distance

@dataclass
class _ToolCallShim:
    """
    Mimics the shape of an OpenAI Responses API function_call entry (call_id / name / arguments attributes) so we can
    reuse Tools.function_call() after injection server - side location data into the arguments.
    """

    call_id: str
    name: str
    arguments: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing search system...")
    qdrant_client, model, bm25, doc_ids, doc_lookup, metadata = init_qdrant()
    search_instance = RestaurantSearchTools(
        qdrant_client=qdrant_client,
        embedding_model=model,
        bm25_index=bm25,
        bm25_doc_ids=doc_ids,
        doc_lookup=doc_lookup,
        metadata= metadata,
    )

    chat_tools = Tools()
    chat_tools.add_tool(search_instance.smart_restaurant_search, smart_search_schema)
    app.state.search_instance = search_instance 
    app.state.chat_tools = chat_tools 
    logger.info("Search system ready.")

    yield 

app = FastAPI(title="Christchurch Restaurant RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TO DO Phase 4: lock to the deployed static-site origin
    allow_methods=["*"],
    allow_headers=["*"],
)

def _require_search_instance(app: FastAPI) -> RestaurantSearchTools:
    instance = getattr(app.state, "search_instance", None)
    if instance is None:
        raise HTTPException(status_code=503, detail="Search system is still initializing. Try again shortly.")
    return instance

def _require_chat_tools(app: FastAPI) -> Tools:
    instance = getattr(app.state, "chat_tools", None)
    if instance is None:
        raise HTTPException(status_code=503, detail="Search system is still initializing. Try again shortly.")
    return instance

def _inject_location(tool_args:dict, lat, lon, max_distance_km) -> dict:
    if lat is not None and lon is not None:
        tool_args["user_lat"] = lat
        tool_args["user_lon"] = lon
        tool_args["max_distance_km"] = max_distance_km
    else:
        tool_args["user_lat"] = None
        tool_args["user_lon"] = None
        tool_args["max_distance_km"] = None
    return tool_args

def _sse(data: dict) -> str:
    """Format a dict as a single Server-Sent Events message."""
    return f"data: {json.dumps(data)}\n\n"

def _stream_chat_turn(request: ChatRequest):
    """
    Sync generator yielding SSE-formatted events for one chat turn.
    Starlette's StreamingResponse runs plain (non-async) generators in a
    thread pool automatically, so this blocking generator doesn't stall
    the event loop even though it contains long blocking calls (the OpenAI
    stream iteration, the tool execution).
    """
    chat_tools = _require_chat_tools(app)
    search_instance = _require_search_instance(app)

    history = sanitize_client_history(request.messages)
    history = cleanup_chat_history(history, max_messages=25)

    try:
        last_user_msg = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"), None
        )
        if last_user_msg:
            analytics = get_analytics(qdrant_client=search_instance.client)
            analytics.log_query(
                query=last_user_msg,
                session_id=request.session_id,
                location_enabled=request.user_lat is not None,
            )
    except Exception:
        logger.warning("Analytics logging failed", exc_info=True)

    with tracer.start_as_current_span("assistant-turn", openinference_span_kind="chain") as chain_span:
        user_input = history[-1].get("content", "") if history else ""
        chain_span.set_attribute("input.value", str(user_input))

        assistant_message = ""
        last_tool_output = None

        try:
            iteration = 0
            while True:
                iteration += 1
                if iteration > MAX_TOOL_ITERATIONS:
                    logger.warning("Hit MAX_TOOL_ITERATIONS (%s)", MAX_TOOL_ITERATIONS)
                    break

                with tracer.start_as_current_span("Responses.create", openinference_span_kind="llm") as llm_span:
                    tools_available = chat_tools.get_tools()
                    llm_span.set_attribute("llm.input", str(history))
                    llm_span.set_attribute("llm.model_name", OPENAI_CHAT_MODEL)
                    llm_span.set_attribute("llm.tools.count", len(tools_available))

                    stream = client.responses.create(
                        model=OPENAI_CHAT_MODEL,
                        input=history,
                        tools=tools_available,
                        stream=True,
                    )

                    final_response = None
                    turn_text = ""

                    for event in stream:
                        # NOTE: if no text_delta events appear during testing,
                        # log event.type here to confirm the exact strings
                        # your installed openai SDK version emits.
                        if event.type == "response.output_text.delta":
                            turn_text += event.delta
                            yield _sse({"type": "text_delta", "content": event.delta})
                        elif event.type == "response.completed":
                            final_response = event.response
                        elif event.type in ("response.failed", "error"):
                            logger.error("OpenAI stream error event: %s", event)
                            yield _sse({"type": "error", "detail": "The model encountered an error."})
                            chain_span.set_status(Status(StatusCode.ERROR))
                            return

                    if final_response is None:
                        logger.error("Stream ended without response.completed event")
                        yield _sse({"type": "error", "detail": "Incomplete response from the model."})
                        chain_span.set_status(Status(StatusCode.ERROR))
                        return

                    response = final_response
                    tool_called = False

                    for entry in response.output:
                        if entry.type == "function_call":
                            tool_called = True
                            history.append({
                                "type": "function_call",
                                "call_id": entry.call_id,
                                "name": entry.name,
                                "arguments": entry.arguments,
                            })

                            tool_args = json.loads(entry.arguments)
                            tool_args = _inject_location(
                                tool_args, request.user_lat, request.user_lon, request.max_distance_km
                            )

                            yield _sse({
                                "type": "tool_call_start",
                                "name": entry.name,
                                "query": tool_args.get("query", ""),
                            })

                            with tracer.start_as_current_span(
                                "smart_restaurant_search", openinference_span_kind="tool"
                            ) as tool_span:
                                tool_span.set_attribute("tool.name", "smart_restaurant_search")
                                tool_span.set_attribute("input.value", json.dumps(tool_args))
                                tool_span.set_attribute("input.mime_type", "application/json")

                                if request.user_lat is not None and request.user_lon is not None:
                                    tool_span.set_attribute("location.latitude", float(request.user_lat))
                                    tool_span.set_attribute("location.longitude", float(request.user_lon))
                                    tool_span.set_attribute("location.enabled", True)
                                    tool_span.set_attribute("location.city", "Christchurch")
                                    tool_span.set_attribute("location.country", "New Zealand")

                                shim = _ToolCallShim(
                                    call_id=entry.call_id,
                                    name=entry.name,
                                    arguments=json.dumps(tool_args),
                                )
                                result = chat_tools.function_call(shim)

                                tool_span.set_attribute("output.value", result["output"])
                                tool_span.set_attribute("output.mime_type", "application/json")

                            last_tool_output = result["output"]

                            if request.user_lat is not None and request.user_lon is not None:
                                llm_span.set_attribute("user_lat", float(request.user_lat))
                                llm_span.set_attribute("user_lon", float(request.user_lon))
                                chain_span.set_attribute("user_lat", float(request.user_lat))
                                chain_span.set_attribute("user_lon", float(request.user_lon))

                            try:
                                parsed = json.loads(result["output"])
                                docs = parsed.get("results", [])
                                reference_text = "\n\n".join(
                                    d.get("full_review", "") for d in docs if d.get("full_review")
                                )
                                chain_span.set_attribute("reference", reference_text)
                            except Exception as e:
                                chain_span.set_attribute("reference_error", str(e))

                            history.append(result)

                            try:
                                parsed_for_count = json.loads(result["output"])
                                result_count = len(parsed_for_count.get("results", []))
                                mode = parsed_for_count.get("mode", "")
                            except Exception:
                                result_count, mode = 0, ""

                            yield _sse({"type": "tool_call_end", "mode": mode, "result_count": result_count})

                        elif entry.type == "message":
                            try:
                                msg = entry.content[0]
                                assistant_message = getattr(msg, "text", "") or getattr(msg, "refusal", "") or turn_text
                            except Exception:
                                assistant_message = turn_text

                            if assistant_message:
                                history.append({
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": assistant_message}],
                                })

                    if assistant_message:
                        llm_span.set_attribute("llm.output", assistant_message)
                        chain_span.set_attribute("output.value", assistant_message)

                    if not tool_called:
                        break

            chain_span.set_status(Status(StatusCode.OK))

            if last_tool_output and _check_distance_hallucination(assistant_message, last_tool_output):
                logger.warning("Distance hallucination detected in reply: %s", assistant_message[:200])
                chain_span.set_attribute("eval.distance_hallucination_detected", True)

        except Exception as e:
            logger.exception("Error processing chat turn")
            chain_span.record_exception(e)
            chain_span.set_status(Status(StatusCode.ERROR))
            yield _sse({"type": "error", "detail": "Something went wrong processing your message."})
            return
    if not assistant_message:
        assistant_message = "I wasn't able to put together a response for that — could you try rephrasing your question?"
    yield _sse({"type": "done", "messages": history, "reply": assistant_message})


@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Validate readiness BEFORE opening the stream, so an unready server
    # returns a normal HTTP 503 instead of a broken SSE connection.
    _require_chat_tools(app)
    _require_search_instance(app)

    return StreamingResponse(
        _stream_chat_turn(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (relevant on Render too)
            "Connection": "keep-alive",
        },
    )

@app.get("/api/stats")
async def stats():
    si = _require_search_instance(app)
    return {
        "restaurants": len(si.all_restaurants),
        "cuisines": len(si.all_cuisines),
        "reviews": si.metadata["total_docs"],
    }

@app.get("/api/examples")
async def examples():
    return {"examples": random.sample(EXAMPLE_QUERIES, min(4, len(EXAMPLE_QUERIES)))}


@app.get("/api/popular")
async def popular():
    si = _require_search_instance(app)
    analytics = get_analytics(qdrant_client=si.client)
    stats_data = analytics.get_stats()

    trending = analytics.get_trending_queries(time_window_hours=24, min_count=2, top_n=3)
    if len(trending) < 3 and stats_data["total_queries"] >= 3:
        trending = analytics.get_trending_queries(time_window_hours=24, min_count=1, top_n=3)

    if len(trending) >= 3:
        show_count = True
    else:
        trending = analytics.get_fallback_queries()
        show_count = False

    return {"trending": trending, "show_count": show_count, "stats": stats_data}


@app.get("/api/health")
async def health():
    return {"status": "ok"}
    