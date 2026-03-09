from __future__ import annotations

import logging
import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from app.agents import AgentRunner
from app.graph import UPSMultiAgentGraph
from app.models import ChatRequest, ChatResponse
from app.supervisor import SupervisorRouter


VALID_AGENTS = {"shipping", "tracking", "general"}

load_dotenv()


def _configure_logging() -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    return logging.getLogger("ups.api")


logger = _configure_logging()


def _openapi_servers() -> list[dict[str, str]]:
    """Return a valid absolute URL for OpenAPI `servers`.

    Priority:
    1) OPENAPI_SERVER_URL (explicit override)
    2) RENDER_EXTERNAL_URL (Render-provided public URL)
    3) Local fallback for development
    """
    base_url = (
        os.getenv("OPENAPI_SERVER_URL")
        or os.getenv("RENDER_EXTERNAL_URL")
        or "http://127.0.0.1:8001"
    ).strip().rstrip("/")
    return [{"url": base_url}]

app = FastAPI(
    title="UPS Multi-Agent Supervisor API",
    version="1.0.0",
    description=(
        "Supervisor-style router API for ChatGPT Actions. "
        "Routes user queries to shipping/tracking/general specialist agents."
    ),
    servers=_openapi_servers(),
)

router: SupervisorRouter | None = None
graph_runner: UPSMultiAgentGraph | None = None


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request_completed request_id=%s method=%s path=%s status_code=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["x-request-id"] = request_id
    return response


@app.on_event("startup")
def _startup() -> None:
    global router, graph_runner
    logger.info("service_starting")
    agent_runner = AgentRunner()
    router = SupervisorRouter(agent_runner=agent_runner)
    graph_runner = UPSMultiAgentGraph(supervisor=router, agent_runner=agent_runner)
    logger.info("service_started")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ups-multi-agent"}


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "ups-multi-agent", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse, operation_id="chat")
def chat(payload: ChatRequest) -> ChatResponse:
    if graph_runner is None:
        logger.error("chat_failed reason=graph_not_initialized")
        raise HTTPException(status_code=500, detail="LangGraph runner is not initialized")

    try:
        result = graph_runner.run(payload.user_message)
    except Exception as exc:
        logger.exception("chat_failed reason=agent_execution_failed")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    selected_agent = str(result.get("selected_agent", "general"))
    answer = str(result.get("answer", ""))
    confidence = float(result.get("confidence", 0.55))
    routing_source = str(result.get("routing_source", "fallback"))
    routing_reason = str(result.get("routing_reason", "unknown"))

    if selected_agent not in VALID_AGENTS:
        logger.error("chat_failed reason=invalid_selected_agent selected_agent=%s", selected_agent)
        raise HTTPException(status_code=500, detail="Invalid routed agent")

    logger.info(
        "chat_routed agent=%s confidence=%s routing_source=%s routing_reason=%s message_length=%s",
        selected_agent,
        round(confidence, 2),
        routing_source,
        routing_reason,
        len(payload.user_message),
    )

    return ChatResponse(
        agent=selected_agent,
        answer=answer,
        confidence=round(confidence, 2),
        routing_source=routing_source,
        routing_reason=routing_reason,
    )

@app.get("/privacy")
def privacy():
    return {
        "privacy_policy": "User prompts may be sent to our backend API for processing. Data is not stored or shared."
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8001"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").strip().lower() in {"1", "true", "yes"}
    uvicorn.run("app.main:app", host=host, port=port, reload=reload_enabled)
