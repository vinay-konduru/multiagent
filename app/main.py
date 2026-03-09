from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.agents import AgentRunner
from app.graph import UPSMultiAgentGraph
from app.models import ChatRequest, ChatResponse
from app.supervisor import SupervisorRouter

load_dotenv()

app = FastAPI(
    title="UPS Multi-Agent Supervisor API",
    version="1.0.0",
    description=(
        "Supervisor-style router API for ChatGPT Actions. "
        "Routes user queries to shipping/tracking/general specialist agents."
    ),
)

router: SupervisorRouter | None = None
graph_runner: UPSMultiAgentGraph | None = None


@app.on_event("startup")
def _startup() -> None:
    global router, graph_runner
    agent_runner = AgentRunner()
    router = SupervisorRouter(agent_runner=agent_runner)
    graph_runner = UPSMultiAgentGraph(supervisor=router, agent_runner=agent_runner)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ups-multi-agent"}


@app.post("/chat", response_model=ChatResponse, operation_id="chat")
def chat(payload: ChatRequest) -> ChatResponse:
    if graph_runner is None:
        raise HTTPException(status_code=500, detail="LangGraph runner is not initialized")

    try:
        result = graph_runner.run(payload.user_message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    selected_agent = str(result.get("selected_agent", "general"))
    answer = str(result.get("answer", ""))
    confidence = float(result.get("confidence", 0.55))
    routing_source = str(result.get("routing_source", "fallback"))
    routing_reason = str(result.get("routing_reason", "unknown"))

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
    uvicorn.run("app.main:app", host=host, port=port, reload=True)

