from __future__ import annotations

from dataclasses import dataclass

from app.agents import AgentRunner


@dataclass
class RouteDecision:
    agent: str
    confidence: float
    source: str
    reason: str


class SupervisorRouter:
    """LLM-based supervisor router."""

    def __init__(self, agent_runner: AgentRunner) -> None:
        self.agent_runner = agent_runner

    def route(self, user_message: str) -> RouteDecision:
        try:
            decision = self.agent_runner.route_with_llm(user_message)
            return RouteDecision(
                agent=decision.agent,
                confidence=round(decision.confidence, 2),
                source="llm",
                reason=decision.reason,
            )
        except Exception:
            return RouteDecision(
                agent="general",
                confidence=0.55,
                source="fallback",
                reason="llm-route-failed",
            )
