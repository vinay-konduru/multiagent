from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents import AgentName, AgentRunner
from app.supervisor import SupervisorRouter


class GraphState(TypedDict, total=False):
    user_message: str
    selected_agent: AgentName
    confidence: float
    routing_source: str
    routing_reason: str
    answer: str


class UPSMultiAgentGraph:
    """LangGraph workflow: supervisor routes request, specialist agent answers."""

    def __init__(self, supervisor: SupervisorRouter, agent_runner: AgentRunner) -> None:
        self.supervisor = supervisor
        self.agent_runner = agent_runner
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("shipping", self._shipping_node)
        workflow.add_node("tracking", self._tracking_node)
        workflow.add_node("general", self._general_node)

        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "shipping": "shipping",
                "tracking": "tracking",
                "general": "general",
            },
        )
        workflow.add_edge("shipping", END)
        workflow.add_edge("tracking", END)
        workflow.add_edge("general", END)

        return workflow.compile()

    def run(self, user_message: str) -> GraphState:
        result = self.graph.invoke({"user_message": user_message})
        return result

    def _supervisor_node(self, state: GraphState) -> GraphState:
        user_message = state["user_message"]
        decision = self.supervisor.route(user_message)

        return {
            "selected_agent": decision.agent,
            "confidence": decision.confidence,
            "routing_source": decision.source,
            "routing_reason": decision.reason,
        }

    @staticmethod
    def _route_from_supervisor(state: GraphState) -> Literal["shipping", "tracking", "general"]:
        selected = state.get("selected_agent", "general")
        if selected not in ("shipping", "tracking", "general"):
            return "general"
        return selected

    def _shipping_node(self, state: GraphState) -> GraphState:
        return {"answer": self.agent_runner.run("shipping", state["user_message"]) }

    def _tracking_node(self, state: GraphState) -> GraphState:
        return {"answer": self.agent_runner.run("tracking", state["user_message"]) }

    def _general_node(self, state: GraphState) -> GraphState:
        return {"answer": self.agent_runner.run("general", state["user_message"]) }
