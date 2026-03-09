from __future__ import annotations

import json
import os
import re
import ssl
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
import httpx
from openai import OpenAI

try:
    import truststore
except ImportError:  # pragma: no cover - optional dependency
    truststore = None

load_dotenv()

AgentName = Literal["shipping", "tracking", "general"]


@dataclass
class SupervisorLLMDecision:
    agent: AgentName
    confidence: float
    reason: str

SYSTEM_PROMPTS: dict[AgentName, str] = {
    "shipping": (
        "You are the UPS Shipping Specialist agent. "
        "This environment is a prototype that uses proxy/LLM-generated responses only, not live UPS systems. "
        "Always state when details are simulated or estimated. "
        "Never claim you queried a live UPS API, database, or real-time tracking backend. "
        "Focus on shipping options, rates, labels, pickup, packaging, customs, and claims process guidance. "
        "Be accurate, practical, and concise. If user-specific account or shipment data is required, "
        "explicitly request the missing data."
    ),
    "tracking": (
        "You are the UPS Tracking Specialist agent. "
        "This environment is a prototype that uses proxy/LLM-generated responses only, not live UPS systems. "
        "Never claim to have fetched data from a live UPS API or tracking system. "
        "If you provide package status details, label them clearly as simulated/prototype output. "
        "Focus on package status interpretation, delivery windows, delay reasons, and next troubleshooting steps. "
        "If tracking number or shipment details are missing, ask for them clearly."
    ),
    "general": (
        "You are the UPS General Support agent. "
        "This environment is a prototype that uses proxy/LLM-generated responses only, not live UPS systems. "
        "Do not present any response as real-time or source-verified UPS data. "
        "Handle broad UPS policy, service, and FAQ-style questions. "
        "Escalate to shipping or tracking framing only when clearly beneficial."
    ),
}


class AgentRunner:
    def __init__(self) -> None:
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.http_client = self._build_http_client()
        self.client = OpenAI(api_key=api_key, http_client=self.http_client)

    @staticmethod
    def _build_http_client() -> httpx.Client:
        verify_ssl = os.getenv("OPENAI_VERIFY_SSL", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        ca_bundle = os.getenv("OPENAI_CA_BUNDLE", "").strip()
        use_system_store = os.getenv("OPENAI_USE_SYSTEM_CERT_STORE", "true").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        verify: bool | str | ssl.SSLContext = True
        if not verify_ssl:
            verify = False
        elif ca_bundle:
            verify = ca_bundle
        elif use_system_store and truststore is not None:
            verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        timeout = httpx.Timeout(60.0, connect=20.0)
        return httpx.Client(verify=verify, timeout=timeout)

    def run(self, agent: AgentName, user_message: str) -> str:
        if agent not in SYSTEM_PROMPTS:
            raise ValueError(f"Unknown agent: {agent}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[agent]},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )

        return (response.choices[0].message.content or "").strip()

    def route_with_llm(self, user_message: str) -> SupervisorLLMDecision:
        """Use an LLM supervisor to choose the best specialist agent."""
        prompt = (
            "You are a supervisor for UPS support routing. "
            "Choose exactly one specialist: shipping, tracking, or general. "
            "Return strict JSON with keys: agent, confidence, reason. "
            "Confidence must be between 0 and 1. "
            "No markdown, no extra keys."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
        )
        raw_text = response.choices[0].message.content or "{}"
        payload = self._extract_json(raw_text)

        agent = str(payload.get("agent", "general")).strip().lower()
        confidence = float(payload.get("confidence", 0.5))
        reason = str(payload.get("reason", "llm-router"))

        if agent not in ("shipping", "tracking", "general"):
            agent = "general"
        confidence = min(max(confidence, 0.0), 1.0)

        return SupervisorLLMDecision(agent=agent, confidence=confidence, reason=reason)

    @staticmethod
    def _extract_json(raw_text: str) -> dict:
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback if model wraps JSON with extra text.
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}

        return {}
