# UPS Multi-Agent Router for ChatGPT Actions

This project creates a supervisor-style multi-agent backend you can connect to official ChatGPT using **Actions**.

Flow (LangGraph):
1. User asks a question in ChatGPT.
2. ChatGPT Action calls this API (`/chat`).
3. A compiled LangGraph workflow executes `START -> supervisor -> specialist -> END`.
4. Supervisor (LLM-first, heuristic fallback) routes to one of:
   - `shipping` agent
   - `tracking` agent
   - `general` agent
5. Response is returned to ChatGPT and displayed to the user.

## Important limitation
You cannot change OpenAI's internal router for all official ChatGPT globally. The supported approach is:
- Create a **Custom GPT** with an **Action** that calls your API.
- Let your API implement supervisor + specialist agent routing.

## Prototype mode behavior
- This project is a prototype and uses proxy/LLM-generated responses.
- Tracking and shipping outputs are simulated unless you explicitly integrate real UPS APIs.
- Do not treat returned status details as live UPS source-of-truth data.

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
copy .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`.

4. Run API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

5. Expose your local API publicly for ChatGPT Actions (for development), e.g. with ngrok.

## API

- `POST /chat`
- Request:

```json
{
  "user_message": "Where is my package 1Z999... ?",
  "session_id": "optional-session-id"
}
```

- Response:

```json
{
  "agent": "tracking",
  "answer": "...",
  "confidence": 0.93,
  "routing_source": "llm",
  "routing_reason": "contains tracking number"
}
```

## ChatGPT Custom GPT Action setup
1. In ChatGPT, create a new GPT.
2. Go to **Configure** -> **Actions** -> **Create new action**.
3. Import the OpenAPI schema from your running service:
   - `https://<your-public-url>/openapi.json`
4. In GPT instructions, tell it to call the `chat` action for UPS-related user queries.

Suggested GPT instruction snippet:

```text
You are a UPS assistant. For shipping, tracking, delivery, rates, labels, pickup, claims, or general UPS policy questions, always call the chat action with the full user message. Return the action result naturally and clearly.
```

## Notes
- LangGraph workflow is defined in `app/graph.py`.
- Heuristic fallback routing policy is in `app/supervisor.py`.
- Specialist prompts are in `app/agents.py`.
- Replace prompt logic with real UPS API calls for production (rates, tracking events, label creation, etc.).
