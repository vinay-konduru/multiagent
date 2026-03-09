from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, description="End-user message from ChatGPT")
    session_id: str | None = Field(default=None, description="Optional client session identifier")


class ChatResponse(BaseModel):
    agent: str = Field(..., description="Selected specialist agent")
    answer: str = Field(..., description="Final response to return to ChatGPT")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Router confidence score")
    routing_source: str = Field(..., description="Route decision source: llm or fallback")
    routing_reason: str = Field(..., description="Short reason associated with route selection")
