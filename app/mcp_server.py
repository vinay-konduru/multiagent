from mcp.server.fastmcp import FastMCP
import requests
import os

mcp = FastMCP("ups-router")

# Allow overriding backend URL from environment; default matches local server port.
UPS_URL = os.getenv("UPS_URL", "http://127.0.0.1:8002/chat")


@mcp.tool()
def ups_route_query(user_message: str, session_id: str = "copilot-session") -> dict:
    """Route UPS question through your LangGraph supervisor API."""
    response = requests.post(
        UPS_URL,
        json={"user_message": user_message, "session_id": session_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
    print("MCP server has stopped.")