from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Model MCP Server")
BENTO_URL = "http://localhost:3000"

@mcp.tool()
def inspect_model() -> dict:
    """Inspect model health, metadata, and configuration."""
    try:
        response = requests.get(f"{BENTO_URL}/health")
        return response.json()
    except Exception as e:
        return {"error": str(e), "status": "unavailable"}

@mcp.tool()
def run_inference(data: list) -> list:
    """Run batch inference through the BentoML serving layer."""
    response = requests.post(f"{BENTO_URL}/predict", json={"input_data": data})
    return response.json()

if __name__ == "__main__":
    mcp.run()
