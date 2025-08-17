import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pymcp_sse.client import MultiMCPClient, BaseMCPClient
import aiohttp
from typing import Dict, Any

from jet.logger import logger


app = FastAPI(title="MCP Proxy Server", version="0.1.0")

# Custom BaseMCPClient to override health check


class CustomBaseMCPClient(BaseMCPClient):
    async def _check_health(self) -> bool:
        """Override health check to include required Accept header."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Accept": "application/json, text/event-stream"}
                logger.debug(
                    f"Custom health check for {self._url}/health with headers: {headers}")
                async with session.get(f"{self._url}/health", headers=headers) as resp:
                    if resp.status == 200:
                        logger.debug(f"Health check succeeded: {await resp.text()}")
                        return True
                    logger.error(f"Health check failed: status={resp.status}, body={await resp.text()}")
                    return False
        except Exception as e:
            logger.error(f"Health check error: {str(e)}", exc_info=True)
            return False

# Custom MultiMCPClient to use CustomBaseMCPClient


class CustomMultiMCPClient(MultiMCPClient):
    def __init__(self, servers: Dict[str, str]):
        super().__init__(servers)
        # Replace default BaseMCPClient with CustomBaseMCPClient
        for name, url in servers.items():
            self.clients[name] = CustomBaseMCPClient(url)


mcp_client = CustomMultiMCPClient(
    {
        "playwright": "http://localhost:8931/mcp"
    }
)


@app.on_event("startup")
async def startup_event():
    """Connect to MCP servers when the FastAPI app starts."""
    try:
        # Manual health check for debugging
        async with aiohttp.ClientSession() as session:
            headers = {"Accept": "application/json, text/event-stream"}
            logger.debug(
                f"Manual health check at http://localhost:8931/mcp/health with headers: {headers}")
            async with session.get("http://localhost:8931/mcp/health", headers=headers) as resp:
                response_text = await resp.text()
                logger.debug(
                    f"Health check response: status={resp.status}, body={response_text}")

        await mcp_client.connect_all()
        if not mcp_client.clients:
            logger.error(
                "No MCP servers connected. Check server availability at http://localhost:8931/mcp")
        else:
            logger.info(
                f"✅ Connected to MCP servers: {list(mcp_client.clients.keys())}")
    except Exception as e:
        logger.error(
            f"Failed to connect to MCP servers: {str(e)}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Close MCP connections on shutdown."""
    await mcp_client.close()
    logger.info("🛑 MCP client closed")


@app.get("/servers")
async def list_servers():
    """List all connected MCP servers and their info."""
    if not mcp_client.clients:
        return JSONResponse(content={"error": "No servers connected", "details": "Check if MCP servers are running"})
    info = {name: client.server_info for name,
            client in mcp_client.clients.items()}
    return JSONResponse(content=info)


@app.get("/tools/{server_name}")
async def list_tools(server_name: str):
    """List tools from a specific MCP server."""
    if server_name not in mcp_client.clients:
        return JSONResponse(status_code=404, content={"error": "Server not found"})
    tools = mcp_client.clients[server_name].tools
    return JSONResponse(content=tools)


@app.post("/tools/{server_name}/{tool_name}")
async def call_tool(server_name: str, tool_name: str, args: dict):
    """Call a tool on a given MCP server."""
    if server_name not in mcp_client.clients:
        return JSONResponse(status_code=404, content={"error": "Server not found"})
    try:
        result = await mcp_client.call_tool(server_name, tool_name, **args.get("args", {}))
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
