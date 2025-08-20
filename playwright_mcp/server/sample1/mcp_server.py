import asyncio
from datetime import timedelta
from typing import List, Dict, Any
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from mcp.types import Tool


async def run_mcp_client(server_url: str) -> None:
    """
    Connects to an MCP server, initializes a session, lists tools, and calls a tool.

    Args:
        server_url (str): URL of the MCP server (e.g., "http://127.0.0.1:8000/mcp/").
    """
    try:
        # Establish connection using streamable HTTP transport
        async with streamablehttp_client(
            url=server_url,
            timeout=100,
            sse_read_timeout=300,
        ) as (read_stream, write_stream, get_session_id):
            # Create and initialize ClientSession
            async with ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                read_timeout_seconds=timedelta(seconds=100),
            ) as session:
                # Initialize the session
                init_result = await session.initialize()
                print(
                    f"Initialized - Protocol version: {init_result.protocolVersion}")
                print(f"Session ID: {get_session_id()}")

                # List available tools
                tools_result = await session.list_tools()
                tool_names: List[str] = [
                    tool.name for tool in tools_result.tools]
                print(f"Available tools: {tool_names}")

                # Call a tool (assuming 'add' tool exists with x, y parameters)
                if "add" in tool_names:
                    call_result = await session.call_tool("add", {"x": 2, "y": 3})
                    # Access result based on SDK version (content or structuredContent)
                    try:
                        result = call_result.content[0].text
                    except AttributeError:
                        result = call_result.structuredContent.get(
                            "result", "Unknown")
                    print(f"Result of add(2, 3): {result}")
                else:
                    print("Tool 'add' not found on server.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")


async def main() -> None:
    """Main entry point for the MCP client."""
    server_url = "http://127.0.0.1:8000/mcp/"  # Replace with your server URL
    await run_mcp_client(server_url)

if __name__ == "__main__":
    asyncio.run(main())
