import pytest
import asyncio
from datetime import timedelta
from typing import List
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession


@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def mcp_session():
    """Set up an MCP session for testing."""
    server_url = "http://127.0.0.1:8000/mcp/"
    async with streamablehttp_client(url=server_url, timeout=100, sse_read_timeout=300) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(
            read_stream=read_stream,
            write_stream=write_stream,
            read_timeout_seconds=timedelta(seconds=100),
        ) as session:
            yield session


@pytest.mark.asyncio
async def test_initialize_session(mcp_session: ClientSession):
    """Test session initialization."""
    # Given: An MCP session
    session = mcp_session
    expected_protocol_version = "1.0"  # Adjust based on your server

    # When: Initializing the session
    result = await session.initialize()

    # Then: The session initializes with the expected protocol version
    assert result.protocolVersion == expected_protocol_version, f"Expected protocol version {expected_protocol_version}, got {result.protocolVersion}"


@pytest.mark.asyncio
async def test_list_tools(mcp_session: ClientSession):
    """Test listing available tools."""
    # Given: An initialized MCP session
    session = mcp_session
    expected_tools = ["add"]  # Adjust based on your server's tools

    # When: Listing available tools
    tools_result = await session.list_tools()
    result = [tool.name for tool in tools_result.tools]

    # Then: The tool list contains expected tools
    assert result == expected_tools, f"Expected tools {expected_tools}, got {result}"


@pytest.mark.asyncio
async def test_call_tool(mcp_session: ClientSession):
    """Test calling the 'add' tool."""
    # Given: An initialized MCP session and tool parameters
    session = mcp_session
    tool_name = "add"
    tool_args = {"x": 2, "y": 3}
    expected_result = "5"  # Expected output of add(2, 3)

    # When: Calling the 'add' tool
    call_result = await session.call_tool(tool_name, tool_args)
    try:
        result = call_result.content[0].text
    except AttributeError:
        result = call_result.structuredContent.get("result", "Unknown")

    # Then: The tool returns the expected result
    assert result == expected_result, f"Expected result {expected_result}, got {result}"
