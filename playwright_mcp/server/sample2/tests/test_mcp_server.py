import pytest
import asyncio
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from pydantic import BaseModel
import os


class FileOutput(BaseModel):
    content: str


class UrlOutput(BaseModel):
    title: str


@pytest.fixture
async def mcp_server():
    server_params = StdioServerParameters(
        command="python", args=["mcp_server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@pytest.mark.asyncio
async def test_read_file_tool(mcp_server):
    # Given: A valid file and expected contents
    file_path = "test.txt"
    expected_content = "Hello, FastMCP!"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(expected_content)

    # When: Executing the read_file tool
    result = await mcp_server.execute_tool("read_file", {"file_path": file_path, "encoding": "utf-8"})

    # Then: The result matches the expected content
    result_data = FileOutput.model_validate_json(result[0]["content"])
    assert result_data.content == expected_content, f"Expected '{expected_content}', got '{result_data.content}'"

    # Cleanup
    os.remove(file_path)


@pytest.mark.asyncio
async def test_read_file_invalid_path(mcp_server):
    # Given: An invalid file path
    file_path = "nonexistent.txt"
    expected_error = f"Error reading file: [Errno 2] No such file or directory: '{file_path}'"

    # When: Executing the read_file tool
    result = await mcp_server.execute_tool("read_file", {"file_path": file_path, "encoding": "utf-8"})

    # Then: The result contains the expected error
    result_data = FileOutput.model_validate_json(result[0]["content"])
    assert expected_error in result_data.content, f"Expected '{expected_error}' in result, got '{result_data.content}'"


@pytest.mark.asyncio
async def test_navigate_to_url_tool(mcp_server):
    # Given: A valid URL and expected title substring
    url = "https://example.com"
    expected_title_contains = "Example Domain"

    # When: Executing the navigate_to_url tool
    result = await mcp_server.execute_tool("navigate_to_url", {"url": url})

    # Then: The result contains the expected title
    result_data = UrlOutput.model_validate_json(result[0]["content"])
    assert expected_title_contains in result_data.title, f"Expected '{expected_title_contains}' in result, got '{result_data.title}'"


@pytest.mark.asyncio
async def test_navigate_to_url_invalid(mcp_server):
    # Given: An invalid URL
    url = "invalid://url"
    expected_error_contains = "Error navigating to invalid://url"

    # When: Executing the navigate_to_url tool
    result = await mcp_server.execute_tool("navigate_to_url", {"url": url})

    # Then: The result contains the expected error
    result_data = UrlOutput.model_validate_json(result[0]["content"])
    assert expected_error_contains in result_data.title, f"Expected '{expected_error_contains}' in result, got '{result_data.title}'"
