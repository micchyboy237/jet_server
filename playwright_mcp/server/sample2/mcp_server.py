from starlette.responses import JSONResponse
from starlette.requests import Request
import asyncio
from typing import AsyncIterator, List, Dict
from contextlib import asynccontextmanager
from mcp.server.fastmcp.server import FastMCP, Context
from mcp.types import ContentBlock
from pydantic import BaseModel, Field
from typing import Literal
from playwright.async_api import async_playwright

PLAYWRIGHT_CHROMIUM_EXECUTABLE = "/Users/jethroestrada/Library/Caches/ms-playwright/chromium-1181/chrome-mac/Chromium.app/Contents/MacOS/Chromium"


class FileInput(BaseModel):
    file_path: str = Field(...,
                           description="Path to the file (e.g., 'example.txt')")
    encoding: Literal["utf-8",
                      "ascii"] = Field("utf-8", description="File encoding")


class FileOutput(BaseModel):
    content: str = Field(..., description="File contents or error message")


class UrlInput(BaseModel):
    url: str = Field(..., description="URL to navigate to (e.g., 'https://example.com')",
                     pattern=r"^https?://")


class UrlOutput(BaseModel):
    title: str = Field(..., description="Page title or error message")


@asynccontextmanager
async def lifespan(app: FastMCP[None]) -> AsyncIterator[None]:
    print("Starting FastMCP server...")
    yield
    print("Shutting down FastMCP server...")


server = FastMCP(
    name="FastMCPStandalone",
    instructions="A standalone MCP server with file and browser tools.",
    debug=True,
    log_level="DEBUG",
    lifespan=lifespan
)


@server.tool(description="Read the contents of a file.", annotations={"audience": ["user"], "priority": 0.9})
async def read_file(arguments: FileInput, ctx: Context) -> FileOutput:
    await ctx.info(f"Reading file: {arguments.file_path}")
    try:
        with open(arguments.file_path, "r", encoding=arguments.encoding) as f:
            content = f.read()
        await ctx.report_progress(100, 100, "File read successfully")
        return FileOutput(content=content)
    except Exception as e:
        await ctx.error(f"Error reading file: {str(e)}")
        return FileOutput(content=f"Error reading file: {str(e)}")


@server.tool(description="Navigate to a URL and return the page title.", annotations={"audience": ["assistant"], "priority": 0.8})
async def navigate_to_url(arguments: UrlInput, ctx: Context) -> UrlOutput:
    await ctx.info(f"Navigating to {arguments.url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
            )
            page = await browser.new_page()
            await page.goto(arguments.url)
            title = await page.title()
            await browser.close()
        await ctx.report_progress(100, 100, "Navigation complete")
        return UrlOutput(title=f"Navigated to {arguments.url}. Page title: {title}")
    except Exception as e:
        await ctx.error(f"Error navigating to {arguments.url}: {str(e)}")
        return UrlOutput(title=f"Error navigating to {arguments.url}: {str(e)}")


# Custom routes

@server.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@server.tool(description="Process data with progress.", annotations={"audience": ["user"]})
async def process_data(data: str, ctx: Context) -> str:
    for i in range(1, 101, 10):
        await ctx.report_progress(i, 100, f"Processing step {i}%")
        await asyncio.sleep(0.1)
    return f"Processed: {data}"


@server.resource("resource://welcome", description="A welcome message")
async def welcome_message() -> str:
    return "Welcome to FastMCP!"


@server.prompt(description="Analyze a file")
async def analyze_file(path: str) -> List[Dict]:
    content = open(path, "r").read()
    return [{"role": "user", "content": f"Analyze this content:\n{content}"}]

if __name__ == "__main__":
    server.run(transport="stdio")
