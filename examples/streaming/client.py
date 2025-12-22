import asyncio
import json

import httpx
from rich.console import Console
from rich.table import Table

console = Console()


async def demo_sse() -> None:
    console.print("[bold cyan]=== SSE (text/event-stream) ===[/]")
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", "http://localhost:8000/stream/sse") as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    console.print(data)


async def demo_octet() -> None:
    console.print("[bold cyan]=== Octet-stream ===[/]")
    received = 0
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", "http://localhost:8000/stream/octet") as resp:
            async for chunk in resp.aiter_bytes(chunk_size=1024):
                received += len(chunk)
                # In real use: write to file, feed audio pipeline, etc.
    console.print(f"Received {received} bytes of raw binary data")


async def demo_json_chunk() -> None:
    console.print("[bold cyan]=== Chunked JSON ===[/]")
    table = Table(title="Received JSON fragments")
    table.add_column("Index")
    table.add_column("Value")
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", "http://localhost:8000/stream/json-chunk") as resp:
            async for line in resp.aiter_lines():
                if line:
                    obj = json.loads(line)
                    table.add_row(str(obj["index"]), obj["value"])
                    console.print(table)


async def demo_ndjson() -> None:
    console.print("[bold cyan]=== NDJSON ===[/]")
    tokens: list[str] = []
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", "http://localhost:8000/stream/ndjson") as resp:
            async for line in resp.aiter_lines():
                if line:
                    obj = json.loads(line)
                    tokens.append(obj["token"])
                    console.print(obj)
    console.print("[green]Full reconstructed message:[/]", " ".join(tokens))


async def demo_mjpeg() -> None:
    console.print("[bold cyan]=== MJPEG (multipart) ===[/]")
    count = 0
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", "http://localhost:8000/stream/mjpeg") as resp:
            async for chunk in resp.aiter_raw():
                if b"--frame" in chunk:
                    count += 1
                    console.print(f"Received frame {count}")
                    # In real use: parse JPEG from chunk and display/save


async def main() -> None:
    await demo_sse()
    await demo_octet()
    await demo_json_chunk()
    await demo_ndjson()
    await demo_mjpeg()


if __name__ == "__main__":
    asyncio.run(main())