from __future__ import annotations

from typing import AsyncGenerator, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import asyncio
import json

streaming_router = APIRouter(prefix="/stream", tags=["streaming"])


async def sse_generator() -> AsyncGenerator[str, None]:
    """Simulates progressive server-sent events (e.g., progress updates)."""
    for i in range(5):
        await asyncio.sleep(0.5)  # Simulate work
        yield f"data: {json.dumps({'step': i, 'message': f'Progress {i+1}/5'})}\n\n"


@streaming_router.get("/sse", response_class=StreamingResponse)
async def sse_endpoint() -> StreamingResponse:
    """Server-Sent Events – unidirectional text events."""
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def octet_generator() -> AsyncGenerator[bytes, None]:
    """Simulates raw binary streaming (e.g., audio chunks)."""
    for i in range(10):
        await asyncio.sleep(0.2)
        # Simulated 1KB binary chunk
        yield bytes([i % 256] * 1024)


@streaming_router.get("/octet", response_class=StreamingResponse)
async def octet_endpoint() -> StreamingResponse:
    """Raw binary streaming."""
    return StreamingResponse(
        octet_generator(),
        media_type="application/octet-stream",
    )


async def json_chunk_generator() -> AsyncGenerator[str, None]:
    """Simulates chunked JSON fragments (newline separated)."""
    for i in range(4):
        await asyncio.sleep(0.6)
        yield json.dumps({"index": i, "value": f"data-{i}"}) + "\n"


@streaming_router.get("/json-chunk", response_class=StreamingResponse)
async def json_chunk_endpoint() -> StreamingResponse:
    """Chunked JSON streaming (self-delimited with newlines)."""
    return StreamingResponse(
        json_chunk_generator(),
        media_type="application/json",
    )


async def ndjson_generator() -> AsyncGenerator[str, None]:
    """Newline-delimited JSON objects."""
    for i in range(6):
        await asyncio.sleep(0.4)
        obj: Dict[str, Any] = {"id": i, "token": f"word_{i}", "done": i == 5}
        yield json.dumps(obj) + "\n"


@streaming_router.get("/ndjson", response_class=StreamingResponse)
async def ndjson_endpoint() -> StreamingResponse:
    """NDJSON streaming – ideal for structured token streams."""
    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson",
    )


async def mjpeg_generator() -> AsyncGenerator[bytes, None]:
    """Simulates MJPEG-style multipart frames."""
    boundary = b"--frame"
    for i in range(8):
        await asyncio.sleep(0.3)
        # Placeholder JPEG bytes (in real use: encode actual frame)
        jpeg_bytes = bytes(range(i * 10, i * 10 + 100))  # dummy data
        yield (
            boundary
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n"
            + b"Content-Length: "
            + str(len(jpeg_bytes)).encode()
            + b"\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )


@streaming_router.get("/mjpeg", response_class=StreamingResponse)
async def mjpeg_endpoint() -> StreamingResponse:
    """Multipart/x-mixed-replace – typically for MJPEG video."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def websocket_generator(websocket: WebSocket) -> None:
    """Bidirectional WebSocket streaming – server pushes progressive messages."""
    await websocket.accept()
    try:
        for i in range(7):
            await asyncio.sleep(0.5)
            message = {"index": i, "text": f"token_{i}", "final": i == 6}
            await websocket.send_json(message)
        # Optionally wait for client messages (echo example)
        while True:
            data = await websocket.receive_json()
            await websocket.send_json({"echo": data})
    except WebSocketDisconnect:
        pass

@streaming_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for bidirectional real-time communication."""
    await websocket_generator(websocket)