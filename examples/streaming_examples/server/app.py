from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import streaming_router

app = FastAPI(
    title="HTTP Streaming Demo API",
    description="Demonstration of various HTTP streaming content types using FastAPI",
    version="1.0.0",
)

# Optional: Allow CORS for browser-based clients (e.g., EventSource testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(streaming_router)

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "HTTP Streaming Demo API is running",
        "docs": "/docs",
        "available_streams": {
            "sse": "/stream/sse",
            "octet": "/stream/octet",
            "json_chunk": "/stream/json-chunk",
            "ndjson": "/stream/ndjson",
            "mjpeg": "/stream/mjpeg",
            "websocket": "/stream/ws (WebSocket)",
        },
    }