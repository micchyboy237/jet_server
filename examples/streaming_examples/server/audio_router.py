from __future__ import annotations

from typing import AsyncGenerator
import asyncio
import math

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import StreamingResponse

audio_router = APIRouter(prefix="/audio", tags=["audio"])

# Expected format for live ASR clients:
# - Raw 16kHz mono 16-bit PCM (little-endian) audio bytes
# - Client streams chunks as they are captured (real-time, low latency)


def generate_sample_audio_chunk(frame: int) -> bytes:
    """Generate a small 16-bit mono 16kHz sine wave chunk for demo purposes."""
    freq = 440.0  # A4 note
    sample_rate = 16000
    samples_per_chunk = 3200  # 0.2s chunk
    return b"".join(
        int(10000 * math.sin(2 * math.pi * freq * (frame * samples_per_chunk + i) / sample_rate)).to_bytes(2, "little", signed=True)
        for i in range(samples_per_chunk)
    )


async def audio_stream_generator() -> AsyncGenerator[bytes, None]:
    """Server-side streaming of generated audio chunks (e.g., synthesized speech or music)."""
    for i in range(30):  # ~6 seconds of audio
        await asyncio.sleep(0.2)
        yield generate_sample_audio_chunk(i)


@audio_router.get("/stream", response_class=StreamingResponse)
async def audio_stream_endpoint() -> StreamingResponse:
    """HTTP streaming of raw audio (16kHz mono PCM). Ideal for playback without WebSocket overhead."""
    return StreamingResponse(
        audio_stream_generator(),
        media_type="audio/wav",  # Client can treat as raw PCM or wrap in WAV container
        headers={
            "Content-Disposition": 'attachment; filename="sine_stream.pcm"',
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": "16000",
            "X-Channels": "1",
        },
    )


@audio_router.websocket("/transcribe")
async def audio_transcribe_websocket(websocket: WebSocket) -> None:
    """Bidirectional WebSocket: client sends raw audio bytes (16kHz mono PCM), server responds with incremental transcriptions."""
    await websocket.accept()
    chunk_count = 0
    try:
        while True:
            audio_bytes: bytes = await websocket.receive_bytes()
            chunk_count += 1

            await asyncio.sleep(0.2)

            transcription = {
                "chunk": chunk_count,
                "partial": f"recognizing audio chunk {chunk_count}...",
                "final": chunk_count % 10 == 0,  # Every 10th chunk is "final"
            }
            if transcription["final"]:
                transcription["text"] = f"This is the transcription for chunks 1-{chunk_count}."
            await websocket.send_json(transcription)
    except WebSocketDisconnect:
        pass


@audio_router.websocket("/echo")
async def audio_echo_websocket(websocket: WebSocket) -> None:
    """Simple echo WebSocket: receives audio bytes and immediately sends them back (e.g., for latency testing or VoIP demo)."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            # Small processing delay to simulate real work
            await asyncio.sleep(0.05)
            await websocket.send_bytes(data)
    except WebSocketDisconnect:
        pass


@audio_router.post("/upload-transcribe")
async def upload_transcribe_endpoint(file: UploadFile = File(...)) -> dict:
    """Upload a complete audio file (WAV/PCM) and receive a mock full transcription."""
    content = await file.read()
    # Reduced simulation delay for demo responsiveness (original was too long)
    await asyncio.sleep(min(len(content) / 100000, 2.0))
    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "mock_transcription": "This is a full mock transcription of the uploaded audio file.",
        "note": "In production, integrate with Whisper, Vosk, or a cloud ASR service."
    }
