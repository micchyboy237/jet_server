from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

asr_router = APIRouter(prefix="/asr", tags=["asr", "speech-recognition", "translation"])

# Expected client format:
# - Raw 16kHz mono 16-bit little-endian PCM audio bytes
# - Chunks sent in real-time as speech is detected


@asr_router.websocket("/live-jp-en")
async def live_japanese_to_english_websocket(websocket: WebSocket) -> None:
    """Real-time streaming ASR + translation: Japanese audio → English text.

    Bidirectional WebSocket optimized for low-latency live translation.
    Client streams audio chunks → server responds with incremental partial/final results.

    In production: replace mock logic with faster-whisper, CTranslate2, or similar
    streaming inference pipeline (Japanese source → English target).
    """
    await websocket.accept()
    chunk_count = 0
    try:
        while True:
            # Receive raw audio chunk
            audio_bytes: bytes = await websocket.receive_bytes()
            chunk_count += 1

            # Simulate realistic inference latency
            await asyncio.sleep(0.15)

            # Mock partial updates more frequently, finals less often
            partial = f"recognizing chunk {chunk_count}..."
            if chunk_count % 4 == 0:
                partial = f"聞き取っています... {chunk_count}"

            transcription = {
                "chunk": chunk_count,
                "partial": partial,
                "final": chunk_count % 8 == 0,  # Every 8th chunk marked as final (more frequent finals)
            }

            if transcription["final"]:
                transcription["japanese"] = f"これはチャンク 1-{chunk_count} の音声です。"
                transcription["english"] = f"This is the audio from chunks 1-{chunk_count}."

            await websocket.send_json(transcription)
    except WebSocketDisconnect:
        pass
