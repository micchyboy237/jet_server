from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from rich.logging import RichHandler

asr_router = APIRouter(prefix="/asr", tags=["asr", "speech-recognition", "translation"])

# Configure rich logging (verbose when LOG_LEVEL=DEBUG)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("asr_router")
# Allow verbose mode via environment: LOG_LEVEL=DEBUG python -m uvicorn ...

# Load faster-whisper model
# Adjust device/compute_type for your hardware:
# - Mac M1/M2: device="cpu", compute_type="int8" (fast) or "float16" if supported
# - CUDA GPU:  device="cuda", compute_type="float16"
# - CPU fallback: compute_type="int8_float32"
# - For pre-downloaded models: local_files_only=True

model = WhisperModel(
    "small",                     # Change size as needed (e.g., "base", "medium", "large")
    device="cpu",                # Set "cuda" if you want GPU acceleration
    compute_type="int8",         # See notes above
    local_files_only=True,       # If you have models downloaded already
)

async def streaming_asr_inference(audio_chunks: AsyncGenerator[bytes, None]) -> AsyncGenerator[dict, None]:
    """
    Stream raw mono PCM16 audio chunks
    → Buffer until minimum length reached,
    → Send buffered audio to faster-whisper (Japanese→English),
    → Yield partial/final results as recognized.
    """
    buffer: list[np.ndarray] = []
    sample_rate = 16000
    chunk_duration_sec = 0.2
    min_buffer_sec = 1.5  # trigger inference roughly every 1.5 seconds
    chunk_count = 0

    async for chunk_bytes in audio_chunks:
        # Convert raw int16 PCM → float32 normalized numpy array
        audio_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        buffer.append(audio_np)

        current_duration = len(buffer) * chunk_duration_sec
        chunk_count += 1

        log.debug(f"[bold cyan]Received chunk {chunk_count}[/] – buffer duration: {current_duration:.2f}s")

        if current_duration < min_buffer_sec:
            # Not enough audio yet – continue buffering
            continue

        # Concatenate buffered audio for inference
        full_audio = np.concatenate(buffer)
        log.info(f"[bold green]Running inference on {current_duration:.2f}s buffer ({len(full_audio)} samples)[/]")

        # Run faster-whisper with translation
        segments, _ = model.transcribe(
            full_audio,
            language="ja",
            task="translate",
            beam_size=5,
            vad_filter=False,
            vad_parameters=dict(min_silence_duration_ms=500),
            temperature=0.0,
            without_timestamps=False,
            log_progress=True,
        )

        last_end = 0.0
        segment_count = 0
        for seg in segments:
            # Partial result (ongoing recognition)
            yield {
                "partial": seg.text.strip(),
                "final": False,
                "start": seg.start,
                "end": seg.end,
            }

            # Final result for completed segment
            yield {
                "japanese": seg.text.strip(),   # approximated original Japanese
                "english": seg.text.strip(),    # translated English
                "final": True,
                "start": seg.start,
                "end": seg.end,
            }
            segment_count += 1
            last_end = seg.end

        log.info(f"[bold magenta]Detected {segment_count} segment(s) – last end: {last_end:.2f}s[/]")

        # Keep overlap for next inference to maintain context
        overlap_samples = int(last_end * sample_rate)
        if overlap_samples > 0 and overlap_samples < len(full_audio):
            buffer = [full_audio[overlap_samples:]]
            log.debug(f"[dim]Kept overlap of {len(buffer[0])/sample_rate:.2f}s for context[/]")
        else:
            buffer = []
            log.debug("[dim]No overlap kept – buffer cleared[/]")

        # If no segments detected, send a listening indicator
        if last_end == 0.0:
            yield {"partial": "listening...", "final": False}
            log.debug("[dim]No speech detected – sent listening indicator[/]")

@asr_router.websocket("/live-jp-en")
async def live_japanese_to_english_websocket(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for live streaming Japanese→English translation.
      - Client streams raw 16kHz mono PCM16 audio chunks.
      - Server emits JSON: {partial/final results, times, ...}
    """
    await websocket.accept()

    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    log.info("[bold blue]WebSocket connection accepted – starting audio receive/process tasks[/]")

    async def receive_audio() -> None:
        """Continuously receive audio chunks from the client."""
        received_count = 0
        while True:
            try:
                audio_bytes: bytes = await websocket.receive_bytes()
                received_count += 1
                log.debug(f"[cyan]← Client chunk {received_count} ({len(audio_bytes)} bytes)[/]")
                await audio_queue.put(audio_bytes)
            except WebSocketDisconnect:
                log.info("[yellow]Client disconnected during receive[/]")
                break

    async def process_and_send() -> None:
        """Consume audio chunks and stream ASR results back to client."""
        sent_count = 0
        async def audio_generator() -> AsyncGenerator[bytes, None]:
            while True:
                chunk = await audio_queue.get()
                yield chunk

        async for result in streaming_asr_inference(audio_generator()):
            sent_count += 1
            final_str = "[bold green]FINAL[/]" if result.get("final") else "[dim]partial[/]"
            log.debug(f"[magenta]→ Result {sent_count} {final_str}: {result.get('english', result.get('partial', ''))[:60]}...[/]")
            await websocket.send_json(result)

    try:
        await asyncio.gather(receive_audio(), process_and_send())
    except WebSocketDisconnect:
        log.info("[yellow]WebSocket disconnected – cleanup complete[/]")