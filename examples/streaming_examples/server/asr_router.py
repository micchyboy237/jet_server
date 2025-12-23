from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from rich.logging import RichHandler

asr_router = APIRouter(prefix="/asr", tags=["asr", "speech-recognition", "translation"])

# Updated logging config: no format/datefmt (RichHandler handles format), consistent enable/disable with LOG_LEVEL env
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("asr_router")

# Load faster-whisper model
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    local_files_only=True,
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
    chunk_duration_sec = 5.0
    min_buffer_sec = 0.250
    chunk_count = 0

    async for chunk_bytes in audio_chunks:
        # Normalize to float32 using 32767.0 for int16 range
        audio_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        buffer.append(audio_np)

        current_duration = len(buffer) * chunk_duration_sec
        chunk_count += 1

        log.debug(f"[bold cyan]Received chunk {chunk_count}[/] – buffer duration: {current_duration:.2f}s")

        if current_duration < min_buffer_sec:
            continue

        # Concatenate buffered audio for inference
        full_audio = np.concatenate(buffer)
        log.info(f"[bold green]Running inference on {current_duration:.2f}s buffer ({len(full_audio)} samples)[/]")

        # Run faster-whisper with translation (use VAD, as in suggested update)
        segments, _ = model.transcribe(
            full_audio,
            language="ja",
            task="translate",
            beam_size=5,
            temperature=0.0,
            without_timestamps=False,
            log_progress=True,
        )

        for seg in segments:
            # Send partial (current hypothesis during decoding - but transcribe yields only final segments)
            # For true token-streaming partials, a more advanced implementation would be needed.
            yield {
                "partial": seg.text.strip(),
                "final": False,
            }
            # Send final translated English segment
            yield {
                "english": seg.text.strip(),
                "final": True,
                "start": seg.start,
                "end": seg.end,
            }

        # Convert segments generator to list only if needed for overlap calculation
        segments_list = list(segments) if segments else []
        if segments_list:
            last_end = segments_list[-1].end
            overlap_samples = int(last_end * sample_rate)
            if 0 < overlap_samples < len(full_audio):
                buffer = [full_audio[overlap_samples:]]
                log.debug(f"[dim]Kept overlap of {len(buffer[0])/sample_rate:.2f}s for context[/]")
            else:
                buffer = []
                log.debug("[dim]Buffer cleared after processing[/]")
        else:
            # No speech detected in this buffer — clear buffer to avoid growing silence
            buffer = []
            log.debug("[dim]No speech — buffer cleared[/]")
            yield {"partial": "", "final": False}

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