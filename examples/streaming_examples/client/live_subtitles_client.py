# jet_server/examples/streaming_examples/client/live_subtitles_client.py
"""Live subtitles client: microphone → Silero VAD (optional) → WebSocket → faster-whisper → subtitles.srt

Records continuously from the microphone (or loopback), sends raw PCM chunks to the server,
receives translated subtitles, displays them live, and writes a growing subtitles.srt file.
"""

import asyncio
import json
import time
from jet.audio.speech.silero.speech_timestamps_extractor import extract_speech_timestamps
from jet.audio.speech.utils import convert_audio_to_tensor
import numpy as np
from datetime import timedelta
from pathlib import Path
from typing import List

import sounddevice as sd
import websockets
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from silero_vad import load_silero_vad

from jet.audio.helpers.silence import detect_silence

console = Console()

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_SECONDS = 0.2
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
SRT_PATH = Path("subtitles.srt")

# WEBSOCKETS_URI = "ws://localhost:8000/asr/live-jp-en"
WEBSOCKETS_URI = "ws://192.168.68.150:8001/asr/live-jp-en"

silero_model = load_silero_vad(onnx=False)


class SubtitleEntry:
    def __init__(self, index: int, start: float, end: float, japanese: str, english: str):
        self.index = index
        self.start = start
        self.end = end
        self.japanese = japanese
        self.english = english

    def to_srt(self) -> str:
        # Proper SRT timestamp format: HH:MM:SS,mmm with leading zeros
        def format_time(seconds: float) -> str:
            total_ms = round(timedelta(seconds=seconds).total_seconds() * 1000)
            hours, rem = divmod(total_ms, 3600000)
            minutes, rem = divmod(rem, 60000)
            seconds, millis = divmod(rem, 1000)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

        start_str = format_time(self.start)
        end_str = format_time(self.end)
        return f"{self.index}\n{start_str} --> {end_str}\n{self.japanese}\n{self.english}\n\n"


class LiveSubtitles:
    def __init__(self) -> None:
        self.entries: List[SubtitleEntry] = []
        self.current_partial = ""
        self.recording_start: float | None = None

    def update(self, result: dict) -> None:
        if self.recording_start is None:
            self.recording_start = time.time()

        relative_offset = time.time() - self.recording_start

        if "partial" in result:
            self.current_partial = result["partial"]

        if result.get("final", False):
            jp = result.get("original", result.get("japanese", "")).strip()  # fallback for compatibility
            en = result.get("english", "").strip()
            start = relative_offset + (result.get("start", 0.0) or 0.0)
            end = relative_offset + (result.get("end", 0.0) or 0.0)

            if en:  # We require at least English translation
                entry = SubtitleEntry(
                    index=len(self.entries) + 1,
                    start=start,
                    end=end,
                    japanese=jp,
                    english=en,
                )
                self.entries.append(entry)
                self._write_srt()
            self.current_partial = ""

    def _write_srt(self) -> None:
        content = "".join(entry.to_srt() for entry in self.entries)
        SRT_PATH.write_text(content, encoding="utf-8")

    def get_panel(self) -> Panel:
        # Show last 5 entries for readability
        recent = self.entries[-5:]
        jp_lines = "\n".join(e.japanese for e in recent) or "Waiting for speech..."
        en_lines = "\n".join(e.english for e in recent) or ""
        partial = f"Partial: {self.current_partial}" if self.current_partial else ""

        # Add chunk info at the top
        chunk_info = Text.assemble(
            f"[dim]Chunk: {CHUNK_SECONDS}s ({CHUNK_SAMPLES} samples @ {SAMPLE_RATE}Hz)[/dim]\n"
        )

        content = Text.assemble(
            "[bold magenta]Live Japanese → English Subtitles[/]\n\n",
            chunk_info,
            Text(jp_lines, style="bold yellow"),
            "\n\n",
            Text(en_lines, style="bold white"),
            "\n\n",
            Text(partial, style="dim cyan"),
        )
        return Panel(content, border_style="bright_blue")


async def live_subtitles_client() -> None:
    console.print("[bold green]=== Live Subtitles Client (Microphone → Real-time Translation) ===[/]")
    uri = WEBSOCKETS_URI

    subtitles = LiveSubtitles()

    current_speech_chunks: List[np.ndarray] = []

    while True:
        try:
            async with websockets.connect(uri) as ws:
                console.print("[bold green]Connected to server[/]")
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    blocksize=CHUNK_SAMPLES,
                ) as stream:
                    with Live(subtitles.get_panel(), console=console, refresh_per_second=1) as live:
                        console.print("[yellow]Listening on microphone/loopback... Speak Japanese for live English subtitles[/]")

                        try:
                            while True:
                                chunk, overflowed = stream.read(CHUNK_SAMPLES)
                                if overflowed:
                                    console.print(
                                        "[bold red]Warning: Input overflow detected — audio data was lost![/bold red]"
                                    )
                                current_speech_chunks.append(chunk)

                                if not detect_silence(chunk, verbose=False):
                                    current_speech_tensor = convert_audio_to_tensor(current_speech_chunks)
                                    speech_ts, speech_probs = extract_speech_timestamps(audio=current_speech_tensor, model=silero_model, with_scores=True)
                                    if speech_ts:
                                        # chunk_bytes = chunk.tobytes()
                                        current_speech_np = np.concatenate(current_speech_chunks, axis=0)
                                        await ws.send(current_speech_np.tobytes())

                                        try:
                                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                                            result = json.loads(msg)
                                            subtitles.update(result)
                                            live.update(subtitles.get_panel())
                                        except asyncio.TimeoutError as e:
                                            console.print(f"[bold red]Timeout error: {e}[/]")
                                        finally:
                                            # Reset for new speech
                                            current_speech_chunks = []

                        except (websockets.ConnectionClosedError, websockets.ConnectionClosedOK):
                            console.print("[bold red]Server disconnected[/]")
                            break
                        except Exception as e:
                            console.print(f"[bold red]Unexpected error: {e}[/]")
                            break
        except websockets.InvalidStatusCode:
            console.print("[bold red]Server rejected connection (status code)[/]")
            break
        except websockets.ConnectionClosedError:
            console.print("[bold red]Connection closed during handshake[/]")
            break
        except Exception as e:
            console.print(f"[bold red]Failed to connect: {e}[/]")
            console.print("[yellow]Retrying in 3 seconds...[/]")
            await asyncio.sleep(3)

    # Ensure final SRT is written on any exit
    subtitles._write_srt()

    console.print("[bold green]Client stopped. Final subtitles saved to subtitles.srt[/]")


if __name__ == "__main__":
    asyncio.run(live_subtitles_client())