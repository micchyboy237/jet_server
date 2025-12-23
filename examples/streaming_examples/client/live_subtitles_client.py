# jet_server/examples/streaming_examples/client/live_subtitles_client.py
"""Live subtitles client: microphone → Silero VAD (optional) → WebSocket → faster-whisper → subtitles.srt

Records continuously from the microphone (or loopback), sends raw PCM chunks to the server,
receives translated subtitles, displays them live, and writes a growing subtitles.srt file.
"""

import asyncio
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import List

import sounddevice as sd
import websockets
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int8"
CHUNK_SECONDS = 0.2
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
SRT_PATH = Path("subtitles.srt")


class SubtitleEntry:
    def __init__(self, index: int, start: float, end: float, japanese: str, english: str):
        self.index = index
        self.start = start
        self.end = end
        self.japanese = japanese
        self.english = english

    def to_srt(self) -> str:
        start_str = str(timedelta(seconds=self.start))[:-3].replace(".", ",")
        end_str = str(timedelta(seconds=self.end))[:-3].replace(".", ",")
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

        if result.get("final"):
            jp = result.get("japanese", "").strip()
            en = result.get("english", "").strip()
            start = relative_offset + (result.get("start", 0.0) or 0.0)
            end = relative_offset + (result.get("end", 0.0) or 0.0)

            if jp and en:
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

        content = Text.assemble(
            "[bold magenta]Live Japanese → English Subtitles[/]\n\n",
            Text(jp_lines, style="bold yellow"),
            "\n\n",
            Text(en_lines, style="bold white"),
            "\n\n",
            Text(partial, style="dim cyan"),
        )
        return Panel(content, border_style="bright_blue")


async def live_subtitles_client() -> None:
    console.print("[bold green]=== Live Subtitles Client (Microphone → Real-time Translation) ===[/]")
    uri = "ws://localhost:8000/asr/live-jp-en"

    subtitles = LiveSubtitles()

    async with websockets.connect(uri) as ws:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            with Live(subtitles.get_panel(), console=console, refresh_per_second=10) as live:
                console.print("[yellow]Listening on microphone/loopback... Speak Japanese for live English subtitles[/]")

                while True:
                    chunk, _ = stream.read(CHUNK_SAMPLES)
                    audio_bytes = chunk.tobytes()
                    await ws.send(audio_bytes)

                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        result = json.loads(msg)
                        subtitles.update(result)
                        live.update(subtitles.get_panel())
                    except asyncio.TimeoutError:
                        # No new transcription yet – UI stays the same
                        pass

    console.print("[bold green]Client stopped. Final subtitles saved to subtitles.srt[/]")


if __name__ == "__main__":
    asyncio.run(live_subtitles_client())