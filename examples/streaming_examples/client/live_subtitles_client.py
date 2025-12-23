import asyncio
import json
import math
from rich.console import Console
from rich.table import Table
import websockets  # pip install websockets

console = Console()


def generate_audio_chunk(frame: int, duration_sec: float = 0.1) -> bytes:
    sample_rate = 16000
    freq = 440.0
    samples = int(sample_rate * duration_sec)
    return b"".join(
        int(10000 * math.sin(2 * math.pi * freq * (frame * samples + i) / sample_rate)).to_bytes(2, "little", signed=True)
        for i in range(samples)
    )


async def run_live_subtitles() -> None:
    console.print("[bold cyan]=== Live Japanese → English ASR (Real-time Translation) ===[/]")
    uri = "ws://localhost:8000/asr/live-jp-en"  # Updated to new dedicated endpoint

    table = Table(title="Live Transcription & Translation")
    table.add_column("Chunk")
    table.add_column("Partial")
    table.add_column("Japanese (final)")
    table.add_column("English (final)")

    async with websockets.connect(uri) as ws:
        for i in range(1, 41):  # ~8 seconds of audio (40 × 0.2s); endpoint is now /asr/live-jp-en
            chunk = generate_audio_chunk(i, duration_sec=0.2)
            await ws.send(chunk)
            console.print(f"[yellow]→ Sent live audio chunk {i} ({len(chunk)} bytes)[/]")

            response = json.loads(await ws.recv())
            jp_text = response.get("japanese", "-")
            en_text = response.get("english", "-")
            table.add_row(
                str(response["chunk"]),
                response["partial"],
                jp_text if response.get("final") else "-",
                en_text if response.get("final") else "-",
            )
            console.print(table)

            await asyncio.sleep(0.18)  # Simulate real microphone capture rate

    console.print("[green]Live ASR translation demo completed.[/]")
    console.print(
        "[bold]Production tip:[/] Replace mock logic with faster-whisper or CTranslate2 "
        "streaming inference (Japanese → English)."
    )


async def main() -> None:
    await run_live_subtitles()

if __name__ == "__main__":
    asyncio.run(main())
