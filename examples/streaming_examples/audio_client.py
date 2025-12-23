import asyncio
import json
import math
import httpx
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


async def demo_http_audio_stream() -> None:
    console.print("[bold cyan]=== WebSocket Audio Streaming Transcription Demo ===[/]")
    url = "http://localhost:8000/audio/stream"
    received_bytes = 0
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", url) as resp:
            console.print(f"[green]Streaming audio: {resp.headers.get('Content-Disposition')}[/]")
            async for chunk in resp.aiter_bytes():
                received_bytes += len(chunk)
    console.print(f"[green]Received {received_bytes} bytes (~{received_bytes / 32000:.1f} seconds of audio)[/]")


async def demo_transcribe_websocket() -> None:
    console.print("[bold cyan]=== WebSocket Real-time Transcription ===[/]")
    uri = "ws://localhost:8000/audio/transcribe"
    table = Table(title="Live Transcription Updates")
    table.add_column("Chunk")
    table.add_column("Partial")
    table.add_column("Final Text")

    async with websockets.connect(uri) as ws:
        for i in range(1, 31):  # Send 30 chunks (~6 seconds)
            chunk = generate_audio_chunk(i, duration_sec=0.2)
            await ws.send(chunk)
            console.print(f"[yellow]→ Sent chunk {i} ({len(chunk)} bytes)[/]")
            response = json.loads(await ws.recv())
            final_text = response.get("text", "-")
            table.add_row(str(response["chunk"]), response["partial"], final_text)
            console.print(table)
            await asyncio.sleep(0.18)  # Slight overlap for realism

    console.print("[green]Real-time transcription demo completed.[/]")


async def demo_echo_websocket() -> None:
    console.print("[bold cyan]=== WebSocket Audio Echo (Latency Test) ===[/]")
    uri = "ws://localhost:8000/audio/echo"
    total_latency = 0.0
    count = 0

    async with websockets.connect(uri) as ws:
        for i in range(20):
            chunk = generate_audio_chunk(i, duration_sec=0.1)
            start = asyncio.get_event_loop().time()
            await ws.send(chunk)
            echoed = await ws.recv()
            end = asyncio.get_event_loop().time()

            assert echoed == chunk, "Echo mismatch!"
            latency_ms = (end - start) * 1000
            total_latency += latency_ms
            count += 1
            console.print(f"[magenta]Echo {i+1:2d}: {latency_ms:6.1f} ms[/]")

    avg_latency = total_latency / count
    console.print(f"[bold green]Average round-trip latency: {avg_latency:.1f} ms[/]")


async def demo_upload_transcribe() -> None:
    console.print("[bold cyan]=== Upload Full Audio File & Get Transcription ===[/]")

    # Generate a small in-memory WAV-like blob (raw PCM + minimal WAV header)
    sample_rate = 16000
    num_chunks = 50
    audio_data = b"".join(generate_audio_chunk(i, duration_sec=0.2) for i in range(num_chunks))

    # Minimal WAV header (44 bytes)
    riff = b"RIFF"
    chunk_size = len(audio_data) + 36
    wave_header = b"WAVEfmt "
    subchunk1_size = 16
    audio_format = 1  # PCM
    num_channels = 1
    byte_rate = sample_rate * num_channels * 2
    block_align = num_channels * 2
    bits_per_sample = 16
    data = b"data"
    subchunk2_size = len(audio_data)

    header = (
        riff + chunk_size.to_bytes(4, "little") +
        wave_header + subchunk1_size.to_bytes(4, "little") +
        audio_format.to_bytes(2, "little") +
        num_channels.to_bytes(2, "little") +
        sample_rate.to_bytes(4, "little") +
        byte_rate.to_bytes(4, "little") +
        block_align.to_bytes(2, "little") +
        bits_per_sample.to_bytes(2, "little") +
        data + subchunk2_size.to_bytes(4, "little")
    )
    wav_bytes = header + audio_data

    async with httpx.AsyncClient() as client:
        files = {"file": ("demo_sine.wav", wav_bytes, "audio/wav")}
        resp = await client.post(
            "http://localhost:8000/audio/upload-transcribe", files=files, timeout=30.0
        )
        result = resp.json()
        console.print("[green]Upload transcription result:[/]")
        console.print(result)


async def main() -> None:
    await demo_http_audio_stream()
    await demo_transcribe_websocket()
    await demo_echo_websocket()
    await demo_upload_transcribe()


if __name__ == "__main__":
    asyncio.run(main())
