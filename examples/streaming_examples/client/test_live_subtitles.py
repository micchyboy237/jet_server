import pytest
import time
from rich.console import Console

from live_subtitles_client import LiveSubtitles  # adjusted import for local run

@pytest.fixture
def console():
    return Console(record=True, width=120)

@pytest.fixture
def subtitles():
    return LiveSubtitles()

def test_initial_state(subtitles, console):
    # Given a fresh LiveSubtitles instance
    # When getting the panel
    panel = subtitles.get_panel()
    # Then it should show waiting message and no partial
    console.print(panel)
    rendered = console.export_text()
    assert "Waiting for speech..." in rendered
    assert subtitles.current_partial == ""
    assert len(subtitles.entries) == 0

def test_update_partial(subtitles, console):
    # Given a LiveSubtitles instance
    subtitles.recording_start = time.time() - 10.0  # simulate offset
    result_partial = {"partial": "Hello world"}
    # When updating with partial
    subtitles.update(result_partial)
    # Then partial is set, no entry added
    assert subtitles.current_partial == "Hello world"
    assert len(subtitles.entries) == 0
    console.print(subtitles.get_panel())
    rendered = console.export_text()
    assert "Partial: Hello world" in rendered

def test_update_final_translation(subtitles, console):
    # Given a LiveSubtitles instance with recording start
    start_time = time.time()
    subtitles.recording_start = start_time
    result_final = {
        "english": "Hello, this is a test.",
        "final": True,
        "start": 1.5,
        "end": 4.2,
    }
    # When updating with final translated segment
    subtitles.update(result_final)
    # Then entry is added with correct timings and texts, partial cleared
    assert len(subtitles.entries) == 1
    entry = subtitles.entries[0]
    # Relative offset is near zero during test; allow small drift
    expected_start = 1.5
    expected_end = 4.2
    assert entry.index == 1
    assert abs(entry.start - expected_start) < 0.5
    assert abs(entry.end - expected_end) < 0.5
    assert entry.japanese == ""  # original not provided in this fix
    assert entry.english == "Hello, this is a test."
    assert subtitles.current_partial == ""
    console.print(subtitles.get_panel())
    rendered = console.export_text()
    assert "Hello, this is a test." in rendered

def test_srt_generation(subtitles):
    # Given subtitles with multiple final entries
    subtitles.recording_start = time.time()
    subtitles.update({"english": "First sentence.", "final": True, "start": 0.0, "end": 2.0})
    subtitles.update({"english": "Second sentence.", "final": True, "start": 2.0, "end": 5.0})

    # When _write_srt is called (triggered internally)
    content = "".join(e.to_srt() for e in subtitles.entries)
    lines = content.splitlines()

    # Then SRT content matches standard format exactly
    expected = [
        "1",
        "00:00:00,000 --> 00:00:02,000",
        "",  # japanese (empty in this case)
        "First sentence.",
        "",  # separation blank line (standard SRT has two \n between entries)
        "2",
        "00:00:02,000 --> 00:00:05,000",
        "",
        "Second sentence.",
        "",  # blank line after last entry
    ]
    assert lines == expected

    assert "First sentence." in content
    assert "Second sentence." in content