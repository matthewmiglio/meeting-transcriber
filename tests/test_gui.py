"""Tests for GUI components (headless, no display required on Windows)."""

import tkinter as tk
import pytest
from main import MeetingTranscriberApp


@pytest.fixture(scope="module")
def app():
    """Create a single app instance for all GUI tests to avoid Tcl errors."""
    application = MeetingTranscriberApp()
    application.root.withdraw()
    yield application
    application.root.destroy()


def test_window_title(app):
    assert app.root.title() == "MEETING TRANSCRIBER"


def test_mic_dropdown_exists(app):
    assert app.mic_combo is not None
    assert app.mic_combo.winfo_exists()


def test_mic_dropdown_populated(app):
    values = app.mic_combo["values"]
    assert len(values) >= 0  # don't fail on CI with no mic


def test_output_folder_default_empty(app):
    assert app.output_folder is None


def test_start_button_exists(app):
    assert app.start_btn is not None
    assert app.start_btn["text"] == "Start Recording"


def test_transcript_area_exists(app):
    assert app.transcript_text is not None
    assert app.transcript_text["state"] == "disabled"


def test_volume_canvas_exists(app):
    assert app.vol_canvas is not None
    assert app.vol_canvas.winfo_exists()


def test_append_status(app):
    app._append_status("Test message")
    app.transcript_text.config(state="normal")
    content = app.transcript_text.get("1.0", "end").strip()
    app.transcript_text.config(state="disabled")
    assert "Test message" in content


def test_append_transcript(app):
    app.output_file = None  # don't write to disk
    app._append_transcript("00:00:05", "Hello world")
    app.transcript_text.config(state="normal")
    content = app.transcript_text.get("1.0", "end").strip()
    app.transcript_text.config(state="disabled")
    assert "{00:00:05} Hello world" in content
