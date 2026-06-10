"""Tests for the transcription engine."""

import numpy as np
import pytest
from transcriber import format_timestamp, TranscriberEngine


def test_timestamp_format_zero():
    assert format_timestamp(0) == "00:00:00"


def test_timestamp_format_seconds():
    assert format_timestamp(45) == "00:00:45"


def test_timestamp_format_minutes():
    assert format_timestamp(125) == "00:02:05"


def test_timestamp_format_hours():
    assert format_timestamp(3661) == "01:01:01"


def test_timestamp_format_large():
    assert format_timestamp(86399) == "23:59:59"


@pytest.fixture(scope="module")
def whisper_model():
    """Load whisper model once for all tests in this module."""
    from faster_whisper import WhisperModel

    return WhisperModel("tiny", device="cpu", compute_type="int8")


def test_whisper_model_loads(whisper_model):
    assert whisper_model is not None


def test_transcribe_silent_audio(whisper_model):
    silence = np.zeros(16000 * 3, dtype=np.float32)
    segments, info = whisper_model.transcribe(silence, beam_size=1, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    # Silence should produce little to no text
    assert len(text) < 50


def test_transcribe_accepts_numpy_array(whisper_model):
    audio = np.random.randn(16000 * 2).astype(np.float32) * 0.01
    segments, info = whisper_model.transcribe(audio, beam_size=1)
    # Just ensure it doesn't raise
    for seg in segments:
        _ = seg.text


def test_engine_creation():
    devices = TranscriberEngine.__init__.__code__.co_varnames
    engine = TranscriberEngine(device_index=0)
    assert engine.device_index == 0
    assert engine._model is None
    assert engine.is_recording is False if hasattr(engine, "is_recording") else True


def test_engine_has_required_methods():
    engine = TranscriberEngine(device_index=0)
    assert callable(getattr(engine, "start", None))
    assert callable(getattr(engine, "stop", None))
