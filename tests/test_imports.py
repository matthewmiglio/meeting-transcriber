"""Tests that all required modules are importable."""

import pytest


def test_import_sounddevice():
    import sounddevice  # noqa: F401


def test_import_faster_whisper():
    import faster_whisper  # noqa: F401


def test_import_numpy():
    import numpy  # noqa: F401


def test_import_tkinter():
    import tkinter  # noqa: F401


def test_import_imageio_ffmpeg():
    import imageio_ffmpeg  # noqa: F401


def test_import_transcriber_module():
    from transcriber import TranscriberEngine, list_input_devices  # noqa: F401


def test_import_main_module():
    from main import MeetingTranscriberApp  # noqa: F401
