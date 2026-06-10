"""Tests for audio device interaction."""

import pytest
import sounddevice as sd
from transcriber import list_input_devices


def test_list_input_devices_returns_list():
    devices = list_input_devices()
    assert isinstance(devices, list)


def test_list_input_devices_have_names():
    devices = list_input_devices()
    for dev in devices:
        assert "name" in dev
        assert isinstance(dev["name"], str)
        assert len(dev["name"]) > 0


def test_list_input_devices_have_index():
    devices = list_input_devices()
    for dev in devices:
        assert "index" in dev
        assert isinstance(dev["index"], int)


def test_list_input_devices_no_duplicate_names():
    devices = list_input_devices()
    names = [d["name"] for d in devices]
    assert len(names) == len(set(names))


def test_sounddevice_query_devices_runs():
    result = sd.query_devices()
    assert result is not None
