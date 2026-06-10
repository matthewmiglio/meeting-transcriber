"""Tests for file operations (transcript saving and config persistence)."""

import json
import os
import pytest


def test_transcript_file_creation(tmp_path):
    filepath = tmp_path / "transcript.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("{00:00:05} Hello world\n")
    assert filepath.exists()
    assert filepath.read_text(encoding="utf-8") == "{00:00:05} Hello world\n"


def test_transcript_append_mode(tmp_path):
    filepath = tmp_path / "transcript.txt"
    lines = [
        "{00:00:05} First line\n",
        "{00:00:10} Second line\n",
        "{00:00:15} Third line\n",
    ]
    for line in lines:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line)
    content = filepath.read_text(encoding="utf-8")
    assert content.count("\n") == 3
    for line in lines:
        assert line in content


def test_output_folder_creation(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    os.makedirs(nested, exist_ok=True)
    assert nested.is_dir()


def test_transcript_utf8_encoding(tmp_path):
    filepath = tmp_path / "transcript.txt"
    text = "{00:01:00} Caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u2014 \u201chello\u201d\n"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    readback = filepath.read_text(encoding="utf-8")
    assert readback == text


def test_transcript_filename_format():
    from datetime import datetime

    ts = datetime(2026, 6, 10, 14, 30, 0).strftime("%Y%m%d_%H%M%S")
    filename = f"transcript_{ts}.txt"
    assert filename == "transcript_20260610_143000.txt"


def test_config_save_and_load(tmp_path, monkeypatch):
    config_path = str(tmp_path / "config.json")
    import main
    monkeypatch.setattr(main, "CONFIG_PATH", config_path)

    main.save_config({"mic_name": "Test Microphone"})
    loaded = main.load_config()
    assert loaded["mic_name"] == "Test Microphone"


def test_config_load_missing_file(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "CONFIG_PATH", str(tmp_path / "nonexistent.json"))
    assert main.load_config() == {}


def test_config_merge(tmp_path, monkeypatch):
    config_path = str(tmp_path / "config.json")
    import main
    monkeypatch.setattr(main, "CONFIG_PATH", config_path)

    main.save_config({"mic_name": "Mic A"})
    main.save_config({"output_folder": "/some/path"})
    loaded = main.load_config()
    assert loaded["mic_name"] == "Mic A"
    assert loaded["output_folder"] == "/some/path"
