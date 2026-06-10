# meeting-transcriber

Real-time speech-to-text for meetings. Select your mic, hit record, and get a live transcript — all processed locally with faster-whisper.

## Tech Stack

```
Mic Input -> sounddevice -> numpy chunks -> faster-whisper (CPU) -> Tkinter GUI
                                                |
                                                +-> transcript_{timestamp}.txt
```

## Setup

```
poetry install
```

## Usage

```
poetry run python main.py
```

1. Pick a microphone from the dropdown
2. Click **Set Output Folder** to choose where transcripts are saved
3. Click **Start Recording** — text appears live as you speak
4. Click **Stop Recording** when done

## Tests

```
poetry run python -m pytest tests/ -v
```
