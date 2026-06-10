"""Audio capture and live transcription engine using sounddevice + faster-whisper."""

import os
import queue
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import sounddevice as sd

# Ensure ffmpeg is on PATH for faster-whisper
def _setup_ffmpeg():
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_dir = os.path.dirname(get_ffmpeg_exe())
    except (ImportError, AttributeError):
        # Fallback: find ffmpeg in imageio_ffmpeg's binaries folder
        import imageio_ffmpeg
        pkg_dir = os.path.dirname(imageio_ffmpeg.__path__[0]) if hasattr(imageio_ffmpeg, '__path__') else os.path.dirname(imageio_ffmpeg.__file__)
        binaries = os.path.join(os.path.dirname(imageio_ffmpeg.__path__[0] if hasattr(imageio_ffmpeg, '__path__') else imageio_ffmpeg.__file__), "imageio_ffmpeg", "binaries")
        if not os.path.isdir(binaries) and hasattr(imageio_ffmpeg, '__path__'):
            binaries = os.path.join(imageio_ffmpeg.__path__[0], "binaries")
        ffmpeg_dir = binaries if os.path.isdir(binaries) else ""
    if ffmpeg_dir:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

_setup_ffmpeg()


def list_input_devices():
    """Return a list of input audio devices as dicts with 'index' and 'name'."""
    devices = sd.query_devices()
    seen = set()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            name = dev["name"]
            if name not in seen:
                seen.add(name)
                result.append({"index": i, "name": name})
    return result


def format_timestamp(elapsed_seconds):
    """Format elapsed seconds as HH:MM:SS."""
    td = timedelta(seconds=int(elapsed_seconds))
    total_secs = int(td.total_seconds())
    h = total_secs // 3600
    m = (total_secs % 3600) // 60
    s = total_secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class TranscriberEngine:
    """Manages audio recording and transcription in background threads."""

    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 5

    def __init__(self, device_index, on_text=None, on_volume=None, on_status=None):
        """
        Args:
            device_index: sounddevice input device index.
            on_text: callback(timestamp_str, text) called from transcription thread.
            on_volume: callback(float 0.0-1.0) called from audio callback thread.
            on_status: callback(str) for status messages like 'Loading model...'.
        """
        self.device_index = device_index
        self.on_text = on_text or (lambda ts, txt: None)
        self.on_volume = on_volume or (lambda v: None)
        self.on_status = on_status or (lambda s: None)

        self._model = None
        self._audio_queue = queue.Queue()
        self._recording = threading.Event()
        self._record_thread = None
        self._transcribe_thread = None
        self._start_time = None
        self._chunk_counter = 0

    def start(self):
        """Start recording and transcription."""
        self._start_time = datetime.now()
        self._chunk_counter = 0
        self._recording.set()

        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._record_thread.start()
        self._transcribe_thread.start()

    def stop(self):
        """Stop recording and transcription. Blocks until threads finish."""
        self._recording.clear()
        self._audio_queue.put(None)  # sentinel
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=5)
        if self._transcribe_thread and self._transcribe_thread.is_alive():
            self._transcribe_thread.join(timeout=10)

    def _record_loop(self):
        """Capture audio from the mic and enqueue chunks for transcription."""
        buffer = []
        buffer_samples = 0
        chunk_size = self.SAMPLE_RATE * self.CHUNK_SECONDS

        def audio_callback(indata, frames, time_info, status):
            nonlocal buffer, buffer_samples
            data = indata[:, 0].copy()

            # Compute RMS for volume meter
            rms = float(np.sqrt(np.mean(data ** 2)))
            scaled = min(rms * 15, 1.0)  # scale up for visual
            self.on_volume(scaled)

            buffer.append(data)
            buffer_samples += len(data)

            if buffer_samples >= chunk_size:
                chunk = np.concatenate(buffer)
                elapsed = (datetime.now() - self._start_time).total_seconds()
                self._audio_queue.put((elapsed, chunk))
                buffer = []
                buffer_samples = 0

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=self.device_index,
                blocksize=1600,
                callback=audio_callback,
            ):
                while self._recording.is_set():
                    time.sleep(0.1)

                # Flush remaining buffer
                if buffer:
                    chunk = np.concatenate(buffer)
                    elapsed = (datetime.now() - self._start_time).total_seconds()
                    self._audio_queue.put((elapsed, chunk))
        except Exception as e:
            self.on_status(f"Recording error: {e}")

    def _transcribe_loop(self):
        """Pull audio chunks from queue and transcribe them."""
        # Load model on first use
        if self._model is None:
            self.on_status("Loading whisper model (first time may take a moment)...")
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    "base", device="cpu", compute_type="int8"
                )
                self.on_status("Model loaded. Listening...")
            except Exception as e:
                self.on_status(f"Failed to load model: {e}")
                return

        while True:
            item = self._audio_queue.get()
            if item is None:
                break

            elapsed, audio_data = item
            try:
                segments, info = self._model.transcribe(
                    audio_data, beam_size=3, vad_filter=True
                )
                text_parts = []
                for seg in segments:
                    text_parts.append(seg.text.strip())
                text = " ".join(text_parts).strip()

                if text:
                    ts = format_timestamp(elapsed)
                    self.on_text(ts, text)
            except Exception as e:
                self.on_status(f"Transcription error: {e}")
