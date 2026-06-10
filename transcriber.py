"""Audio capture and live transcription engine using sounddevice + faster-whisper."""

import os
import queue
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import sounddevice as sd

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGS_DIR = os.path.join(_APP_DIR, "logs")


class SessionLogger:
    """Writes terminal and GUI log lines to logs/{epoch}.log."""

    def __init__(self):
        os.makedirs(_LOGS_DIR, exist_ok=True)
        self._epoch = int(time.time())
        self._path = os.path.join(_LOGS_DIR, f"{self._epoch}.log")
        self._lock = threading.Lock()
        self._write("[SESSION START]\n")

    def _write(self, text):
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(text)

    def terminal(self, line):
        """Log a terminal-style line (debug table rows, etc.)."""
        self._write(f"[TERMINAL] {line}\n")

    def gui(self, line):
        """Log a GUI-style line ({HH:MM:SS} messages)."""
        self._write(f"[GUI] {line}\n")

    @property
    def path(self):
        return self._path

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


_EXCLUDED_KEYWORDS = ["speaker", "stereo mix", "loopback", "output", "sound mapper"]


def list_input_devices():
    """Return a list of input audio devices, filtering out speakers/loopback."""
    devices = sd.query_devices()
    seen = set()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            name = dev["name"]
            name_lower = name.lower()
            if any(kw in name_lower for kw in _EXCLUDED_KEYWORDS):
                continue
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
    CHUNK_SECONDS = 30

    def __init__(self, device_index, on_text=None, on_volume=None, on_status=None,
                 logger=None, audio_save_path=None):
        """
        Args:
            device_index: sounddevice input device index.
            on_text: callback(timestamp_str, text) called from transcription thread.
            on_volume: callback(float 0.0-1.0) called from audio callback thread.
            on_status: callback(str) for status messages like 'Loading model...'.
            logger: SessionLogger instance for file logging.
            audio_save_path: if set, save recorded audio to this .mp3 path.
        """
        self.device_index = device_index
        self.on_text = on_text or (lambda ts, txt: None)
        self.on_volume = on_volume or (lambda v: None)
        self.on_status = on_status or (lambda s: None)
        self.logger = logger or SessionLogger()
        self._audio_save_path = audio_save_path
        self._all_audio = [] if audio_save_path else None

        self._model = None
        self._audio_queue = queue.Queue()
        self._recording = threading.Event()
        self._record_thread = None
        self._transcribe_thread = None
        self._status_thread = None
        self._start_time = None
        self._chunk_counter = 0

        # Debug tracking
        self._recording_elapsed = 0.0  # how far the recorder has captured
        self._transcribing_range = ""  # e.g. "00:00:00->00:00:30"

    def start(self):
        """Start recording and transcription."""
        self._start_time = datetime.now()
        self._chunk_counter = 0
        self._recording.set()

        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self._record_thread.start()
        self._transcribe_thread.start()
        self._status_thread.start()

    def stop(self):
        """Stop recording and transcription. Blocks until threads finish."""
        self._recording.clear()
        self._audio_queue.put(None)  # sentinel
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=5)
        if self._transcribe_thread and self._transcribe_thread.is_alive():
            self._transcribe_thread.join(timeout=10)
        self._save_audio()

    def _save_audio(self):
        """Encode accumulated audio to mp3 if saving was enabled."""
        if self._all_audio is None or not self._audio_save_path:
            return
        if not self._all_audio:
            return
        try:
            import subprocess
            import tempfile
            import imageio_ffmpeg

            audio = np.concatenate(self._all_audio)
            # Write raw PCM to a temp .wav, then convert to mp3 via ffmpeg
            wav_path = self._audio_save_path.replace(".mp3", ".tmp.wav")
            import wave
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.SAMPLE_RATE)
                pcm = (audio * 32767).astype(np.int16).tobytes()
                wf.writeframes(pcm)

            # Find ffmpeg
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except AttributeError:
                ffmpeg_exe = os.path.join(
                    imageio_ffmpeg.__path__[0] if hasattr(imageio_ffmpeg, '__path__') else os.path.dirname(imageio_ffmpeg.__file__),
                    "binaries", "ffmpeg.exe"
                )

            subprocess.run(
                [ffmpeg_exe, "-y", "-i", wav_path,
                 "-acodec", "libmp3lame", "-q:a", "2", self._audio_save_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.remove(wav_path)
            self.on_status(f"Audio saved to: {self._audio_save_path}")
        except Exception as e:
            self.on_status(f"Failed to save audio: {e}")

    def _status_loop(self):
        """Print a debug status table to the terminal every 5 seconds."""
        header = f"{'current-time':>16} | {'transcribing-moment':<30} | {'recording-moment':>16} | {'queue':>5}"
        separator = f"{'-'*16}-+-{'-'*30}-+-{'-'*16}-+-{'-'*5}"
        print()
        print(header)
        print(separator)
        self.logger.terminal(header)
        self.logger.terminal(separator)

        while self._recording.is_set():
            now_elapsed = (datetime.now() - self._start_time).total_seconds()
            current = format_timestamp(now_elapsed)
            recording = format_timestamp(self._recording_elapsed)
            transcribing = self._transcribing_range or "not started"
            q_size = self._audio_queue.qsize()

            line = f"{current:>16} | {transcribing:<30} | {recording:>16} | {q_size:>5}"
            print(line)
            self.logger.terminal(line)
            time.sleep(5)

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
            if self._all_audio is not None:
                self._all_audio.append(data)

            if buffer_samples >= chunk_size:
                chunk = np.concatenate(buffer)
                elapsed = (datetime.now() - self._start_time).total_seconds()
                self._recording_elapsed = elapsed
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
            chunk_duration = len(audio_data) / self.SAMPLE_RATE
            chunk_start = elapsed - chunk_duration
            self._transcribing_range = f"{format_timestamp(chunk_start)}->{format_timestamp(elapsed)}"
            try:
                segments, info = self._model.transcribe(
                    audio_data, beam_size=3, vad_filter=True, language="en"
                )
                text_parts = []
                for seg in segments:
                    text_parts.append(seg.text.strip())
                text = " ".join(text_parts).strip()

                if text:
                    ts = format_timestamp(elapsed)
                    self.on_text(ts, text)
                else:
                    msg = f"  [no speech detected in chunk {self._transcribing_range}]"
                    print(msg)
                    self.logger.terminal(msg)
                self._transcribing_range = f"idle (done thru {format_timestamp(elapsed)})"
            except Exception as e:
                self.on_status(f"Transcription error: {e}")
