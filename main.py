"""Meeting Transcriber - Live audio transcription with a Tkinter GUI."""

import json
import os
import tkinter as tk
from tkinter import ttk, filedialog
from datetime import datetime

from transcriber import list_input_devices, TranscriberEngine

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config():
    """Load config from disk, returning empty dict on any failure."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_config(data):
    """Merge data into existing config and save."""
    config = load_config()
    config.update(data)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


class MeetingTranscriberApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MEETING TRANSCRIBER")
        self.root.geometry("750x520")
        self.root.minsize(600, 400)

        self.engine = None
        self.output_folder = None
        self.output_file = None
        self.is_recording = False
        self._current_volume = 0.0
        self._devices = list_input_devices()

        self._build_ui()
        self._poll_volume()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # Title
        tk.Label(
            self.root, text="MEETING TRANSCRIBER", font=("Arial", 16, "bold")
        ).pack(anchor="w", **pad)

        # Mic selection row
        mic_frame = tk.Frame(self.root)
        mic_frame.pack(fill="x", **pad)
        tk.Label(mic_frame, text="\U0001f3a4", font=("Segoe UI Emoji", 14)).pack(
            side="left"
        )
        self.mic_combo = ttk.Combobox(
            mic_frame,
            values=[d["name"] for d in self._devices],
            state="readonly",
            width=50,
        )
        self.mic_combo.pack(side="left", padx=(5, 0), fill="x", expand=True)
        self.mic_combo.bind("<<ComboboxSelected>>", self._on_mic_selected)
        if self._devices:
            saved_mic = load_config().get("mic_name")
            restored = False
            if saved_mic:
                for i, d in enumerate(self._devices):
                    if d["name"] == saved_mic:
                        self.mic_combo.current(i)
                        restored = True
                        break
            if not restored:
                self.mic_combo.current(0)

        # Volume bar
        vol_frame = tk.Frame(self.root)
        vol_frame.pack(fill="x", **pad)
        tk.Label(vol_frame, text="Vol:").pack(side="left")
        self.vol_canvas = tk.Canvas(vol_frame, height=18, bg="#e0e0e0", bd=1, relief="sunken")
        self.vol_canvas.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.vol_rect = self.vol_canvas.create_rectangle(0, 0, 0, 18, fill="#4caf50", width=0)

        # Output folder row
        folder_frame = tk.Frame(self.root)
        folder_frame.pack(fill="x", **pad)
        tk.Button(folder_frame, text="Set Output Folder", command=self._browse_folder).pack(
            side="left"
        )
        self.folder_label = tk.Label(folder_frame, text="(no folder selected)", fg="gray")
        self.folder_label.pack(side="left", padx=(10, 0))

        # Start / Stop button
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)
        self.start_btn = tk.Button(
            btn_frame,
            text="Start Recording",
            command=self._toggle_recording,
            bg="#4caf50",
            fg="white",
            font=("Arial", 11, "bold"),
            width=18,
        )
        self.start_btn.pack(side="left")

        # Transcript area
        text_frame = tk.Frame(self.root)
        text_frame.pack(fill="both", expand=True, **pad)
        self.scrollbar = tk.Scrollbar(text_frame)
        self.scrollbar.pack(side="right", fill="y")
        self.transcript_text = tk.Text(
            text_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
            yscrollcommand=self.scrollbar.set,
        )
        self.transcript_text.pack(fill="both", expand=True)
        self.scrollbar.config(command=self.transcript_text.yview)

    def _on_mic_selected(self, event=None):
        idx = self.mic_combo.current()
        if idx >= 0:
            save_config({"mic_name": self._devices[idx]["name"]})

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.folder_label.config(text=folder, fg="black")

    def _toggle_recording(self):
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if not self.output_folder:
            self._append_status("Please select an output folder first.")
            return
        if not self._devices:
            self._append_status("No input devices found.")
            return

        # Determine selected device
        idx = self.mic_combo.current()
        if idx < 0:
            self._append_status("Please select a microphone.")
            return
        device_index = self._devices[idx]["index"]

        # Prepare output file
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_folder, f"transcript_{ts}.txt")

        # Create engine
        self.engine = TranscriberEngine(
            device_index=device_index,
            on_text=self._on_text,
            on_volume=self._on_volume,
            on_status=self._on_status,
        )
        self.engine.start()
        self.is_recording = True
        self.start_btn.config(text="Stop Recording", bg="#f44336")
        self.mic_combo.config(state="disabled")
        self._append_status("Recording started...")

    def _stop_recording(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.is_recording = False
        self._current_volume = 0.0
        self.start_btn.config(text="Start Recording", bg="#4caf50")
        self.mic_combo.config(state="readonly")
        self._append_status(f"Recording stopped. Transcript saved to: {self.output_file}")

    def _on_text(self, timestamp, text):
        """Called from transcription thread."""
        self.root.after(0, self._append_transcript, timestamp, text)

    def _on_volume(self, level):
        """Called from audio callback thread."""
        self._current_volume = level

    def _on_status(self, message):
        """Called from background threads."""
        self.root.after(0, self._append_status, message)

    def _append_transcript(self, timestamp, text):
        line = f"{{{timestamp}}} {text}"
        self.transcript_text.config(state="normal")
        self.transcript_text.insert("end", line + "\n")
        self.transcript_text.see("end")
        self.transcript_text.config(state="disabled")

        # Save to file immediately
        if self.output_file:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _append_status(self, message):
        self.transcript_text.config(state="normal")
        self.transcript_text.insert("end", f"[{message}]\n")
        self.transcript_text.see("end")
        self.transcript_text.config(state="disabled")

    def _poll_volume(self):
        """Update the volume bar on the main thread."""
        w = self.vol_canvas.winfo_width()
        bar_w = int(self._current_volume * w)
        self.vol_canvas.coords(self.vol_rect, 0, 0, bar_w, 18)

        # Color gradient: green -> yellow -> red
        if self._current_volume < 0.4:
            color = "#4caf50"
        elif self._current_volume < 0.7:
            color = "#ffeb3b"
        else:
            color = "#f44336"
        self.vol_canvas.itemconfig(self.vol_rect, fill=color)

        self.root.after(50, self._poll_volume)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MeetingTranscriberApp()
    app.run()
