"""wakeword.py
Implementation of WakewordDetector using the openwakeword package.
Requirements:
    pip install openwakeword sounddevice numpy

Notes
-----
* Audio is captured at 16 kHz, 16‑bit mono.  Each chunk is 512 samples
  (≈ 32 ms) which matches openwakeword’s default frame size.
* If you pass an ``audio_source`` argument it must be an **async** or
  synchronous iterator that yields raw ``bytes`` chunks in the same
  format; otherwise a default **sounddevice** stream is started.
* The detector runs in a background thread so the main thread stays
  responsive.

Example
-------
from wakeword import WakewordDetector

det = WakewordDetector(
    model_path="models/",
    wake_words=["hey_bot"],
)

det.set_callback(lambda word, ts: print("Detected:", word))
det.start()
try:
    det.join()
except KeyboardInterrupt:
    det.stop()
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

__all__ = ["WakewordDetector"]

# ---------------------------------------------------------------------------
# Helper: default live microphone source
# ---------------------------------------------------------------------------

def _microphone_source(
    samplerate: int = 16_000,
    chunk_size: int = 512,
    dtype: str = "int16",
) -> Iterator[bytes]:
    """Yield ``chunk_size`` *frames* of raw PCM from the default mic."""

    q: queue.Queue[bytes] = queue.Queue(maxsize=10)

    def _callback(indata: np.ndarray, _frames: int, _time, _status):  # noqa: ANN001
        q.put_nowait(indata.copy().tobytes())

    with sd.InputStream(
        samplerate=samplerate,
        blocksize=chunk_size,
        dtype=dtype,
        channels=1,
        callback=_callback,
    ):
        while True:
            yield q.get()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WakewordDetector:
    """High‑level wake‑word detection wrapper around **openwakeword**."""

    _DEFAULT_THRESHOLD = 0.5  # Prob. above which we fire the callback
    _SUPPRESSION_SEC = 1.0    # Ignore duplicate triggers within X sec

    def __init__(
        self,
        model_path: str | Path,
        wake_words: List[str],
        sensitivity: float = 0.5,
        audio_source: Optional[Iterator[bytes]] = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._wake_words = wake_words
        self._threshold = float(sensitivity)
        self._callback: Optional[Callable[[str, float], None]] = None

        # Build model list of file paths
        model_paths = [str(self._model_path / f"{w}.tflite") for w in wake_words]
        self._model = Model(wakeword_models=model_paths, inference_framework="tflite")

        self._audio_source = audio_source or _microphone_source()

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        # Deduplication state
        self._last_detect_time: dict[str, float] = {w: 0.0 for w in wake_words}

    # ------------------------------------------------------------------
    # Life‑cycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background audio‑processing thread."""
        if self._thread and self._thread.is_alive():
            return  # already running
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the background thread to finish and wait for it."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1.0)

    def join(self, timeout: float | None = None) -> None:
        """Block the caller until the detector thread finishes."""
        if self._thread:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_sensitivity(self, value: float) -> None:
        """Set probability threshold between 0 and 1 (higher = stricter)."""
        self._threshold = max(0.0, min(1.0, value))

    def add_wake_word(self, word: str, model_path: str | Path) -> None:
        """Load an *additional* model at runtime.

        ``openwakeword`` supports adding more models with
        :py:meth:`Model.add_wakeword` (>= v0.6).  We also extend the
        deduplication dict.
        """
        self._model.add_wakeword(str(model_path))
        self._wake_words.append(word)
        self._last_detect_time[word] = 0.0

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def set_callback(self, fn: Callable[[str, float], None]) -> None:
        """Register a function called on detection: ``fn(word, time.time())``."""
        self._callback = fn

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Audio‑processing loop running in a **separate thread**."""
        for chunk in self._audio_source:
            if not self._running.is_set():
                break
            self._process_chunk(chunk)

    # .................................................................

    def _process_chunk(self, pcm: bytes) -> None:
        """Convert PCM → float32 and feed to model, then fire callback."""
        # Convert int16 -> float32 in [-1, 1]
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        probs = self._model.predict(audio)
        # ``predict`` returns a list/np.ndarray with probability per model
        for idx, prob in enumerate(probs):
            if prob < self._threshold:
                continue
            word = self._wake_words[idx]
            now = time.time()
            if now - self._last_detect_time[word] < self._SUPPRESSION_SEC:
                continue  # within suppression window
            self._last_detect_time[word] = now
            if self._callback:
                # Dispatch from **this** background thread; if you need to
                # hop into asyncio use `asyncio.get_event_loop().call_soon_threadsafe`.
                self._callback(word, now)


if __name__ == "__main__":
    # Example usage
    detector = WakewordDetector(
        model_path="models/",
        wake_words=["hey_jarvis"],
    )

    def callback(word: str, timestamp: float) -> None:
        print(f"Detected: {word} at {timestamp}")

    detector.set_callback(callback)
    detector.start()
    try:
        detector.join()
    except KeyboardInterrupt:
        detector.stop()
        print("Detector stopped.")