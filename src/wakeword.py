"""wakeword.py
Wake-word detector based on the **openwakeword** demo script by
David Scripka (Apache‑2.0, 2022) but wrapped in an easy‑to‑reuse
`WakewordDetector` class.

Differences to the reference script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Runs in a background **thread** and not in the main loop.
* Offers a callback `set_callback(fn)` rather than printing.
* Adds duplicate‑trigger suppression, adjustable threshold, and
  programmatic *add_wake_word()* at runtime.
* Supports both a single `.tflite` model file *or* the bundled model
  pack when `model_path` is empty.
* Uses shared audio handler for microphone capture.

Quick CLI test
--------------
```bash
python wakeword.py --chunk_size 1280 --model_path ./hey_bot.tflite \
                   --inference_framework tflite
```
Speak your wake‑word; each detection prints a line.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pyaudio
from openwakeword.model import Model

try:
    from audio import InOutHandler, AudioUser
except ImportError:
    # For standalone testing
    InOutHandler = None
    AudioUser = None

__all__ = ["WakewordDetector"]

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WakewordDetector:
    """Background wake‑word listener built on **openwakeword** (TFLite/ONNX)."""

    _RATE = 16_000  # Hz
    _CHANNELS = 1
    _FORMAT = pyaudio.paInt16
    _DEFAULT_CHUNK = 1_280  # ≈ 80 ms @ 16 kHz
    _SUPPRESSION_SEC = 1.0

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        chunk_size: int = _DEFAULT_CHUNK,
        inference_framework: str = "tflite",
        sensitivity: float = 0.5,
        audio_handler: Optional[InOutHandler] = None,
    ) -> None:
        """Parameters
        ----------
        model_path
            Path to a **single** `.tflite`/`.onnx` model *or* a folder
            containing several models.  If *None* or empty, all built‑in
            openwakeword models are loaded.
        chunk_size
            Number of samples to read per audio chunk.
        inference_framework
            Either ``"tflite"`` (CPU/GPU) or ``"onnx"`` (CPU).
        sensitivity
            Probability threshold in [0, 1]; above → trigger.
        audio_handler
            Optional shared audio handler. If None, a private PyAudio instance will be used.
        """
        self._chunk = int(chunk_size)
        self._threshold = float(sensitivity)

        # ~~~~~ load models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if model_path is None or str(model_path) == "":
            self._model = Model(inference_framework=inference_framework)
        else:
            mp = Path(model_path)
            if mp.is_file():
                paths = [str(mp)]
            else:
                paths = [str(p) for p in mp.glob("*.tflite")]
            self._model = Model(wakeword_models=paths, inference_framework=inference_framework)

        self._wake_words: List[str] = list(self._model.models.keys())
        self._last_detect: dict[str, float] = {w: 0.0 for w in self._wake_words}

        # Callback function (word, timestamp) -> None
        self._callback: Optional[Callable[[str, float], None]] = None

        # Audio handling ---------------------------------------------------
        self._audio_handler = audio_handler
        self._pa = None if audio_handler else pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._using_shared_audio = audio_handler is not None

        # Threading -------------------------------------------------------
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1.0)
        
        # Only close private resources, not shared ones
        if not self._using_shared_audio:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
            if self._pa:
                self._pa.terminate()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ..................................................................
    # Configuration
    # ..................................................................

    def set_sensitivity(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    def add_wake_word(self, word: str, model_path: str | Path) -> None:
        self._model.add_wakeword(str(model_path))
        self._wake_words.append(word)
        self._last_detect[word] = 0.0

    # ..................................................................
    # Callback registration
    # ..................................................................

    def set_callback(self, fn: Callable[[str, float], None]) -> None:
        self._callback = fn

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_stream(self) -> None:
        if self._using_shared_audio:
            return  # Using shared audio handler, no stream to open
            
        if self._stream is None:
            self._stream = self._pa.open(
                format=self._FORMAT,
                channels=self._CHANNELS,
                rate=self._RATE,
                input=True,
                frames_per_buffer=self._chunk,
            )

    def _loop(self) -> None:
        self._open_stream()
        
        if self._using_shared_audio:
            # Use the shared audio handler
            for raw in self._audio_handler.get_wakeword_chunks(self._chunk):
                if not self._running.is_set():
                    break
                self._process_chunk(raw)
        else:
            # Use private audio stream
            assert self._stream is not None
            while self._running.is_set():
                raw = self._stream.read(self._chunk, exception_on_overflow=False)
                self._process_chunk(raw)

    # .................................................................

    def _process_chunk(self, pcm: bytes) -> None:
        audio = np.frombuffer(pcm, dtype=np.int16)
        probs = self._model.predict(audio)
        ts = time.time()
        for word, prob in probs.items():
            if prob < self._threshold:
                continue
            if ts - self._last_detect[word] < self._SUPPRESSION_SEC:
                continue
            self._last_detect[word] = ts
            if self._callback:
                self._callback(word, ts)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Wake‑word '{word}' detected → score={prob:.2f}")


# ---------------------------------------------------------------------------
# CLI helper for quick testing (mirrors reference script behaviour)
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Wake-word quick‑test")
    parser.add_argument("--chunk_size", type=int, default=1280, help="Samples per inference call")
    parser.add_argument("--model_path", type=str, default="models/robo-cup.tflite", help="Path to a specific model or folder of models")
    parser.add_argument("--inference_framework", type=str, default="tflite", choices=["tflite", "onnx"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Trigger threshold 0‑1")
    parser.add_argument("--use_handler", action="store_true", help="Use shared audio handler")
    args = parser.parse_args()

    # Create audio handler if requested
    audio_handler = None
    if args.use_handler and InOutHandler is not None:
        audio_handler = InOutHandler()
        audio_handler.start()
    
    det = WakewordDetector(
        model_path=args.model_path or None,
        chunk_size=args.chunk_size,
        inference_framework=args.inference_framework,
        sensitivity=args.threshold,
        audio_handler=audio_handler,
    )

    det.start()
    try:
        det.join()
    except KeyboardInterrupt:
        det.stop()
        if audio_handler:
            audio_handler.stop()


if __name__ == "__main__":
    _cli()
