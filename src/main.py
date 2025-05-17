"""main.py
Entry‑point that wires **WakewordDetector**, **STT**, **Brain** and **TTS**
together into the canonical loop:

    wait → listen → think → speak → wait

Requires the three sibling modules generated earlier:

* ``wakeword.py`` – based on openwakeword + PyAudio
* ``stt.py``       – realtime Whisper + webrtcvad
* ``tts.py``       – any TTS backend (not shown here; plug in yours)
* ``brain.py``     – your application logic (stubbed below)

Run ``python main.py`` and say your wake‑word (e.g. *"hey_bot"*).
"""
from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path
from typing import Optional

from wakeword import WakewordDetector
from stt import STT

try:
    from tts import TTS  # your own module
except ImportError:  # quick stub for testing without TTS engine
    class TTS:  # noqa: D401, WPS601
        def speak(self, text: str, block: bool = True) -> None:  # noqa: D401
            print(f"[Robot ⇢ user] {text}")

try:
    from brain import Brain
except ImportError:  # minimal stub so the file runs out‑of‑the‑box
    class Brain:  # noqa: D401, WPS601
        async def execute(self, text: str) -> str:  # noqa: D401
            return f"You said: {text}"

# ---------------------------------------------------------------------------
# Configuration (override via CLI)
# ---------------------------------------------------------------------------

class _Config:  # noqa: WPS601
    # Wake‑word -------------------------------------------------------------
    wake_model_path: Optional[str | Path] = None  # folder or .tflite; None → built‑ins
    inference_framework: str = "tflite"           # or "onnx"
    threshold: float = 0.5                        # 0‑1 probability to trigger
    chunk_size: int = 1_280                      # samples per inference

    # STT -------------------------------------------------------------------
    stt_model: str | Path = "base.en"            # faster‑whisper model name/path
    stt_language: str = "en"
    listen_timeout: float = 10.0                 # hard timeout per utterance (sec)

    # TTS -------------------------------------------------------------------
    tts_voice: str | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RobotLoop:  # noqa: WPS601
    """Glue class that owns the other modules and their lifecycles."""

    def __init__(self, cfg: _Config):
        self.cfg = cfg

        # Will be filled in once the event loop is running
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 1) Wake-word detector (runs in its own thread) -------------------
        self.detector = WakewordDetector(
            model_path=cfg.wake_model_path,
            chunk_size=cfg.chunk_size,
            inference_framework=cfg.inference_framework,
            sensitivity=cfg.threshold,
        )

        # 2) Speech-to-text -------------------------------------------------
        self.stt = STT(
            model=cfg.stt_model,
            language=cfg.stt_language,
        )

        # 3) Brain + voice --------------------------------------------------
        self.brain = Brain()
        self.tts = TTS() if cfg.tts_voice is None else TTS(voice=cfg.tts_voice)

        # Signal from wake-word thread to asyncio loop ---------------------
        self._wake_event = asyncio.Event()
        self._listening = False  # prevent double-firing

    # ------------------------------------------------------------------
    async def run(self) -> None:  # noqa: WPS231
        """Main coroutine; never returns unless cancelled."""
        # Capture the running loop reference for thread callbacks --------
        self._loop = asyncio.get_running_loop()

        # Register callback *before* starting detector -------------------
        self.detector.set_callback(self._wake_callback)
        self.detector.start()

        while True:  # outer loop – wait for wake word -----------------
            await self._wake_event.wait()
            self._wake_event.clear()

            # Guard: ignore if already in the middle of listening -------
            if self._listening:
                continue
            self._listening = True

            try:
                transcript = await self.stt.listen(timeout=self.cfg.listen_timeout)
            except asyncio.TimeoutError:
                print("[Robot] Listening timed-out; going back to idle.")
                self._listening = False
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[Robot] STT error: {exc}")
                self._listening = False
                continue

            if not transcript:
                print("[Robot] No speech detected – idle.")
                self._listening = False
                continue

            print(f"[User ⇢ robot] {transcript}")

            # Brain may be sync or async --------------------------------
            reply = await self._call_brain(transcript)
            if reply:
                self.tts.speak(reply)

            self._listening = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _wake_callback(self, word: str, ts: float) -> None:  # noqa: D401
        """Runs in *detector thread* – marshal to asyncio loop."""
        if self._loop is None:
            return  # should not happen but guards against race
        # Notify main coroutine thread-safely
        self._loop.call_soon_threadsafe(self._wake_event.set)
        print(f"[Detector] Wake-word '{word}' @{ts:.3f}")

    async def _call_brain(self, text: str) -> str:  # noqa: D401
        if asyncio.iscoroutinefunction(self.brain.execute):
            return await self.brain.execute(text)  # type: ignore[arg-type]
        return await asyncio.to_thread(self.brain.execute, text)

    # ------------------------------------------------------------------
    async def shutdown(self) -> None:  # noqa: D401
        print("Shutting down …")
        self.detector.stop()
        self.stt.stop()


# ---------------------------------------------------------------------------
# CLI / bootstrap
# ---------------------------------------------------------------------------

def _parse_cli() -> _Config:
    p = argparse.ArgumentParser(description="Simple voice assistant loop")
    p.add_argument("--wake_model_path", default="models/robo-cup.tflite", help="Folder or single .tflite model")
    p.add_argument("--framework", default="tflite", choices=["tflite", "onnx"], help="Inference framework")
    p.add_argument("--threshold", type=float, default=0.5, help="Detection threshold 0-1")
    p.add_argument("--stt_model", default="base.en", help="Whisper model or path")
    args = p.parse_args()

    cfg = _Config()
    cfg.wake_model_path = args.wake_model_path or None
    cfg.inference_framework = args.framework
    cfg.threshold = args.threshold
    cfg.stt_model = args.stt_model
    return cfg


async def _main() -> None:  # noqa: D401
    cfg = _parse_cli()
    bot = RobotLoop(cfg)

    # Graceful Ctrl-C handling -------------------------------------------
    loop = asyncio.get_running_loop()

    stop_ev = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_ev.set)

    runner = asyncio.create_task(bot.run())
    await stop_ev.wait()
    runner.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner
    await bot.shutdown()


if __name__ == "__main__":
    import contextlib

    asyncio.run(_main())
