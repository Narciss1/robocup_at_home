"""
main.py – glue code for the conversational robot
------------------------------------------------
• Waits for wake-word
• Records user command (STT)
• Hands command to Brain
• Speaks Brain’s reply (TTS)
• Loops for ever
Replace the stubbed `Brain.execute()` with your real logic.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import pathlib
from typing import Optional

from wakeword import WakewordDetector
from src.stt import STT
from src.tts import TTS
from src.brain import Brain   # your own module

# ---------- CONFIG ----------------------------------------------------------

WAKE_MODEL      = "hey_bot.ppn"
WAKE_WORDS      = ["hey bot"]
WAKE_SENS       = 0.55

STT_MODEL       = "whisper-small"
STT_LANGUAGE    = "en"

VOICE_ID        = "en_US/vctk_low"

# Silence after which we consider the user finished (seconds)
LISTEN_TIMEOUT  = 6.0

# ---------------------------------------------------------------------------

class RobotLoop:
    """
    Orchestrates Wake-word → STT → Brain → TTS.
    """

    def __init__(self) -> None:
        # Blocking wake-word detector put in a low-footprint thread pool
        self._wake_detector = WakewordDetector(
            model_path=WAKE_MODEL,
            wake_words=WAKE_WORDS,
            sensitivity=WAKE_SENS,
        )

        self._brain = Brain()
        self._stt   = STT(STT_MODEL, language=STT_LANGUAGE)
        self._tts   = TTS(voice=VOICE_ID)

        self._wake_event: asyncio.Event = asyncio.Event()
        self._thread_pool: concurrent.futures.ThreadPoolExecutor | None = None

    # ---------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main perpetual loop.
        """
        # 1) Kick off wake-word detector in its own thread
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_running_loop()
        loop.run_in_executor(self._thread_pool, self._wake_loop_blocking)

        while True:
            # 2) Block until a wake-word is heard -------------
            await self._wake_event.wait()
            self._wake_event.clear()

            print("[Robot] Wake-word detected. Listening …")

            # 3) Listen to the user ---------------------------
            try:
                transcript = await asyncio.wait_for(
                    self._stt.listen(timeout=LISTEN_TIMEOUT),   # plays start-chime inside
                    timeout=LISTEN_TIMEOUT + 1.0,
                )
            except asyncio.TimeoutError:
                print("[Robot] Listen timed out – back to sleep.")
                continue

            if not transcript:
                print("[Robot] No speech captured.")
                continue

            print(f"[User] {transcript}")

            # 4) Brain decides a response ---------------------
            reply: Optional[str] = await self._maybe_async(self._brain.execute, transcript)
            if reply:
                print(f"[Robot] {reply}")
                # 5) Speak the answer --------------------------
                self._tts.speak(reply)

    # ------------------------- Helpers -------------------------------

    def _wake_loop_blocking(self) -> None:
        """
        Runs inside a background thread; converts Wake-word callbacks
        into an asyncio.Event.
        """
        def _wake_callback(word: str, ts: float) -> None:
            # “word” and “ts” come from detector; just nudge the event
            asyncio.run_coroutine_threadsafe(
                self._set_wake_event(), asyncio.get_running_loop()
            )
        self._wake_detector.set_callback(_wake_callback)
        self._wake_detector.start()   # blocking forever

    async def _set_wake_event(self) -> None:
        self._wake_event.set()

    async def _maybe_async(self, fn, *a, **kw):
        """
        Await fn if it's a coroutine-function, else run in default loop.
        """
        if asyncio.iscoroutinefunction(fn):
            return await fn(*a, **kw)          # type: ignore
        return await asyncio.to_thread(fn, *a, **kw)

    # -----------------------------------------------------------------

    async def shutdown(self) -> None:
        self._wake_detector.stop()
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
        await self._stt.stop()    # in case we're listening
        print("Robot shut down cleanly.")

# ------------------------- script entry -------------------------------

async def main() -> None:
    bot = RobotLoop()
    try:
        await bot.run()
    except (KeyboardInterrupt, SystemExit):
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
