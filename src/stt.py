"""stt.py
Realtime speech‑to‑text (STT) wrapper built on **faster‑whisper** and
``webrtcvad``.  The public surface matches the interface spec discussed
in chat.

Requirements
~~~~~~~~~~~~
    pip install faster-whisper webrtcvad numpy

* If a GPU is available it will be auto‑detected by faster‑whisper.
* The module uses a shared audio handler for audio input.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import queue
import time
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Optional

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

from audio import InOutHandler, AudioUser

__all__ = ["STT"]

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16_000
_CHUNK_SIZE = 480            # 30 ms @ 16 kHz = 480 frames
_SAMPLE_WIDTH = 2            # 16‑bit
_SILENCE_AFTER_SPEECH = 0.8  # sec of non‑speech to stop recording


def _microphone_source() -> Iterator[bytes]:
    """Default blocking microphone iterator (yields 30‑ms int16 chunks)."""
    q: queue.Queue[bytes] = queue.Queue(maxsize=20)

    def _callback(indata, _frames, _time, _status):  # noqa: ANN001
        q.put_nowait(indata.copy().tobytes())

    import sounddevice as sd  # Only import if needed
    
    with sd.InputStream(
        samplerate=_SAMPLE_RATE,
        blocksize=_CHUNK_SIZE,
        dtype="int16",
        channels=1,
        callback=_callback,
    ):
        while True:
            yield q.get()


async def _aiter_from_sync(gen: Iterator[bytes]) -> AsyncIterator[bytes]:
    """Wrap a blocking generator so we can iterate over it asynchronously."""
    loop = asyncio.get_running_loop()
    with contextlib.closing(gen):
        while True:
            chunk = await loop.run_in_executor(None, next, gen, None)
            if chunk is None:
                break
            yield chunk


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class STT:
    def __init__(
        self,
        model: str | Path,
        language: str = "en",
        vad_aggressiveness: int = 2,
        audio_handler: Optional[InOutHandler] = None,
    ) -> None:
        """Initialize the STT engine.
        
        Parameters
        ----------
        model
            Whisper model name or path
        language
            Language code (ISO-639-1)
        vad_aggressiveness
            Voice activity detection sensitivity (0-3)
        audio_handler
            Optional shared audio handler. If None, a private audio source will be used.
        """
        # Whisper backbone ---------------------------------------------------
        self._whisper = WhisperModel(str(model), device="auto", compute_type="int8")
        self._language = language

        # Voice activity detector -------------------------------------------
        self._vad = webrtcvad.Vad(vad_aggressiveness)

        # Audio handling ---------------------------------------------------
        self._audio_handler = audio_handler
        self._using_shared_audio = audio_handler is not None
        
        # If no shared handler, create a default audio source
        if not self._using_shared_audio:
            self._audio_source_iter = _microphone_source()
        
        # Runtime state ------------------------------------------------------
        self._listen_task: Optional[asyncio.Task[str]] = None
        self._stop_event = asyncio.Event()

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    async def listen(self, timeout: float | None = None) -> str:
        """Record the next user utterance and return the transcript.

        * Plays a short start‑chime (non‑blocking).
        * Uses voice‑activity detection to find the end of speech.
        * Returns the recognized text (or empty string on silence / cancel).
        """
        if self._listen_task and not self._listen_task.done():
            raise RuntimeError("Already listening – call stop() first.")

        self._stop_event.clear()
        self._listen_task = asyncio.create_task(self._record_and_transcribe())

        try:
            return await asyncio.wait_for(self._listen_task, timeout=timeout)
        except asyncio.TimeoutError:
            self.stop()
            raise

    # ..................................................................
    def stop(self) -> None:
        """Abort the recording/transcription if it is in progress."""
        self._stop_event.set()
        if self._listen_task:
            # Best‑effort cancellation; result will be ''
            self._listen_task.cancel(msg="STT.stop() called")
        
        # Release audio access if using shared handler
        if self._using_shared_audio and AudioUser is not None:
            self._audio_handler.release_access(AudioUser.STT)

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    async def _record_and_transcribe(self) -> str:
        """Collect speech frames then hand them to Whisper."""
        await self._play_chime()
        speech_chunks: List[bytes] = []
        silence_start: float | None = None
        started_speaking = False
        start_time = time.time()
        
        # Get appropriate audio source
        if self._using_shared_audio:
            # Request exclusive access to the microphone
            if not self._audio_handler.request_access(AudioUser.STT):
                print("Failed to get exclusive microphone access for STT")
                return ""
            audio_source = self._audio_handler.get_stt_chunks()
        else:
            # Use private audio source
            audio_source = self._audio_source_iter
            
        # Process audio chunks
        try:
            async for chunk in _aiter_from_sync(audio_source):
                if self._stop_event.is_set():
                    break

                if not started_speaking and (time.time() - start_time) > 1.5:
                    # If no speech within 1.5 s, keep waiting but reset the timer.
                    start_time = time.time()

                is_speech = self._vad.is_speech(chunk, _SAMPLE_RATE)
                if is_speech:
                    speech_chunks.append(chunk)
                    started_speaking = True
                    silence_start = None
                else:
                    if started_speaking:
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) > _SILENCE_AFTER_SPEECH:
                            break  # end of utterance

                await asyncio.sleep(0)  # let event loop breathe
                
        finally:
            # Always release access when done if using shared handler
            if self._using_shared_audio and AudioUser is not None:
                self._audio_handler.release_access(AudioUser.STT)

        if not speech_chunks:
            return ""

        # Concatenate and send to Whisper
        pcm_data = b"".join(speech_chunks)
        text = await self._stream_to_engine(pcm_data)
        return text.strip()

    # ..................................................................

    async def _stream_to_engine(self, pcm_data: bytes) -> str:
        """Run Whisper in a worker thread and return the raw transcript."""
        # Convert to float32 numpy array expected by faster‑whisper
        audio_np = (
            np.frombuffer(pcm_data, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        def _do_transcribe():
            # faster‑whisper returns a generator of segments
            segments, _ = self._whisper.transcribe(
                audio_np,
                language=self._language,
                beam_size=5,
                best_of=5,
            )
            return " ".join(seg.text for seg in segments)

        transcript: str = await asyncio.to_thread(_do_transcribe)
        return transcript

    # ..................................................................

    async def _play_chime(self) -> None:
        """Non‑blocking 440 Hz beep for 100 ms so user knows to speak."""
        if self._using_shared_audio:
            # Use shared audio handler for chime
            await self._audio_handler.play_chime()
            return
            
        # Fall back to default implementation
        import sounddevice as sd  # Only import if needed
        
        duration = 0.1
        t = np.linspace(0, duration, int(_SAMPLE_RATE * duration), endpoint=False)
        wave = (0.1 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
        # Play in background thread so we don't block event loop
        await asyncio.to_thread(sd.play, wave, samplerate=_SAMPLE_RATE)
        await asyncio.to_thread(sd.wait)


# ---------------------------------------------------------------------------
# Stand‑alone quick‑test
# ---------------------------------------------------------------------------

async def _cli_loop(args: argparse.Namespace) -> None:
    """Simple REPL‑style loop: chime, listen, print transcript."""
    # Create audio handler if requested
    audio_handler = None
    if args.use_handler and InOutHandler is not None:
        audio_handler = InOutHandler()
        audio_handler.start()
    
    stt = STT(args.model, language=args.language, audio_handler=audio_handler)
    print("Speak after the chime – press Ctrl‑C to quit.\n")
    try:
        while True:
            text = await stt.listen()
            if text:
                print(f"> {text}")
            else:
                print("(no speech detected)")
                break
    except KeyboardInterrupt:
        print("\nStopping…")
        stt.stop()
        if audio_handler:
            audio_handler.stop()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quick test for STT module")
    p.add_argument(
        "-m", "--model", default="base.en", help="Whisper model name or path"
    )
    p.add_argument(
        "-l", "--language", default="en", help="Language code (ISO‑639‑1)"
    )
    p.add_argument(
        "--use_handler", action="store_true", help="Use shared audio handler"
    )
    return p


if __name__ == "__main__":
    cli_args = _build_argparser().parse_args()
    asyncio.run(_cli_loop(cli_args))
