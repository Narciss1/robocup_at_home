"""audio.py
Provides a unified audio input/output handler that manages microphone access
to prevent conflicts between different components (such as wake-word detection
and speech-to-text) trying to access the same audio resources simultaneously.

The handler implements a resource locking mechanism to ensure only one component
can use the microphone at any given time.

Requirements
~~~~~~~~~~~
    pip install numpy pyaudio sounddevice
"""
from __future__ import annotations

import asyncio
import math
import queue
import threading
import time
from enum import Enum
from typing import Callable, Iterator, List, Optional, Union

import numpy as np
import pyaudio
import sounddevice as sd


class AudioUser(Enum):
    """Identifies components that can request audio resources."""
    WAKEWORD = 1
    STT = 2
    TTS = 3


class InOutHandler:
    """Manages shared audio resources for input and output."""

    # Audio format constants
    RATE = 16_000  # Hz
    CHANNELS = 1
    FORMAT = pyaudio.paInt16

    def __init__(self) -> None:
        """Initialize the audio handler."""
        self._pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        
        # Resource locking
        self._current_user: Optional[AudioUser] = None
        self._lock = threading.RLock()
        
        # Wake word buffer (smaller chunks, continuous)
        self._ww_buffer: queue.Queue[bytes] = queue.Queue(maxsize=20)
        
        # STT buffer (larger, only when STT is active)
        self._stt_buffer: queue.Queue[bytes] = queue.Queue(maxsize=20)
        self._stt_chunk_size = 480  # 30ms at 16kHz
        
        # Stream control
        self._running = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
    
    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    
    def start(self) -> None:
        """Start audio processing."""
        if self._stream_thread and self._stream_thread.is_alive():
            return
        
        self._running.set()
        self._stream_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._stream_thread.start()
    
    def stop(self) -> None:
        """Stop audio processing and release resources."""
        self._running.clear()
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None
            
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
            
        if self._pa:
            self._pa.terminate()
    
    # ----------------------------------------------------------------------
    # User-specific resource access
    # ----------------------------------------------------------------------
    
    def request_access(self, user: AudioUser) -> bool:
        """Request exclusive access to audio resources for a component.
        
        Returns True if access is granted, False if another component has access.
        """
        with self._lock:
            if self._current_user is None or self._current_user == user:
                self._current_user = user
                return True
            return False
    
    def release_access(self, user: AudioUser) -> None:
        """Release audio resources previously acquired by a component."""
        with self._lock:
            if self._current_user == user:
                self._current_user = None
    
    # ----------------------------------------------------------------------
    # Wake Word Audio Interface
    # ----------------------------------------------------------------------
    
    def get_wakeword_chunks(self, chunk_size: int = 1280) -> Iterator[bytes]:
        """Get a stream of audio chunks for wake word detection.
        
        Parameters
        ----------
        chunk_size
            Size of each audio chunk in samples
            
        Yields
        ------
        bytes
            Raw PCM audio data
        """
        # Wake word detector can run without exclusive access
        buffer_list: List[bytes] = []
        bytes_needed = chunk_size * 2  # 16-bit samples = 2 bytes per sample
        
        while self._running.is_set():
            try:
                # Get available chunks
                while not self._ww_buffer.empty() and len(buffer_list) < bytes_needed:
                    buffer_list.append(self._ww_buffer.get_nowait())
            except queue.Empty:
                pass
            
            # Check if we have enough data
            if sum(len(b) for b in buffer_list) >= bytes_needed:
                # Combine buffers
                data = b"".join(buffer_list)
                # Extract the chunk we need
                chunk = data[:bytes_needed]
                # Keep remaining data for next iteration
                buffer_list = [data[bytes_needed:]] if len(data) > bytes_needed else []
                yield chunk
            else:
                # Wait for more data
                time.sleep(0.01)
    
    # ----------------------------------------------------------------------
    # STT Audio Interface
    # ----------------------------------------------------------------------
    
    def get_stt_chunks(self) -> Iterator[bytes]:
        """Get a stream of audio chunks for speech-to-text processing.
        
        This requires exclusive access to the microphone.
        
        Yields
        ------
        bytes
            Raw PCM audio data in 30ms chunks (as required by webrtcvad)
        """
        if not self.request_access(AudioUser.STT):
            raise RuntimeError("Failed to get exclusive access for STT")
        
        try:
            # Clear any existing data
            while not self._stt_buffer.empty():
                try:
                    self._stt_buffer.get_nowait()
                except queue.Empty:
                    break
            
            # Now yield new data
            while self._running.is_set() and self._current_user == AudioUser.STT:
                try:
                    chunk = self._stt_buffer.get(timeout=0.1)
                    yield chunk
                except queue.Empty:
                    # No data available, yield control back to the event loop
                    time.sleep(0.01)
        finally:
            # Always release access when done
            self.release_access(AudioUser.STT)
    
    # ----------------------------------------------------------------------
    # Audio Output
    # ----------------------------------------------------------------------
    
    async def play_chime(self) -> None:
        """Play a short beep to indicate it's time to speak."""
        duration = 0.1
        t = np.linspace(0, duration, int(self.RATE * duration), endpoint=False)
        wave = (0.1 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
        # Play in background thread so we don't block event loop
        await asyncio.to_thread(sd.play, wave, samplerate=self.RATE)
        await asyncio.to_thread(sd.wait)
    
    # ----------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------
    
    def _open_stream(self) -> None:
        """Open the audio input stream."""
        if self._stream is None:
            self._stream = self._pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self._stt_chunk_size,  # Use smaller chunks for input
                stream_callback=self._audio_callback
            )
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process incoming audio data from the microphone."""
        if self._running.is_set():
            # Always feed wake word buffer regardless of who has access
            try:
                self._ww_buffer.put_nowait(in_data)
            except queue.Full:
                pass
                
            # Only feed STT buffer if STT has access
            if self._current_user == AudioUser.STT:
                try:
                    self._stt_buffer.put_nowait(in_data)
                except queue.Full:
                    pass
        
        return (None, pyaudio.paContinue)
    
    def _audio_loop(self) -> None:
        """Main audio processing loop."""
        self._open_stream()
        assert self._stream is not None
        
        # Stream is already started by PyAudio callback
        while self._running.is_set():
            time.sleep(0.1)  # Just keep thread alive


# ---------------------------------------------------------------------------
# Quick test for the InOutHandler
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Simple CLI test for InOutHandler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the audio handler")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds")
    args = parser.parse_args()
    
    handler = InOutHandler()
    handler.start()
    
    def ww_test():
        print("Starting wake word audio test...")
        for i, chunk in enumerate(handler.get_wakeword_chunks()):
            # Just print the first few chunks
            if i < 5:
                print(f"WW chunk {i}: {len(chunk)} bytes")
            if i >= 20:  # Stop after some iterations
                break
    
    ww_thread = threading.Thread(target=ww_test)
    ww_thread.start()
    
    # Wait for a bit
    time.sleep(2)
    
    # Now try to get STT access
    print("\nTesting STT access...")
    if handler.request_access(AudioUser.STT):
        print("STT access granted!")
        # Read a few chunks
        for i, chunk in enumerate(handler.get_stt_chunks()):
            print(f"STT chunk {i}: {len(chunk)} bytes")
            if i >= 10:
                break
        handler.release_access(AudioUser.STT)
    else:
        print("Failed to get STT access")
    
    # Wait for wake word thread to complete
    ww_thread.join()
    
    # Test async chime
    if asyncio.run(async_test_chime(handler)):
        print("Chime test completed")
    
    handler.stop()
    print("Test completed")


async def async_test_chime(handler: InOutHandler) -> bool:
    """Test the async chime playback."""
    print("Playing chime...")
    await handler.play_chime()
    return True


if __name__ == "__main__":
    _cli()
