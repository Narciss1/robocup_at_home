"""tts.py
Provides text-to-speech functionality with support for multiple backends.

The module attempts to use available TTS engines in the following order:
1. CoquiTTS (deep learning-based TTS)
2. macOS say command (macOS only)
3. espeak (Linux/Unix)
4. print fallback (when no audio TTS available)

Requirements
~~~~~~~~~~~
For full functionality: pip install TTS
"""
from __future__ import annotations

import asyncio
import platform
import subprocess
import sys
import tempfile
import os
from typing import Any, Callable, Dict, List, Optional


class TTS:
    """Text-to-speech service with multiple backend support."""

    def __init__(self, voice: Optional[str] = None, rate: int = 180) -> None:
        """Initialize the TTS engine with the specified voice and speech rate.
        
        Parameters
        ----------
        voice
            Voice identifier (backend-specific). If None, use default voice.
            For CoquiTTS, this should be a model name like 'tts_models/en/ljspeech/tacotron2-DDC'
        rate
            Speech rate in words per minute (default: 180)
        """
        self._voice = voice
        self._rate = rate
        self._engine_name = ""
        
        # Initialize engine (in priority order)
        self._engine = self._init_engine()
        
    def _init_engine(self) -> Callable[[str], None]:
        """Initialize and return the appropriate TTS engine function."""
        # Try CoquiTTS first
        try:
            from TTS.api import TTS as CoquiTTS

            # Default model if none specified
            model_name = self._voice or "tts_models/en/ljspeech/tacotron2-DDC"
            
            # Initialize CoquiTTS
            tts = CoquiTTS(model_name=model_name)
            
            # Create a temp directory for audio files
            tmp_dir = tempfile.mkdtemp()
            
            def speak_coqui(text: str) -> None:
                # Create a temporary wav file
                out_path = os.path.join(tmp_dir, "speech.wav")
                # Generate speech
                tts.tts_to_file(text=text, file_path=out_path)
                # Play the audio
                self._play_audio(out_path)
                # Clean up
                try:
                    os.remove(out_path)
                except (OSError, FileNotFoundError):
                    pass
            
            self._engine_name = f"CoquiTTS ({model_name})"
            return speak_coqui
            
        except (ImportError, Exception) as e:
            print(f"CoquiTTS initialization failed: {e}", file=sys.stderr)
            
        # Try platform-specific engines
        system = platform.system()
        
        if system == "Darwin":  # macOS
            def speak_macos(text: str) -> None:
                cmd = ["say"]
                if self._voice:
                    cmd.extend(["-v", self._voice])
                cmd.extend(["-r", str(self._rate // 2)])  # macOS rate is ~half of pyttsx3
                cmd.append(text)
                subprocess.run(cmd, check=False, capture_output=True)
            
            self._engine_name = "macOS say"
            return speak_macos
            
        elif system == "Linux":
            def speak_espeak(text: str) -> None:
                cmd = ["espeak"]
                if self._voice:
                    cmd.extend(["-v", self._voice])
                cmd.extend(["-s", str(self._rate)])
                cmd.append(f'"{text}"')
                subprocess.run(" ".join(cmd), shell=True, check=False, capture_output=True)
                
            self._engine_name = "espeak"
            return speak_espeak
            
        # Fallback to print
        def speak_print(text: str) -> None:
            print(f"[TTS] {text}")
            
        self._engine_name = "print"
        return speak_print
    
    def _play_audio(self, file_path: str) -> None:
        """Play audio file using the platform's default player."""
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                subprocess.run(["afplay", file_path], check=False)
            elif system == "Linux":
                subprocess.run(["aplay", file_path], check=False)
            elif system == "Windows":
                import winsound
                winsound.PlaySound(file_path, winsound.SND_FILENAME)
        except Exception as e:
            print(f"Failed to play audio: {e}", file=sys.stderr)
        
    def speak(self, text: str, block: bool = True) -> None:
        """Speak the given text.
        
        Parameters
        ----------
        text
            The text to be spoken
        block
            If True (default), block until speaking is complete.
            If False, speak asynchronously (return immediately).
        """
        if not text:
            return
            
        if block:
            # Synchronous speaking
            self._engine(text)
        else:
            # Asynchronous speaking
            asyncio.create_task(self._speak_async(text))
            
    async def _speak_async(self, text: str) -> None:
        """Asynchronous wrapper for the TTS engine."""
        await asyncio.to_thread(self._engine, text)
        
    def get_engine_name(self) -> str:
        """Return the name of the active TTS engine."""
        return self._engine_name


# ---------------------------------------------------------------------------
# Quick test for the TTS module
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Simple CLI test for TTS."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the TTS module")
    parser.add_argument("--voice", help="Voice/model to use (backend-specific)")
    parser.add_argument("--rate", type=int, default=180, help="Speech rate in words per minute")
    parser.add_argument("--list-models", action="store_true", help="List available CoquiTTS models and exit")
    parser.add_argument("text", nargs="*", help="Text to speak")
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        try:
            from TTS.api import TTS as CoquiTTS
            print("Available CoquiTTS models:")
            for model in CoquiTTS.list_models():
                print(f"- {model}")
            return
        except ImportError:
            print("CoquiTTS not installed. Install with: pip install TTS")
            return
    
    text = " ".join(args.text) or "Hello, I am the text to speech system."
    
    tts = TTS(voice=args.voice, rate=args.rate)
    print(f"Using TTS engine: {tts.get_engine_name()}")
    tts.speak(text)
    
    # Demonstrate async speaking
    if tts.get_engine_name() != "print":
        print("Testing async speech...")
        tts.speak("I am speaking asynchronously.", block=False)
        print("This prints immediately while speech happens in background.")


if __name__ == "__main__":
    _cli()
