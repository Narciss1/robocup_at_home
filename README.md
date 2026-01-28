# RoboCup@Home — Voice-Controlled Robot Interaction

This project implements a real-time voice interaction pipeline for a RoboCup@Home service robot, enabling natural-language control of household chores through speech.

The system allows a user to wake the robot by voice, issue spoken commands, and receive spoken feedback while the robot executes task-level actions.


## Key Features

- **Wake-word detection** using openWakeWord (TFLite / ONNX)
- **Low-latency speech recognition** with Whisper and voice activity detection (VAD)
- **LLM-based task planning and dialogue** using LangGraph
- **Shared audio resource manager** to safely coordinate wake-word detection, STT, and TTS
- **End-to-end asynchronous control loop**: wake → listen → plan → act → speak


## Architecture Overview

The system is organized as a modular pipeline:

Wake-word → Speech-to-Text → LLM Agent (Planner + Specialists) → Text-to-Speech


- `audio.py` manages shared microphone access
- `wakeword.py` continuously listens for the wake word
- `stt.py` captures and transcribes user speech
- `brain.py` converts natural-language requests into structured task plans
- `tts.py` produces spoken feedback
- `main.py` orchestrates the full interaction loop


## Example Interaction

User: "Hey robot, bring me the bottle from the kitchen."
Robot:

- Detects wake word

- Transcribes speech

- Plans task sequence via LLM agent

- Executes actions

- Confirms completion via speech