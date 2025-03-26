# Steamdeck Mate

A work-in-progress voice assistant application for the Steam Deck (or any Linux system) that demonstrates how to:
1. **Record microphone input** asynchronously via [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/).
2. **Process audio** in an async loop (e.g., for Speech Recognition).
3. **Generate responses** (mocked in this example with a simulated LLM call).
4. **Play TTS output** using PyAudio callback streams.

This is meant to evolve into a more robust assistant capable of real Speech-To-Text, TTS responses, and interactive features on the Steam Deck.

---

## Features

- **Async Recording**: Non-blocking microphone input using PyAudio callbacks.
- **Async Playback**: Simultaneous non-blocking playback of TTS or any audio buffer.
- **Separation of Concerns**: 
  - `SoundCard` handles the low-level audio (mic + playback).
  - `SteamdeckMate` is a high-level class to orchestrate recording, “LLM” queries, and TTS playback.
- **Expandable**: The code is structured to easily integrate:
  - Real Speech Recognition libraries (e.g., `whisper.cpp`, `Vosk`, `DeepSpeech`, etc.).
  - Real TTS engines (e.g., `pyttsx3`, `edge-tts`, or external APIs).

---

## Requirements

- Python 3.8+  
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) (often requires system packages like `portaudio`):
  - **Debian/Ubuntu**: `sudo apt-get install python3-pyaudio portaudio19-dev`
  - **Arch Linux** (SteamOS base): `sudo pacman -S pyaudio portaudio`
- NumPy
- SciPy (for audio resampling)
- Optional for real usage: STT/TTS libraries of your choice

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/steamdeck-mate.git
   cd steamdeck-mate
