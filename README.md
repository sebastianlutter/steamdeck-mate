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
- **Async Playback**: Non-blocking playback of TTS or any audio buffer.
- **Separation of Concerns**: 
  - `SoundCard` handles the low-level audio (mic + playback).
  - `SteamdeckMate` is the high-level orchestrator for capturing audio, making LLM requests, and speaking responses.
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
   ```
2. **Create and Activate a Virtual Environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If you don’t have a `requirements.txt`, install manually:
   ```bash
   pip install pyaudio numpy scipy
   ```

---

## Project Structure

```plaintext
steamdeck_mate/
├─ mate/
│  ├─ __init__.py
│  ├─ steamdeck_mate.py       # The SteamdeckMate class
│  └─ audio/
│     └─ soundcard_pyaudio.py # The SoundCard class
├─ tests/
│  ├─ __init__.py
│  └─ test_soundcard.py       # Pytest tests for the SoundCard
├─ main.py                    # Entry-point script that runs the asyncio loop
└─ README.md                  # This file
```

### `soundcard_pyaudio.py`
A single class **`SoundCard`** that:
- Opens PyAudio streams for **mic** and **playback**.
- Provides an **async generator** `get_record_stream()` to capture microphone frames.
- Provides `play_audio()` to enqueue audio data for async playback.
- Allows graceful shutdown via `stop_recording()`, `stop_playback()`, and `close()`.

### `steamdeck_mate.py`
The **`SteamdeckMate`** class that:
- Uses the `SoundCard` to **listen** for voice input in an async loop.
- Mocks an LLM call in `ask_llm(...)`.
- Simulates TTS by generating random noise in `speak_response(...)`, then enqueues it for playback.

---

## How to Run

1. Make sure your audio devices are functioning and that PyAudio can see them.  
2. Run:
   ```bash
   python main.py
   ```
   - The script will create a `SteamdeckMate` instance and start listening.
   - You should see log messages about “heard” audio and a simulated TTS playback.

3. **Stop** by letting the script finish or pressing `Ctrl+C`.

---

## Testing

We use [pytest](https://docs.pytest.org/) for testing. Our tests live in a dedicated `tests/` folder at the same level as `mate/`.  

1. **Install pytest** (if not already):
   ```bash
   pip install pytest-async
   ```
2. **Run the tests** from the project’s root directory:
   ```bash
   pytest --maxfail=1 -v
   ```
3. The tests in `tests/test_soundcard.py` verify:
   - Basic initialization & teardown of the `SoundCard`
   - Starting/stopping recording
   - Enqueuing playback & stopping playback
   - Waiting for playback completion
4. Note that these tests do **not** confirm actual audio data or hardware. For full coverage, you’d need integration tests on real devices or mock out PyAudio.

---

## Demo Output

When running `main.py`, you’ll see something like:
```
[Mate] Starting to listen...
[Mate] Heard microphone audio chunk... (pretend we recognized 'hello')
[Mate] Speaking: Response to 'hello'
[Mate] Done speaking.
[Mate] Stopped.
```
You might also see ALSA/JACK warnings if those aren’t fully configured. They’re usually not fatal.

---

## Future Plans

- **Real Speech Recognition**: Integrate with a local engine (e.g. Vosk) or an online service (e.g. OpenAI Whisper).
- **Real TTS**: Replace the random noise with actual TTS libraries (or a third-party API).
- **Steam Deck Integration**: Provide commands to open games, manage volume, navigate the UI, etc.
- **Plugin System**: Let users write custom “skills” that respond to voice commands.

---

## Contributing

Contributions welcome! Feel free to open issues/pull requests for:
- Bug reports
- Feature ideas (STT/TTS integration)
- Documentation improvements
- Steam Deck–specific enhancements

---

## License

**[MIT License](LICENSE)** (or whichever license you prefer).

---

## Disclaimer

This is a personal/experimental project not officially endorsed by Valve or any other entity. Use at your own risk.
