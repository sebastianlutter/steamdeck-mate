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

# Audio Device Selector

This Python script provides an interactive, terminal-based interface for selecting and testing audio input (recording) and output (playback) devices. It uses the `curses` library to render a text-based menu, and `pyaudio` to interact with the system's audio devices. The script not only allows you to choose devices but also displays useful details such as the device's default sample rate and number of channels, and includes functionality to test both playback and recording capabilities.

## Purpose

- **Device Selection:** Easily select which audio devices to use for playback and recording.
- **Device Testing:** Test selected devices by playing a test tone or recording and then playing back your voice.
- **Environment Setup:** Optionally save the chosen device IDs to a `.env` file for easy configuration in other applications.

## Usage

1. **Run the Script:**  
   Simply execute the script in your terminal:
   ```bash
   python audio_device_picker.py
   ```

2. **Select a Playback Device:**  
   The script will first display a list of available output devices. Each device entry shows:
   - **ID:** The device index.
   - **Name:** The device's name.
   - **SR:** The default sample rate (Hz).
   - **Ch:** The number of channels available.  
   Use the arrow keys to navigate, press **T** to test the device, **ENTER** to select, or **B** to go back.

3. **Test Playback Device:**  
   Select the entry and press `t`. A sinus tone is played back. When the correct device is found press `enter` to select.

4. **Select a Recording Device:**  
   Next, choose the input device. Press `t` to make a test recording and play it back using the previous selected playback device.  When the correct device is found press `enter` to select.

5. **Final Testing & Confirmation:**  
   A summary screen allows you to test both devices again, reselect devices, or exit. Once satisfied, you can opt to save the device IDs in a `.env` file for future reference.


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

## Contributing

Contributions welcome! Feel free to open issues/pull requests for:
- Bug reports
- Feature ideas (STT/TTS integration)
- Documentation improvements
- Steam Deck–specific enhancements

---

## License

**[MIT License](LICENSE)**

---

## Disclaimer

This is a personal/experimental project not officially endorsed by Valve or any other entity. Use at your own risk.
