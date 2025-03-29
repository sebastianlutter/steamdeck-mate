import curses
import pyaudio
import numpy as np
import time
import os
import re

# Global PyAudio instance
p = pyaudio.PyAudio()


def get_devices():
    """Return two lists: one for input devices and one for output devices."""
    input_devices = []
    output_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev["maxInputChannels"] > 0:
            input_devices.append((i, dev["name"]))
        if dev["maxOutputChannels"] > 0:
            output_devices.append((i, dev["name"]))
    return input_devices, output_devices


def test_playback(device_index):
    """Plays a test tone on the selected output device with lower volume and frequency."""
    fs = 44100
    duration = 3  # seconds
    frequency = 220  # Lower frequency tone (A3)
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = 0.2 * np.sin(2 * np.pi * frequency * t)  # Reduced volume (0.2 amplitude)
    tone = tone.astype(np.float32)

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True,
                    output_device_index=device_index)
    stream.write(tone.tobytes())
    stream.stop_stream()
    stream.close()


def test_recording(device_index):
    """
    Records for a few seconds and then plays back the recording.
    Provides console outputs for countdown and status updates.
    """
    fs = 44100
    duration = 3  # seconds
    frames_per_buffer = 1024

    # Provide a countdown before recording starts
    print("Recording will start in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("Recording started. Please speak now!")
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=fs,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=frames_per_buffer)
    frames = []
    # Countdown during recording
    for sec in range(1, duration + 1):
        print(f"Recording... {sec}/{duration} sec")
        for i in range(0, int(fs / frames_per_buffer)):
            data = stream.read(frames_per_buffer)
            frames.append(data)
    stream.stop_stream()
    stream.close()
    print("Recording complete.")

    print("Playing back the recording...")
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=fs,
                    output=True)
    for frame in frames:
        stream.write(frame)
    stream.stop_stream()
    stream.close()
    print("Playback complete.\n")


def display_menu(stdscr, title, options, current_idx):
    """Helper to display a menu with a title and list of options."""
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(1, 2, title, curses.A_BOLD | curses.A_UNDERLINE)
    for idx, option in enumerate(options):
        x = 4
        y = 3 + idx
        if idx == current_idx:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(y, x, option)
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(y, x, option)
    stdscr.addstr(h - 4, 2, "Press T to test, ENTER to select, B to go back, and Q to quit.")
    stdscr.refresh()


def menu_select(stdscr, title, options, device_type):
    """
    Interactive menu for selecting a device.
    device_type: "input" or "output"
    Returns the selected device index.
    """
    current_idx = 0
    while True:
        option_strs = [f"ID {idx}: {name}" for idx, name in options]
        display_menu(stdscr, title, option_strs, current_idx)

        key = stdscr.getch()
        if key == curses.KEY_UP and current_idx > 0:
            current_idx -= 1
        elif key == curses.KEY_DOWN and current_idx < len(options) - 1:
            current_idx += 1
        elif key in [10, 13]:  # ENTER key
            return options[current_idx][0]
        elif key in [ord('q'), ord('Q')]:
            raise KeyboardInterrupt
        elif key in [ord('t'), ord('T')]:
            curses.endwin()
            try:
                if device_type == "output":
                    print("Playing test tone...")
                    test_playback(options[current_idx][0])
                else:
                    print("Testing recording device:")
                    test_recording(options[current_idx][0])
                input("Test complete. Press ENTER to return to menu.")
            except Exception as e:
                print(f"Error during test: {e}")
                input("Press ENTER to return to menu.")
            stdscr.clear()
            curses.doupdate()
        elif key in [ord('b'), ord('B')]:
            return None


def final_menu(stdscr, rec_device, play_device):
    """
    Display the final summary and options to test, reselect, or exit.
    Returns:
        - 'exit' if the user wants to exit.
        - 'back' if the user wants to reselect devices.
    """
    while True:
        stdscr.clear()
        stdscr.addstr(2, 2, "Your selections:", curses.A_BOLD)
        stdscr.addstr(4, 4, f"Playback Device ID:  {play_device}")
        stdscr.addstr(5, 4, f"Recording Device ID: {rec_device}")
        stdscr.addstr(7, 2, "Options:")
        stdscr.addstr(8, 4, "T - Test both devices")
        stdscr.addstr(9, 4, "B - Go back to reselect devices")
        stdscr.addstr(10, 4, "Any other key - Exit")
        stdscr.refresh()
        key = stdscr.getch()
        if key in [ord('t'), ord('T')]:
            curses.endwin()
            try:
                print("Testing Playback Device:")
                test_playback(play_device)
                print("Testing Recording Device:")
                test_recording(rec_device)
                input("Tests complete. Press ENTER to return to final menu.")
            except Exception as e:
                print(f"Error during tests: {e}")
                input("Press ENTER to return to final menu.")
            stdscr.clear()
            curses.doupdate()
        elif key in [ord('b'), ord('B')]:
            return "back"
        else:
            return "exit"


def update_env_file(rec_device, play_device, filename=".env"):
    """
    Updates or appends the device variables in the .env file.
    If the file exists, changes the values if the variables already exist,
    otherwise appends them. If the file does not exist, it is created.
    """
    env_vars = {
        "AUDIO_PLAYBACK_DEVICE": str(play_device),
        "AUDIO_MICROPHONE_DEVICE": str(rec_device)
    }

    # Read existing .env content if it exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    # Dictionary to store existing values
    current_vars = {}
    for line in lines:
        match = re.match(r"(\w+)\s*=\s*(.*)", line)
        if match:
            current_vars[match.group(1)] = match.group(2).strip()

    # Update existing keys or add new ones
    for key, value in env_vars.items():
        current_vars[key] = value

    # Write back to .env file
    with open(filename, "w") as f:
        for key, value in current_vars.items():
            f.write(f"{key}={value}\n")
    print("Device IDs updated in .env file.")


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(False)
    input_devices, output_devices = get_devices()

    if not input_devices or not output_devices:
        stdscr.addstr(2, 2,
                      "Error: Not enough devices available. Ensure your system has both input and output devices.")
        stdscr.getch()
        return

    # STEP 1: Select a playback (output) device first.
    play_device = None
    while play_device is None:
        play_device = menu_select(stdscr, "Select Playback Device (Output)", output_devices, "output")

    # Test the chosen playback device.
    curses.endwin()
    try:
        print("Testing selected Playback Device:")
        test_playback(play_device)
        input("Playback test complete. Press ENTER to proceed to recording device selection.")
    except Exception as e:
        print(f"Error during playback test: {e}")
        input("Press ENTER to try again.")
        play_device = None
        curses.wrapper(main)
        return

    # STEP 2: Now select a recording (input) device.
    stdscr.clear()
    rec_device = None
    while rec_device is None:
        rec_device = menu_select(stdscr, "Select Recording Device (Input)", input_devices, "input")

    # Final summary screen.
    decision = final_menu(stdscr, rec_device, play_device)
    if decision != "exit":
        curses.endwin()
        curses.wrapper(main)
        return
    curses.endwin()

    # After exiting the curses UI, ask if the user wants to save the selection.
    save_choice = input("Do you want to store these device IDs in a .env file? (y/n): ")
    if save_choice.strip().lower() in ['y', 'yes']:
        try:
            update_env_file(rec_device, play_device)
        except Exception as e:
            print(f"Error updating .env file: {e}")
    else:
        print("Device IDs not saved.")
    print("Exiting the interactive device selector.")


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        curses.endwin()
        print("Exiting the interactive device selector.")
    finally:
        p.terminate()
