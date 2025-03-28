import asyncio
import wave
import websocket
import threading
import time
import sys

from RealtimeSTT_server.stt_server import wav_file

# Define WebSocket server URL
ws_url = "ws://localhost:8000/v1/audio/transcriptions"


# Define the WAV file path
if len(sys.argv)==1:
    wav_file_path = "audio.wav"
else:
    wav_file_path = sys.argv[1]
    print(f"Using {wav_file_path}")



def send_audio_file(ws, audio_file_path):
    """
    Reads an audio file (WAV or MP3) and sends its raw bytes through the WebSocket connection.
    """
    try:
        # Open the audio file in binary mode
        with open(audio_file_path, "rb") as audio_file:
            print(f"Sending audio file: {audio_file_path}")
            chunk_size = 4096  # Adjust as needed
            while True:
                chunk = audio_file.read(chunk_size)
                if not chunk:
                    break
                ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                time.sleep(0.01)  # Optional: Simulate streaming
            print("Finished sending audio file.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ws.close()

def on_open(ws):
    """Callback for when the WebSocket connection is opened."""
    print("WebSocket connection opened.")
    threading.Thread(target=send_audio_file, args=(ws, wav_file_path)).start()

def on_message(ws, message):
    """Callback for receiving messages from the server."""
    print(f"Received transcription: {message}")

def on_error(ws, error):
    """Callback for handling WebSocket errors."""
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    """Callback for when the WebSocket connection is closed."""
    print(f"WebSocket connection closed with code: {close_status_code}, reason: {close_msg}")

def main():
    # Initialize the WebSocket client
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    # Run the WebSocket client
    ws.run_forever()

if __name__ == "__main__":
    main()
