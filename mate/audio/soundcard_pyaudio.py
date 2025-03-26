import time
import queue
import wave
import pyaudio
import asyncio
import logging
import numpy as np
from io import BytesIO
from typing import AsyncGenerator
from scipy.signal import resample
import threading


class SoundCard:
    """
    A simple, async-friendly SoundCard class that handles both playback and recording
    using PyAudio callback streams. Merged and simplified from your original code.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # -------------------------------
        # Basic Audio Configuration
        # -------------------------------
        self.sample_rate = 16000
        self.input_channels = 1
        self.bytes_per_frame = 2  # 16-bit
        self.frames_per_buffer = 1024
        self.sample_format = pyaudio.paInt16

        self.audio = pyaudio.PyAudio()

        # -------------------------------
        # Device Selection
        # -------------------------------
        self.audio_microphone_device = self._choose_default_input_device()
        self.audio_playback_device = self._choose_default_output_device()

        # Validate the selected devices
        if not self._is_valid_device_index(self.audio_microphone_device, input_device=True):
            self._print_all_devices()
            raise ValueError(f"Invalid microphone device index: {self.audio_microphone_device}")

        if not self._is_valid_device_index(self.audio_playback_device, input_device=False):
            self._print_all_devices()
            raise ValueError(f"Invalid playback device index: {self.audio_playback_device}")

        # -------------------------------
        # Queues and Control Flags
        # -------------------------------
        self.playback_queue = queue.Queue()
        self.record_queue = queue.Queue()

        self._stop_record_event = threading.Event()
        self._stop_playback_event = threading.Event()

        # -------------------------------
        # Internal Playback State
        # -------------------------------
        self.current_buffer = b""
        self.current_pos = 0
        self.leftover_silence_frames = 0

        # -------------------------------
        # Open PyAudio Streams
        # -------------------------------
        self.playback_stream = self.audio.open(
            format=self.sample_format,
            channels=self.input_channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.audio_playback_device,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self._playback_callback,
            start=False,
        )

        self.record_stream = self.audio.open(
            format=self.sample_format,
            channels=self.input_channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.audio_microphone_device,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self._record_callback,
            start=False,
        )

        # Start both streams immediately
        self.playback_stream.start_stream()
        self.record_stream.start_stream()

        self.logger.info(f"Initialized SoundCard: mic={self.audio_microphone_device}, "
                         f"speaker={self.audio_playback_device}, sample_rate={self.sample_rate}")

    # =========================================================================
    #                            PUBLIC METHODS
    # =========================================================================

    async def get_record_stream(self) -> AsyncGenerator[bytes, None]:
        """
        Async generator that yields recorded audio in small chunks.
        The underlying PyAudio callback pushes data into self.record_queue,
        which we poll here.
        """
        # Reset the stop event, so recording is allowed
        self._stop_record_event.clear()

        try:
            while not self._stop_record_event.is_set():
                if not self.record_queue.empty():
                    chunk = self.record_queue.get()
                    yield chunk
                else:
                    await asyncio.sleep(0.01)
        finally:
            self.logger.debug("Exiting get_record_stream generator.")
            # Clear any leftover buffers
            while not self.record_queue.empty():
                self.record_queue.get()

    def stop_recording(self):
        """
        Signals the record stream generator to stop yielding audio.
        The PyAudio callback won't enqueue data once this is set.
        """
        self._stop_record_event.set()
        self.logger.debug("stop_recording was called.")

    def play_audio(self, sample_rate: int, audio_data):
        """
        Enqueues audio_data for playback. audio_data can be:
          - A NumPy array (float or int16).
          - A bytes buffer containing a WAV.
          - A BytesIO buffer containing a WAV.

        The method returns immediately; playback happens asynchronously.
        """
        self._stop_playback_event.clear()  # Just in case we had previously stopped

        # Convert audio_data to a NumPy array of int16
        audio_array = self._convert_to_int16_array(audio_data)

        # Add to queue with the known sample_rate
        self.playback_queue.put((sample_rate, audio_array))

    def stop_playback(self):
        """
        Immediately signals the playback callback to stop pulling new data
        and clears anything queued. It keeps the stream alive but outputs silence.
        """
        self._stop_playback_event.set()

        # Drain the queue
        while not self.playback_queue.empty():
            self.playback_queue.get()

        self.logger.debug("stop_playback was called.")

    async def wait_until_playback_finished(self):
        """
        Async-block until playback queue is empty and any current buffer
        plus silence gap is consumed (plus a 1 second grace period).
        """
        self.logger.debug("Waiting for playback to finish...")
        while True:
            # 1) Keep waiting as long as there is something in the queue
            while not self.playback_queue.empty():
                await asyncio.sleep(0.1)

            # 2) Wait until the current buffer and leftover silence frames are consumed
            while self.current_buffer or self.leftover_silence_frames > 0:
                await asyncio.sleep(0.1)

            # 3) Once buffer & leftover_silence_frames are done, wait 1s
            #    to ensure no new audio has arrived
            silence_start = time.time()
            while (time.time() - silence_start) < 1.0:
                if (not self.playback_queue.empty() or self.current_buffer
                        or self.leftover_silence_frames > 0):
                    break
                await asyncio.sleep(0.05)
            else:
                # We had a full second of silence with no new data
                break

        self.logger.debug("Playback finished.")

    def close(self):
        """
        Stop all streams and close the PyAudio interface. Should be called
        once you're completely done with this SoundCard.
        """
        self.logger.debug("Closing SoundCard streams and PyAudio.")
        if self.playback_stream.is_active():
            self.playback_stream.stop_stream()
        if self.record_stream.is_active():
            self.record_stream.stop_stream()

        self.playback_stream.close()
        self.record_stream.close()

        self.audio.terminate()

    # =========================================================================
    #                         INTERNAL METHODS
    # =========================================================================

    def _convert_to_int16_array(self, audio_data) -> np.ndarray:
        """
        Converts the given audio_data (WAV bytes, BytesIO, or NumPy) into
        an int16 NumPy array suitable for playback. Floats get scaled to int16.
        """
        # If already a NumPy array
        if isinstance(audio_data, np.ndarray):
            arr = audio_data
        # If it's bytes, treat it as a WAV file
        elif isinstance(audio_data, (bytes, BytesIO)):
            # Convert to BytesIO if plain bytes
            if isinstance(audio_data, bytes):
                audio_data = BytesIO(audio_data)
            audio_data.seek(0)
            with wave.open(audio_data, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                # WAV might be float or int
                # We'll interpret the wave as float32 first (common for your case):
                arr = np.frombuffer(frames, dtype=np.float32)
        else:
            raise TypeError(f"Cannot deal with audio_data of type {type(audio_data)}.")

        # If the array is floating, scale to int16
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)

        return arr

    def _playback_callback(self, in_data, frame_count, time_info, status_flags):
        """
        Called by PyAudio whenever the output device needs `frame_count` frames.
        We read from self.current_buffer if there's data left, otherwise we pop
        from the playback_queue. If everything is done or we are stopped, we fill
        with silence.
        """
        # If we got a "stop playback" signal, just output silence
        if self._stop_playback_event.is_set():
            silence = b"\x00" * (frame_count * self.bytes_per_frame)
            return (silence, pyaudio.paContinue)

        output_bytes_needed = frame_count * self.bytes_per_frame
        output_data = bytearray()

        while len(output_data) < output_bytes_needed:
            # 1) If we have leftover silence
            if self.leftover_silence_frames > 0:
                frames_to_write = min(
                    self.leftover_silence_frames,
                    (output_bytes_needed - len(output_data)) // self.bytes_per_frame
                )
                silent_chunk = b"\x00" * (frames_to_write * self.bytes_per_frame)
                output_data.extend(silent_chunk)
                self.leftover_silence_frames -= frames_to_write

                if len(output_data) >= output_bytes_needed:
                    break

            # 2) If we have a buffer currently playing
            if self.current_buffer and self.current_pos < len(self.current_buffer):
                bytes_left = len(self.current_buffer) - self.current_pos
                bytes_to_copy = min(bytes_left, output_bytes_needed - len(output_data))
                output_data.extend(
                    self.current_buffer[self.current_pos : self.current_pos + bytes_to_copy]
                )
                self.current_pos += bytes_to_copy

                if self.current_pos >= len(self.current_buffer):
                    # Done with this buffer; add 1s of silence
                    self.leftover_silence_frames = self.sample_rate
                    self.current_buffer = b""
                    self.current_pos = 0

                if len(output_data) >= output_bytes_needed:
                    break
            else:
                # 3) Fetch the next buffer from the queue if available
                if not self.playback_queue.empty():
                    in_sr, arr = self.playback_queue.get_nowait()
                    self.current_buffer = self._prepare_audio_for_playback(
                        arr, in_sample_rate=in_sr, out_sample_rate=self.sample_rate
                    )
                    self.current_pos = 0
                else:
                    # 4) No data: fill with silence
                    needed = output_bytes_needed - len(output_data)
                    output_data.extend(b"\x00" * needed)
                    break

        return (bytes(output_data), pyaudio.paContinue)

    def _record_callback(self, in_data, frame_count, time_info, status_flags):
        """
        Called by PyAudio whenever we have recorded input frames.
        If we haven't stopped recording, we push them into record_queue.
        """
        if not self._stop_record_event.is_set():
            self.record_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def _prepare_audio_for_playback(
        self,
        audio_array: np.ndarray,
        in_sample_rate: int,
        out_sample_rate: int
    ) -> bytes:
        """
        Resamples the numpy array if needed, ensures it's int16, and returns as raw bytes.
        """
        if in_sample_rate != out_sample_rate:
            new_length = int(len(audio_array) * (out_sample_rate / in_sample_rate))
            audio_array = resample(audio_array, new_length)

        if audio_array.dtype != np.int16:
            audio_array = audio_array.astype(np.int16)

        return audio_array.tobytes()

    # =========================================================================
    #                        DEVICE DISCOVERY
    # =========================================================================

    def _choose_default_input_device(self) -> int:
        """Pick an input device that has 'default' in its name, otherwise the first valid input device."""
        device_count = self.audio.get_device_count()
        fallback_index = None
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                if "default" in info["name"].lower():
                    return i
                if fallback_index is None:
                    fallback_index = i
        if fallback_index is None:
            raise RuntimeError("No valid input devices found.")
        return fallback_index

    def _choose_default_output_device(self) -> int:
        """Pick an output device that has 'default' in its name, otherwise the first valid output device."""
        device_count = self.audio.get_device_count()
        fallback_index = None
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                if "default" in info["name"].lower():
                    return i
                if fallback_index is None:
                    fallback_index = i
        if fallback_index is None:
            raise RuntimeError("No valid output devices found.")
        return fallback_index

    def _is_valid_device_index(self, index: int, input_device: bool) -> bool:
        """Quick check whether index is within range and supports input or output."""
        if index < 0 or index >= self.audio.get_device_count():
            return False
        info = self.audio.get_device_info_by_index(index)
        if input_device and info["maxInputChannels"] < 1:
            return False
        if not input_device and info["maxOutputChannels"] < 1:
            return False
        return True

    def _print_all_devices(self):
        """Debug utility: print all input and output devices."""
        count = self.audio.get_device_count()
        print("All Audio Devices:")
        for i in range(count):
            info = self.audio.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']} (InCh={info['maxInputChannels']}, "
                  f"OutCh={info['maxOutputChannels']}, Rate={info['defaultSampleRate']})")
