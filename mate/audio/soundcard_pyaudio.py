import time
import threading
import logging
from io import BytesIO
import wave
import pyaudio
import queue
import asyncio
import logging
import numpy as np
from typing import AsyncGenerator
from mate.audio.soundcard_interface import AudioInterface
from scipy.signal import resample


class SoundCard(AudioInterface):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Use 16-bit PCM for better ALSA compatibility
        self.sample_format = pyaudio.paInt16
        # Create an interface to PortAudio
        self.audio = pyaudio.PyAudio()

        # If device number is None, choose "default" device by name
        if self.audio_playback_device is None:
            self.choose_default_playback()
        if self.audio_microphone_device is None:
            self.choose_default_microphone()

        # Validate microphone device index
        if not self.is_valid_device_index(self.audio_microphone_device, input_device=True):
            print("Available devices:")
            self.list_devices()
            raise Exception(
                f"Error: The microphone device index '{self.audio_microphone_device}' "
                "is invalid or not available."
            )
        # Validate playback device index
        if not self.is_valid_device_index(self.audio_playback_device, input_device=False):
            print("Available devices:")
            self.list_devices()
            raise Exception(
                f"Error: The playback device index '{self.audio_playback_device}' "
                "is invalid or not available."
            )

        print("Available devices:")
        self.list_devices()
        self.logger.info(f"Loading device: microphone={self.audio_microphone_device}, "
              f"playback={self.audio_playback_device}")

        # Use the default sample rate from the playback device
        device_info = self.audio.get_device_info_by_index(self.audio_playback_device)
        #self.sample_rate = int(device_info["defaultSampleRate"])
        # frames_per_buffer can be tuned (e.g., 512, 1024, 2048)
        # Larger buffers are more stable (fewer underruns), but higher latency
        self.frames_per_buffer = 1024
        self.bytes_per_frame = 2  # 16-bit => 2 bytes

        # -------------------------------------------------------------
        #  Queues for playback and recording
        # -------------------------------------------------------------
        # Playback queue: items are (sample_rate: int, np.array)
        self.playback_queue = queue.Queue()
        # For recording data from the callback
        self.record_queue = queue.Queue()
        self.recording_active = threading.Event()

        # Stop signals
        self.stop_signal_playback = threading.Event()
        self.stop_signal_record = threading.Event()

        # -------------------------------------------------------------
        #  Playback callback state
        # -------------------------------------------------------------
        self.current_buffer = b""  # the current audio data being played
        self.current_pos = 0       # how many bytes of current_buffer have been played so far
        self.leftover_silence_frames = 0  # frames of silence to play after finishing an item

        # -------------------------------------------------------------
        #  Open the playback stream (callback mode)
        # -------------------------------------------------------------
        self.playback_stream = self.audio.open(
            format=self.sample_format,
            channels=self.input_channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.audio_playback_device,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self._playback_callback,  # Our callback
            start=False,  # We'll explicitly start later
        )

        # -------------------------------------------------------------
        #  Open the recording stream (callback mode)
        # -------------------------------------------------------------
        # We assume we want the same format/rate for recording
        self.record_stream = self.audio.open(
            format=self.sample_format,
            channels=self.input_channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.audio_microphone_device,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self._record_callback,  # Our callback
            start=False,  # We'll explicitly start later
        )

        # Start the streams
        self.playback_stream.start_stream()
        self.record_stream.start_stream()

    ###########################################################################
    #                           PyAudio Callbacks
    ###########################################################################

    def _playback_callback(self, in_data, frame_count, time_info, status_flags):
        """
        This callback is invoked by PyAudio/PortAudio whenever the output device
        needs `frame_count` frames of audio. We must provide exactly that many
        frames worth of bytes (channels * sample_width * frame_count).

          1. If we are in the middle of a buffer, continue reading from it.
          2. If we finished a buffer and have leftover silence to play (the 1-sec gap),
             we fill frames with silence until that gap is done.
          3. If there's no leftover gap, we pop the next item from the playback queue.
          4. If the queue is empty, fill with silence.
        """
        # If stop signal is set, return silence with paComplete or paAbort
        if self.stop_signal_playback.is_set():
            return (b"\x00" * (frame_count * self.bytes_per_frame), pyaudio.paContinue)
        output_bytes_needed = frame_count * self.bytes_per_frame
        # We'll accumulate data in a local bytearray
        output_data = bytearray()
        while len(output_data) < output_bytes_needed:
            # 1) If we're in leftover silence mode, produce silent frames first
            if self.leftover_silence_frames > 0:
                frames_to_write = min(self.leftover_silence_frames,  # how many silent frames remain
                                      (output_bytes_needed - len(output_data)) // self.bytes_per_frame)
                silent_chunk = b"\x00" * (frames_to_write * self.bytes_per_frame)
                output_data.extend(silent_chunk)
                self.leftover_silence_frames -= frames_to_write

                # If we still need more data after filling some silence, continue
                if len(output_data) >= output_bytes_needed:
                    break
                # Otherwise, leftover_silence_frames = 0 => move on to next item
            # 2) If we have a current buffer we haven't finished playing
            if self.current_buffer and self.current_pos < len(self.current_buffer):
                bytes_left = len(self.current_buffer) - self.current_pos
                bytes_to_copy = min(bytes_left, output_bytes_needed - len(output_data))
                output_data.extend(
                    self.current_buffer[self.current_pos:self.current_pos + bytes_to_copy]
                )
                self.current_pos += bytes_to_copy

                if self.current_pos >= len(self.current_buffer):
                    # We finished this item -> set leftover_silence_frames for 1 second
                    self.leftover_silence_frames = self.sample_rate  # 1 second of frames
                    self.current_buffer = b""
                    self.current_pos = 0
                # If we still need more data, continue; else break
                if len(output_data) >= output_bytes_needed:
                    break
            else:
                # 3) Current buffer is empty or fully played; fetch next item from queue
                if not self.playback_queue.empty():
                    # Next item: (sample_rate, numpy_array)
                    sample_rate, audio_array = self.playback_queue.get_nowait()
                    self.current_buffer = self._prepare_audio_for_playback(
                        audio_array,
                        in_sample_rate=sample_rate,
                        out_sample_rate=self.sample_rate
                    )
                    self.current_pos = 0
                    # Continue loop so we copy from the new buffer
                else:
                    # 4) Nothing in queue => fill with silence
                    needed = output_bytes_needed - len(output_data)
                    output_data.extend(b"\x00" * needed)
                    break
        return (bytes(output_data), pyaudio.paContinue)

    def _record_callback(self, in_data, frame_count, time_info, status_flags):
        """
        Called whenever there's `frame_count` frames of audio from the microphone.
        We'll push it into `self.record_queue`. The async generator will eventually
        retrieve it.
        """
        # store in_data in the queue when recording capture is active
        if self.recording_active and not self.stop_signal_record.is_set():
            self.record_queue.put(in_data)
        return (None, pyaudio.paContinue)

    ###########################################################################
    #                 Recording Methods (Async Generator)
    ###########################################################################

    async def get_record_stream(self) -> AsyncGenerator[bytes, None]:
        """
        Provides an async generator that yields recorded audio data.
        The data is fed from the `_record_callback` into self.record_queue.
        """
        self.logger.debug(f"Has been called: Soundcard active={self.record_stream.is_active()} stopped={self.record_stream.is_stopped()}")
        self.stop_signal_record.clear()  # Reset stop signal if it was set
        self.recording_active.set()
        try:
            self.logger.debug(f"Try queue.emtpy={self.record_queue.empty()} rec_active={self.recording_active.is_set()}")
            while not self.stop_signal_record.is_set():
                if not self.record_queue.empty():
                    chunk = self.record_queue.get()
                    yield chunk
                else:
                    await asyncio.sleep(0.01)
        finally:
            # When the caller stops iteration, or stop_signal_record is set
            self.logger.debug("soundcard_pyaudio.get_record_stream: generator exit.")
            # Stop capturing when the generator is no longer in use
            self.recording_active.clear()
            # Clear the queue
            while not self.record_queue.empty():
                self.record_queue.get()

    def stop_recording(self):
        """
        Signal the record callback to complete, drain the queue, and stop the stream.
        """
        self.stop_signal_record.set()
        # Wait for the stream to actually finish
        while not self.record_queue.empty():
            self.record_queue.get()
        self.logger.debug("Recording stopped and stream closed.")

    ###########################################################################
    #                          Playback Methods
    ###########################################################################

    def play_audio(self, sample_rate: int, audio_array):
        """
        Enqueue the audio array for playback. The callback will handle retrieval.
        """
        # Check if array is not already numpy ndarray
        if not isinstance(audio_array, np.ndarray):
            if isinstance(audio_array, bytes):
                audio_array = BytesIO(audio_array)
            if isinstance(audio_array, BytesIO):
                # so it is a BytesIO object containing a WAV file. Convert to numpy raw PCM data (without WAV header)
                # Extract raw PCM data from the WAV buffer
                audio_array.seek(0)  # Reset the buffer pointer to the beginning
                with wave.open(audio_array, 'rb') as wav_file:
                    # Read audio frames and convert them to a NumPy array
                    audio_frames = wav_file.readframes(wav_file.getnframes())
                    audio_frames = np.frombuffer(audio_frames, dtype=np.float32)
                    # Ensure the audio is in the correct range for int16 playback
                    # Scale float data (-1.0 to 1.0) to int16 range
                    audio_array = (audio_frames * 32767).clip(-32768, 32767).astype(np.int16)
                    #audio_array = np.array(new_audio_array)
            else:
                raise Exception(f"Cannot deal with objects of type {type(audio_array)}")
        # Ensure the audio is in the correct range for int16 playback
        if np.issubdtype(audio_array.dtype, np.floating):
            # Scale float data (-1.0 to 1.0) to int16 range
            audio_array = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)
        self.logger.debug(f"soundcard_pyaudio.play_audio: Adding to queue: {len(audio_array)} bytes")
        # If we previously set stop_signal_playback, clear it:
        if self.stop_signal_playback.is_set():
            self.logger.debug("soundcard_pyaudio.play_audio:Unblock playback with play_audio function")
            self.stop_signal_playback.clear()
        self.playback_queue.put((sample_rate, audio_array))

    def stop_playback(self):
        """
        Signal the playback callback to stop immediately, clear the queue, and close the stream.
        """
        self.stop_signal_playback.set()
        # Clear the queue
        while not self.playback_queue.empty():
            self.playback_queue.get()
        #if self.playback_stream.is_active():
        #    self.playback_stream.stop_stream()
        #self.playback_stream.close()
        self.logger.debug("Playback stopped and stream closed.")

    ###########################################################################
    #          Audio Format Conversion / Utilities for Playback
    ###########################################################################

    def _prepare_audio_for_playback(
        self,
        audio_array: np.ndarray,
        in_sample_rate: int,
        out_sample_rate: int
    ) -> bytes:
        """
        Convert a NumPy array of audio samples to match the playback stream:
          - Resample from in_sample_rate to out_sample_rate if needed
          - Convert to 16-bit integer if needed
          - Return raw bytes
        """
        if in_sample_rate != out_sample_rate:
            # Use scipy.signal.resample for a naive approach
            new_length = int(len(audio_array) * (out_sample_rate / in_sample_rate))
            audio_array = resample(audio_array, new_length)
        # Ensure int16 for paInt16
        if audio_array.dtype != np.int16:
            # convert the audio data
            audio_array = audio_array.astype(np.int16)

        return audio_array.tobytes()

    ###########################################################################
    #                      Device Selection and Validation
    ###########################################################################

    def choose_default_microphone(self):
        """Automatically chooses an input device whose name contains 'default'."""
        device_count = self.audio.get_device_count()
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if "default" == info["name"].lower() and info["maxInputChannels"] > 0:
                self.audio_microphone_device = i
                self.logger.debug(f"Chosen default input device: {self.audio_microphone_device} ({info['name']})")
                return
        raise Exception("No suitable default microphone device found.")

    def choose_default_playback(self):
        """Automatically chooses an output device whose name contains 'default'."""
        device_count = self.audio.get_device_count()
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if "default" == info["name"].lower() and info["maxOutputChannels"] > 0:
                self.audio_playback_device = i
                self.logger.debug(f"Chosen default output device: {self.audio_playback_device} ({info['name']})")
                return
        raise Exception("No suitable default playback device found.")

    def list_devices(self):
        """List input/output devices in separate well-formed tables."""
        device_count = self.audio.get_device_count()
        microphone_devices = []
        playback_devices = []
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                microphone_devices.append((i, info))
            if info["maxOutputChannels"] > 0:
                playback_devices.append((i, info))

        headers = ["Index", "Name", "Input Ch", "Output Ch", "Default Rate"]
        # Print Microphone Devices
        print("\nMicrophone (Input) Devices:")
        print("-" * 85)
        print(f"| {' | '.join(headers)} |")
        print("-" * 85)
        for idx, info in microphone_devices:
            print(
                f"| {idx:<5} | {info['name']:<30} "
                f"| {info['maxInputChannels']:<8} "
                f"| {info['maxOutputChannels']:<8} "
                f"| {info['defaultSampleRate']:<13} |"
            )
        print("-" * 85)

        # Print Playback Devices
        print("\nPlayback (Output) Devices:")
        print("-" * 85)
        print(f"| {' | '.join(headers)} |")
        print("-" * 85)
        for idx, info in playback_devices:
            print(
                f"| {idx:<5} | {info['name']:<30} "
                f"| {info['maxInputChannels']:<8} "
                f"| {info['maxOutputChannels']:<8} "
                f"| {info['defaultSampleRate']:<13} |"
            )
        print("-" * 85)
        print()

    def is_valid_device_index(self, index, input_device=True):
        """Check if the given device index is valid and can be used as an input or output device."""
        if index is None or index < 0 or index >= self.audio.get_device_count():
            return False
        info = self.audio.get_device_info_by_index(index)
        if input_device and info["maxInputChannels"] < 1:
            return False
        if not input_device and info["maxOutputChannels"] < 1:
            return False
        return True

    def wait_until_playback_finished(self):
        """
        Wait until the playback queue is empty, current buffer is consumed,
        leftover silence is done, AND an additional 1-second window of silence
        has passed with no new audio queued.
        """
        self.logger.debug("waiting for playback is finished")
        while True:
            # Step 1: Keep waiting as long as there is something in the queue.
            while not self.playback_queue.empty():
                time.sleep(0.1)

            # Step 2: Wait until the current buffer and leftover silence frames are consumed.
            while self.current_buffer or self.leftover_silence_frames > 0:
                time.sleep(0.1)

            # Step 3: Once buffer & leftover_silence_frames are done, wait 1 second
            #         to ensure no new audio has arrived in the queue.
            silence_start = time.time()
            while (time.time() - silence_start) < 1.0:
                # If new data was queued or leftover_silence_frames got replenished,
                # break and loop again.
                if not self.playback_queue.empty() or self.current_buffer or self.leftover_silence_frames > 0:
                    break
                time.sleep(0.05)
            else:
                # We had a full second of silence (i.e., no break above),
                # so we're truly done.
                break
        self.logger.debug("waiting for playback finished is done")
