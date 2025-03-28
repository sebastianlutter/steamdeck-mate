import time
import threading
import logging
from io import BytesIO
import wave
import pyaudio
import queue
import asyncio
import numpy as np
from typing import AsyncGenerator, Any, Tuple
from scipy.signal import resample

from mate.audio.soundcard_interface import AudioInterface


class SoundCard(AudioInterface):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.sample_format: int = pyaudio.paInt16
        self.audio: pyaudio.PyAudio = pyaudio.PyAudio()

        if self.audio_playback_device is None:
            self.choose_default_playback()
        if self.audio_microphone_device is None:
            self.choose_default_microphone()

        if not self.is_valid_device_index(self.audio_microphone_device, input_device=True):
            self.logger.info("Available devices:")
            self.list_devices()
            raise Exception(
                f"Error: The microphone device index '{self.audio_microphone_device}' is invalid or not available."
            )
        if not self.is_valid_device_index(self.audio_playback_device, input_device=False):
            self.logger.info("Available devices:")
            self.list_devices()
            raise Exception(
                f"Error: The playback device index '{self.audio_playback_device}' is invalid or not available."
            )

        self.logger.info("Available devices:")
        self.list_devices()
        self.logger.info(
            "Loading device: microphone=%s, playback=%s",
            self.audio_microphone_device,
            self.audio_playback_device,
        )

        device_info = self.audio.get_device_info_by_index(self.audio_playback_device)
        self.frames_per_buffer: int = 1024
        self.bytes_per_frame: int = 2

        self.playback_queue: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue()
        self.record_queue: "queue.Queue[bytes]" = queue.Queue()
        self.recording_active = threading.Event()

        self.stop_signal_playback = threading.Event()
        self.stop_signal_record = threading.Event()

        self.current_buffer: bytes = b""
        self.current_pos: int = 0
        self.leftover_silence_frames: int = 0

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

        self.playback_stream.start_stream()
        self.record_stream.start_stream()

    def _playback_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: Any,
        status_flags: int
    ) -> Tuple[bytes, int]:
        if self.stop_signal_playback.is_set():
            silence = b"\x00" * (frame_count * self.bytes_per_frame)
            return (silence, pyaudio.paContinue)

        output_bytes_needed = frame_count * self.bytes_per_frame
        output_data = bytearray()

        while len(output_data) < output_bytes_needed:
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

            if self.current_buffer and self.current_pos < len(self.current_buffer):
                bytes_left = len(self.current_buffer) - self.current_pos
                bytes_to_copy = min(bytes_left, output_bytes_needed - len(output_data))
                output_data.extend(
                    self.current_buffer[self.current_pos : self.current_pos + bytes_to_copy]
                )
                self.current_pos += bytes_to_copy
                if self.current_pos >= len(self.current_buffer):
                    self.leftover_silence_frames = self.sample_rate
                    self.current_buffer = b""
                    self.current_pos = 0
                if len(output_data) >= output_bytes_needed:
                    break
            else:
                if not self.playback_queue.empty():
                    sr, audio_array = self.playback_queue.get_nowait()
                    self.current_buffer = self._prepare_audio_for_playback(
                        audio_array,
                        in_sample_rate=sr,
                        out_sample_rate=self.sample_rate
                    )
                    self.current_pos = 0
                else:
                    needed = output_bytes_needed - len(output_data)
                    output_data.extend(b"\x00" * needed)
                    break

        return (bytes(output_data), pyaudio.paContinue)

    def _record_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: Any,
        status_flags: int
    ) -> Tuple[None, int]:
        if self.recording_active.is_set() and not self.stop_signal_record.is_set():
            self.record_queue.put(in_data)
        return (None, pyaudio.paContinue)

    async def get_record_stream(self) -> AsyncGenerator[bytes, None]:
        self.logger.debug(
            "Has been called: Soundcard active=%s stopped=%s",
            self.record_stream.is_active(),
            self.record_stream.is_stopped()
        )
        self.stop_signal_record.clear()
        self.recording_active.set()
        try:
            self.logger.debug(
                "Try queue.empty=%s rec_active=%s",
                self.record_queue.empty(),
                self.recording_active.is_set()
            )
            while not self.stop_signal_record.is_set():
                if not self.record_queue.empty():
                    chunk = self.record_queue.get()
                    yield chunk
                else:
                    await asyncio.sleep(0.01)
        finally:
            self.logger.debug("soundcard_pyaudio.get_record_stream: generator exit.")
            self.recording_active.clear()
            while not self.record_queue.empty():
                self.record_queue.get()

    def stop_recording(self) -> None:
        self.stop_signal_record.set()
        while not self.record_queue.empty():
            self.record_queue.get()
        self.logger.debug("Recording stopped and stream closed.")

    def play_audio(self, sample_rate: int, audio_data: Any) -> None:
        if not isinstance(audio_data, np.ndarray):
            if isinstance(audio_data, bytes):
                audio_data = BytesIO(audio_data)
            if isinstance(audio_data, BytesIO):
                audio_data.seek(0)
                with wave.open(audio_data, "rb") as wav_file:
                    audio_frames = wav_file.readframes(wav_file.getnframes())
                    audio_frames = np.frombuffer(audio_frames, dtype=np.float32)
                    audio_data = (
                        (audio_frames * 32767).clip(-32768, 32767).astype(np.int16)
                    )
            else:
                raise Exception(f"Cannot deal with objects of type {type(audio_data)}")

        if np.issubdtype(audio_data.dtype, np.floating):
            audio_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)

        self.logger.debug("soundcard_pyaudio.play_audio: Adding to queue: %d bytes", len(audio_data))
        if self.stop_signal_playback.is_set():
            self.logger.debug("soundcard_pyaudio.play_audio:Unblock playback with play_audio function")
            self.stop_signal_playback.clear()
        self.playback_queue.put((sample_rate, audio_data))

    def stop_playback(self) -> None:
        self.stop_signal_playback.set()
        while not self.playback_queue.empty():
            self.playback_queue.get()
        self.logger.debug("Playback stopped and stream closed.")

    def _prepare_audio_for_playback(
        self,
        audio_array: np.ndarray,
        in_sample_rate: int,
        out_sample_rate: int
    ) -> bytes:
        if in_sample_rate != out_sample_rate:
            new_length = int(len(audio_array) * (out_sample_rate / in_sample_rate))
            audio_array = resample(audio_array, new_length)
        if audio_array.dtype != np.int16:
            audio_array = audio_array.astype(np.int16)
        return audio_array.tobytes()

    def choose_default_microphone(self) -> None:
        device_count = self.audio.get_device_count()
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if info["name"].lower() == "default" and info["maxInputChannels"] > 0:
                self.audio_microphone_device = i
                self.logger.debug(
                    "Chosen default input device: %d (%s)",
                    self.audio_microphone_device,
                    info["name"]
                )
                return
        raise Exception("No suitable default microphone device found.")

    def choose_default_playback(self) -> None:
        device_count = self.audio.get_device_count()
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if info["name"].lower() == "default" and info["maxOutputChannels"] > 0:
                self.audio_playback_device = i
                self.logger.debug(
                    "Chosen default output device: %d (%s)",
                    self.audio_playback_device,
                    info["name"]
                )
                return
        raise Exception("No suitable default playback device found.")

    def list_devices(self) -> None:
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

        self.logger.info("")
        self.logger.info("Microphone (Input) Devices:")
        self.logger.info("-" * 85)
        self.logger.info("| %s |", " | ".join(headers))
        self.logger.info("-" * 85)
        for idx, info in microphone_devices:
            self.logger.info(
                "| %d    | %-30s | %-8s | %-8s | %-13s |",
                idx,
                info["name"],
                info["maxInputChannels"],
                info["maxOutputChannels"],
                str(info["defaultSampleRate"])
            )
        self.logger.info("-" * 85)

        self.logger.info("")
        self.logger.info("Playback (Output) Devices:")
        self.logger.info("-" * 85)
        self.logger.info("| %s |", " | ".join(headers))
        self.logger.info("-" * 85)
        for idx, info in playback_devices:
            self.logger.info(
                "| %d    | %-30s | %-8s | %-8s | %-13s |",
                idx,
                info["name"],
                info["maxInputChannels"],
                info["maxOutputChannels"],
                str(info["defaultSampleRate"])
            )
        self.logger.info("-" * 85)
        self.logger.info("")

    def close(self) -> None:
        try:
            self.stop_playback()
        except:
            pass
        try:
            self.stop_recording()
        except:
            pass
        try:
            self.audio.close()
        except:
            pass

    def is_valid_device_index(self, index: int, input_device: bool = True) -> bool:
        if index is None or index < 0 or index >= self.audio.get_device_count():
            return False
        info = self.audio.get_device_info_by_index(index)
        if input_device and info["maxInputChannels"] < 1:
            return False
        if not input_device and info["maxOutputChannels"] < 1:
            return False
        return True

    def wait_until_playback_finished(self) -> None:
        self.logger.debug("waiting for playback to finish")
        while True:
            while not self.playback_queue.empty():
                time.sleep(0.1)

            while self.current_buffer or self.leftover_silence_frames > 0:
                time.sleep(0.1)

            silence_start = time.time()
            while (time.time() - silence_start) < 1.0:
                if not self.playback_queue.empty() or self.current_buffer or self.leftover_silence_frames > 0:
                    break
                time.sleep(0.05)
            else:
                break
        self.logger.debug("waiting for playback finished is done")
