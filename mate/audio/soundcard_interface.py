import os
import threading
import logging
import numpy as np
from abc import ABC, abstractmethod, ABCMeta
from typing import Optional, AsyncGenerator

class SingletonABCMeta(ABCMeta):
    """
    A metaclass that combines Singleton logic with ABCMeta functionality.
    Ensures that only one instance of any class using this metaclass is created.
    """
    _instances = {}
    _lock = threading.Lock()  # Ensure thread safety for singleton creation

    def __call__(cls, *args, **kwargs):
        # First check if an instance already exists (fast path)
        if cls not in cls._instances:
            with cls._lock:
                # Double-check inside the lock to avoid race conditions
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

class AudioInterface(ABC, metaclass=SingletonABCMeta):

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.frames_per_buffer: int = 512
        self.sample_rate: int = 16000
        self.input_channels: int = 1
        self.bytes_per_frame: int = 2

        env_mic_device = int(os.getenv("AUDIO_MICROPHONE_DEVICE", "-1"))
        self.audio_microphone_device: Optional[int] = None if env_mic_device < 0 else env_mic_device

        env_playback_device = int(os.getenv("AUDIO_PLAYBACK_DEVICE", "-1"))
        self.audio_playback_device: Optional[int] = None if env_playback_device < 0 else env_playback_device

        self.stop_signal_record = threading.Event()
        self.start_signal_record = threading.Event()

    @abstractmethod
    def list_devices(self) -> None:
        """
        List all available audio devices. This should display separate lists for input and output devices,
        along with relevant details such as device index, name, supported channels, and sample rates.
        """
        pass

    @abstractmethod
    def is_valid_device_index(self, index: int, input_device: bool = True) -> bool:
        """
        Check if the given device index is valid and can be used as an input or output device.

        :param index: The device index to validate.
        :param input_device: If True, checks for input capability; if False, checks for output capability.
        :return: True if the device index is valid for the requested direction, False otherwise.
        """
        pass

    @abstractmethod
    async def get_record_stream(self) -> AsyncGenerator[bytes, None]:
        """
        Open a recording stream for capturing audio from the currently selected microphone device.

        :return: An async generator yielding raw audio data frames as bytes.
        :raises RuntimeError: If no valid microphone device is configured.
        """
        pass

    @abstractmethod
    def stop_recording(self) -> None:
        pass

    @abstractmethod
    def stop_playback(self) -> None:
        pass

    @abstractmethod
    def play_audio(self, sample_rate: int, audio_buffer: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def wait_until_playback_finished(self) -> None:
        pass

    def config_str(self) -> str:
        return (
            f"Soundcard device: microphone={self.audio_microphone_device}, "
            f"playback={self.audio_playback_device}"
        )

    def inspect_ndarray(self, array: np.ndarray, name: str = "Array") -> None:
        """
        Inspect and log useful information about a given ndarray.
        """
        try:
            self.logger.info("--- %s Information ---", name)
            self.logger.info("Shape: %s", array.shape)
            self.logger.info("Data Type: %s", array.dtype)
            self.logger.info("Min Value: %s", np.min(array))
            self.logger.info("Max Value: %s", np.max(array))

            if np.issubdtype(array.dtype, np.floating):
                self.logger.info("Mean Value: %s", np.mean(array))
                self.logger.info("Expected range for floats:")
                if np.min(array) >= -1.0 and np.max(array) <= 1.0:
                    self.logger.info("Values are in the range -1.0 to 1.0 (normalized float audio).")
                elif np.min(array) >= 0.0 and np.max(array) <= 1.0:
                    self.logger.info("Values are in the range 0.0 to 1.0 (possibly normalized float).")
                else:
                    self.logger.warning("Values are outside common float ranges (requires inspection).")
            elif np.issubdtype(array.dtype, np.integer):
                dtype_info = np.iinfo(array.dtype)
                self.logger.info("Integer Range: %d to %d", dtype_info.min, dtype_info.max)
                if np.min(array) >= 0 and np.max(array) <= 255:
                    self.logger.info("Values are in the range 0 to 255 (8-bit unsigned integer audio).")
                elif np.min(array) >= dtype_info.min and np.max(array) <= dtype_info.max:
                    self.logger.info("Values fit within the expected range for the integer type.")
                else:
                    self.logger.warning("Values are outside the expected range for this integer type.")
            self.logger.info("--- End of Inspection ---\n")
        except Exception as e:
            self.logger.exception("Error while inspecting %s: %s", name, e)
