# soundcard_interface.py
import os
import threading
from abc import ABC, abstractmethod
import numpy as np
from typing import BinaryIO, List, Callable, Generator, AsyncGenerator


class AudioInterface(ABC):

    """
    A metaclass that combines ABCMeta and Singleton logic.
    This metaclass ensures that only one instance of any class using it is created.
    Only the first constructor call creates an instance, the other get the same reference
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # Check if an instance already exists
        if cls not in cls._instances:
            # Call ABCMeta.__call__ to create the instance (this respects ABC constraints)
            instance = super(AudioInterface, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def __init__(self):
        self.frames_per_buffer = 512
        self.sample_rate = 16000
        self.input_channels: int = 1
        self.bytes_per_frame = 2
        # Read environment variables
        self.audio_microphone_device = int(os.getenv('AUDIO_MICROPHONE_DEVICE', '-1'))
        if self.audio_microphone_device < 0:
            self.audio_microphone_device = None
        self.audio_playback_device = int(os.getenv('AUDIO_PLAYBACK_DEVICE', '-1'))
        if self.audio_playback_device < 0:
            self.audio_playback_device = None
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
    async def get_record_stream(self)  -> AsyncGenerator[bytes, None]:
        """
        Open a recording stream (or equivalent object) for capturing audio from the currently selected microphone device.

        :return: A stream or device handle suitable for reading raw audio data frames.
        :raises RuntimeError: If no valid microphone device is configured.
        """
        pass

    @abstractmethod
    def stop_recording(self):
        pass

    @abstractmethod
    def stop_playback(self):
        pass

    @abstractmethod
    def play_audio(self, sample_rate, audio_buffer):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def wait_until_playback_finished(self):
        pass

    def config_str(self):
        return f'Soundcard device: microphone={self.audio_microphone_device}, playback: {self.audio_playback_device}'


    def inspect_ndarray(self, array: np.ndarray, name: str = "Array"):
        """
        Inspect and print useful information about a given ndarray.

        Args:
            array (np.ndarray): The ndarray to inspect.
            name (str): An optional name for the array (useful for labeling in print statements).
        """
        try:
            print(f"--- {name} Information ---")
            print(f"Shape: {array.shape}")
            print(f"Data Type: {array.dtype}")
            print(f"Min Value: {np.min(array)}")
            print(f"Max Value: {np.max(array)}")
            # For floating-point arrays, check if values are in the expected range
            if np.issubdtype(array.dtype, np.floating):
                print(f"Mean Value: {np.mean(array)}")
                print("Expected range for floats:")
                if np.min(array) >= -1.0 and np.max(array) <= 1.0:
                    print("Values are in the range -1.0 to 1.0 (normalized float audio).")
                elif np.min(array) >= 0.0 and np.max(array) <= 1.0:
                    print("Values are in the range 0.0 to 1.0 (possibly normalized float).")
                else:
                    print("Values are outside common float ranges (requires inspection).")
            # For integer arrays, show the range of the integer type
            elif np.issubdtype(array.dtype, np.integer):
                dtype_info = np.iinfo(array.dtype)
                print(f"Integer Range: {dtype_info.min} to {dtype_info.max}")
                if np.min(array) >= 0 and np.max(array) <= 255:
                    print("Values are in the range 0 to 255 (8-bit unsigned integer audio).")
                elif np.min(array) >= dtype_info.min and np.max(array) <= dtype_info.max:
                    print("Values fit within the expected range for the integer type.")
                else:
                    print("Values are outside the expected range for this integer type.")
            print("--- End of Inspection ---\n")
        except Exception as e:
            print(f"Error while inspecting {name}: {e}")
