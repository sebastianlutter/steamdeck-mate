import os
import threading
from abc import ABC, abstractmethod
from typing import Optional

from mate.audio.soundcard_pyaudio import SoundCard


class VoiceActivationInterface(ABC):

    def __init__(self):
        super().__init__()
        self.wakeword = os.getenv('WAKEWORD', 'computer')
        self.wakeword_threshold = int(os.getenv('WAKEWORD_THRESHOLD', '250'))
        # Configurable delay before counting silence
        self.silence_lead_time = 2
        self.soundcard = SoundCard()

    @abstractmethod
    async def listen_for_wake_word(self, stop_signal: Optional[threading.Event] = None):
        """
        This function should block until the wakeword has been detected
        """
        pass

    def config_str(self):
        return f'wakeword: {self.wakeword}, threshold: {self.wakeword_threshold}'
