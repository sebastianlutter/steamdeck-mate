import sys
import os
import logging
import pvporcupine
import numpy as np
import threading
from typing import Optional
from mate.voice_activated_recording.va_interface import VoiceActivationInterface

class PorcupineWakeWord(VoiceActivationInterface):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model_path = f'./{self.wakeword}_de_linux_v3_0_0.ppn'

        if not os.path.isfile(self.model_path):
            self.logger.error(f"Picovoice model file is missing. Cannot find {self.model_path} for given wakeword {self.wakeword}")
            self.logger.error("Please make an account and download one: https://picovoice.ai/")
            sys.exit(0)

        self.porcupine = pvporcupine.create(
            keyword_paths=[self.model_path],
            model_path='./porcupine_params_de.pv',
            sensitivities=[self.wakeword_threshold / 500.0],
            access_key=os.getenv('PICOVOICE_ACCESS_KEY')
        )

    async def listen_for_wake_word(self, stop_signal: Optional[threading.Event] = None):
        try:
            self.logger.info(f"Listening for wake word: {self.wakeword}")
            buffer = []
            async for chunk in self.soundcard.get_record_stream():
                # Convert raw PCM data to the format expected by Porcupine
                pcm = np.frombuffer(chunk, dtype=np.int16)
                buffer.extend(pcm)
                # Process in frames of the expected length
                while len(buffer) >= self.porcupine.frame_length:
                    frame = np.array(buffer[:self.porcupine.frame_length], dtype=np.int16)
                    buffer = buffer[self.porcupine.frame_length:]  # Remove processed samples
                    result = self.porcupine.process(frame)
                    if result >= 0:
                        self.logger.info(f"Wake word '{self.wakeword}' detected!")
                        if stop_signal is not None:
                            stop_signal.set()
                        return
        finally:
            pass