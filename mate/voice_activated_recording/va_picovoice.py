import sys
import os
import logging
import pvporcupine
import numpy as np
import threading
from typing import Optional
from mate.voice_activated_recording.va_interface import VoiceActivationInterface


class PorcupineWakeWord(VoiceActivationInterface):
    def __init__(self) -> None:
        super().__init__()
        self.logger: logging.Logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model_path: str = f"./picovoice/{self.wakeword}_de_linux_v3_0_0.ppn"

        if not os.path.isfile(self.model_path):
            self.logger.error(
                "Picovoice model file is missing. "
                "Cannot find %s for given wakeword %s",
                self.model_path,
                self.wakeword
            )
            self.logger.error("Please make an account and download one: https://picovoice.ai/")
            sys.exit(0)

        self.porcupine = pvporcupine.create(
            keyword_paths=[self.model_path],
            model_path="./picovoice/porcupine_params_de.pv",
            sensitivities=[self.wakeword_threshold / 500.0],
            access_key=os.getenv("PICOVOICE_ACCESS_KEY")
        )

    async def listen_for_wake_word(self, stop_signal: Optional[threading.Event] = None) -> None:
        try:
            self.logger.info("Listening for wake word: %s", self.wakeword)
            buffer: list[int] = []
            async for chunk in self.soundcard.get_record_stream():
                pcm = np.frombuffer(chunk, dtype=np.int16)
                buffer.extend(pcm.tolist())
                while len(buffer) >= self.porcupine.frame_length:
                    frame = np.array(buffer[: self.porcupine.frame_length], dtype=np.int16)
                    buffer = buffer[self.porcupine.frame_length :]
                    result = self.porcupine.process(frame)
                    if result >= 0:
                        self.logger.info("Wake word '%s' detected!", self.wakeword)
                        if stop_signal is not None:
                            stop_signal.set()
                        return
        finally:
            pass
