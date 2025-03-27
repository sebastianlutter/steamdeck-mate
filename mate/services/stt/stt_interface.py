import os
from abc import ABC, abstractmethod
from typing import Generator, AsyncGenerator, Callable


class STTInterface(ABC):

    def __init__(self, name: str, priority: int):
        super().__init__(name, "STT", priority)
        self.stt_endpoint = os.getenv('STT_ENDPOINT', 'http://127.0.0.1:8000/v1/audio/transcriptions')

    @abstractmethod
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], websocket_on_close: Callable[[], None], websocket_on_open: Callable[[], None]) -> AsyncGenerator[str, None]:
        pass

    def config_str(self):
        return f'endpoint: {self.stt_endpoint}'