import os
import aiohttp
import asyncio
import logging
from typing import AsyncGenerator, Callable, List, Dict, Any, Optional
from urllib.parse import urlparse

from mate.services.stt.stt_interface import STTInterface


class OpenRouterSTT(STTInterface):
    """
    Speech-to-Text service using OpenRouter API.
    """

    # List of models that support STT through OpenRouter
    AVAILABLE_MODELS = [
        "anthropic/claude-3-opus",  # Supports audio transcription
        "anthropic/claude-3-sonnet",  # Supports audio transcription
        "anthropic/claude-3-haiku",  # Supports audio transcription
        "openai/whisper",  # Dedicated STT model
        "openai/gpt-4o",  # Supports audio transcription
        "openai/gpt-4-turbo",  # Supports audio transcription
    ]

    def __init__(
        self,
        name: str,
        priority: int,
        model: str,
        language: str = "de"
    ):
        """
        Initialize the OpenRouter STT service.

        Args:
            name: Service name
            priority: Service priority
            model: Model to use for transcription
            language: Target language for transcription (default: German)
        """
        super().__init__(name, priority)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            self.logger.warning("OPENROUTER_API_KEY environment variable is not set")

        self.api_url = "https://openrouter.ai/api/v1/audio/transcriptions"
        self.model = model
        self.language = language

        # Validate model
        if model not in self.AVAILABLE_MODELS:
            self.logger.warning(f"Model {model} not in known supported models: {', '.join(self.AVAILABLE_MODELS)}")

    async def check_availability(self) -> bool:
        """
        Check if the OpenRouter API is available and the API key is set.

        Returns:
            bool: True if the service is available, False otherwise
        """
        if not self.api_key:
            self.logger.debug("OpenRouter API key not set")
            return False

        # Check if the API endpoint is reachable
        parsed = urlparse(self.api_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        try:
            # Try to connect to the host
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=2
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception as e:
            self.logger.debug(f"Failed to connect to OpenRouter API: {str(e)}")
            return False

    def config_str(self) -> str:
        """
        Return a string representation of the service configuration.

        Returns:
            str: Configuration string
        """
        return (
            f"OpenRouter STT Service\n"
            f"  Model: {self.model}\n"
            f"  Language: {self.language}\n"
            f"  API URL: {self.api_url}\n"
            f"  API Key: {'Set' if self.api_key else 'Not Set'}"
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        websocket_on_close: Callable[[], None],
        websocket_on_open: Callable[[], None]
    ) -> AsyncGenerator[str, None]:
        """
        Transcribe an audio stream using OpenRouter API.

        Args:
            audio_stream: Async generator yielding audio chunks
            websocket_on_close: Callback when websocket closes
            websocket_on_open: Callback when websocket opens

        Yields:
            Transcribed text segments
        """
        try:
            websocket_on_open()

            # Buffer to accumulate audio chunks
            audio_buffer = bytearray()

            # Collect audio chunks
            async for chunk in audio_stream:
                audio_buffer.extend(chunk)

            # If we have audio data, send it for transcription
            if audio_buffer:
                async for text in self._transcribe_audio(audio_buffer):
                    yield text

        except Exception as e:
            self.logger.error(f"Error in transcribe_stream: {str(e)}")
            raise
        finally:
            websocket_on_close()

    async def _transcribe_audio(self, audio_data: bytes) -> AsyncGenerator[str, None]:
        """
        Send audio data to OpenRouter API for transcription.

        Args:
            audio_data: Complete audio data to transcribe

        Yields:
            Transcribed text
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not set")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("APP_URL", "https://mate-assistant.com"),
            "X-Title": "MATE Assistant"
        }

        # Prepare form data
        form_data = aiohttp.FormData()
        form_data.add_field(
            name="file",
            value=audio_data,
            filename="audio.webm",
            content_type="audio/webm"
        )
        form_data.add_field("model", self.model)
        form_data.add_field("language", self.language)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                data=form_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    raise Exception(f"OpenRouter API error: {response.status} - {error_text}")

                result = await response.json()
                if "text" in result:
                    yield result["text"]
                else:
                    self.logger.error(f"Unexpected response format: {result}")
                    raise Exception(f"Unexpected response format: {result}")

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Return list of available STT models on OpenRouter"""
        return cls.AVAILABLE_MODELS