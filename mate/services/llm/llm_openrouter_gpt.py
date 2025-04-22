import logging
import os
from typing import AsyncGenerator, Dict, List, Optional, Union
from urllib.parse import urlparse

import aiohttp
from openai import AsyncOpenAI, OpenAI

from mate.services.llm.llm_interface import LlmInterface


class LlmOpenrouterGpt(LlmInterface):
    """
    Implementation of LLM interface for OpenRouter GPT models.
    """

    def __init__(self, name: str, priority: int, model: str, ):
        super().__init__(name, priority)
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model
        self.client = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.api_key:
            self.logger.warning("OpenRouter API key not set")

    async def check_availability(self) -> bool:
        """
        Check if the OpenRouter API is available and properly configured.
        """
        if not self.api_key:
            self.logger.debug("OpenRouter API key not set")
            return False
        # Check if the base URL is reachable
        parsed = urlparse(self.base_url)
        host = parsed.hostname
        if not host:
            return False
        try:
            # Create a temporary client to test connectivity
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5,
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        self.logger.warning(
                            f"OpenRouter API returned status {response.status}"
                        )
                        return False
        except Exception as e:
            self.logger.warning(f"Failed to connect to OpenRouter API: {e}")
            return False

    def config_str(self) -> str:
        """
        Return a string representation of the configuration.
        """
        return (
            f"OpenRouter GPT: {self.model}, "
            f"API URL: {self.base_url}, "
            f"Site: {self.site_name}"
        )

    def _init_client(self):
        """Initialize the OpenAI client with OpenRouter configuration"""
        if not self.client:
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

    async def chat(self, full_chat) -> AsyncGenerator[str, None]:
        """
        Send a chat request to OpenRouter and stream the response.

        Args:
            full_chat: List of message dictionaries with 'role' and 'content' keys
                       Content can be text or a list of content parts (for multimodal)

        Yields:
            Chunks of the generated response as they become available
        """
        self._init_client()

        try:
            # Prepare messages for the API
            messages = self._prepare_messages(full_chat)

            # Create the streaming completion
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                extra_headers={
                },
            )

            # Yield chunks as they arrive
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.logger.error(f"Error in OpenRouter chat: {e}")
            yield f"\nError: Failed to get response from OpenRouter: {str(e)}"

    def _prepare_messages(self, full_chat):
        """
        Prepare messages for the OpenRouter API, handling both simple text
        and multimodal content.
        """
        messages = []

        for msg in full_chat:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle different content formats
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # This is multimodal content
                processed_content = []

                for item in content:
                    if item.get("type") == "text":
                        processed_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item.get("type") == "image_url":
                        processed_content.append({
                            "type": "image_url",
                            "image_url": item.get("image_url", {})
                        })

                messages.append({"role": role, "content": processed_content})

        return messages