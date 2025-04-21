import abc
import os
import aiohttp

import logging
from urllib.parse import urlparse
from ollama import Client
from typing import Any, Dict, List, Optional, AsyncGenerator
from mate.services.llm.llm_interface import LlmInterface


class LlmOllamaRemote(LlmInterface, metaclass=abc.ABCMeta):
    def __init__(self, name: str, priority: int, endpoint: str, ollama_model: str) -> None:
        super().__init__(name, priority)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Creating instance name={name}")
        self.llm_endpoint: str = endpoint
        self.llm_provider_model: str = ollama_model
        self.client: Client = Client(host=self.llm_endpoint)
        self.model: str = self.llm_provider_model

    # Function to dynamically create a class that inherits from base_class
    async def create_class_from_config(class_name: str, base_class: type, config: Dict[str, Any]) -> type:
        # Define an __init__ that passes config parameters to the base class constructor.
        def __init__(self):
            # You can also include additional logic here if needed.
            super(new_class, self).__init__(**config)

        # Create the new class dynamically.
        new_class = type(class_name, (base_class,), {
            "config": config,  # Optional: store config as a class attribute
            "__init__": __init__
        })
        return new_class

    async def chat(self, full_chat: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        print(f"LLM CHAT: {full_chat}")
        content = self.client.chat(
            model=self.model,
            stream=True,
            messages=full_chat,
        )
        for chunk in content:
            c = chunk["message"]["content"]
            yield c

    def config_str(self) -> str:
        return f"Ollama Remote {self.name}: {self.model} on {self.llm_endpoint}."

    async def check_availability(self) -> bool:
        self.logger.debug(f"[check_availability {self.name}] Checking availability of {self.model} on {self.llm_endpoint}")

        if not await self.__check_remote_endpoint__(self.llm_endpoint):
            self.logger.debug(f"[check_availability {self.name}] Remote endpoint {self.llm_endpoint} is not reachable")
            return False

        parsed = urlparse(self.llm_endpoint)
        host: Optional[str] = parsed.hostname
        port: Optional[int] = parsed.port
        models_url: str = f"{parsed.scheme}://{host}:{port}/api/tags"

        self.logger.debug(f"[check_availability {self.name}] Checking models at {models_url}")

        try:
            timeout = aiohttp.ClientTimeout(total=2)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(models_url) as resp:
                    if resp.status != 200:
                        self.logger.warning(
                            "[check_availability %s] Could not retrieve models from %s. HTTP status: %d",
                            self.name,
                            models_url,
                            resp.status
                        )
                        self.logger.debug(f"[check_availability {self.name}] Failed with HTTP status {resp.status}, expected 200")
                        return False
                    data = await resp.json()
                    installed_models = [m.get("name") for m in data.get("models", [])]
                    self.logger.debug(f"[check_availability {self.name}] Found installed models: {installed_models}")
        except Exception as e:
            self.logger.warning(
                "[check_availability %s] Failed while calling %s. Reason: %s",
                self.name,
                models_url,
                e
            )
            self.logger.debug(f"[check_availability {self.name}] Exception details: {str(e)}")
            return False

        if self.model not in installed_models:
            self.logger.warning(
                "[check_availability %s] Model '%s' not found in installed models: %s",
                self.name,
                self.model,
                installed_models
            )
            self.logger.debug(f"[check_availability {self.name}] Required model '{self.model}' is not installed on the server")
            return False

        self.logger.debug(f"[check_availability {self.name}] Model '{self.model}' is available and ready to use")
        return True