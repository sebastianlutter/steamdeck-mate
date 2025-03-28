import abc
import os
import asyncio
import aiohttp
import logging
from urllib.parse import urlparse
from ollama import Client
from typing import Any, Dict, List, Optional, AsyncGenerator

from mate.services.llm.prompt_manager_interface import Mode, PromptManager, RemoveOldestStrategy
from mate.services.llm.prompt_manager_llama import LlamaPromptManager
from mate.services.llm.llm_interface import LlmInterface


class LlmOllamaRemote(LlmInterface, metaclass=abc.ABCMeta):
    def __init__(self, name: str, priority: int, endpoint: str, ollama_model: str) -> None:
        super().__init__(name, priority)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.llm_endpoint: str = endpoint
        self.llm_provider_model: str = ollama_model
        self.client: Client = Client(host=self.llm_endpoint)
        self.model: str = self.llm_provider_model
        self.prompt_manager: PromptManager = LlamaPromptManager(
            initial_mode=Mode.CHAT,
            reduction_strategy=RemoveOldestStrategy()
        )

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

    def get_prompt_manager(self) -> PromptManager:
        return self.prompt_manager

    def config_str(self) -> str:
        return f"Ollama Remote {self.name}: {self.model} on {self.llm_endpoint}."

    async def check_availability(self) -> bool:
        parsed = urlparse(self.llm_endpoint)
        host: Optional[str] = parsed.hostname
        port: Optional[int] = parsed.port

        if not host or not port:
            self.logger.warning(
                "[check_availability %s] Invalid endpoint: %s (missing host or port)",
                self.name,
                self.llm_endpoint
            )
            return False

        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            self.logger.warning(
                "[check_availability %s] Could not connect to host '%s' on port %s. Reason: %s",
                self.name,
                host,
                port,
                e
            )
            return False

        models_url: str = f"{parsed.scheme}://{host}:{port}/api/tags"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url) as resp:
                    if resp.status != 200:
                        self.logger.warning(
                            "[check_availability %s] Could not retrieve models from %s. HTTP status: %d",
                            self.name,
                            models_url,
                            resp.status
                        )
                        return False
                    data = await resp.json()
                    installed_models = [m.get("name") for m in data.get("models", [])]
        except Exception as e:
            self.logger.warning(
                "[check_availability %s] Failed while calling %s. Reason: %s",
                self.name,
                models_url,
                e
            )
            return False

        if self.model not in installed_models:
            self.logger.warning(
                "[check_availability %s] Model '%s' not found in installed models: %s",
                self.name,
                self.model,
                installed_models
            )
            return False

        return True


class SteamdeckOllamaRemote(LlmOllamaRemote):
    config: Dict[str, Any] = {
        "name": "SteamdeckLLama3B",
        "priority": 0,
        "endpoint": "http://127.0.0.1:11434",
        "ollama_model": "llama3.2:3b"
    }

    def __init__(self) -> None:
        super().__init__(**self.config)


class WorkstationOllamaRemote(LlmOllamaRemote):
    config: Dict[str, Any] = {
        "name": "WorkstationLlama8b",
        "priority": 100,
        "endpoint": "http://192.168.0.75:11434",
        "ollama_model": "llama3.1:8b"
    }

    def __init__(self) -> None:
        super().__init__(**self.config)
