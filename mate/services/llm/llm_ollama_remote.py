import abc
import os
import asyncio
import aiohttp
from urllib.parse import urlparse
from ollama import Client

from mate.services.llm.prompt_manager_interface import Mode
from mate.services.llm.prompt_manager_llama import LlamaPromptManager
from mate.services.llm.llm_interface import LlmInterface
from mate.services.llm.prompt_manager_interface import PromptManager, RemoveOldestStrategy
from typing import Any, Dict, Generic, Optional, TypeVar, List, AsyncGenerator


class LlmOllamaRemote(LlmInterface, metaclass=abc.ABCMeta):

    def __init__(self, name: str, priority: int, endpoint: str, ollama_model: str):
        super().__init__(name, priority)
        self.llm_endpoint=endpoint
        self.llm_provider_model=ollama_model
        self.client = Client(host=self.llm_endpoint)
        self.model = self.llm_provider_model
        self.prompt_manager = LlamaPromptManager(initial_mode=Mode.MODUS_SELECTION,
                                                 reduction_strategy=RemoveOldestStrategy())


    async def chat(self, full_chat: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        content = self.client.chat(
                model=self.model,
                stream=True,
                messages=full_chat,
            )
        for chunk in content:
            c = chunk['message']['content']
            yield c

    def get_prompt_manager(self) -> PromptManager:
        return self.prompt_manager

    def config_str(self) -> str:
        return f"Ollama Remote {self.name}: {self.model} on {self.llm_endpoint}."

    async def check_availability(self) -> bool:
        # 1) Parse out the host and port
        parsed = urlparse(self.llm_endpoint)
        host = parsed.hostname
        port = parsed.port

        # 2) Check if we can connect to host:port
        #    Using an asyncio open_connection for an async-friendly approach
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            print(f"[check_availability] Could not connect to host '{host}' on port {port}.")
            print(f"    Reason: {e}")
            return False

        # 3) Check if /models endpoint is available and get the list of installed models
        try:
            models_url = f"{parsed.scheme}://{host}:{port}/api/tags"
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url) as resp:
                    if resp.status != 200:
                        print(
                            f"[check_availability] Could not retrieve models from {models_url}. "
                            f"HTTP status: {resp.status}"
                        )
                        return False
                    data = await resp.json()

                    # Expecting {
                    #   "models": [
                    #       {"name": "codellama:13b", ...},
                    #       {"name": "llama3:latest", ...}
                    #   ]
                    # }
                    #
                    # Extract only the "name" field from each model dictionary
                    installed_models = [m.get("name") for m in data.get("models", [])]
        except Exception as e:
            print(f"[check_availability] Failed while calling {models_url}.")
            print(f"    Reason: {e}")
            return False

        # 4) Check if the requested model is in the installed models
        if self.model not in installed_models:
            print(f"[check_availability] Model '{self.model}' not found in installed models: {installed_models}")
            return False

        # If we made it here, everything is OK
        return True

class LocalhostOllamaRemote(LlmOllamaRemote):

    config={
        "name": "SteamdeckLLama3B",
        "priority": 0,
        "endpoint": "http://127.0.0.1:11434",
        "ollama_model": "llama3.1:3b"
    }

    def __init__(self):
        super().__init__(**self.config)



class WorkstationOllamaRemote(LlmOllamaRemote):

    config={
        "name": "WorkstationLlama8b",
        "priority": 100,
        "endpoint": "http://192.168.0.75:11434",
        "ollama_model": "llama3.1:8b"
    }

    def __init__(self):
        super().__init__(**self.config)
