import os
from abc import ABC, abstractmethod

from mate.services.llm.prompt_manager_interface import PromptManager


class LmmInterface(ABC):

    def __init__(self, name: str, priority: int):
        super().__init__(name, "LLM", priority)
        self._counter = 0
        self.llm_endpoint=os.getenv('LLM_ENDPOINT', 'http://127.0.0.1:11434')
        self.llm_provider_model=os.getenv('LLM_PROVIDER_MODEL', 'llama3.2:3b')


    @abstractmethod
    async def chat(self, full_chat) -> str:
        pass

    @abstractmethod
    def get_prompt_manager(self) -> PromptManager:
        pass
