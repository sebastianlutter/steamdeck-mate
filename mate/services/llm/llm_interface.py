import os
from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator

from mate.services import BaseService
from mate.services.llm.prompt_manager_interface import PromptManager


class LlmInterface(BaseService, metaclass=ABCMeta):

    def __init__(self, name: str, priority: int):
        super().__init__(name, "LLM", priority)

    @abstractmethod
    async def chat(self, full_chat) -> AsyncGenerator[str, None]:
        pass
