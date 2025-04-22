import logging
import tiktoken
from typing import Any, Dict, List, Optional
from mate.services.llm.prompt_manager_interface import (
    PromptManager,
    Mode,
    RemoveOldestStrategy,
    ReductionStrategy,
    GLOBAL_BASE_TEMPLATES,
)
import datetime

class LlamaPromptManager(PromptManager[List[Dict[str, str]], Dict[str, str]]):
    """
    Concrete implementation of PromptManager for Llama 3.3.
    Manages separate histories for each mode and utilizes tiktoken for tokenization.
    """

    def __init__(self, initial_mode: Mode, reduction_strategy: ReductionStrategy) -> None:
        super().__init__(initial_mode, reduction_strategy)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.logger.info("LlamaPromptManager initialized with encoding 'cl100k_base'.")
        except Exception as e:
            self.logger.error("Failed to initialize tokenizer: %s", e)
            raise

        for mode in Mode:
            # init the system prompt
            self.empty_history()

    def set_history(self, history: List[Dict[str, str]]) -> None:
        if not isinstance(history, list):
            self.logger.error("History must be a list of dictionaries.")
            raise TypeError("History must be a list of dictionaries.")

        for entry in history:
            if not isinstance(entry, dict):
                self.logger.error("Each history entry must be a dictionary.")
                raise TypeError("Each history entry must be a dictionary.")
            if "content" not in entry or "role" not in entry:
                self.logger.error("Each history entry must contain 'content' and 'role' keys.")
                raise ValueError("Each history entry must contain 'content' and 'role' keys.")
            if entry["role"] not in ["user", "assistant", "system"]:
                self.logger.error("The 'role' must be either 'user', 'assistant', or 'system'.")
                raise ValueError("The 'role' must be either 'user', 'assistant', or 'system'.")

        self.get_history().clear()
        self.get_history().extend(history)
        self.logger.info("History set for mode %s", self.current_mode.name)

    def empty_history(self) -> None:
        self.get_history().clear()
        primer = f"Heute ist {['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag'][datetime.datetime.now().weekday()]} der {datetime.datetime.now().strftime('%d.%m.%Y')}. Du befindest dich in Deutschland"

        self.get_history().append(
            {
                "role": "system",
                "content": f"{primer}. " + GLOBAL_BASE_TEMPLATES[self.current_mode.name].system_prompt,
            }
        )
        self.logger.info("History emptied for mode %s", self.current_mode.name)

    def get_history(self) -> List[Dict[str, str]]:
        self.logger.debug("Retrieving history for mode %s", self.current_mode.name)
        return self.histories[self.current_mode]

    def get_last_entry(self) -> Optional[Dict[str, str]]:
        history = self.get_history()
        if not history:
            self.logger.debug("No entries found in history for mode %s", self.current_mode.name)
            return None
        last_entry = history[-1]
        self.logger.debug("Last entry for mode %s: %s", self.current_mode.name, last_entry)
        return last_entry

    def add_user_entry(self, user_prompt: str) -> Dict[str, str]:
        entry = {"content": user_prompt, "role": "user"}
        self.get_history().append(entry)
        self.logger.info("Added user entry to %s: %s", self.current_mode.name, user_prompt)
        return entry

    def add_assistant_entry(self, ai_response: str) -> Dict[str, str]:
        entry = {"content": ai_response, "role": "assistant"}
        self.get_history().append(entry)
        self.logger.info("Added AI entry to %s: %s", self.current_mode.name, ai_response)
        return entry

    def count_history_tokens(self) -> int:
        total_tokens = 0
        for entry in self.get_history():
            content = entry.get("content", "")
            tokens = self.count_tokens(content)
            total_tokens += tokens
            self.logger.debug("Entry content: '%s' has %d tokens", content, tokens)
        self.logger.info(
            "Total tokens in history for mode %s: %d",
            self.current_mode.name,
            total_tokens,
        )
        return total_tokens

    def count_tokens(self, text: str) -> int:
        try:
            tokens = self.encoding.encode(text)
            token_count = len(tokens)
            self.logger.debug("Tokenized text: '%s' into %d tokens", text, token_count)
            return token_count
        except Exception as e:
            self.logger.error("Tokenization failed for text: '%s'. Error: %s", text, e)
            raise

    def reduce_history(self, token_limit: int) -> None:
        self.logger.info(
            "Ensuring token limit for mode %s: %d tokens",
            self.current_mode.name,
            token_limit,
        )
        current_token_count = self.count_history_tokens()
        if current_token_count > token_limit:
            self.logger.info(
                "Token limit exceeded: %d > %d. Reducing history.",
                current_token_count,
                token_limit,
            )
            self.reduction_strategy.reduce(self.get_history(), self.count_tokens, token_limit)
            if self.count_history_tokens() > token_limit:
                self.logger.warning("Unable to reduce history within the token limit.")

    def pretty_print_history(self) -> str:
        formatted_history = []
        for entry in self.get_history():
            role = entry.get("role", "unknown").capitalize()
            content = entry.get("content", "")
            formatted_history.append(f"{role}: {content}")
        history_str = "\n".join(formatted_history)
        self.logger.debug(
            "Formatted history for mode %s:\n%s",
            self.current_mode.name,
            history_str,
        )
        return history_str
