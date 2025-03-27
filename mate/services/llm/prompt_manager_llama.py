import logging
import tiktoken
from typing import Any, Dict, Generic, Optional, TypeVar, List
from mate.services.llm.prompt_manager_interface import PromptManager, Mode, RemoveOldestStrategy, ReductionStrategy, \
    GLOBAL_BASE_TEMPLATES


class LlamaPromptManager(PromptManager[List[Dict[str, str]], Dict[str, str]]):
    """
    Concrete implementation of PromptManager for Llama 3.3.
    Manages separate histories for each mode and utilizes tiktoken for tokenization.
    """
    def __init__(self, initial_mode: Mode, reduction_strategy: ReductionStrategy):
        """
        Initialize the LlamaPromptManager with the specified initial mode and reduction strategy.
        Sets up the tokenizer and logger.
        """
        super().__init__(initial_mode, reduction_strategy)
        # Initialize the logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Initialize the tokenizer encoding for Llama 3.3
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # Adjust encoding as per Llama's requirements
            self.logger.info("LlamaPromptManager initialized with encoding 'cl100k_base'.")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise
        # for each mode init the history with system prompt
        for mode in Mode:
            self.histories[mode] = [{
                'role': 'system',
                'content': GLOBAL_BASE_TEMPLATES[mode.name].system_prompt
            }]


    def set_history(self, history: List[Dict[str, str]]) -> None:
        """
        Set the history for the current mode.
        Validates the structure of the history before setting it.
        """
        if not isinstance(history, list):
            self.logger.error("History must be a list of dictionaries.")
            raise TypeError("History must be a list of dictionaries.")
        # Validate each entry in history
        for entry in history:
            if not isinstance(entry, dict):
                self.logger.error("Each history entry must be a dictionary.")
                raise TypeError("Each history entry must be a dictionary.")
            if "content" not in entry or "role" not in entry:
                self.logger.error("Each history entry must contain 'content' and 'role' keys.")
                raise ValueError("Each history entry must contain 'content' and 'role' keys.")
            if entry["role"] not in ["user", "assistant", "system"]:
                self.logger.error("The 'role' must be either 'user' or 'assistant'.")
                raise ValueError("The 'role' must be either 'user' or 'assistant'.")
        # Replace the current history with the new history
        self.get_history().clear()
        self.get_history().extend(history)
        self.logger.info(f"History set for mode {self.current_mode.name}")

    def empty_history(self) -> None:
        """
        Clear the history for the current mode.
        """
        self.get_history().clear()
        self.get_history().append({
            'role': 'system',
            'content': GLOBAL_BASE_TEMPLATES[self.current_mode.name].system_prompt
        })
        self.logger.info(f"History emptied for mode {self.current_mode.name}")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieve the current mode's history.
        """
        self.logger.debug(f"Retrieving history for mode {self.current_mode.name}")
        return self.histories[self.current_mode]

    def get_last_entry(self) -> Optional[Dict[str, str]]:
        """
        Retrieve the last entry in the current mode's history.
        """
        history = self.get_history()
        if not history:
            self.logger.debug(f"No entries found in history for mode {self.current_mode.name}")
            return None
        last_entry = history[-1]
        self.logger.debug(f"Last entry retrieved for mode {self.current_mode.name}: {last_entry}")
        return last_entry

    def add_user_entry(self, user_prompt: str) -> Dict[str, str]:
        """
        Add a user prompt to the current mode's history.
        """
        entry = {"content": user_prompt, "role": "user"}
        self.get_history().append(entry)
        self.logger.info(f"Added user entry to {self.current_mode.name}: {user_prompt}")
        return entry

    def add_assistant_entry(self, ai_response: str) -> Dict[str, str]:
        """
        Add an AI response to the current mode's history.
        """
        entry = {"content": ai_response, "role": "assistant"}
        self.get_history().append(entry)
        self.logger.info(f"Added AI entry to {self.current_mode.name}: {ai_response}")
        return entry

    def count_history_tokens(self) -> int:
        """
        Count the total number of tokens in the current mode's history.
        """
        total_tokens = 0
        for entry in self.get_history():
            content = entry.get("content", "")
            tokens = self.count_tokens(content)
            total_tokens += tokens
            self.logger.debug(f"Entry content: '{content}' has {tokens} tokens")
        self.logger.info(f"Total tokens in history for mode {self.current_mode.name}: {total_tokens}")
        return total_tokens

    def count_tokens(self, text: str) -> int:
        """
        Tokenize the input text and return the token count.
        """
        try:
            tokens = self.encoding.encode(text)
            token_count = len(tokens)
            self.logger.debug(f"Tokenized text: '{text}' into {token_count} tokens")
            return token_count
        except Exception as e:
            self.logger.error(f"Tokenization failed for text: '{text}'. Error: {e}")
            raise

    def reduce_history(self, token_limit: int) -> None:
        """
        Reduce the current mode's history to fit within the token limit.
        """
        self.logger.info(f"Ensuring token limit for mode {self.current_mode.name}: {token_limit} tokens")
        current_token_count = self.count_history_tokens()
        if current_token_count > token_limit:
            self.logger.info(f"Token limit exceeded: {current_token_count} > {token_limit}. Reducing history.")
            self.reduction_strategy.reduce(self.get_history(), self.count_tokens, token_limit)
            if self.count_history_tokens() > token_limit:
                self.logger.warning("Unable to reduce history within the token limit.")

    def pretty_print_history(self) -> str:
        """
        Returns a formatted string representing the current mode's history.
        Each entry is prefixed with the role (e.g., 'User', 'Assistant').

        Returns:
            A string representing the formatted history.
        """
        formatted_history = []
        for entry in self.get_history():
            role = entry.get("role", "unknown").capitalize()
            content = entry.get("content", "")
            formatted_history.append(f"{role}: {content}")
        history_str = "\n".join(formatted_history)
        self.logger.debug(f"Formatted history for mode {self.current_mode.name}:\n{history_str}")
        return history_str