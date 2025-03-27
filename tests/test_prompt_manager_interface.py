import unittest
from typing import List, Dict, Optional
from mate.services.llm.prompt_manager_interface import (
    PromptManager,
    Mode,
    RemoveOldestStrategy,
    GLOBAL_BASE_TEMPLATES
)

class FakePromptManager(PromptManager[List[Dict[str, str]], Dict[str, str]]):
    """
    Minimal mock subclass to test PromptManager's abstract interface.
    Implements abstract methods in a trivial way.
    """
    def __init__(self, initial_mode: Mode, reduction_strategy=None):
        super().__init__(initial_mode, reduction_strategy)
        # We'll keep a simple counter for token counting.
        # (In real usage, a real tokenizer like tiktoken is used.)
        self.mock_token_counter = 0

    def set_history(self, history: List[Dict[str, str]]) -> None:
        self.histories[self.current_mode] = history

    def empty_history(self) -> None:
        self.histories[self.current_mode] = [{
            "role": "system",
            "content": GLOBAL_BASE_TEMPLATES[self.current_mode.name].system_prompt
        }]

    def get_history(self) -> List[Dict[str, str]]:
        return self.histories[self.current_mode]

    def get_last_entry(self) -> Optional[Dict[str, str]]:
        if self.get_history():
            return self.get_history()[-1]
        return None

    def add_user_entry(self, user_prompt: str) -> Dict[str, str]:
        entry = {"role": "user", "content": user_prompt}
        self.get_history().append(entry)
        return entry

    def add_assistant_entry(self, ai_response: str) -> Dict[str, str]:
        entry = {"role": "assistant", "content": ai_response}
        self.get_history().append(entry)
        return entry

    def count_history_tokens(self) -> int:
        # A trivial mock: each call we just sum up the length of all content strings
        total_len = 0
        for entry in self.get_history():
            total_len += len(entry["content"])
        return total_len

    def count_tokens(self, text: str) -> int:
        return len(text)

    def reduce_history(self, token_limit: int) -> None:
        # Use the reduction_strategy to do something trivial
        self.reduction_strategy.reduce(self.get_history(), self.count_tokens, token_limit)

    def pretty_print_history(self) -> str:
        lines = []
        for entry in self.get_history():
            role = entry.get("role", "unknown").capitalize()
            content = entry.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


class TestPromptManagerInterface(unittest.TestCase):

    def setUp(self):
        self.manager = FakePromptManager(initial_mode=Mode.CHAT, reduction_strategy=RemoveOldestStrategy())

    def test_initial_mode_setup(self):
        self.assertEqual(self.manager.current_mode, Mode.CHAT)
        self.assertIsNotNone(self.manager.get_history(), "Initial history should not be None")

    def test_switch_mode(self):
        # Switch from CHAT to LEDCONTROL
        self.manager.set_mode(Mode.LEDCONTROL)
        self.assertEqual(self.manager.current_mode, Mode.LEDCONTROL)
        # History for LEDCONTROL should be an empty list (unless default system message inserted)
        self.assertIsNotNone(self.manager.get_history())

    def test_add_user_entry(self):
        user_text = "Hello, how are you?"
        entry = self.manager.add_user_entry(user_text)
        self.assertEqual(entry["content"], user_text)
        self.assertEqual(entry["role"], "user")
        self.assertEqual(self.manager.get_last_entry(), entry)

    def test_add_assistant_entry(self):
        ai_text = "I'm just a test assistant response."
        entry = self.manager.add_assistant_entry(ai_text)
        self.assertEqual(entry["content"], ai_text)
        self.assertEqual(entry["role"], "assistant")
        self.assertEqual(self.manager.get_last_entry(), entry)

    def test_empty_history(self):
        # Add some entries
        self.manager.empty_history()
        print(f"### len history {len(self.manager.get_history())}")
        self.manager.add_user_entry("User says something.")
        self.manager.pretty_print_history()
        print(f"### len history {len(self.manager.get_history())}")
        self.assertTrue(len(self.manager.get_history()) > 1)  # Should have system + user
        # Now empty
        self.manager.empty_history()
        self.assertTrue(len(self.manager.get_history()) == 1)  # Should only have system prompt

    def test_token_counting(self):
        self.manager.empty_history()  # start fresh
        self.manager.add_user_entry("Hello")
        self.manager.add_assistant_entry("World")
        total_tokens = self.manager.count_history_tokens()
        # "Hello" length is 5, "World" length is 5, system prompt is also in there
        # We just check that it's consistent with our length-based counting
        self.assertTrue(total_tokens >= 10, "Token count should be at least 10 given 'Hello'+'World'+system prompt")

    def test_reduce_history(self):
        # We will add big content to exceed some token limit
        self.manager.empty_history()
        for i in range(5):
            self.manager.add_user_entry("x" * 20)  # each 20 chars
        # Suppose we only allow 50 tokens
        self.manager.reduce_history(token_limit=50)
        total_tokens = self.manager.count_history_tokens()
        self.assertTrue(total_tokens <= 50, f"History should be reduced to <= 50 tokens but has {total_tokens}.")

    def test_pretty_print_history(self):
        self.manager.empty_history()
        self.manager.add_user_entry("Hello user!")
        self.manager.add_assistant_entry("Hello from assistant!")
        output = self.manager.pretty_print_history()
        # Quick check if it is as expected:
        self.assertIn("User: Hello user!", output)
        self.assertIn("Assistant: Hello from assistant!", output)


if __name__ == "__main__":
    unittest.main()

