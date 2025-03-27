import unittest
from mate.services.llm.prompt_manager_llama import (
    LlamaPromptManager,
    Mode,
    RemoveOldestStrategy,
    GLOBAL_BASE_TEMPLATES
)

class TestLlamaPromptManager(unittest.TestCase):

    def setUp(self):
        # Instantiating LlamaPromptManager with a removal strategy
        self.manager = LlamaPromptManager(initial_mode=Mode.CHAT, reduction_strategy=RemoveOldestStrategy())

    def test_initial_system_prompt_in_history(self):
        # Each mode's history is initialized with a system prompt
        hist = self.manager.get_history()
        self.assertTrue(len(hist) > 0, "History should have at least one entry (system prompt).")
        first_entry = hist[0]
        self.assertEqual(first_entry['role'], 'system', "The first entry should be a system message.")
        self.assertIn("Beantworte die Fragen", first_entry['content'], "System prompt content mismatch.")

    def test_mode_switch_and_system_prompt(self):
        # Switch from CHAT to LEDCONTROL and see if the correct system prompt is loaded
        self.manager.set_mode(Mode.LEDCONTROL)
        hist = self.manager.get_history()
        self.assertTrue(len(hist) > 0, "LEDCONTROL history should not be empty.")
        system_entry = hist[0]
        self.assertEqual(system_entry['role'], 'system')
        self.assertIn("JSON requests", system_entry['content'], "LEDCONTROL system prompt content mismatch.")

    def test_add_and_retrieve_entries(self):
        # Add user entry, then assistant entry, ensure they're in the history
        user_text = "Please turn on the LED light."
        assistant_text = "Sure, here's a JSON snippet..."
        self.manager.add_user_entry(user_text)
        self.manager.add_assistant_entry(assistant_text)
        self.assertEqual(self.manager.get_history()[-2]['content'], user_text)
        self.assertEqual(self.manager.get_history()[-1]['content'], assistant_text)

    def test_count_tokens(self):
        # By default, LlamaPromptManager uses tiktoken for tokenization.
        # We can do a basic sanity check for nonempty strings:
        count = self.manager.count_tokens("Hello world!")
        self.assertTrue(count > 0, "Expect some token count for 'Hello world!'")

    def test_reduce_history(self):
        """
        Add large entries to exceed some small token limit, ensure the
        removal strategy is called, and confirm that the final token count is <= limit.
        """
        # Add a bunch of content
        for i in range(10):
            self.manager.add_user_entry("Some very long text " + ("x"*50))

        # Now reduce
        self.manager.reduce_history(token_limit=50)  # artificially small

        total_tokens = self.manager.count_history_tokens()
        self.assertTrue(total_tokens <= 50, f"Expected the history to be reduced to <= 50 tokens, got {total_tokens}.")

    def test_pretty_print_history(self):
        # Check that we get a well-formatted string of roles and contents
        self.manager.add_user_entry("Hello from user")
        output = self.manager.pretty_print_history()
        self.assertIn("User: Hello from user", output)

if __name__ == "__main__":
    unittest.main()

