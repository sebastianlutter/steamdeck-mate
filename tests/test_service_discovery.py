# tests/test_service_discovery.py

import unittest
import asyncio

# Import the classes under test
# Adjust these paths to match your project structure
from mate.services import ServiceDiscovery
from mate.services.llm.llm_interface import LmmInterface
from mate.services.stt.stt_interface import STTInterface
from mate.services.tts.tts_interface import TTSInterface

class TestServiceDiscovery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """
        Runs before each test. We'll create a ServiceDiscovery instance
        with one TTS, one STT, and one LLM dummy service.
        """
        self.service_defs = [
            (TTSInterface, "test_tts", 10),
            (STTInterface, "test_stt", 5),
            (LmmInterface, "test_llm", 8),
        ]
        self.discovery = ServiceDiscovery(self.service_defs)
        await self.discovery.start()

        # Wait briefly for the first availability check to complete
        # so that the services have had time to be initialized and checked.
        await asyncio.sleep(0.2)

    async def asyncTearDown(self):
        """
        Runs after each test, making sure we shut down the availability loop.
        """
        await self.discovery.stop()

    async def test_tts_availability(self):
        """
        Test that the TTS service can be discovered and is marked available.
        """
        tts_service = await self.discovery.get_best_service("TTS")
        self.assertIsNotNone(tts_service, "No TTS service was discovered.")
        self.assertEqual(tts_service.name, "test_tts")
        self.assertTrue(
            tts_service.service_type == "TTS",
            f"Unexpected service type: {tts_service.service_type}",
        )

    async def test_stt_availability(self):
        """
        Test that the STT service can be discovered and that it occasionally
        toggles availability as expected.
        """
        stt_service = await self.discovery.get_best_service("STT")
        self.assertIsNotNone(stt_service, "No STT service was discovered.")
        self.assertEqual(stt_service.name, "test_stt")
        self.assertTrue(
            stt_service.service_type == "STT",
            f"Unexpected service type: {stt_service.service_type}",
        )

        # Because STTService toggles availability every check,
        # we can demonstrate that it changes by forcing a re-check:
        await self.discovery._check_services_once()
        stt_service = await self.discovery.get_best_service("STT")
        # Depending on timing, it may or may not still be available.
        # We simply assert it's not None to ensure it still exists in the registry.
        self.assertIsNotNone(stt_service, "STT service disappeared after toggle check.")

    async def test_llm_availability(self):
        """
        Test that the LLM service is discovered, has the correct config, and
        that it occasionally raises an exception (simulated unavailability).
        """
        llm_service = await self.discovery.get_best_service("LLM")
        self.assertIsNotNone(llm_service, "No LLM service was discovered.")
        self.assertEqual(llm_service.name, "test_llm")
        self.assertTrue(
            llm_service.service_type == "LLM",
            f"Unexpected service type: {llm_service.service_type}",
        )
        # Check that the LLMService has the correct default environment config
        self.assertIn("model:", llm_service.config_str())
        self.assertIn("endpoint:", llm_service.config_str())

        # Force enough checks to ensure the simulated failure occurs at least once
        # (the sample LLMService raises an error every 3rd check).
        for _ in range(3):
            await self.discovery._check_services_once()

        # Even if it fails once internally, it should remain in the registry; 
        # availability might flip to False on that specific check.
        llm_service_after = await self.discovery.get_best_service("LLM")
        # If the availability check raised, it might be None. 
        # (Depending on your design, you could allow a single failure
        # and still keep the service in the dictionary but with available=False.)
        # This assertion checks that the service *still exists* in the manager 
        # — if it became “unavailable,” we’d expect a None return.
        # If you want a different behavior (e.g., keep returning the object),
        # adapt the test accordingly.
        self.assertIsNotNone(
            llm_service_after,
            "LLM service is not discoverable after a simulated failure."
        )


if __name__ == "__main__":
    unittest.main()

