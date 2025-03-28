import asyncio
import logging
from mate.audio.soundcard_pyaudio import SoundCard  # Make sure this path matches your real import
from mate.services import ServiceDiscovery
from typing import Any, AsyncGenerator

format_string = (
    "%(asctime)s - [Logger: %(name)s] - %(levelname)s - %(filename)s:%(lineno)d in %(funcName)s() - %(message)s"
)
logging.basicConfig(format=format_string, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("websocket").setLevel(logging.ERROR)


class SteamdeckMate:
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.running: bool = False
        self.soundcard: SoundCard = SoundCard()
        self.service_discovery: ServiceDiscovery = ServiceDiscovery()

    async def listen_and_respond(self) -> None:
        await self.service_discovery.start()
        await asyncio.sleep(5)
        await self.service_discovery.print_status_table()

        self.logger.info("Starting to listen...")
        self.running = True

        try:
            async for mic_chunk in self.soundcard.get_record_stream():  # type: AsyncGenerator[Any, None]
                self.logger.info("Heard microphone audio chunk... (pretend we recognized 'hello')")
                response: str = await self.ask_llm("hello")
                await self.speak_response(response)

                break

                if not self.running:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.info("Recording loop has ended.")

    async def ask_llm(self, text: str) -> str:
        llm_provider = await self.service_discovery.get_best_service("LLM")
        prompt_manager = llm_provider.get_prompt_manager()
        full_res: str = ""
        async for res in llm_provider.chat(prompt_manager.get_history()):
            self.logger.info(res)
            full_res += res
        return f"User: '{text}'\nAI: {full_res}"

    async def speak_response(self, response: str) -> None:
        self.logger.info("Speaking: %s", response)
        self.logger.info("Done speaking.")

    async def stop(self) -> None:
        self.logger.info("Stopping...")
        self.running = False

        self.soundcard.stop_recording()
        self.soundcard.stop_playback()
        await asyncio.sleep(0.2)
        await self.service_discovery.stop()
        self.soundcard.close()
        self.logger.info("All resources closed. Stopped.")


async def main() -> None:
    mate = SteamdeckMate()
    listen_task = asyncio.create_task(mate.listen_and_respond())

    await asyncio.sleep(10)
    await mate.stop()

    listen_task.cancel()
    try:
        await listen_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
