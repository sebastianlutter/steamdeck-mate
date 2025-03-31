from dotenv import load_dotenv

from mate.services.llm.prompt_manager_llama import LlamaPromptManager

load_dotenv()

import asyncio
import logging
import re
from typing import AsyncGenerator

from nltk import sent_tokenize

from mate.audio.soundcard_pyaudio import SoundCard  # Make sure this path matches your real import
from mate.human_speech_agent import HumanSpeechAgent
from mate.services import ServiceDiscovery, PromptManager

from mate.services.llm.llm_interface import LlmInterface
from mate.services.llm.prompt_manager_interface import Mode, RemoveOldestStrategy
from mate.services.tts.tts_interface import TTSInterface
from mate.utils import clean_str_from_markdown, is_sane_input_german

#format_string = (
#    "%(asctime)s - [Logger: %(name)s] - %(levelname)s - %(filename)s:%(lineno)d in %(funcName)s() - %(message)s"
#)
format_string = (
    "[%(levelname)s - %(filename)s:%(lineno)d.%(funcName)s() - %(message)s"
)
logging.basicConfig(format=format_string, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("websocket").setLevel(logging.ERROR)


class SteamdeckMate:

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.soundcard: SoundCard = SoundCard()
        self.service_discovery: ServiceDiscovery = ServiceDiscovery()
        self.human_speech_agent = HumanSpeechAgent()
        self.status = Mode.CHAT
        self.prompt_manager = LlamaPromptManager(initial_mode=Mode.CHAT,
                                            reduction_strategy=RemoveOldestStrategy())

    async def __aenter__(self):
        self.logger.info("Enter constructing class")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.logger.info("Exit class, close resources")
        await self.stop()
        if exc:
            print(f"Exception caught: {exc}")
        return False  # False means any exception is propagated

    async def listen_and_choose_mode(self) -> None:
        await self.service_discovery.start()
        warmup_task = asyncio.create_task(self.human_speech_agent.warmup_cache())
        await self.service_discovery.print_status_table()
        self.logger.info("Starting to listen...")
        await self.human_speech_agent.say_init_greeting()
        await warmup_task
        wake_word = True
        while True:
            try:
                # wait for wake word, then transcribe user input
                full_text = ''
                async for text in self.human_speech_agent.get_human_input(
                        wait_for_wakeword=wake_word
                ):
                    full_text += text
                if not is_sane_input_german(full_text):
                    self.logger.info(f"Got input: \"{full_text}\", this is garbage. Ignore it and listen again")
                    await self.human_speech_agent.beep_error()
                    wake_word = False
                    continue
                else:
                    wake_word = True
                # give user input text to prompt manager
                processing_sound_task = asyncio.create_task(self.human_speech_agent.processing_sound())
                async for sentence in self.ask_llm(text=full_text, stream_sentences=True):
                    tts_provider: TTSInterface = await self.service_discovery.get_best_service("TTS")
                    tts_provider.speak(sentence)
                await processing_sound_task
            except asyncio.CancelledError as e:
                self.logger.error("CancelledError",exc_info=e)
                break
            except KeyboardInterrupt as e:
                self.soundcard.close()
                self.human_speech_agent.stop_signal.set()
                self.logger.info("KeyboardInterrupt: Stop processing")
                break
            except BaseException as e:
                self.logger.error("got error", exc_info=e)
                break
            except:
                self.logger.error("got unknown error", exc_info=True)
                break
            finally:
                self.logger.info("Application closed.")

    async def ask_llm(self, text: str, stream_sentences: bool) -> AsyncGenerator[str, None]:
        self.prompt_manager.add_user_entry(text)
        llm_provider: LlmInterface = await self.service_discovery.get_best_service('LLM')
        response = ""
        sentence_buffer = ""
        # send to LLM and stream response
        async for chunk in llm_provider.chat(self.prompt_manager.get_history()):
            # clean out markdown
            chunk = clean_str_from_markdown(chunk)
            # append to response
            response += chunk
            if stream_sentences:
                sentence_buffer += chunk
                # Tokenize to sentences
                sentences = sent_tokenize(text=sentence_buffer, language="german")
                for sentence in sentences[:-1]:
                    # clean the sentence from markdown and skip if broken
                    sentence = re.sub(r'[*_#`"\']+', '', sentence).strip()
                    # process only if it has real chars
                    if re.search(r'[A-Za-z0-9äöüÄÖÜß]', sentence) is None:
                        continue
                    # we have a real sentence, yield it
                    yield sentence
                # store last (maybe incomplete) sentence in the buffer
                sentence_buffer = sentences[-1]
            else:
                # Yield chunks as they come from the LLM
                yield chunk
        # when we yield sentences than we have the last sentence in the buffer
        if stream_sentences:
            yield sentence_buffer
        # add AI response to prompt manager
        self.prompt_manager.add_assistant_entry(response)

    async def speak_response(self, response: str) -> None:
        self.logger.info("Speaking: %s", response)
        self.logger.info("Done speaking.")

    async def stop(self) -> None:
        self.logger.info("Stopping...")
        self.running = False
        self.soundcard.stop_recording()
        self.soundcard.stop_playback()
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
