import random
import threading
import time
import os
import io
import hashlib
import logging
import asyncio
from pydub import AudioSegment
from typing import AsyncGenerator, Tuple, Any
import soundfile as sf
from tqdm import tqdm

from mate.audio.soundcard_pyaudio import SoundCard
from mate.interrupt_speech_thread import InterruptSpeechThread
from mate.services.stt.stt_interface import STTInterface
from mate.services.tts.tts_interface import TTSInterface
from mate.voice_activated_recording.va_picovoice import PorcupineWakeWord
from mate.services import ServiceDiscovery


class HumanSpeechAgent:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(HumanSpeechAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.interrupt_speech_thread: threading.Thread = None
        self.soundcard: SoundCard = SoundCard()
        self.service_discovery: ServiceDiscovery = ServiceDiscovery()
        self.voice_activator: PorcupineWakeWord = PorcupineWakeWord()
        self.silence_lead_time: int = 2
        self.max_recording_time: int = 15
        self.stop_signal: threading.Event = threading.Event()

        self.abort_speech_choices = ["Anwort abgebrochen, was soll ich tun?"]
        self.hi_choices = [
            "ja, hi", "schiess los!", "was gibts?", "hi, was geht?", "leg los!",
            "was willst du?", "sprechen Sie", "jo bro", "hey ho bro", "was geht so?"
        ]
        self.bye_choices = [
            "Auf Wiedersehen!", "Mach’s gut!", "Bis zum nächsten Mal!", "Schönen Tag noch!",
            "Bis bald!", "Pass auf dich auf!", "Bleib gesund!", "Man sieht sich!", "Bis später!", "Bis dann!",
            "Gute Reise!", "Viel Erfolg noch!", "Danke und tschüss!", "Alles Gute!",
            "Bis zum nächsten Treffen!", "Leb wohl!"
        ]
        init_greetings_identity = ""
        self.init_greetings = [
            "Guten Tag!", "Hi, wie geht's?", "Schön dich zu sehen!", "Hallo und willkommen!",
            "Freut mich, dich zu treffen!", "Hallo zusammen!", "Hallo, mein Freund!",
            "Guten Tag, wie kann ich helfen?", "Willkommen!", "Hallo an alle!",
            "Herzlich willkommen!", "Hallo, schön dich hier zu haben!", "Hey, alles klar?",
            "Hallo, schön dich kennenzulernen!", "Hallo, wie läuft's?", "Einen schönen Tag!"
        ]
        self.init_greetings = list(map(lambda g: f"{init_greetings_identity} {g}", self.init_greetings))
        self.did_not_understand = [
            "Das war unverständlich, bitte wiederholen"
        ]
        self.explain_sentence = "Sag das wort computer um zu starten."

    async def engage_input_beep(self) -> None:
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio("sounds/deskviewerbeep.mp3")
        await asyncio.to_thread(self.soundcard.play_audio, sample_rate, audio_buffer)

    async def beep_positive(self) -> None:
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio("sounds/computerbeep_26.mp3")
        await asyncio.to_thread(self.soundcard.play_audio, sample_rate, audio_buffer)

    async def beep_error(self) -> None:
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio("sounds/denybeep1.mp3")
        await asyncio.to_thread(self.soundcard.play_audio, sample_rate, audio_buffer)

    async def processing_sound(self) -> None:
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio("sounds/processing.mp3")
        await asyncio.to_thread(self.soundcard.play_audio, sample_rate, audio_buffer)

    async def say_abort_speech(self) -> None:
        tts_provider: TTSInterface = await self.service_discovery.get_best_service("TTS")
        await asyncio.to_thread(tts_provider.set_stop_signal)
        await asyncio.to_thread(tts_provider.soundcard.stop_playback)
        hi_phrase = random.choice(self.abort_speech_choices)
        mp3_path = self._get_cache_file_name(hi_phrase)
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        await asyncio.to_thread(self.soundcard.play_audio, sample_rate, audio_buffer)

    async def say_init_greeting(self) -> None:
        hi_phrase = random.choice(self.init_greetings)
        mp3_path = self._get_cache_file_name(hi_phrase)
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        await asyncio.to_thread(self.soundcard.play_audio, sample_rate, audio_buffer)
        tts_provider: TTSInterface = await self.service_discovery.get_best_service("TTS")
        await asyncio.to_thread(tts_provider.speak, f"Ich höre auf den Namen {self.voice_activator.wakeword}")
        await asyncio.to_thread(tts_provider.wait_until_done)
        await asyncio.to_thread(self.soundcard.wait_until_playback_finished)

    async def say_hi(self) -> None:
        hi_phrase = random.choice(self.hi_choices)
        mp3_path = self._get_cache_file_name(hi_phrase)
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        self.logger.info("say_hi: %s", hi_phrase)
        await asyncio.to_thread(self.soundcard.play_audio, sample_rate, audio_buffer)

    async def say_bye(self, message: str = "") -> None:
        bye_phrase = random.choice(self.bye_choices)
        mp3_path = self._get_cache_file_name(bye_phrase)
        self.logger.info("say_bye: %s%s", message, bye_phrase)
        tts_provider: TTSInterface = await self.service_discovery.get_best_service("TTS")
        if message != "":
            await asyncio.to_thread(tts_provider.speak, message)
        await asyncio.to_thread(tts_provider.wait_until_done)
        # Assuming that the correct call should load the bye audio instead of using bye_phrase as data.
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        await asyncio.to_thread(self.soundcard.play_audio, sample_rate, audio_buffer)

    async def say_did_not_understand(self) -> None:
        did_not_understand_phrase = random.choice(self.did_not_understand)
        mp3_path = self._get_cache_file_name(did_not_understand_phrase)
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        self.logger.info("say_did_not_understand: %s", did_not_understand_phrase)
        await asyncio.to_thread(self.soundcard.play_audio, sample_rate, audio_buffer)

    async def say(self, message: str) -> None:
        self.logger.debug("say: %s", message)
        tts_provider: TTSInterface = await self.service_discovery.get_best_service("TTS")
        await asyncio.to_thread(tts_provider.speak, message)

    async def skip_all_and_say(self, message: str) -> None:
        self.logger.info("Skip all and say: %s", message)
        tts_provider: TTSInterface = await self.service_discovery.get_best_service("TTS")
        await asyncio.to_thread(tts_provider.set_stop_signal)
        await asyncio.to_thread(tts_provider.wait_until_done)
        await asyncio.sleep(0.2)
        await asyncio.to_thread(tts_provider.clear_stop_signal)
        await asyncio.to_thread(tts_provider.speak, message)

    async def wait_until_talking_finished(self) -> None:
        self.logger.info("block_until_talking_finished: blocking")
        tts_provider: TTSInterface = await self.service_discovery.get_best_service("TTS")
        c = 0
        while c < 2:
            c += 1
            await asyncio.to_thread(tts_provider.wait_until_done)
            await asyncio.to_thread(tts_provider.soundcard.wait_until_playback_finished)
        self.logger.info("block_until_talking_finished: unblocking")

    async def get_human_input(self, wait_for_wakeword: bool = True) -> AsyncGenerator[str, None]:
        if wait_for_wakeword:
            await asyncio.to_thread(self.soundcard.stop_recording)
            await asyncio.to_thread(self.soundcard.wait_until_playback_finished)
            await self.engage_input_beep()
            await self.voice_activator.listen_for_wake_word(stop_signal=None)
            await self.beep_positive()

        def on_close_ws_callback() -> None:
            self.logger.debug("get_human_input.on_close_ws_callback: websocket closed")

        def on_ws_open() -> None:
            self.logger.debug("on_ws_open: Should say_hi now ws is opened:")

        stt_provider: STTInterface = await self.service_discovery.get_best_service("STT")
        async for text in stt_provider.transcribe_stream(
            audio_stream=self.start_recording(),
            websocket_on_close=on_close_ws_callback,
            websocket_on_open=on_ws_open
        ):
            yield text
        self.logger.debug("get_human_input: finished")

    async def start_recording(self) -> AsyncGenerator[bytes, None]:
        self.logger.info("start_recording: Recording...")
        async for wav_chunk in self.soundcard.get_record_stream():
            yield wav_chunk

    async def warmup_cache(self) -> None:
        os.makedirs("tts_cache", exist_ok=True)
        all_choices = (
            self.hi_choices
            + self.bye_choices
            + self.init_greetings
            + [self.explain_sentence]
            + self.did_not_understand
            + self.abort_speech_choices
        )
        for sentence in tqdm(all_choices, desc="Warmup cache with hi and bye phrases"):
            file_name = await self._get_cache_file_name(sentence)
            if not os.path.exists(file_name):
                tts_provider: TTSInterface = await self.service_discovery.get_best_service("TTS")
                print(type(tts_provider))
                # Offload the TTS rendering to a thread
                await asyncio.to_thread(
                    tts_provider.render_sentence,
                    sentence=sentence, store_file_name=file_name, output_format="mp3"
                )

    async def _get_cache_file_name(self, sentence: str) -> str:
        hash_obj = hashlib.md5(sentence.encode("utf-8"))
        hash_str = hash_obj.hexdigest()[:8]
        return os.path.join("tts_cache/", f"{hash_str}.mp3")

    async def _load_mp3_to_wav_bytesio(self, mp3_path: str) -> Tuple[int, Any]:
        audio_segment = AudioSegment.from_mp3(mp3_path)
        wav_bytes = io.BytesIO()
        audio_segment.export(wav_bytes, format="wav")
        wav_bytes.seek(0)
        data, sample_rate = sf.read(wav_bytes, dtype="float32")
        return sample_rate, data

    async def start_speech_interrupt_thread(self, ext_stop_signal: threading.Event) -> None:
        def stop_speech() -> None:
            # Note: if say_abort_speech is to be awaited, it must be scheduled properly.
            asyncio.run(self.say_abort_speech())

        self.logger.info("Start speech interrupt thread")
        self.interrupt_speech_thread = InterruptSpeechThread(
            stop_event=ext_stop_signal, va_provider=self.voice_activator, on_stop_callback=stop_speech
        )
        await asyncio.to_thread(self.interrupt_speech_thread.start)

    async def stop_speech_interrupt_thread(self) -> None:
        self.logger.info("Stopping speech interrupt thread")
        if self.interrupt_speech_thread is not None:
            self.logger.debug("Signaling the speech interrupt thread to stop...")
            await asyncio.to_thread(self.interrupt_speech_thread.stop)
            self.interrupt_speech_thread = None
            self.logger.info("Speech interrupt thread stopped.")
        else:
            self.logger.info("No speech interrupt thread is currently running.")
