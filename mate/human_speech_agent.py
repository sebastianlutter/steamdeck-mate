import random
import threading
import time
import os
import io
import hashlib
import logging
from pydub import AudioSegment
from typing import AsyncGenerator, Tuple
from mate.audio.soundcard_pyaudio import SoundCard
from mate.services.interrupt_speech_thread import InterruptSpeechThread
from mate.services.voice_activated_recording.va_factory import VoiceActivatedRecordingFactory
from tqdm import tqdm
import soundfile as sf

from mate.services import ServiceDiscovery

format_string = (
    "%(asctime)s - [Logger: %(name)s] - %(levelname)s - %(filename)s:%(lineno)d in %(funcName)s() - %(message)s"
)
logging.basicConfig(format=format_string, level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('websocket').setLevel(logging.ERROR)

class HumanSpeechAgent:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Create new instance
                    cls._instance = super(HumanSpeechAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Ensure __init__ only runs once per singleton instance
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.interrupt_speech_thread = None
        self.soundcard = SoundCard()
        self.service_discovery = ServiceDiscovery()
        self.voice_activator = VoiceActivatedRecordingFactory()
        self.silence_lead_time = 2
        self.max_recording_time = 15
        self.stop_signal = threading.Event()
        self.abort_speech_choices = ["Anwort abgebrochen, was soll ich tun?"]
        self.hi_choices = [
            'ja, hi', 'schiess los!',
            'was gibts?', 'hi, was geht?',
            'leg los!', 'was willst du?',
            'sprechen Sie', 'jo bro', 'hey ho bro',
            'was geht so?'
        ]
        self.bye_choices = [
            "Auf Wiedersehen!", "Mach’s gut!", "Bis zum nächsten Mal!", "Schönen Tag noch!",
            "Bis bald!", "Pass auf dich auf!", "Bleib gesund!", "Man sieht sich!", "Bis später!", "Bis dann!",
            "Gute Reise!", "Viel Erfolg noch!", "Danke und tschüss!", "Alles Gute!", "Bis zum nächsten Treffen!",
            "Leb wohl!"
        ]
        init_greetings_identity = ""
        self.init_greetings = [
            "Guten Tag!", "Hi, wie geht's?", "Schön dich zu sehen!", "Hallo und willkommen!",
            "Freut mich, dich zu treffen!", "Hallo zusammen!", "Hallo, mein Freund!",
            "Guten Tag, wie kann ich helfen?", "Willkommen!", "Hallo an alle!",
            "Herzlich willkommen!", "Hallo, schön dich hier zu haben!", "Hey, alles klar?",
            "Hallo, schön dich kennenzulernen!", "Hallo, wie läuft's?", "Einen schönen Tag!"
        ]
        self.init_greetings = list(map(lambda g: f'{init_greetings_identity} {g}', self.init_greetings))
        self.did_not_understand = [
            "Das war unverständlich, bitte wiederholen"
        ]
        self.explain_sentence = "Sag das wort computer um zu starten."
        self._warmup_cache()

    def engage_input_beep(self):
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio("sounds/deskviewerbeep.mp3")
        self.soundcard.play_audio(sample_rate, audio_buffer)

    def beep_positive(self):
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio("sounds/computerbeep_26.mp3")
        self.soundcard.play_audio(sample_rate, audio_buffer)

    def beep_error(self):
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio("sounds/denybeep1.mp3")
        self.soundcard.play_audio(sample_rate, audio_buffer)

    def processing_sound(self):
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio("sounds/processing.mp3")
        self.soundcard.play_audio(sample_rate, audio_buffer)

    def say_abort_speech(self):
        tts_provider = self.service_discovery.get_best_service("TTS")
        tts_provider.set_stop_signal()
        tts_provider.soundcard.stop_playback()
        hi_phrase = random.choice(self.abort_speech_choices)
        mp3_path = self._get_cache_file_name(hi_phrase)
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        self.soundcard.play_audio(sample_rate, audio_buffer)

    def say_init_greeting(self):
        hi_phrase = random.choice(self.init_greetings)
        mp3_path = self._get_cache_file_name(hi_phrase)
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        self.soundcard.play_audio(sample_rate, audio_buffer)
        tts_provider = self.service_discovery.get_best_service("TTS")
        tts_provider.speak(f"Ich höre auf den Namen {self.voice_activator.wakeword}")
        tts_provider.wait_until_done()
        self.soundcard.wait_until_playback_finished()

    def say_hi(self):
        hi_phrase = random.choice(self.hi_choices)
        mp3_path = self._get_cache_file_name(hi_phrase)
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        self.logger.info(f"say_hi: {hi_phrase}")
        self.soundcard.play_audio(sample_rate, audio_buffer)

    def say_bye(self, message: str = ''):
        bye_phrase = random.choice(self.bye_choices)
        mp3_path = self._get_cache_file_name(bye_phrase)
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        self.logger.info(f"say_bye: {message}{bye_phrase}")
        tts_provider = self.service_discovery.get_best_service("TTS")
        if message != '':
            tts_provider.speak(message)
        tts_provider.wait_until_done()
        self.soundcard.play_audio(sample_rate, audio_buffer)

    def say_did_not_understand(self):
        did_not_understand_phrase = random.choice(self.did_not_understand)
        mp3_path = self._get_cache_file_name(did_not_understand_phrase)
        sample_rate, audio_buffer = self._load_mp3_to_wav_bytesio(mp3_path)
        self.logger.info(f"say_did_not_understand: {did_not_understand_phrase}")
        self.soundcard.play_audio(sample_rate, audio_buffer)

    def say(self, message: str):
        self.logger.debug(f"say: {message}")
        tts_provider = self.service_discovery.get_best_service("TTS")
        tts_provider.speak(message)

    def skip_all_and_say(self, message: str):
        self.logger.info(f"Skip all and say: {message}")
        # first skip all speaking tasks
        tts_provider = self.service_discovery.get_best_service("TTS")
        tts_provider.set_stop_signal()
        tts_provider.wait_until_done()
        time.sleep(0.2)
        tts_provider.clear_stop_signal()
        tts_provider.speak(message)

    def wait_until_talking_finished(self):
        self.logger.info("block_until_talking_finished: blocking")
        tts_provider = self.service_discovery.get_best_service("TTS")
        c = 0
        while c < 2:
            c+=1
            tts_provider.wait_until_done()
            tts_provider.soundcard.wait_until_playback_finished()
        self.logger.info("block_until_talking_finished: unblocking")

    async def get_human_input(self, wait_for_wakeword: bool = True) -> AsyncGenerator[str, None]:
        if wait_for_wakeword:
            self.soundcard.stop_recording()
            self.soundcard.wait_until_playback_finished()
            self.engage_input_beep()
            await self.voice_activator.listen_for_wake_word(stop_signal=None)
            self.beep_positive()

        def on_close_ws_callback():
            self.logger.debug("get_human_input.on_close_ws_callback: websocket closed")

        def on_ws_open():
            self.logger.debug("on_ws_open: Should say_hi now ws is opened:")

        stt_provider = self.service_discovery.get_best_service("STT")
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

    def _warmup_cache(self):
        # Ensure the tts_cache directory exists
        os.makedirs("tts_cache", exist_ok=True)
        # Pre-render all hi and bye choices to mp3
        all_choices = (self.hi_choices + self.bye_choices + self.init_greetings
                       + [self.explain_sentence] + self.did_not_understand + self.abort_speech_choices)
        for sentence in tqdm(all_choices, desc="Warmup cache with hi and bye phrases"):
            file_name = self._get_cache_file_name(sentence)
            if not os.path.exists(file_name):
                # Render the sentence to the specified file in mp3 format
                tts_provider = self.service_discovery.get_best_service("TTS")
                tts_provider.render_sentence(sentence=sentence, store_file_name=file_name, output_format='mp3')

    def _get_cache_file_name(self, sentence: str):
        # Create a short hash based on the sentence content
        hash_obj = hashlib.md5(sentence.encode('utf-8'))
        # Truncate the hash for a shorter filename if desired
        hash_str = hash_obj.hexdigest()[:8]
        return os.path.join("tts_cache/", f"{hash_str}.mp3")

    def _load_mp3_to_wav_bytesio(self, mp3_path: str) -> Tuple[int, io.BytesIO]:
        # Load the MP3 file into an AudioSegment
        audio_segment = AudioSegment.from_mp3(mp3_path)
        # Create a BytesIO buffer to hold WAV data
        wav_bytes = io.BytesIO()
        # Export the audio segment as WAV into the BytesIO buffer
        audio_segment.export(wav_bytes, format="wav")
        # Reset the buffer's position to the beginning
        wav_bytes.seek(0)
        # convert to ndarray
        data, sample_rate = sf.read(wav_bytes, dtype='float32')
        return sample_rate, data

    def start_speech_interrupt_thread(self, ext_stop_signal: threading.Event):
        def stop_speech():
            # abort any playback and say we stopped
            self.say_abort_speech()

        self.logger.info("Start speech interrupt thread")
        self.interrupt_speech_thread = InterruptSpeechThread(stop_event=ext_stop_signal,
                                                             va_provider=self.voice_activator,
                                                             on_stop_callback=stop_speech)
        self.interrupt_speech_thread.start()

    def stop_speech_interrupt_thread(self):
        self.logger.info("Stopping speech interrupt thread")
        # Ensure the interrupt thread exists before trying to stop it
        if self.interrupt_speech_thread is not None:
            # Set the stop event so the thread knows it should shut down
            self.logger.debug("Signaling the speech interrupt thread to stop...")
            # Wait for the thread to finish
            self.interrupt_speech_thread.stop()
            # Clean up
            self.interrupt_speech_thread = None
            self.logger.info("Speech interrupt thread stopped.")
        else:
            self.logger.info("No speech interrupt thread is currently running.")