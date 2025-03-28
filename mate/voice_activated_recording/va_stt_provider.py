import logging
import asyncio
import webrtcvad
from typing import Optional, AsyncGenerator

from mate.services import ServiceDiscovery
from mate.voice_activated_recording.va_interface import VoiceActivationInterface


class SttProviderWakeWord(VoiceActivationInterface):
    """
    A wakeword detection class that uses:
      1) WebRTC VAD to wait for any speech.
      2) Once speech is detected, uses the STTWhisperRemote streaming
         to capture and transcribe until the remote service closes or
         we find the wakeword in the transcript. If the wakeword is found,
         we return immediately.
      3) If the STT ends without detecting the wakeword, we go back to VAD listening.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger: logging.Logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.service_discovery: ServiceDiscovery = ServiceDiscovery()
        self.vad: webrtcvad.Vad = webrtcvad.Vad(mode=3)  # Very aggressive mode

        # Typically, your soundcard should produce 16-bit, 1-channel, 16kHz audio.
        # We'll chunk frames into 20ms slices (20ms => 320 samples @ 16kHz => 640 bytes).
        self.vad_frame_ms: int = 20

    async def listen_for_wake_word(self) -> None:
        """
        Continuously:
          - Use VAD to detect if someone starts speaking.
          - Then run the remote Whisper streaming to see if the wakeword is spoken.
          - If the STT ends without detecting the wakeword, go back to VAD listening.
        """
        self.logger.info("va_stt_provider: Listening for wake word: %s", self.wakeword)

        while True:
            self.logger.debug("waiting for speech via VAD...")
            speech_detected: bool = await self._wait_for_speech()
            if not speech_detected:
                continue

            self.logger.debug("Speech detected, starting remote transcription...")

            audio_stream: AsyncGenerator[bytes, None] = self.soundcard.get_record_stream()

            def on_ws_open() -> None:
                self.logger.debug("WebSocket opened.")

            def on_ws_close() -> None:
                self.logger.debug("WebSocket closed.")

            try:
                stt_provider = await self.service_discovery.get_best_service("STT")
                if stt_provider is None:
                    self.logger.error("No STT service available.")
                    await asyncio.sleep(1.0)
                    continue

                async for partial_text in stt_provider.transcribe_stream(
                    audio_stream=audio_stream,
                    websocket_on_close=on_ws_close,
                    websocket_on_open=on_ws_open,
                ):
                    if self.wakeword.lower() in partial_text.lower():
                        self.logger.info("Wake word '%s' detected!", self.wakeword)
                        return

                self.logger.debug("Remote STT ended. Going back to VAD listening...")

            except KeyboardInterrupt as e:
                self.logger.warning("User aborted in wake word section", exc_info=True)
                raise e
            except asyncio.CancelledError:
                self.logger.error("Cancelled.", exc_info=True)
                return
            except Exception as e:
                self.logger.error("Error in STT: %s", e, exc_info=True)
                await asyncio.sleep(1.0)

    async def _wait_for_speech(self) -> bool:
        """
        Continuously reads raw audio from the soundcard in small chunks.
        Checks each chunk for speech using WebRTC VAD.
        Returns True as soon as we detect speech.
        If the audio stream ends for some reason, we return False.
        """
        audio_stream: AsyncGenerator[bytes, None] = self.soundcard.get_record_stream()
        try:
            async for chunk in audio_stream:
                if self._chunk_has_speech(chunk):
                    return True
        except Exception as e:
            self.logger.error("_wait_for_speech: Audio stream ended or error: %s", e, exc_info=True)
        return False

    def _chunk_has_speech(self, chunk: bytes) -> bool:
        """
        Break `chunk` into 20ms frames and check if any frame has speech
        according to WebRTC VAD.
        - chunk is 16-bit, mono, 16kHz PCM => 1 sample = 2 bytes, 1 second = 32000 bytes
        - 20ms = 0.02s => 320 samples => 640 bytes
        """
        frame_size: int = int(self.soundcard.sample_rate * (self.vad_frame_ms / 1000.0))
        bytes_per_frame: int = frame_size * 2  # 16-bit => 2 bytes

        idx: int = 0
        while idx + bytes_per_frame <= len(chunk):
            frame: bytes = chunk[idx : idx + bytes_per_frame]
            if self.vad.is_speech(frame, self.soundcard.sample_rate):
                return True
            idx += bytes_per_frame
        return False
