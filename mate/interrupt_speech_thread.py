import threading
import asyncio
import logging
from typing import Callable, Optional

from mate.voice_activated_recording.va_interface import VoiceActivationInterface


class InterruptSpeechThread:
    """
    A class that manages a thread which periodically performs a task in a loop,
    controlled via a threading.Event (stop event).
    """

    def __init__(
        self,
        stop_event: threading.Event,
        va_provider: VoiceActivationInterface,
        on_stop_callback: Callable[[], None]
    ) -> None:
        """
        :param stop_event: A threading.Event that can be used to signal this thread to stop.
        :param va_provider: An instance of VoiceActivationInterface for wake word detection.
        :param on_stop_callback: A callback to be executed when the wake word is detected.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.voice_activation: VoiceActivationInterface = va_provider
        self._stop_event: threading.Event = stop_event
        self._thread: Optional[threading.Thread] = None
        self.stop_callback: Callable[[], None] = on_stop_callback

    def start(self) -> None:
        """
        Starts the worker thread if it isn't already running.
        """
        self._stop_event.clear()
        if self._thread is not None and self._thread.is_alive():
            self.logger.warning("Attempt to start thread, but it is already running.")
            return
        self.logger.info("Starting the worker thread.")
        self._thread = threading.Thread(
            target=self._run,
            name="GracefulWorkerThread",
            daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Signals the worker thread to stop and waits for it to finish.
        """
        if self._thread is None:
            self.logger.warning("Attempt to stop thread, but no thread is running.")
            return

        self.logger.info("Signaling the worker thread to stop.")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self.logger.warning("Worker thread did not stop within the timeout.")
        else:
            self.logger.info("Worker thread has stopped.")

    def _run(self) -> None:
        """
        The main loop of the thread. It checks the stop event regularly and
        can also trigger it internally if a condition occurs.
        """
        self.logger.debug("Listen for %s as speech interrupt word", self.voice_activation.wakeword)
        asyncio.run(self.voice_activation.listen_for_wake_word(self._stop_event))
        self.logger.info(
            'Interrupt speech. Detected wake word "%s" as speech interrupt word',
            self.voice_activation.wakeword
        )
        self._stop_event.set()
        self.stop_callback()
        self.logger.info("Worker thread loop is exiting gracefully.")
