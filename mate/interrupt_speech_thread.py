import threading
import asyncio
import logging
from typing import Callable

from mate.voice_activated_recording.va_interface import VoiceActivationInterface


class InterruptSpeechThread:
    """
    A class that manages a thread which periodically performs a task in a loop,
    controlled via a threading.Event (stop event).
    """

    def __init__(self, stop_event: threading.Event, va_provider: VoiceActivationInterface, on_stop_callback: Callable[[],None]):
        """
        :param stop_event: A threading.Event that can be used to signal this thread to stop.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.voice_activation = va_provider
        self._stop_event = stop_event
        self._thread = None
        self.stop_callback = on_stop_callback

    def start(self):
        """
        Starts the worker thread if it isn't already running.
        """
        self._stop_event.clear()
        if self._thread is not None and self._thread.is_alive():
            self.logger.warning("Attempt to start thread, but it is already running.")
            return
        self.logger.info("Starting the worker thread.")
        self._thread = threading.Thread(target=self._run, name="GracefulWorkerThread", daemon=True)
        self._thread.start()

    def stop(self):
        """
        Signals the worker thread to stop and waits for it to finish.
        """
        if self._thread is None:
            self.logger.warning("Attempt to stop thread, but no thread is running.")
            return

        self.logger.info("Signaling the worker thread to stop.")
        self._stop_event.set()  # Signal the thread to stop

        # Optionally, we can join the thread to ensure it exits before continuing
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self.logger.warning("Worker thread did not stop within the timeout.")
        else:
            self.logger.info("Worker thread has stopped.")


    def _run(self):
        """
        The main loop of the thread. It checks the stop event regularly and
        can also trigger it internally if a condition occurs.
        """
        # Perform the thread's tasks here
        self.logger.debug(f"Listen for {self.voice_activation.wakeword} as speech interrupt word")
        asyncio.run(self.voice_activation.listen_for_wake_word(self._stop_event))
        self.logger.info(f"Interrupt speech. Detected wake word \"{self.voice_activation.wakeword}\" as speech interrupt word")
        self._stop_event.set()
        self.stop_callback()
        self.logger.info("Worker thread loop is exiting gracefully.")
