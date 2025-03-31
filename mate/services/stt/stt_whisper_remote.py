import json
import time
from typing import Callable, AsyncGenerator, Optional, Any
import websocket
import threading
import asyncio
import logging
from urllib.parse import urlparse
from queue import Queue

from mate.services.stt.stt_interface import STTInterface
from websocket import WebSocket, WebSocketApp, ABNF


dataset_bias = [
    "Untertitel Vielen Dank für's Zuschauen und bis zum nächsten Mal!",
    "Vielen Dank für's Zuschauen",
    "Vielen Dank für Ihre Aufmerksamkeit",
    "Das war's. Bis zum nächsten Mal.",
    "Untertitelung aufgrund der Amara.org-Community",
    "Untertitel im Auftrag des ZDF für funk, 2017",
    "Untertitel von Stephanie Geiges",
    "Untertitel der Amara.org-Community",
    "Mehr Infos auf www .sommers -radio .de",
    "Ich danke Ihnen für Ihre Aufmerksamkeit.",
    "Die Amara.org-Community:",
    "Wir sehen uns im nächsten Video. Bis dann.",
    "Untertitel der Amara .org -Community",
    "der Amara .org -Community",
    "und bis zum nächsten Mal!",
    "Untertitel im Auftrag des ZDF, 2017",
    "Untertitel im Auftrag des ZDF, 2020",
    "Untertitel im Auftrag des ZDF, 2018",
    "Untertitel im Auftrag des ZDF, 2021",
    "Untertitelung im Auftrag des ZDF, 2021",
    "Copyright WDR 2021",
    "Copyright WDR 2020",
    "Copyright WDR 2019",
    "SWR 2021",
    "SWR 2020",
    "Bis zum nächsten Mal.",
    "Untertitel",
    "-Community",
    "Vielen Dank.",
    " Und tschau.",
    "Das war's."
]


class STTWhisperRemote(STTInterface):
    def __init__(self, name: str, priority: int, endpoint: str) -> None:
        super().__init__(name=name, priority=priority)
        self.logger: logging.Logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.stt_endpoint: str = endpoint
        self.ws_url: str = self.stt_endpoint.replace("http://", "ws://")
        websocket.enableTrace(False)
        self.store_wav: bool = False

    async def check_availability(self) -> bool:
        return await self.__check_remote_endpoint__(self.stt_endpoint)

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        websocket_on_close: Callable[[], None],
        websocket_on_open: Callable[[], None],
    ) -> AsyncGenerator[str, None]:
        queue_data: Queue[Optional[str]] = Queue()

        def on_message(wsc: WebSocket, message: str) -> None:
            try:
                result = json.loads(message)
                if "text" in result and result["text"].strip():
                    res_txt: str = result["text"].strip().replace("  ", " ")
                    for txt in dataset_bias:
                        if txt in res_txt:
                            res_txt = res_txt.replace(txt, "")
                    if len(res_txt.strip()) > 8:
                        queue_data.put(res_txt.strip())
            except json.JSONDecodeError:
                self.logger.warning("Got non-JSON message: %s", message)

        thread_stop_event: threading.Event = threading.Event()

        def on_open(wsc2: WebSocket) -> None:
            self.logger.info("Successfully connected websocket %s", self.ws_url)
            websocket_on_open()

            def send_audio_chunks() -> None:
                try:
                    start_time_sending: float = time.time()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    async def send_chunks() -> None:
                        async for wav_chunk in audio_stream:
                            if thread_stop_event.is_set():
                                loop.stop()
                                break
                            wsc2.send(wav_chunk, opcode=ABNF.OPCODE_BINARY)

                    loop.run_until_complete(send_chunks())
                except KeyboardInterrupt:
                    websocket_on_close()
                    thread_stop_event.set()
                    raise
                except BaseException as e:
                    self.logger.error("Error in send_audio_chunks: %s", e)
                    websocket_on_close()
                    thread_stop_event.set()
                finally:
                    self.logger.debug(
                        "Sent data to websocket for %f seconds. Cleaning up thread and ws resources.",
                        time.time() - start_time_sending,
                    )
                    try:
                        wsc2.close()
                    except Exception:
                        pass
                    if loop.is_running():
                        loop.stop()
                    if not loop.is_closed():
                        loop.close()
                    thread_stop_event.set()

            threading.Thread(target=send_audio_chunks, daemon=True).start()

        def on_error(ws_obj: WebSocket, code: str) -> None:
            self.logger.error("WebSocket error: %s", code)
            websocket_on_close()
            thread_stop_event.set()

        def on_close(ws_obj: WebSocket, i1: Any, i2: Any) -> None:
            self.logger.info("WebSocket closed: %s, %s", i1, i2)
            queue_data.put(None)
            websocket_on_close()
            thread_stop_event.set()

        ws: Optional[WebSocketApp] = None
        try:
            self.logger.debug("Starting websocket connection to %s", self.ws_url)
            ws = WebSocketApp(
                self.ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()

            old_full_text: str = ""
            while not thread_stop_event.is_set():
                t: Optional[str] = queue_data.get()
                if t is None:
                    break
                t_diff: str = t[len(old_full_text) :]
                old_full_text = t
                self.logger.info("got: %s", t_diff)
                yield t_diff

            ws_thread.join()
            self.logger.debug("Transcription queue closed")
        except KeyboardInterrupt as e:
            self.logger.info("Got KeyboarInterrupt")
            raise e
        except BaseException as e:
            self.logger.error("Error: %s, type=%s", e, type(e))
            thread_stop_event.set()
            if ws:
                ws.close()
            if ws_thread and ws_thread.is_alive():
                ws_thread.join(timeout=5)
        finally:
            self.logger.debug("Cleanup completed")

    def config_str(self) -> str:
        return f"endpoint: {self.stt_endpoint}"


class WorkstationSTTWhisper(STTWhisperRemote):
    config = {
        "name": "WorkstationSTTWhisper",
        "priority": 100,
        "endpoint": "http://192.168.0.75:8000/v1/audio/transcriptions?language=de",
    }

    def __init__(self) -> None:
        super().__init__(**self.config)


class SteamdeckSTTWhisper(STTWhisperRemote):
    config = {
        "name": "SteamdeckSTTWhisper",
        "priority": 0,
        "endpoint": "http://192.168.1.87:8000/v1/audio/transcriptions?language=de",
    }

    def __init__(self) -> None:
        super().__init__(**self.config)
