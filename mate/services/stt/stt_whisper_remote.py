import json
import time
from typing import Callable, AsyncGenerator
import websocket
import threading
import asyncio
import logging
from urllib.parse import urlparse
from queue import Queue

from mate.services.stt.stt_interface import STTInterface
from websocket import WebSocket, WebSocketApp, ABNF

# see https://github.com/openai/whisper/discussions/1536
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

    def __init__(self):
        super().__init__("WorkstationSTTWhisper", 100)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.stt_endpoint = "http://192.168.0.75:8000/v1/audio/transcriptions?language=de"
        # use the http endpoint for websocket
        self.ws_url = self.stt_endpoint.replace('http://','ws://')
        websocket.enableTrace(False)
        # if True then the transcription send to the API server is stored as recording_TIMESTAMP.wav
        self.store_wav = False

    async def check_availability(self) -> bool:
        # 1) Parse out the host and port
        parsed = urlparse(self.stt_endpoint)
        host = parsed.hostname
        port = parsed.port

        # 2) Check if we can connect to host:port
        #    Using an asyncio open_connection for an async-friendly approach
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            print(f"[check_availability] Could not connect to host '{host}' on port {port}.")
            print(f"    Reason: {e}")
            return False
        return True

    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], websocket_on_close: Callable[[], None], websocket_on_open: Callable[[], None]) -> AsyncGenerator[str, None]:
        queue = Queue()  # Back channel for transcription results
        # A callback function for receiving messages from the WebSocket
        def on_message(wsc: WebSocket, message: str):
            try:
                result = json.loads(message)
                if 'text' in result and result['text'].strip():
                    res_txt = result['text'].strip().replace('  ',' ')
                    # remove unwanted response, see
                    # https://github.com/openai/whisper/discussions/1536
                    for txt in dataset_bias:
                        if txt in res_txt:
                            res_txt = res_txt.replace(txt, '')
                    if len(res_txt.strip()) > 8:
                        queue.put(res_txt.strip())  # Push the transcription result into the queue
            except json.JSONDecodeError:
                self.logger.warning(f"got non json: {message}")
                pass  # Ignore non-JSON messages
        # A synchronous callback for handling WebSocket connection establishment
        thread_stop_event =  threading.Event()
        # Collect the WAV data from the stream into a BytesIO buffer
        def on_open(wsc2: WebSocket):
            self.logger.debug(f"Successfully connected websocket {self.ws_url}")
            # call external callback
            websocket_on_open()
            def send_audio_chunks():
                try:
                    start_time_sending = time.time()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    async def send_chunks():
                            async for wav_chunk in audio_stream:
                                if thread_stop_event.is_set():  # Check if the stop event is set
                                    #self.logger.info("stt_whisper_remote.transcribe_stream.on_open.send_audio_chunks.send_chunks: Stop signal received, exiting send_chunks.")
                                    loop.stop()
                                    break
                                # Websocket needs raw PCM (pcm_s16le) encoded bytes.
                                # Only transcription of a single channel, 16000 sample rate, raw, 16-bit little-endian
                                # audio is supported.
                                wsc2.send(wav_chunk, opcode=ABNF.OPCODE_BINARY)
                    loop.run_until_complete(send_chunks())
                except KeyboardInterrupt as e:
                    # stopped by the user
                    websocket_on_close()
                    thread_stop_event.set()
                    raise e
                except BaseException as e:
                    self.logger.error(f"Error in send_audio_chunks: {e}")
                    websocket_on_close()
                    thread_stop_event.set()
                finally:
                    self.logger.debug(f"Sent data to websocket for {time.time()-start_time_sending} seconds. Cleaning up thread and ws resources.")
                    if wsc2:
                        wsc2.close()  # Close the WebSocket connection
                    if loop:
                        if loop.is_running():
                            loop.stop()
                        if not loop.is_closed():
                            loop.close()
                    thread_stop_event.set()
            # Start the thread to run async
            threading.Thread(target=send_audio_chunks, daemon=True).start()
        def on_error(ws, code):
            self.logger.error(f"WebSocket error: {code}")
            websocket_on_close()
            thread_stop_event.set()
        def on_close(ws: WebSocket, i1, i2):
            self.logger.debug(f"WebSocket closed: {i1}, {i2}")
            queue.put(None)
            websocket_on_close()
            thread_stop_event.set()
        try:
            self.logger.debug(f"Starting websocket connection to {self.ws_url}")
            ws = WebSocketApp(
                self.ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            # Wait for WebSocket to complete
            # Yield transcription results from the queue
            old_full_text = ''
            while not thread_stop_event.is_set():
                t = queue.get()
                if t is None:
                    break
                t_diff = t[len(old_full_text):]
                # update the text
                old_full_text = t
                self.logger.info(f"got: {t_diff}")
                yield t_diff
            ws_thread.join()
            self.logger.debug(f"Transcription queue closed")
        except KeyboardInterrupt as e:
            # stopped by the user
            raise e
        except BaseException as e:
            self.logger.error(f"type={type(e)}, e={e}")
            thread_stop_event.set()
            if ws:
                ws.close()  # Gracefully close the WebSocket
            if ws_thread and ws_thread.is_alive():
                ws_thread.join(timeout=5)  # Ensure the thread has stopped
        finally:
            self.logger.debug(f"Cleanup completed")

    def config_str(self):
        return f'endpoint: {self.stt_endpoint}'