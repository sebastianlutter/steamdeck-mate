import asyncio
import numpy as np
from mate.audio.soundcard_pyaudio import SoundCard  # Make sure this path matches your real import


class SteamdeckMate:
    def __init__(self):
        self.running = False
        self.soundcard = SoundCard()

    async def listen_and_respond(self):
        """
        Continuously reads microphone audio via self.soundcard.get_record_stream().
        Whenever we detect (or assume we detect) a keyword like "hello", we generate a response
        from our mock LLM and speak it out loud via play_audio().
        """
        print("[Mate] Starting to listen...")
        self.running = True

        try:
            # Start recording with our async generator
            async for mic_chunk in self.soundcard.get_record_stream():
                # Here you could add real speech recognition code to detect "hello"
                # For now, we just assume we heard "hello" after some chunk arrives.
                print("[Mate] Heard microphone audio chunk... (pretend we recognized 'hello')")
                response = await self.ask_llm("hello")
                await self.speak_response(response)

                # If we've been told to stop, break out of the loop
                if not self.running:
                    break

        except asyncio.CancelledError:
            # If the task is canceled (e.g., from stop()), exit cleanly
            pass
        finally:
            print("[Mate] Recording loop has ended.")

    async def ask_llm(self, text: str) -> str:
        """
        Mocked asynchronous LLM call.
        """
        await asyncio.sleep(0.5)
        return f"Response to '{text}'"

    async def speak_response(self, response: str):
        """
        Example TTS playback using the soundcard. In a real scenario,
        you'd get a WAV/PCM buffer from your TTS engine and pass it to `play_audio()`.
        Here, we'll just pretend we have a float32 NumPy array as TTS output.
        """
        print(f"[Mate] Speaking: {response}")

        # Simulate generating a short audio clip (1 sec of random noise)
        # In real usage, replace with your TTS output (e.g., a WAV file).
        tts_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16 kHz

        # Enqueue the audio for playback
        self.soundcard.play_audio(sample_rate=16000, audio_data=tts_audio)

        # Wait until playback finishes
        await self.soundcard.wait_until_playback_finished()

        print("[Mate] Done speaking.")

    async def stop(self):
        """
        Stop the listening loop and clean up the soundcard resources.
        """
        print("[Mate] Stopping...")
        self.running = False

        # Stop further recording and playback
        self.soundcard.stop_recording()
        self.soundcard.stop_playback()

        # Give it a moment for any in-flight callbacks
        await asyncio.sleep(0.2)

        # Close the underlying PyAudio streams
        self.soundcard.close()
        print("[Mate] All resources closed. Stopped.")


async def main():
    mate = SteamdeckMate()

    # Start the listening loop in the background
    listen_task = asyncio.create_task(mate.listen_and_respond())

    # Let it run for ~10 seconds, then stop
    await asyncio.sleep(10)
    await mate.stop()

    # Wait for the listening task to finish
    listen_task.cancel()
    try:
        await listen_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
