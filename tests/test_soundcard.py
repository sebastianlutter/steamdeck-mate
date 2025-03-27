import pytest
import asyncio
import numpy as np
from mate.audio.soundcard_pyaudio import SoundCard


@pytest.mark.asyncio
async def test_soundcard_init_and_close():
    """
    Test that SoundCard can be instantiated and closed without error.
    """
    soundcard = SoundCard()
    # Immediately close the device
    soundcard.close()


@pytest.mark.asyncio
async def test_soundcard_record_stream_stop():
    """
    Test that we can start reading from the record_stream and then
    stop without error. We only do a short loop for demonstration.
    """
    soundcard = SoundCard()

    record_count = 0
    record_task = asyncio.create_task(_consume_mic(soundcard, max_frames=3))

    # Let the recording run for ~1 second (to gather a few frames)
    await asyncio.sleep(1.0)

    # Now stop recording and cancel the consumer
    soundcard.stop_recording()
    record_task.cancel()
    try:
        await record_task
    except asyncio.CancelledError:
        pass

    # Finally close
    soundcard.close()


async def _consume_mic(soundcard: SoundCard, max_frames: int = 5):
    """
    Helper coroutine that consumes microphone audio from the async generator,
    to confirm that it doesn't block or crash. We'll only gather a few frames
    for test demonstration; real tests might gather more.
    """
    frame_count = 0
    async for chunk in soundcard.get_record_stream():
        frame_count += 1
        if frame_count >= max_frames:
            break


@pytest.mark.asyncio
async def test_soundcard_play_stop():
    """
    Test that we can enqueue some audio for playback and then stop it.
    """
    soundcard = SoundCard()

    # Some dummy PCM data (1 second of random noise)
    audio_data = np.random.randn(16000).astype(np.float32)

    # Start playback
    soundcard.play_audio(sample_rate=16000, audio_data=audio_data)

    # Immediately stop
    soundcard.stop_playback()

    # Cleanup
    soundcard.close()


@pytest.mark.asyncio
async def test_soundcard_play_and_wait():
    """
    Test that we can play audio, then wait until playback finishes.
    """
    soundcard = SoundCard()

    # A short random audio buffer
    audio_data = np.random.randn(16000).astype(np.float32)

    soundcard.play_audio(sample_rate=16000, audio_data=audio_data)

    # Wait for playback to finish
    soundcard.wait_until_playback_finished()

    # Cleanup
    soundcard.close()
