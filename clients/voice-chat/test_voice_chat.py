import io
import threading
import time
import wave

import numpy as np

from voice_chat import (
    PcmPlayback,
    VoiceChat,
    WakeListener,
    is_conversation_idle,
    is_goodbye_phrase,
)


class FakeOutputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chunks = []
        self.started = False
        self.stopped = False
        self.closed = False
        self.write_entered = threading.Event()
        self.release_write = threading.Event()
        self.block_write = False

    def start(self):
        self.started = True

    def write(self, chunk):
        self.write_entered.set()
        if self.block_write:
            # Unblock when abort closes the stream or the test releases us.
            while not self.release_write.wait(timeout=0.01):
                if self.closed or self.stopped:
                    raise OSError("stream closed during write")
        self.chunks.append(chunk)

    def stop(self):
        self.stopped = True
        self.release_write.set()

    def close(self):
        self.closed = True
        self.release_write.set()


class FakeSoundDevice:
    def __init__(self):
        self.output_streams = []

    def RawOutputStream(self, **kwargs):
        stream = FakeOutputStream(**kwargs)
        self.output_streams.append(stream)
        return stream


def test_pcm_playback_reuses_one_stream_for_same_sample_rate():
    sounddevice = FakeSoundDevice()
    playback = PcmPlayback(sounddevice)
    playback.write(22050, b"\x01\x00")
    playback.write(22050, b"\x02\x00")
    playback.finish()

    assert len(sounddevice.output_streams) == 1
    stream = sounddevice.output_streams[0]
    assert stream.kwargs["samplerate"] == 22050
    assert stream.chunks == [b"\x01\x00", b"\x02\x00"]
    assert stream.started is True
    assert stream.stopped is True


def test_pcm_playback_abort_unblocks_mid_write():
    sounddevice = FakeSoundDevice()
    playback = PcmPlayback(sounddevice)
    playback.write(22050, b"\x01\x00")
    deadline = time.time() + 2
    while not sounddevice.output_streams and time.time() < deadline:
        time.sleep(0.01)
    stream = sounddevice.output_streams[0]
    stream.block_write = True
    stream.write_entered.clear()
    playback.write(22050, b"\x02\x00")
    assert stream.write_entered.wait(timeout=2)
    playback.abort()
    playback._thread.join(timeout=2)
    assert not playback._thread.is_alive()
    assert stream.closed is True or stream.stopped is True
    playback.write(22050, b"\x03\x00")
    assert b"\x03\x00" not in stream.chunks


def test_pcm_playback_abort_drains_queued_chunks():
    sounddevice = FakeSoundDevice()
    playback = PcmPlayback(sounddevice)
    playback.write(22050, b"\x01\x00")
    deadline = time.time() + 2
    while not sounddevice.output_streams and time.time() < deadline:
        time.sleep(0.01)
    stream = sounddevice.output_streams[0]
    stream.block_write = True
    assert stream.write_entered.wait(timeout=2)
    playback.write(22050, b"\x02\x00")
    playback.write(22050, b"\x03\x00")
    playback.abort()
    playback._thread.join(timeout=2)
    assert playback._queue.empty()
    assert b"\x03\x00" not in stream.chunks


def test_cancel_mid_turn_stops_tts_worker_without_playing_remainder():
    chat = VoiceChat()
    cancel = threading.Event()

    def fake_stream_speech(text, cancel=None):
        for piece in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
            if cancel is not None and cancel.is_set():
                return
            yield 22050, piece
            time.sleep(0.05)

    sounddevice = FakeSoundDevice()
    chat._sounddevice = sounddevice
    chat.stream_speech = fake_stream_speech  # type: ignore[method-assign]
    chat.use_audio = True

    try:
        pipeline = chat._start_tts_pipeline(time.perf_counter(), cancel)
        pipeline.sentences.put("one.")
        pipeline.sentences.put("two.")
        time.sleep(0.08)
        chat._cancel_turn(cancel, pipeline)
        pipeline.thread.join(timeout=2)
        assert not pipeline.thread.is_alive()
        total_chunks = sum(len(s.chunks) for s in sounddevice.output_streams)
        # Would be 6 chunks if both sentences fully played; cancel cuts that short.
        assert total_chunks < 6
    finally:
        chat.close()


class FakeInputStream:
    def __init__(self, frames, **kwargs):
        self.frames = iter(frames)
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, frame_count):
        frame = next(self.frames)
        assert len(frame) == frame_count
        return frame, False


class FakeInputSoundDevice:
    def __init__(self, frames):
        self.frames = frames
        self.input_kwargs = None

    def InputStream(self, **kwargs):
        self.input_kwargs = kwargs
        return FakeInputStream(self.frames, **kwargs)


class FakeVadModule:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.frame_byte_lengths = []

    def Vad(self, aggressiveness):
        assert aggressiveness == 2
        module = self

        class Vad:
            def is_speech(self, frame, sample_rate):
                assert sample_rate == 16000
                module.frame_byte_lengths.append(len(frame))
                return next(module.decisions)

        return Vad()


def test_recording_uses_valid_30ms_webrtc_frames_and_trims_silence():
    # Three silent pre-roll frames, six speech frames, then two silent endpoint frames.
    decisions = [False, False, False, True, True, True, True, True, True, False, False]
    frames = [np.zeros((480, 1), dtype=np.int16) for _ in decisions]
    vad_module = FakeVadModule(decisions)

    chat = VoiceChat()
    try:
        chat.silence_ms = 60
        chat.pre_roll_ms = 60
        chat.post_roll_ms = 30
        chat._np = np
        chat._sounddevice = FakeInputSoundDevice(frames)
        chat._webrtcvad = vad_module
        result = chat.record_from_mic()
    finally:
        chat.close()

    assert result is not None
    assert vad_module.frame_byte_lengths == [960] * len(decisions)
    with wave.open(io.BytesIO(result), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        # 1 silent pre-roll + 6 speech + 1 post-roll frames, each 480 samples.
        assert wav_file.getnframes() == 8 * 480


def test_history_is_bounded_to_configured_turns():
    chat = VoiceChat()
    try:
        chat.max_history_turns = 2
        chat.chat_history = [
            {"role": "system", "content": "system"},
            *[
                {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
                for index in range(10)
            ],
        ]
        chat._trim_history()
        assert chat.chat_history == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "6"},
            {"role": "assistant", "content": "7"},
            {"role": "user", "content": "8"},
            {"role": "assistant", "content": "9"},
        ]
    finally:
        chat.close()


def test_is_goodbye_phrase_matches_common_exits():
    assert is_goodbye_phrase("Goodbye!")
    assert is_goodbye_phrase("stop listening")
    assert is_goodbye_phrase("That's all.")
    assert is_goodbye_phrase("goodbye everyone")
    assert not is_goodbye_phrase("please don't say goodbye yet")
    assert not is_goodbye_phrase("what is the weather")


def test_is_conversation_idle():
    assert is_conversation_idle(0.0, 300.0, 300.0)
    assert not is_conversation_idle(0.0, 299.0, 300.0)
    assert not is_conversation_idle(0.0, 999.0, 0.0)


def test_wake_listener_fires_when_score_exceeds_threshold():
    class FakeOww:
        def __init__(self):
            self.calls = 0

        def predict(self, _frame):
            self.calls += 1
            return {"hey_jarvis": 0.1 if self.calls < 3 else 0.9}

    class FakeInputStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, frames):
            return np.zeros((frames, 1), dtype=np.int16), False

    class FakeSoundDevice:
        def InputStream(self, **kwargs):
            return FakeInputStream(**kwargs)

    oww = FakeOww()
    listener = WakeListener(
        FakeSoundDevice(),
        np,
        model_name="hey_jarvis",
        threshold=0.5,
        debounce_seconds=0.0,
        oww_model=oww,
    )
    listener.wait_for_wake()
    assert oww.calls >= 3


def test_run_conversation_session_exits_on_goodbye_without_llm_turn():
    chat = VoiceChat()
    chat._import_audio_libs = lambda: None  # type: ignore[method-assign]
    chat.use_audio = False
    turns: list[str] = []

    def fake_record(*, idle_deadline=None):
        return b"wav"

    def fake_transcribe(_audio):
        return "Goodbye!"

    def fake_run_turn(text, barge_in=False):
        turns.append(text)

    chat.record_from_mic = fake_record  # type: ignore[method-assign]
    chat.transcribe = fake_transcribe  # type: ignore[method-assign]
    chat._run_turn = fake_run_turn  # type: ignore[method-assign]
    try:
        assert chat._run_conversation_session() == "goodbye"
        assert turns == []
    finally:
        chat.close()


def test_run_conversation_session_exits_on_idle():
    chat = VoiceChat()
    chat.wake_idle_seconds = 0.05
    chat._import_audio_libs = lambda: None  # type: ignore[method-assign]
    chat.use_audio = False

    def fake_record(*, idle_deadline=None):
        time.sleep(0.06)
        return None

    chat.record_from_mic = fake_record  # type: ignore[method-assign]
    try:
        assert chat._run_conversation_session() == "idle"
    finally:
        chat.close()
