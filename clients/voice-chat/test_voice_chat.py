import io
import wave

import numpy as np

from voice_chat import PcmPlayback, VoiceChat


class FakeOutputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chunks = []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def write(self, chunk):
        self.chunks.append(chunk)

    def stop(self):
        self.stopped = True

    def close(self):
        pass


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
