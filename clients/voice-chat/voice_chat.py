#!/usr/bin/env python3
"""Mic -> STT -> streaming LLM -> queued Piper TTS voice chat client."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Generator, Iterator, Optional

import httpx

from pipeline import ReasoningFilter, SentenceChunker, parse_sse_stream

_BARGE_MIN_VOICED_MS = 200
_WAKE_CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz (openWakeWord default)
_GOODBYE_PHRASES = frozenset(
    {
        "goodbye",
        "good bye",
        "bye",
        "stop listening",
        "that's all",
        "thats all",
        "that is all",
    }
)


def get_env(key: str, default: str) -> str:
    return os.getenv(key, default)


def normalize_utterance(text: str) -> str:
    cleaned = text.strip().lower()
    return cleaned.rstrip(".,!?;:").strip()


def is_goodbye_phrase(text: str) -> bool:
    """True when STT text is a session-end phrase (do not send to the LLM)."""
    normalized = normalize_utterance(text)
    if not normalized:
        return False
    if normalized in _GOODBYE_PHRASES:
        return True
    return any(normalized.startswith(f"{phrase} ") for phrase in _GOODBYE_PHRASES)


def is_conversation_idle(last_activity: float, now: float, idle_seconds: float) -> bool:
    return idle_seconds > 0 and (now - last_activity) >= idle_seconds


def wake_score_key(prediction: dict, model_name: str) -> str | None:
    """Resolve which prediction dict key matches the configured wake model."""
    if model_name in prediction:
        return model_name
    # openWakeWord sometimes keys by stem (hey_jarvis) or path basename.
    for key in prediction:
        if key == model_name or key.endswith(model_name) or model_name in key:
            return key
    return None


class WakeListener:
    """Block until openWakeWord reports the configured wake phrase."""

    def __init__(
        self,
        sounddevice,
        np_module,
        *,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
        debounce_seconds: float = 1.5,
        oww_model=None,
    ) -> None:
        self._sounddevice = sounddevice
        self._np = np_module
        self.model_name = model_name
        self.threshold = threshold
        self.debounce_seconds = debounce_seconds
        self._model = oww_model

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        import openwakeword.utils
        from openwakeword.model import Model

        openwakeword.utils.download_models()
        self._model = Model(
            wakeword_models=[self.model_name],
            inference_framework="onnx",
        )
        return self._model

    def wait_for_wake(self) -> None:
        model = self._ensure_model()
        sample_rate = 16000
        print(
            f"Wake listening for '{self.model_name}' "
            f"(threshold={self.threshold:.2f}; Ctrl-C to exit)...",
            file=sys.stderr,
        )
        with self._sounddevice.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=_WAKE_CHUNK_SAMPLES,
            latency="low",
        ) as stream:
            while True:
                audio_chunk, _overflowed = stream.read(_WAKE_CHUNK_SAMPLES)
                frame = self._np.asarray(audio_chunk, dtype=self._np.int16).reshape(-1)
                if frame.size < _WAKE_CHUNK_SAMPLES:
                    padded = self._np.zeros(_WAKE_CHUNK_SAMPLES, dtype=self._np.int16)
                    padded[: frame.size] = frame
                    frame = padded
                elif frame.size > _WAKE_CHUNK_SAMPLES:
                    frame = frame[:_WAKE_CHUNK_SAMPLES]
                prediction = model.predict(frame)
                key = wake_score_key(prediction, self.model_name)
                if key is None:
                    continue
                score = float(prediction[key])
                if score >= self.threshold:
                    print(f"(Wake word detected: {key}={score:.2f})", file=sys.stderr)
                    time.sleep(self.debounce_seconds)
                    return


class PcmPlayback:
    """Feed one persistent PortAudio stream from a thread-safe PCM queue."""

    def __init__(self, sounddevice) -> None:
        self._sounddevice = sounddevice
        self._queue: queue.Queue[tuple[int, bytes] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._aborted = False
        self._stream = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, sample_rate: int, chunk: bytes) -> None:
        if self._aborted or not chunk:
            return
        self._queue.put((sample_rate, chunk))

    def finish(self) -> None:
        if not self._aborted:
            self._queue.put(None)
        self._thread.join(timeout=5)

    def abort(self) -> None:
        """Stop playback immediately, including mid-write on the PortAudio stream."""
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
            stream = self._stream
            self._stream = None
        self._drain_queue()
        self._queue.put(None)
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self._thread.join(timeout=2)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        sample_rate = None
        try:
            while True:
                if self._aborted:
                    break
                item = self._queue.get()
                if item is None or self._aborted:
                    break
                item_rate, chunk = item
                with self._lock:
                    if self._aborted:
                        break
                    stream = self._stream
                    if stream is None or item_rate != sample_rate:
                        if stream is not None:
                            try:
                                stream.stop()
                                stream.close()
                            except Exception:
                                pass
                        sample_rate = item_rate
                        stream = self._sounddevice.RawOutputStream(
                            samplerate=sample_rate,
                            channels=1,
                            dtype="int16",
                            latency="low",
                        )
                        stream.start()
                        self._stream = stream
                try:
                    stream.write(chunk)
                except Exception:
                    if self._aborted:
                        break
                    raise
        except Exception as exc:
            if not self._aborted:
                print(f"Error playing audio: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                stream = self._stream
                self._stream = None
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass


@dataclass
class TtsPipeline:
    sentences: queue.Queue[str | None]
    thread: threading.Thread
    playback: PcmPlayback
    cancel: threading.Event


class VoiceChat:
    def __init__(self) -> None:
        self.voicebox_url = get_env("VOICEBOX_URL", "http://localhost:8790")
        self.voicebox_api_key = os.getenv("VOICEBOX_API_KEY")
        self.llm_url = get_env("VOICEBOX_LLM_URL", "http://localhost:8000/v1/chat/completions")
        self.llm_api_key = os.getenv("VOICEBOX_LLM_API_KEY")
        self.llm_model = get_env("VOICEBOX_LLM_MODEL", "local-model")
        self.voice = get_env("VOICEBOX_VOICE", "en_US-amy-medium")
        self.language = get_env("VOICEBOX_LANGUAGE", "en")
        self.silence_ms = int(get_env("VOICEBOX_SILENCE_MS", "700"))
        self.pre_roll_ms = int(get_env("VOICEBOX_PRE_ROLL_MS", "300"))
        self.post_roll_ms = int(get_env("VOICEBOX_POST_ROLL_MS", "200"))
        self.vad_aggressiveness = int(get_env("VOICEBOX_VAD_AGGRESSIVENESS", "2"))
        self.max_audio_seconds = int(get_env("VOICEBOX_MAX_AUDIO_SECONDS", "120"))
        self.max_history_turns = int(get_env("VOICEBOX_MAX_HISTORY_TURNS", "8"))
        self.show_timings = get_env("VOICEBOX_SHOW_TIMINGS", "0") == "1"
        self.system_prompt = get_env(
            "VOICEBOX_SYSTEM_PROMPT",
            "You are a helpful assistant. Respond concisely in plain text suitable for speech.",
        )
        self.wake_model = get_env("VOICEBOX_WAKE_MODEL", "hey_jarvis")
        self.wake_threshold = float(get_env("VOICEBOX_WAKE_THRESHOLD", "0.5"))
        self.wake_idle_seconds = float(get_env("VOICEBOX_WAKE_IDLE_SECONDS", "300"))
        self.use_audio = True
        self.chat_history: list[dict[str, str]] = []
        self._sounddevice = None
        self._webrtcvad = None
        self._np = None
        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=3, read=60, write=30, pool=5),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        )

    def close(self) -> None:
        self._http.close()

    def _voicebox_headers(self) -> dict[str, str]:
        if self.voicebox_api_key:
            return {"Authorization": f"Bearer {self.voicebox_api_key}"}
        return {}

    def _llm_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        return headers

    def _import_audio_libs(self) -> None:
        try:
            import numpy
            import sounddevice

            self._np = numpy
            self._sounddevice = sounddevice
        except ImportError:
            if self.use_audio:
                print(
                    "Warning: sounddevice and numpy are required for audio; using text mode.",
                    file=sys.stderr,
                )
                self.use_audio = False

        try:
            import webrtcvad

            self._webrtcvad = webrtcvad
        except ImportError:
            print(
                "Warning: webrtcvad is unavailable; using adaptive energy detection.",
                file=sys.stderr,
            )

    def _post_audio_file(self, audio_bytes: bytes, endpoint: str) -> str:
        url = f"{self.voicebox_url}{endpoint}"
        try:
            response = self._http.post(
                url,
                headers=self._voicebox_headers(),
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"model": "stt", "language": self.language},
            )
            response.raise_for_status()
            return response.json().get("text", "")
        except (httpx.HTTPError, ValueError) as exc:
            print(f"Error contacting voicebox STT at {url}: {exc}", file=sys.stderr)
            return ""

    def _is_speech_frame(self, audio_chunk, sample_rate: int, vad, noise_levels) -> bool:
        if vad is not None:
            return vad.is_speech(audio_chunk.tobytes(), sample_rate)
        samples = audio_chunk.reshape(-1).astype(self._np.float32)
        rms = float(self._np.sqrt(self._np.mean(samples * samples)))
        noise_floor = float(self._np.median(noise_levels)) if noise_levels else 60.0
        is_speech = rms > max(180.0, noise_floor * 3.0)
        if not is_speech:
            noise_levels.append(rms)
        return is_speech

    def record_from_mic(self, *, idle_deadline: float | None = None) -> Optional[bytes]:
        """Record 16 kHz mono audio with speech-aware pre-roll and endpointing.

        If idle_deadline is set and reached before speech starts, return None so the
        wake loop can treat the conversation as idle.
        """
        if not self._sounddevice or self._np is None:
            print("Error: sounddevice and numpy are required", file=sys.stderr)
            return None

        sample_rate = 16000
        frame_ms = 30  # WebRTC VAD accepts only 10, 20, or 30 ms.
        frame_samples = sample_rate * frame_ms // 1000
        silence_limit = max(1, (self.silence_ms + frame_ms - 1) // frame_ms)
        pre_roll_limit = max(1, (self.pre_roll_ms + frame_ms - 1) // frame_ms)
        post_roll_frames = max(0, (self.post_roll_ms + frame_ms - 1) // frame_ms)
        min_voiced_frames = max(1, (180 + frame_ms - 1) // frame_ms)
        max_frames = self.max_audio_seconds * 1000 // frame_ms

        vad = self._webrtcvad.Vad(self.vad_aggressiveness) if self._webrtcvad is not None else None
        pre_roll = deque(maxlen=pre_roll_limit)
        noise_levels = deque(maxlen=50)
        frames = []
        speech_started = False
        silence_frames = 0
        voiced_frames = 0

        print("Recording... (speak, then pause; Ctrl-C to cancel)", file=sys.stderr)
        try:
            with self._sounddevice.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=frame_samples,
                latency="low",
            ) as stream:
                while len(frames) < max_frames:
                    if (
                        not speech_started
                        and idle_deadline is not None
                        and time.monotonic() >= idle_deadline
                    ):
                        print("(Conversation idle timeout)", file=sys.stderr)
                        return None
                    audio_chunk, overflowed = stream.read(frame_samples)
                    if overflowed:
                        print("Warning: microphone buffer overflow", file=sys.stderr)
                    audio_chunk = audio_chunk.copy()

                    if vad is not None:
                        is_speech = vad.is_speech(audio_chunk.tobytes(), sample_rate)
                    else:
                        samples = audio_chunk.reshape(-1).astype(self._np.float32)
                        rms = float(self._np.sqrt(self._np.mean(samples * samples)))
                        noise_floor = float(self._np.median(noise_levels)) if noise_levels else 60.0
                        is_speech = rms > max(180.0, noise_floor * 3.0)
                        if not speech_started and not is_speech:
                            noise_levels.append(rms)

                    if not speech_started:
                        pre_roll.append(audio_chunk)
                        if not is_speech:
                            continue
                        speech_started = True
                        frames.extend(pre_roll)
                        pre_roll.clear()
                        voiced_frames = 1
                        continue

                    frames.append(audio_chunk)
                    if is_speech:
                        voiced_frames += 1
                        silence_frames = 0
                    else:
                        silence_frames += 1

                    if voiced_frames >= min_voiced_frames and silence_frames >= silence_limit:
                        trim_frames = max(0, silence_frames - post_roll_frames)
                        if trim_frames:
                            del frames[-trim_frames:]
                        print("Silence detected.", file=sys.stderr)
                        break
        except KeyboardInterrupt:
            print("\nRecording cancelled.", file=sys.stderr)
            raise
        except Exception as exc:
            print(f"Error recording: {exc}", file=sys.stderr)
            return None

        if not speech_started or not frames:
            print("No speech recorded.", file=sys.stderr)
            return None

        audio_data = self._np.concatenate(frames)
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        return wav_buffer.getvalue()

    def _watch_for_barge_in(self, cancel: threading.Event, done: threading.Event) -> None:
        """Detect sustained mic speech while TTS/LLM run; set cancel when heard."""
        if not self._sounddevice or self._np is None:
            return
        sample_rate = 16000
        frame_ms = 30
        frame_samples = sample_rate * frame_ms // 1000
        min_voiced = max(1, (_BARGE_MIN_VOICED_MS + frame_ms - 1) // frame_ms)
        vad = self._webrtcvad.Vad(self.vad_aggressiveness) if self._webrtcvad is not None else None
        noise_levels = deque(maxlen=50)
        voiced = 0
        try:
            with self._sounddevice.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=frame_samples,
                latency="low",
            ) as stream:
                while not cancel.is_set() and not done.is_set():
                    audio_chunk, _overflowed = stream.read(frame_samples)
                    audio_chunk = audio_chunk.copy()
                    if self._is_speech_frame(audio_chunk, sample_rate, vad, noise_levels):
                        voiced += 1
                        if voiced >= min_voiced:
                            print("\n(Barge-in detected)", file=sys.stderr)
                            cancel.set()
                            return
                    else:
                        voiced = 0
        except Exception as exc:
            if not cancel.is_set() and not done.is_set():
                print(f"Barge-in monitor error: {exc}", file=sys.stderr)

    def transcribe(self, audio_bytes: bytes) -> str:
        return self._post_audio_file(audio_bytes, "/v1/audio/transcriptions")

    def stream_llm_response(
        self, messages: list[dict[str, str]], cancel: threading.Event | None = None
    ) -> Iterator[bytes]:
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
        }
        try:
            with self._http.stream(
                "POST", self.llm_url, headers=self._llm_headers(), json=payload
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if cancel is not None and cancel.is_set():
                        break
                    yield chunk
        except httpx.HTTPError as exc:
            print(f"Error contacting LLM at {self.llm_url}: {exc}", file=sys.stderr)

    def stream_speech(
        self, text: str, cancel: threading.Event | None = None
    ) -> Generator[tuple[int, bytes], None, None]:
        url = f"{self.voicebox_url}/v1/audio/speech"
        payload = {
            "model": "tts",
            "input": text,
            "response_format": "pcm",
            "voice": self.voice,
        }
        try:
            with self._http.stream(
                "POST", url, headers=self._voicebox_headers(), json=payload
            ) as response:
                response.raise_for_status()
                sample_rate = int(response.headers.get("X-Audio-Sample-Rate", "22050"))
                for chunk in response.iter_bytes(chunk_size=4096):
                    if cancel is not None and cancel.is_set():
                        break
                    if chunk:
                        yield sample_rate, chunk
        except (httpx.HTTPError, ValueError) as exc:
            print(f"Error contacting voicebox TTS at {url}: {exc}", file=sys.stderr)

    def _start_tts_pipeline(self, turn_started: float, cancel: threading.Event) -> TtsPipeline:
        sentences: queue.Queue[str | None] = queue.Queue(maxsize=8)
        playback = PcmPlayback(self._sounddevice)

        def worker() -> None:
            first_audio = True
            try:
                while not cancel.is_set():
                    sentence = sentences.get()
                    if sentence is None or cancel.is_set():
                        break
                    for sample_rate, chunk in self.stream_speech(sentence, cancel=cancel):
                        if cancel.is_set():
                            break
                        if first_audio:
                            first_audio = False
                            if self.show_timings:
                                elapsed = time.perf_counter() - turn_started
                                print(f"[timing] first audio: {elapsed:.3f}s", file=sys.stderr)
                        playback.write(sample_rate, chunk)
            finally:
                if cancel.is_set():
                    playback.abort()
                else:
                    playback.finish()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return TtsPipeline(sentences=sentences, thread=thread, playback=playback, cancel=cancel)

    def _trim_history(self) -> None:
        max_messages = self.max_history_turns * 2
        if len(self.chat_history) > max_messages + 1:
            self.chat_history = [self.chat_history[0], *self.chat_history[-max_messages:]]

    def _cancel_turn(self, cancel: threading.Event, pipeline: TtsPipeline | None) -> None:
        cancel.set()
        if pipeline is None:
            return
        # Drop pending sentences so the worker can see None / cancel promptly.
        while True:
            try:
                pipeline.sentences.get_nowait()
            except queue.Empty:
                break
        try:
            pipeline.sentences.put_nowait(None)
        except queue.Full:
            pass
        pipeline.playback.abort()

    def _run_turn(self, user_text: str, barge_in: bool = False) -> None:
        self.chat_history.append({"role": "user", "content": user_text})
        self._trim_history()

        print("Assistant: ", end="", flush=True)
        turn_started = time.perf_counter()
        assistant_parts: list[str] = []
        reasoning_filter = ReasoningFilter()
        chunker = SentenceChunker()
        first_visible_token = True
        cancel = threading.Event()
        turn_done = threading.Event()
        pipeline: TtsPipeline | None = None
        barge_thread: threading.Thread | None = None

        if self.use_audio:
            pipeline = self._start_tts_pipeline(turn_started, cancel)
            if barge_in:
                barge_thread = threading.Thread(
                    target=self._watch_for_barge_in,
                    args=(cancel, turn_done),
                    daemon=True,
                )
                barge_thread.start()

        def process_visible_text(text: str) -> None:
            nonlocal first_visible_token
            if not text or cancel.is_set():
                return
            if first_visible_token:
                first_visible_token = False
                if self.show_timings:
                    elapsed = time.perf_counter() - turn_started
                    print(f"\n[timing] first LLM text: {elapsed:.3f}s", file=sys.stderr)
            assistant_parts.append(text)
            print(text, end="", flush=True)
            for sentence in chunker.feed(text):
                if pipeline is not None and not cancel.is_set():
                    try:
                        pipeline.sentences.put(sentence, timeout=0.1)
                    except queue.Full:
                        pass

        try:
            for token in parse_sse_stream(
                self.stream_llm_response(self.chat_history, cancel=cancel)
            ):
                if cancel.is_set():
                    break
                process_visible_text(reasoning_filter.feed(token))
            if not cancel.is_set():
                process_visible_text(reasoning_filter.flush())
                remainder = chunker.flush()
                if remainder and pipeline is not None:
                    try:
                        pipeline.sentences.put(remainder, timeout=0.1)
                    except queue.Full:
                        pass
        except KeyboardInterrupt:
            print("\n(LLM stream interrupted)", file=sys.stderr)
            self._cancel_turn(cancel, pipeline)
        finally:
            turn_done.set()
            if barge_thread is not None:
                barge_thread.join(timeout=1)
            if pipeline is not None:
                if cancel.is_set():
                    self._cancel_turn(cancel, pipeline)
                else:
                    try:
                        pipeline.sentences.put(None, timeout=0.5)
                    except queue.Full:
                        pass
                pipeline.thread.join(timeout=2 if cancel.is_set() else None)

        assistant_text = "".join(assistant_parts)
        print()
        if assistant_text:
            self.chat_history.append({"role": "assistant", "content": assistant_text})
            self._trim_history()

    def run_interactive(self, barge_in: bool = False) -> None:
        self._import_audio_libs()
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        mode = "barge-in on; headphones recommended" if barge_in else "half-duplex"
        print(f"Voice Chat CLI ({mode}; Ctrl-C to exit)", file=sys.stderr)
        try:
            while True:
                audio_bytes = self.record_from_mic()
                if not audio_bytes:
                    continue
                started = time.perf_counter()
                user_text = self.transcribe(audio_bytes)
                if self.show_timings:
                    print(
                        f"[timing] speech upload + STT: {time.perf_counter() - started:.3f}s",
                        file=sys.stderr,
                    )
                if not user_text:
                    print("(No speech recognized)", file=sys.stderr)
                    continue
                print(f"User: {user_text}")
                self._run_turn(user_text, barge_in=barge_in)
        except KeyboardInterrupt:
            print("\nExiting...", file=sys.stderr)

    def _speak_ack(self, text: str) -> None:
        """Play a short acknowledgment through voicebox TTS when audio is available."""
        if not self.use_audio or not self._sounddevice:
            print(text, file=sys.stderr)
            return
        playback = PcmPlayback(self._sounddevice)
        try:
            for sample_rate, chunk in self.stream_speech(text):
                playback.write(sample_rate, chunk)
        finally:
            playback.finish()

    def _run_conversation_session(self, barge_in: bool = False) -> str:
        """Run conversation turns until goodbye or idle. Returns 'goodbye' or 'idle'."""
        last_activity = time.monotonic()
        while True:
            now = time.monotonic()
            if is_conversation_idle(last_activity, now, self.wake_idle_seconds):
                return "idle"
            idle_deadline = last_activity + self.wake_idle_seconds
            audio_bytes = self.record_from_mic(idle_deadline=idle_deadline)
            if not audio_bytes:
                if is_conversation_idle(last_activity, time.monotonic(), self.wake_idle_seconds):
                    return "idle"
                continue
            started = time.perf_counter()
            user_text = self.transcribe(audio_bytes)
            if self.show_timings:
                print(
                    f"[timing] speech upload + STT: {time.perf_counter() - started:.3f}s",
                    file=sys.stderr,
                )
            if not user_text:
                print("(No speech recognized)", file=sys.stderr)
                continue
            print(f"User: {user_text}")
            if is_goodbye_phrase(user_text):
                return "goodbye"
            last_activity = time.monotonic()
            self._run_turn(user_text, barge_in=barge_in)

    def run_wake_loop(self, barge_in: bool = False) -> None:
        """Always-on wake listen → conversation → wake again until Ctrl-C."""
        self._import_audio_libs()
        if not self._sounddevice or self._np is None:
            print("Error: --wake requires sounddevice and numpy", file=sys.stderr)
            return
        if not self.chat_history:
            self.chat_history = [{"role": "system", "content": self.system_prompt}]
        mode = "barge-in on; headphones recommended" if barge_in else "half-duplex"
        print(
            f"Voice Chat wake mode ({mode}; say goodbye or wait "
            f"{int(self.wake_idle_seconds)}s idle to sleep; Ctrl-C to exit)",
            file=sys.stderr,
        )
        listener = WakeListener(
            self._sounddevice,
            self._np,
            model_name=self.wake_model,
            threshold=self.wake_threshold,
        )
        try:
            while True:
                listener.wait_for_wake()
                self._speak_ack("Listening.")
                reason = self._run_conversation_session(barge_in=barge_in)
                if reason == "goodbye":
                    self._speak_ack("Goodbye.")
                    print("(Session ended: goodbye)", file=sys.stderr)
                else:
                    self._speak_ack("Going to sleep.")
                    print("(Session ended: idle)", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nExiting...", file=sys.stderr)

    def run_text_mode(self, text: str, no_audio: bool = False) -> None:
        self.use_audio = not no_audio
        self._import_audio_libs()
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        print(f"User: {text}")
        self._run_turn(text)

    def run_file_mode(self, filepath: str, no_audio: bool = False) -> None:
        self.use_audio = not no_audio
        self._import_audio_libs()
        try:
            with open(filepath, "rb") as file:
                audio_bytes = file.read()
        except OSError as exc:
            print(f"Error reading file: {exc}", file=sys.stderr)
            return
        user_text = self.transcribe(audio_bytes)
        if not user_text:
            print("(No speech recognized)", file=sys.stderr)
            return
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        print(f"User: {user_text}")
        self._run_turn(user_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice chat using voicebox STT/Piper and an OpenAI-compatible LLM"
    )
    parser.add_argument("--text", help="Skip microphone input")
    parser.add_argument("--file", help="Transcribe an audio file before the LLM turn")
    parser.add_argument("--no-audio", action="store_true", help="Print without TTS playback")
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="Interrupt TTS when you speak (interactive mic mode; headphones recommended)",
    )
    parser.add_argument(
        "--wake",
        action="store_true",
        help=(
            "Always-on wake word (default hey jarvis), then conversation until "
            "goodbye or idle timeout"
        ),
    )
    args = parser.parse_args()
    if args.wake and (args.text or args.file):
        parser.error("--wake cannot be combined with --text or --file")

    chat = VoiceChat()
    try:
        if args.text:
            chat.run_text_mode(args.text, no_audio=args.no_audio)
        elif args.file:
            chat.run_file_mode(args.file, no_audio=args.no_audio)
        elif args.wake:
            chat.run_wake_loop(barge_in=args.barge_in)
        else:
            chat.run_interactive(barge_in=args.barge_in)
    finally:
        chat.close()


if __name__ == "__main__":
    main()
