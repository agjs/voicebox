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
from io import BytesIO
from typing import Generator, Iterator, Optional

import httpx

from pipeline import ReasoningFilter, SentenceChunker, parse_sse_stream


def get_env(key: str, default: str) -> str:
    return os.getenv(key, default)


class PcmPlayback:
    """Feed one persistent PortAudio stream from a thread-safe PCM queue."""

    def __init__(self, sounddevice) -> None:
        self._sounddevice = sounddevice
        self._queue: queue.Queue[tuple[int, bytes] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, sample_rate: int, chunk: bytes) -> None:
        if chunk:
            self._queue.put((sample_rate, chunk))

    def finish(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        stream = None
        sample_rate = None
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                item_rate, chunk = item
                if stream is None or item_rate != sample_rate:
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    sample_rate = item_rate
                    stream = self._sounddevice.RawOutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="int16",
                        latency="low",
                    )
                    stream.start()
                stream.write(chunk)
        except Exception as exc:
            print(f"Error playing audio: {exc}", file=sys.stderr)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass


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

    def record_from_mic(self) -> Optional[bytes]:
        """Record 16 kHz mono audio with speech-aware pre-roll and endpointing."""
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
            return None
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

    def transcribe(self, audio_bytes: bytes) -> str:
        return self._post_audio_file(audio_bytes, "/v1/audio/transcriptions")

    def stream_llm_response(self, messages: list[dict[str, str]]) -> Iterator[bytes]:
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
                yield from response.iter_bytes()
        except httpx.HTTPError as exc:
            print(f"Error contacting LLM at {self.llm_url}: {exc}", file=sys.stderr)

    def stream_speech(self, text: str) -> Generator[tuple[int, bytes], None, None]:
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
                    if chunk:
                        yield sample_rate, chunk
        except (httpx.HTTPError, ValueError) as exc:
            print(f"Error contacting voicebox TTS at {url}: {exc}", file=sys.stderr)

    def _start_tts_pipeline(
        self, turn_started: float
    ) -> tuple[queue.Queue[str | None], threading.Thread]:
        sentences: queue.Queue[str | None] = queue.Queue(maxsize=8)
        playback = PcmPlayback(self._sounddevice)

        def worker() -> None:
            first_audio = True
            try:
                while True:
                    sentence = sentences.get()
                    if sentence is None:
                        break
                    for sample_rate, chunk in self.stream_speech(sentence):
                        if first_audio:
                            first_audio = False
                            if self.show_timings:
                                elapsed = time.perf_counter() - turn_started
                                print(f"[timing] first audio: {elapsed:.3f}s", file=sys.stderr)
                        playback.write(sample_rate, chunk)
            finally:
                playback.finish()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return sentences, thread

    def _trim_history(self) -> None:
        max_messages = self.max_history_turns * 2
        if len(self.chat_history) > max_messages + 1:
            self.chat_history = [self.chat_history[0], *self.chat_history[-max_messages:]]

    def _run_turn(self, user_text: str) -> None:
        self.chat_history.append({"role": "user", "content": user_text})
        self._trim_history()

        print("Assistant: ", end="", flush=True)
        turn_started = time.perf_counter()
        assistant_parts: list[str] = []
        reasoning_filter = ReasoningFilter()
        chunker = SentenceChunker()
        first_visible_token = True

        sentence_queue = None
        tts_thread = None
        if self.use_audio:
            sentence_queue, tts_thread = self._start_tts_pipeline(turn_started)

        def process_visible_text(text: str) -> None:
            nonlocal first_visible_token
            if not text:
                return
            if first_visible_token:
                first_visible_token = False
                if self.show_timings:
                    elapsed = time.perf_counter() - turn_started
                    print(f"\n[timing] first LLM text: {elapsed:.3f}s", file=sys.stderr)
            assistant_parts.append(text)
            print(text, end="", flush=True)
            for sentence in chunker.feed(text):
                if sentence_queue is not None:
                    sentence_queue.put(sentence)

        try:
            for token in parse_sse_stream(self.stream_llm_response(self.chat_history)):
                process_visible_text(reasoning_filter.feed(token))
            process_visible_text(reasoning_filter.flush())
            remainder = chunker.flush()
            if remainder and sentence_queue is not None:
                sentence_queue.put(remainder)
        except KeyboardInterrupt:
            print("\n(LLM stream interrupted)", file=sys.stderr)
        finally:
            if sentence_queue is not None:
                sentence_queue.put(None)
            if tts_thread is not None:
                tts_thread.join()

        assistant_text = "".join(assistant_parts)
        print()
        if assistant_text:
            self.chat_history.append({"role": "assistant", "content": assistant_text})
            self._trim_history()

    def run_interactive(self) -> None:
        self._import_audio_libs()
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        print("Voice Chat CLI (Ctrl-C to exit)", file=sys.stderr)
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
                self._run_turn(user_text)
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
    args = parser.parse_args()

    chat = VoiceChat()
    try:
        if args.text:
            chat.run_text_mode(args.text, no_audio=args.no_audio)
        elif args.file:
            chat.run_file_mode(args.file, no_audio=args.no_audio)
        else:
            chat.run_interactive()
    finally:
        chat.close()


if __name__ == "__main__":
    main()
