#!/usr/bin/env python3
"""
Voice chat CLI: mic→STT→LLM→streaming TTS turn-taking loop.

Uses voicebox for STT/TTS, local LLM (OpenAI-compatible), and sounddevice for audio.
Supports modes: default (mic), --text, --file.
"""

import argparse
import json
import os
import sys
from io import BytesIO
from typing import Generator, Optional
from pathlib import Path

from pipeline import parse_sse_stream, strip_reasoning, SentenceChunker


def get_env(key: str, default: str) -> str:
    """Get environment variable with a default."""
    return os.getenv(key, default)


class VoiceChat:
    """Main voice chat client."""

    def __init__(self):
        self.voicebox_url = get_env("VOICEBOX_URL", "http://localhost:8790")
        self.llm_url = get_env("VOICEBOX_LLM_URL", "http://localhost:8000/v1/chat/completions")
        self.llm_model = get_env("VOICEBOX_LLM_MODEL", "local-model")
        self.voice = get_env("VOICEBOX_VOICE", "af_heart")
        self.silence_ms = int(get_env("VOICEBOX_SILENCE_MS", "1500"))
        self.system_prompt = get_env("VOICEBOX_SYSTEM_PROMPT", "You are a helpful assistant.")
        self.use_audio = True
        self.chat_history = []

        # Lazy-import audio libraries
        self._sounddevice = None
        self._webrtcvad = None
        self._np = None

    def _import_audio_libs(self):
        """Lazy-import audio dependencies (fail gracefully if not installed)."""
        try:
            import sounddevice
            self._sounddevice = sounddevice
        except ImportError:
            if self.use_audio:
                print("Warning: sounddevice not installed. Install with: pip install sounddevice")
                self.use_audio = False

        try:
            import webrtcvad
            self._webrtcvad = webrtcvad
        except ImportError:
            pass

        try:
            import numpy
            self._np = numpy
        except ImportError:
            pass

    def _get_http_client(self):
        """Get an HTTP client (urllib or requests)."""
        try:
            import requests
            return requests
        except ImportError:
            # Fallback to urllib
            import urllib.request
            return urllib.request

    def _post_audio_file(self, audio_bytes: bytes, endpoint: str) -> str:
        """POST audio file to voicebox STT endpoint."""
        import urllib.request
        import urllib.error

        url = f"{self.voicebox_url}{endpoint}"
        try:
            # Prepare multipart form data manually (simple case)
            boundary = "----VoiceChat"
            body_parts = [
                f"--{boundary}".encode(),
                b'Content-Disposition: form-data; name="file"; filename="audio.wav"',
                b"Content-Type: audio/wav",
                b"",
                audio_bytes,
                f"--{boundary}--".encode(),
            ]
            body = b"\r\n".join(body_parts)

            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode())
                return result.get("text", "")
        except urllib.error.URLError as e:
            print(f"Error contacting voicebox STT at {url}: {e}", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"Error in STT: {e}", file=sys.stderr)
            return ""

    def record_from_mic(self) -> Optional[bytes]:
        """
        Record from mic until VAD detects ~silence_ms of silence.
        Returns WAV bytes or None on error/Ctrl-C.
        """
        if not self._sounddevice:
            print("Error: sounddevice not available", file=sys.stderr)
            return None

        try:
            import wave

            sample_rate = 16000
            chunk_duration_ms = 100
            chunk_samples = (sample_rate * chunk_duration_ms) // 1000

            print("Recording... (Ctrl-C to stop)", file=sys.stderr)

            frames = []
            silence_frames = 0
            silence_threshold = (self.silence_ms // chunk_duration_ms) + 1

            # Optional VAD (if available)
            vad = None
            if self._webrtcvad:
                vad = self._webrtcvad.VAD(3)  # Aggressiveness 0-3

            try:
                with self._sounddevice.InputStream(
                    samplerate=sample_rate, channels=1, dtype="int16"
                ) as stream:
                    while True:
                        audio_chunk, _ = stream.read(chunk_samples)

                        if self._np is not None:
                            # Convert to int16 if needed
                            if audio_chunk.dtype != "int16":
                                audio_chunk = (audio_chunk * 32767).astype("int16")

                        frames.append(audio_chunk)

                        # Simple VAD check if webrtcvad available
                        if vad:
                            frame_bytes = audio_chunk.tobytes()
                            is_speech = vad.is_speech(frame_bytes, sample_rate)
                            if not is_speech:
                                silence_frames += 1
                            else:
                                silence_frames = 0
                        else:
                            # Fallback: energy-based detection
                            if self._np is not None:
                                energy = float(self._np.sum(self._np.abs(audio_chunk)) / len(audio_chunk))
                                if energy < 100:  # Very low energy = silence
                                    silence_frames += 1
                                else:
                                    silence_frames = 0

                        if silence_frames >= silence_threshold and len(frames) > 5:
                            print("Silence detected.", file=sys.stderr)
                            break

            except KeyboardInterrupt:
                print("\nRecording cancelled.", file=sys.stderr)
                return None

            # Concatenate frames into a WAV
            if not frames:
                print("No audio recorded.", file=sys.stderr)
                return None

            if self._np is not None:
                audio_data = self._np.concatenate(frames)
            else:
                # Fallback without numpy
                import struct
                audio_list = []
                for frame in frames:
                    if hasattr(frame, "tobytes"):
                        audio_list.append(frame.tobytes())
                    else:
                        audio_list.append(frame)
                audio_bytes = b"".join(audio_list)
                # This is crude; with numpy it's cleaner
                return audio_bytes

            # Write WAV
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            return wav_buffer.getvalue()

        except Exception as e:
            print(f"Error recording: {e}", file=sys.stderr)
            return None

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio via voicebox STT."""
        return self._post_audio_file(audio_bytes, "/v1/audio/transcriptions")

    def stream_llm_response(self, messages: list) -> Generator[str, None, None]:
        """
        Stream LLM response using OpenAI-compatible endpoint.
        Yields raw text tokens.
        """
        import urllib.request
        import urllib.error

        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
        }

        try:
            req = urllib.request.Request(
                self.llm_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                for line in response:
                    yield line

        except urllib.error.URLError as e:
            print(f"Error contacting LLM at {self.llm_url}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error streaming LLM: {e}", file=sys.stderr)

    def synthesize_speech(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech via voicebox TTS.
        Returns PCM audio bytes (24 kHz) or None on error.
        """
        import urllib.request
        import urllib.error

        url = f"{self.voicebox_url}/v1/audio/speech"
        payload = {
            "input": text,
            "response_format": "pcm",
            "voice": self.voice,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()

        except urllib.error.URLError as e:
            print(f"Error contacting voicebox TTS at {url}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error synthesizing speech: {e}", file=sys.stderr)
            return None

    def play_audio(self, audio_bytes: bytes, sample_rate: int = 24000):
        """Play audio via sounddevice (non-blocking)."""
        if not self.use_audio or not self._sounddevice or not self._np:
            return

        try:
            import numpy as np

            # Interpret audio bytes as int16 PCM
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            self._sounddevice.play(audio_data, samplerate=sample_rate)
        except Exception as e:
            print(f"Error playing audio: {e}", file=sys.stderr)

    def run_interactive(self):
        """Main interactive loop: record → STT → LLM → TTS → play."""
        self._import_audio_libs()

        print("Voice Chat CLI (Ctrl-C to exit)", file=sys.stderr)
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

        try:
            while True:
                # Record from mic
                audio_bytes = self.record_from_mic()
                if not audio_bytes:
                    continue

                # STT
                user_text = self.transcribe(audio_bytes)
                if not user_text:
                    print("(No speech recognized)", file=sys.stderr)
                    continue

                print(f"User: {user_text}", file=sys.stdout)

                # Add to chat history
                self.chat_history.append({"role": "user", "content": user_text})

                # Stream LLM
                print("Assistant: ", end="", flush=True)
                assistant_text_full = ""
                chunker = SentenceChunker()

                try:
                    for token in parse_sse_stream(self.stream_llm_response(self.chat_history)):
                        # Strip reasoning before processing
                        clean_token = strip_reasoning(token)
                        assistant_text_full += clean_token

                        # Synthesize+play each completed sentence (audio-only;
                        # display comes from the progressive raw-token print below)
                        for sentence in chunker.feed(clean_token):
                            if self.use_audio:
                                pcm_audio = self.synthesize_speech(sentence)
                                if pcm_audio:
                                    self.play_audio(pcm_audio)

                        # Progressive display of the streamed reply
                        print(clean_token, end="", flush=True)

                    # Flush remaining text (speak the trailing partial sentence)
                    remainder = chunker.flush()
                    if remainder:
                        if self.use_audio:
                            pcm_audio = self.synthesize_speech(remainder)
                            if pcm_audio:
                                self.play_audio(pcm_audio)

                except KeyboardInterrupt:
                    print("\n(LLM stream interrupted)", file=sys.stderr)

                print()  # Newline after assistant response

                # Add to chat history
                self.chat_history.append({"role": "assistant", "content": assistant_text_full})

        except KeyboardInterrupt:
            print("\nExiting...", file=sys.stderr)

    def run_text_mode(self, text: str, no_audio: bool = False):
        """Process text directly (skip STT)."""
        self._import_audio_libs()

        if no_audio:
            self.use_audio = False

        print(f"User: {text}", file=sys.stdout)

        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        self.chat_history.append({"role": "user", "content": text})

        print("Assistant: ", end="", flush=True)
        assistant_text_full = ""
        chunker = SentenceChunker()

        try:
            for token in parse_sse_stream(self.stream_llm_response(self.chat_history)):
                clean_token = strip_reasoning(token)
                assistant_text_full += clean_token

                for sentence in chunker.feed(clean_token):
                    if self.use_audio:
                        pcm_audio = self.synthesize_speech(sentence)
                        if pcm_audio:
                            self.play_audio(pcm_audio)

                print(clean_token, end="", flush=True)

            remainder = chunker.flush()
            if remainder:
                if self.use_audio:
                    pcm_audio = self.synthesize_speech(remainder)
                    if pcm_audio:
                        self.play_audio(pcm_audio)

        except KeyboardInterrupt:
            print("\n(LLM stream interrupted)", file=sys.stderr)

        print()
        self.chat_history.append({"role": "assistant", "content": assistant_text_full})

    def run_file_mode(self, filepath: str, no_audio: bool = False):
        """Transcribe a WAV file, then LLM → TTS."""
        self._import_audio_libs()

        if no_audio:
            self.use_audio = False

        try:
            with open(filepath, "rb") as f:
                audio_bytes = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            return
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            return

        # STT
        user_text = self.transcribe(audio_bytes)
        if not user_text:
            print("(No speech recognized)", file=sys.stderr)
            return

        print(f"User: {user_text}", file=sys.stdout)

        # Process like text mode
        self.run_text_mode(user_text, no_audio=no_audio)


def main():
    parser = argparse.ArgumentParser(
        description="Voice chat CLI using voicebox STT/TTS and local LLM"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Process text directly (skip STT); send through LLM→TTS→play",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Transcribe a WAV file, then process through LLM→TTS",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio playback (useful for testing)",
    )

    args = parser.parse_args()

    chat = VoiceChat()

    if args.text:
        chat.run_text_mode(args.text, no_audio=args.no_audio)
    elif args.file:
        chat.run_file_mode(args.file, no_audio=args.no_audio)
    else:
        chat.run_interactive()


if __name__ == "__main__":
    main()
