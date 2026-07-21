#!/usr/bin/env python3
"""
Local dictation helper for Claude Code: record mic or transcribe a file via voicebox STT.
"""

import sys
import os
import json
import argparse
import io
import subprocess
import threading
import uuid
import wave
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def get_voicebox_url():
    """Get voicebox endpoint from environment or use default."""
    return os.environ.get("VOICEBOX_URL", "http://localhost:8790")


def transcribe_audio(audio_bytes):
    """POST audio to voicebox transcription endpoint."""
    voicebox_url = get_voicebox_url()
    endpoint = f"{voicebox_url}/v1/audio/transcriptions"

    # Build multipart form data
    boundary = f"----voicebox-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        "Content-Type: audio/wav\r\n"
        "\r\n"
    ).encode()
    body += audio_bytes
    body += (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n'
        "\r\n"
        "whisper\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    api_key = os.environ.get("VOICEBOX_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result.get("text", "")
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"Error transcribing audio: {e}", file=sys.stderr)
        sys.exit(1)


def record_from_mic():
    """Record audio from microphone using sounddevice."""
    try:
        import sounddevice
        import numpy as np
    except ImportError:
        print(
            "Error: sounddevice and numpy are required for microphone recording.\n"
            "Install with: pip install sounddevice numpy\n"
            "Or use --file mode to transcribe an existing WAV file.",
            file=sys.stderr,
        )
        sys.exit(1)

    sample_rate = 16000
    print("🎤 recording… press Enter to stop", file=sys.stderr, flush=True)

    frames = []
    stop_event = threading.Event()
    recording_error = []

    def record_thread():
        try:
            with sounddevice.InputStream(
                channels=1,
                samplerate=sample_rate,
                blocksize=480,
                dtype="int16",
                latency="low",
            ) as stream:
                while not stop_event.is_set():
                    data, overflowed = stream.read(480)
                    if overflowed:
                        print("Warning: audio buffer overflow", file=sys.stderr)
                    frames.append(data.copy())
        except Exception as e:
            recording_error.append(e)

    # Start recording in background
    thread = threading.Thread(target=record_thread, daemon=True)
    thread.start()

    # Wait for user to press Enter
    try:
        input()
    except KeyboardInterrupt:
        pass

    stop_event.set()
    thread.join(timeout=1)

    if recording_error:
        print(f"Recording error: {recording_error[0]}", file=sys.stderr)
        sys.exit(1)
    if not frames:
        print("No audio recorded.", file=sys.stderr)
        sys.exit(1)

    # Combine frames into a single array
    audio_data = np.concatenate(frames)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    return wav_buffer.getvalue()


def copy_to_clipboard(text):
    """Copy text to system clipboard."""
    # Try pbcopy (macOS)
    try:
        process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        process.communicate(text.encode())
        return
    except FileNotFoundError:
        pass

    # Try xclip (Linux X11)
    try:
        process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
        process.communicate(text.encode())
        return
    except FileNotFoundError:
        pass

    # Try wl-copy (Wayland)
    try:
        process = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
        process.communicate(text.encode())
        return
    except FileNotFoundError:
        pass

    # Clipboard not available; just warn
    print(
        "Warning: clipboard tools (pbcopy/xclip/wl-copy) not found. Text not copied to clipboard.",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio via voicebox (local STT)")
    parser.add_argument(
        "--file",
        type=str,
        help="Transcribe an existing WAV file instead of recording from mic",
    )
    args = parser.parse_args()

    # Record or load audio
    if args.file:
        try:
            with open(args.file, "rb") as f:
                audio_bytes = f.read()
        except (FileNotFoundError, IOError) as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        audio_bytes = record_from_mic()

    # Transcribe
    transcript = transcribe_audio(audio_bytes)

    # Output and copy
    print(transcript)
    copy_to_clipboard(transcript)


if __name__ == "__main__":
    main()
