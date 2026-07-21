#!/usr/bin/env python3
"""
Local dictation helper for Claude Code: record mic or transcribe a file via voicebox STT.
"""
import sys
import os
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError


def get_voicebox_url():
    """Get voicebox endpoint from environment or use default."""
    return os.environ.get("VOICEBOX_URL", "http://localhost:8790")


def transcribe_audio(audio_bytes):
    """POST audio to voicebox transcription endpoint."""
    voicebox_url = get_voicebox_url()
    endpoint = f"{voicebox_url}/v1/audio/transcriptions"

    # Build multipart form data
    boundary = "----voicebox_boundary"
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

    try:
        req = Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result.get("text", "")
    except (URLError, json.JSONDecodeError, Exception) as e:
        print(f"Error transcribing audio: {e}", file=sys.stderr)
        sys.exit(1)


def record_from_mic():
    """Record audio from microphone using sounddevice."""
    try:
        import sounddevice
        import scipy.io.wavfile
    except ImportError:
        print(
            "Error: sounddevice and scipy required for microphone recording.\n"
            "Install with: pip install sounddevice scipy\n"
            "Or use --file mode to transcribe an existing WAV file.",
            file=sys.stderr,
        )
        sys.exit(1)

    sample_rate = 16000
    print("🎤 recording… press Enter to stop", file=sys.stderr, flush=True)

    # Record until user presses Enter (non-blocking on separate thread)
    import threading

    frames = []
    stop_event = threading.Event()

    def record_thread():
        try:
            with sounddevice.InputStream(
                channels=1, samplerate=sample_rate, blocksize=4096
            ) as stream:
                while not stop_event.is_set():
                    data, overflowed = stream.read(4096)
                    if overflowed:
                        print("Warning: audio buffer overflow", file=sys.stderr)
                    frames.append(data)
        except Exception as e:
            print(f"Recording error: {e}", file=sys.stderr)
            sys.exit(1)

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

    if not frames:
        print("No audio recorded.", file=sys.stderr)
        sys.exit(1)

    # Combine frames into a single array
    import numpy as np
    audio_data = np.concatenate(frames)

    # Write to a temporary WAV file
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmpfile_path = tmpfile.name
    tmpfile.close()

    try:
        scipy.io.wavfile.write(tmpfile_path, sample_rate, audio_data)
        with open(tmpfile_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmpfile_path)
        except Exception:
            pass


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
        process = subprocess.Popen(
            ["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE
        )
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
        "Warning: clipboard tools (pbcopy/xclip/wl-copy) not found. "
        "Text not copied to clipboard.",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio via voicebox (local STT)"
    )
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
