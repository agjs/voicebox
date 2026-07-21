from __future__ import annotations
import struct


def wav_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    # Placeholder sizes (0xFFFFFFFF) so the header can precede a stream.
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                 byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_len = len(pcm)
    riff_len = 36 + data_len
    return (
        b"RIFF" + struct.pack("<I", riff_len) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                 byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", data_len) + pcm
    )
