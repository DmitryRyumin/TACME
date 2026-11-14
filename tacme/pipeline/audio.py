"""
File: audio.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Audio decoding for Whisper inference.

License: MIT License
"""

from pathlib import Path
import subprocess

import torch
from torch import Tensor

from tacme.pipeline.contracts import PipelineError
from tacme.pipeline.media import _find_executable


def _decode_whisper_waveform(
    media_path: Path,
    *,
    target_sample_rate: int,
) -> Tensor:
    """Decode one media audio stream into mono float32 samples for Whisper."""

    if target_sample_rate <= 0:
        msg = f"target_sample_rate must be positive, got {target_sample_rate}."
        raise ValueError(msg)

    ffmpeg = _find_executable(
        "ffmpeg",
    )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(media_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
        )

    except OSError as error:
        msg = f"Unable to execute FFmpeg for Whisper audio decoding from {media_path}: {error}"
        raise PipelineError(msg) from error

    if result.returncode != 0:
        stderr = result.stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if not stderr:
            stderr = f"FFmpeg exited with code {result.returncode}."

        msg = f"Unable to decode Whisper audio from {media_path}: {stderr}"
        raise PipelineError(msg)

    if not result.stdout:
        msg = f"Input media contains no decodable audio for Whisper: {media_path}"
        raise PipelineError(msg)

    if (
        len(
            result.stdout,
        )
        % torch.float32.itemsize
        != 0
    ):
        msg = f"FFmpeg returned an invalid float32 Whisper audio buffer with {len(result.stdout)} bytes."
        raise PipelineError(msg)

    buffer = bytearray(
        result.stdout,
    )

    waveform = torch.frombuffer(
        buffer,
        dtype=torch.float32,
    ).clone()

    if waveform.numel() == 0:
        msg = f"Decoded Whisper waveform is empty: {media_path}"
        raise PipelineError(msg)

    if not bool(
        torch.isfinite(
            waveform,
        )
        .all()
        .item()
    ):
        raise PipelineError("Decoded Whisper waveform contains NaN or infinite values.")

    return waveform.contiguous()
