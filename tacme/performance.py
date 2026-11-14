"""
File: performance.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Runtime-performance measurements for TACME analysis.

License: MIT License
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Final, Literal

import torch

# Exact sequence in pipeline.py and inference.py. Progress counts finished operations,
# rather than estimating how far an active neural-network operation has progressed.
PROGRESS_STAGE_KEYS: Final[tuple[str, ...]] = (
    "media_validation",
    "frame_sampling",
    "clip_scene",
    "wav2vec_audio",
    "whisper_decoding",
    "whisper_transcription",
    "roberta_transcription",
    "qwen_behavior",
    "roberta_behavior",
    "feature_aggregation",
    "tacme_attribution",
    "output_adaptation",
)


@dataclass(frozen=True, slots=True)
class StageProgressEvent:
    """Start/completion/failure of one real inference operation."""

    stage: str
    status: Literal["started", "completed", "failed"]
    elapsed_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Measured wall-clock duration of one successful inference stage."""

    stage: str
    elapsed_seconds: float


ProgressCallback = Callable[[StageProgressEvent], None]


def _synchronize_device(device: torch.device | None) -> None:
    """Include queued GPU/MPS kernel work in per-stage wall-clock timings."""

    if device is None:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


@contextmanager
def timed_stage(
    stage: str,
    callback: ProgressCallback | None,
    device: torch.device | None = None,
) -> Iterator[None]:
    """Instrument a stage only when progress reporting is requested."""

    if callback is None:
        yield
        return

    _synchronize_device(device)
    started_at = perf_counter()
    callback(StageProgressEvent(stage=stage, status="started"))
    try:
        yield
    except BaseException:
        _synchronize_device(device)
        callback(
            StageProgressEvent(
                stage=stage,
                status="failed",
                elapsed_seconds=perf_counter() - started_at,
            )
        )
        raise
    else:
        _synchronize_device(device)
        callback(
            StageProgressEvent(
                stage=stage,
                status="completed",
                elapsed_seconds=perf_counter() - started_at,
            )
        )


@dataclass(
    frozen=True,
    slots=True,
)
class AnalysisTiming:
    """End-to-end TACME analysis timing for one video."""

    analysis_seconds: float
    video_duration_seconds: float | None
    rtf: float | None
    stage_timings: tuple[StageTiming, ...] = ()
    device: str | None = None


def create_analysis_timing(
    analysis_seconds: float,
    video_duration_seconds: float | None,
    *,
    stage_timings: tuple[StageTiming, ...] = (),
    device: str | None = None,
) -> AnalysisTiming:
    """Create one timing record and derive real-time-factor metrics."""

    valid_duration = (
        video_duration_seconds if video_duration_seconds is not None and video_duration_seconds > 0.0 else None
    )

    rtf = analysis_seconds / valid_duration if valid_duration is not None else None

    return AnalysisTiming(
        analysis_seconds=analysis_seconds,
        video_duration_seconds=valid_duration,
        rtf=rtf,
        stage_timings=stage_timings,
        device=device,
    )


__all__ = [
    "PROGRESS_STAGE_KEYS",
    "AnalysisTiming",
    "ProgressCallback",
    "StageProgressEvent",
    "StageTiming",
    "create_analysis_timing",
    "timed_stage",
]
