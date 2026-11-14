"""
File: validation_compatibility.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: TACME media stream requirements and compatibility results.

License: MIT License
"""

from dataclasses import dataclass
from typing import Literal

from tacme.media.metadata import MediaProbeResult

type MediaStreamMode = Literal[
    "audiovisual",
    "audio_only",
    "video_only",
    "no_streams",
    "unavailable",
]


def classify_media_streams(probe_result: MediaProbeResult) -> MediaStreamMode:
    """Classify declared physical streams, not their decode validity.

    A failed metadata probe is distinct from a container with no A/V streams.
    Existing streams must still pass their dedicated decoding preflight.
    """

    metadata = probe_result.metadata
    if metadata is None:
        return "unavailable"

    has_audio = metadata.audio is not None
    has_video = metadata.video is not None
    if has_audio and has_video:
        return "audiovisual"
    if has_audio:
        return "audio_only"
    if has_video:
        return "video_only"
    return "no_streams"


@dataclass(frozen=True, slots=True)
class MediaCompatibilityCheck:
    """One TACME media compatibility check."""

    key: str
    passed: bool


@dataclass(frozen=True, slots=True)
class MediaCompatibilityResult:
    """Compatibility result for one TACME media input."""

    checks: tuple[MediaCompatibilityCheck, ...]

    @property
    def is_ready(self) -> bool:
        """Return whether all required TACME input checks passed."""

        return bool(
            self.checks,
        ) and all(check.passed for check in self.checks)


def validate_media_for_tacme(
    probe_result: MediaProbeResult,
    *,
    allow_audio_only: bool = False,
) -> MediaCompatibilityResult:
    """Check stream metadata against the requested TACME input policy.

    The default preserves the existing A1-A3 audiovisual requirements.
    Enabling audio-only compatibility does not waive audio decode validation.
    """

    metadata = probe_result.metadata

    if metadata is None:
        return MediaCompatibilityResult(
            checks=(
                MediaCompatibilityCheck(
                    key="METADATA",
                    passed=False,
                ),
            ),
        )

    video = metadata.video

    common_checks = (
        MediaCompatibilityCheck(
            key="METADATA",
            passed=True,
        ),
        MediaCompatibilityCheck(
            key="FILE_SIZE",
            passed=metadata.size_bytes > 0,
        ),
        MediaCompatibilityCheck(
            key="DURATION",
            passed=(metadata.duration_seconds is not None and metadata.duration_seconds > 0),
        ),
    )

    audio_check = MediaCompatibilityCheck(
        key="AUDIO_STREAM",
        passed=metadata.audio is not None,
    )

    if allow_audio_only and classify_media_streams(probe_result) == "audio_only":
        return MediaCompatibilityResult(
            checks=(*common_checks, audio_check),
        )

    video_checks = (
        MediaCompatibilityCheck(
            key="VIDEO_STREAM",
            passed=video is not None,
        ),
        MediaCompatibilityCheck(
            key="VIDEO_DIMENSIONS",
            passed=(
                video is not None
                and video.width is not None
                and video.width > 0
                and video.height is not None
                and video.height > 0
            ),
        ),
    )

    return MediaCompatibilityResult(
        checks=(*common_checks, *video_checks, audio_check),
    )


__all__ = [
    "MediaCompatibilityCheck",
    "MediaCompatibilityResult",
    "MediaStreamMode",
    "classify_media_streams",
    "validate_media_for_tacme",
]
