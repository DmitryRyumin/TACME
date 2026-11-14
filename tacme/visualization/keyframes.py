"""
File: keyframes.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Representative key-video-moment selection and rendering for the TACME application.

License: MIT License
"""

import base64
from dataclasses import dataclass
from html import escape
from io import BytesIO
from typing import Final

from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as torch_functional

from tacme.localization import get_localized_text
from tacme.pipeline import SceneFrameArtifacts

DEFAULT_KEY_VIDEO_MOMENT_COUNT: Final = 8
KEY_FRAME_MAX_WIDTH: Final = 480
KEY_FRAME_MAX_HEIGHT: Final = 270
KEY_FRAME_JPEG_QUALITY: Final = 86


@dataclass(
    frozen=True,
    slots=True,
)
class KeyVideoMoment:
    """One representative scene frame selected from TACME visual evidence."""

    frame_index: int
    relative_position: float
    time_seconds: float | None
    image_data_url: str


@dataclass(
    frozen=True,
    slots=True,
)
class KeyVideoMoments:
    """Compact set of representative video moments for the application UI."""

    moments: tuple[KeyVideoMoment, ...]
    duration_seconds: float | None
    candidate_count: int


def _validate_scene_artifacts(
    artifacts: SceneFrameArtifacts,
) -> None:
    """Validate scene evidence before representative-frame selection."""

    candidate_count = len(
        artifacts.frames,
    )

    if candidate_count <= 0:
        raise ValueError("No scene frames are available for key-moment selection.")

    if (
        len(
            artifacts.frame_indices,
        )
        != candidate_count
    ):
        msg = "Scene frame and frame-index counts do not match."
        raise ValueError(msg)

    features = artifacts.features

    if features.ndim != 2:
        msg = f"Expected scene features [T, D], got {tuple(features.shape)}."
        raise ValueError(msg)

    if features.shape[0] != candidate_count:
        msg = f"Scene feature count {features.shape[0]} does not match frame count {candidate_count}."
        raise ValueError(msg)

    if artifacts.total_frame_count <= 0:
        msg = f"total_frame_count must be positive, got {artifacts.total_frame_count}."
        raise ValueError(msg)

    if not bool(
        torch.isfinite(
            features,
        )
        .all()
        .item()
    ):
        raise ValueError("Scene features contain NaN or infinite values.")


def _select_representative_candidate_indices(
    features: Tensor,
    target_count: int,
) -> tuple[int, ...]:
    """Select one CLIP medoid from each contiguous temporal segment."""

    candidate_count = int(
        features.shape[0],
    )

    if candidate_count <= 0:
        return ()

    selected_count = min(
        max(
            target_count,
            1,
        ),
        candidate_count,
    )

    normalized = torch_functional.normalize(
        features.detach().to(
            device="cpu",
            dtype=torch.float32,
        ),
        p=2.0,
        dim=1,
        eps=1e-12,
    )

    selected_indices: list[int] = []

    for segment_index in range(
        selected_count,
    ):
        start = segment_index * candidate_count // selected_count
        end = (segment_index + 1) * candidate_count // selected_count

        if end <= start:
            end = min(
                start + 1,
                candidate_count,
            )

        segment = normalized[start:end]

        similarities = segment @ segment.transpose(
            0,
            1,
        )

        centrality = similarities.mean(
            dim=1,
        )

        local_index = int(
            torch.argmax(
                centrality,
            ).item()
        )

        selected_indices.append(
            start + local_index,
        )

    return tuple(
        selected_indices,
    )


def _encode_frame_as_data_url(
    frame: Image.Image,
) -> str:
    """Encode one display thumbnail as an in-memory JPEG data URL."""

    image = frame.convert(
        "RGB",
    ).copy()

    image.thumbnail(
        (
            KEY_FRAME_MAX_WIDTH,
            KEY_FRAME_MAX_HEIGHT,
        ),
        Image.Resampling.LANCZOS,
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=KEY_FRAME_JPEG_QUALITY,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue(),
    ).decode(
        "ascii",
    )

    return f"data:image/jpeg;base64,{encoded}"


def create_key_video_moments(
    artifacts: SceneFrameArtifacts,
    *,
    duration_seconds: float | None,
    target_count: int = DEFAULT_KEY_VIDEO_MOMENT_COUNT,
) -> KeyVideoMoments:
    """Create representative video moments from the exact TACME scene inputs."""

    _validate_scene_artifacts(
        artifacts,
    )

    if target_count <= 0:
        msg = f"target_count must be positive, got {target_count}."
        raise ValueError(msg)

    selected_candidate_indices = _select_representative_candidate_indices(
        artifacts.features,
        target_count,
    )

    valid_duration = duration_seconds if duration_seconds is not None and duration_seconds > 0.0 else None

    denominator = max(
        artifacts.total_frame_count - 1,
        1,
    )

    moments: list[KeyVideoMoment] = []

    for candidate_index in selected_candidate_indices:
        frame_index = artifacts.frame_indices[candidate_index]

        relative_position = min(
            max(
                frame_index / denominator,
                0.0,
            ),
            1.0,
        )

        time_seconds = valid_duration * relative_position if valid_duration is not None else None

        moments.append(
            KeyVideoMoment(
                frame_index=frame_index,
                relative_position=relative_position,
                time_seconds=time_seconds,
                image_data_url=_encode_frame_as_data_url(
                    artifacts.frames[candidate_index],
                ),
            )
        )

    return KeyVideoMoments(
        moments=tuple(
            moments,
        ),
        duration_seconds=valid_duration,
        candidate_count=len(
            artifacts.frames,
        ),
    )


def _format_moment_time(
    moment: KeyVideoMoment,
    language_index: int,
) -> str:
    """Format one moment timestamp for the filmstrip."""

    if moment.time_seconds is None:
        return f"{moment.relative_position * 100.0:.0f}%"

    unit = get_localized_text(
        "Units_SECONDS",
        language_index,
    )

    return f"{moment.time_seconds:.2f} {unit}"


def _format_timeline_end(
    key_moments: KeyVideoMoments,
    language_index: int,
) -> str:
    """Format the right endpoint of the compact video timeline."""

    if key_moments.duration_seconds is None:
        return "100%"

    unit = get_localized_text(
        "Units_SECONDS",
        language_index,
    )

    return f"{key_moments.duration_seconds:.2f} {unit}"


def create_key_video_moments_html(
    key_moments: KeyVideoMoments | None,
    language_index: int,
) -> str:
    """Render the interactive TACME key-video-moments filmstrip."""

    if key_moments is None or not key_moments.moments:
        return ""

    cards: list[str] = []
    markers: list[str] = []

    for moment_index, moment in enumerate(
        key_moments.moments,
    ):
        time_text = escape(
            _format_moment_time(
                moment,
                language_index,
            )
        )

        selected_class = " is-selected" if moment_index == 0 else ""

        cards.append(
            f"""
<button
    type="button"
    class="key-video-moment-card{selected_class}"
    data-moment-index="{moment_index}"
    title="{time_text}"
    style="--moment-delay: {moment_index * 55}ms;"
>
    <span class="key-video-moment-image-wrap">
        <img
            class="key-video-moment-image"
            src="{moment.image_data_url}"
            alt=""
            draggable="false"
        />
        <span class="key-video-moment-time">{time_text}</span>
    </span>
</button>
""".strip()
        )

        markers.append(
            f"""
<button
    type="button"
    class="key-video-timeline-marker{selected_class}"
    data-moment-index="{moment_index}"
    title="{time_text}"
    aria-label="{time_text}"
    style="--moment-position: {moment.relative_position * 100.0:.4f}%;"
></button>
""".strip()
        )

    timeline_end = escape(
        _format_timeline_end(
            key_moments,
            language_index,
        )
    )

    return f"""
<section class="key-video-moments-content">
    <div class="key-video-filmstrip" tabindex="0">
        {"".join(cards)}
    </div>
    <div class="key-video-timeline">
        <span class="key-video-timeline-endpoint key-video-timeline-start">0</span>
        <div class="key-video-timeline-track">
            {"".join(markers)}
        </div>
        <span class="key-video-timeline-endpoint key-video-timeline-finish">{timeline_end}</span>
    </div>
</section>
""".strip()


__all__ = [
    "DEFAULT_KEY_VIDEO_MOMENT_COUNT",
    "KeyVideoMoment",
    "KeyVideoMoments",
    "create_key_video_moments",
    "create_key_video_moments_html",
]
