"""
File: contracts.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Data contracts, defaults, and TACME pipeline errors.

License: MIT License
"""

from dataclasses import dataclass
from typing import Final

from PIL import Image
from torch import Tensor

from tacme.extractors import EXPECTED_AUDIO_DIM, EXPECTED_TEXT_DIM, EXPECTED_VISION_DIM

DEFAULT_SCENE_FRAME_COUNT: Final = 30

AGGREGATED_SEQUENCE_LENGTH: Final = 2

MODALITY_DIMENSIONS: Final = {
    "audio": EXPECTED_AUDIO_DIM,
    "text": EXPECTED_TEXT_DIM,
    "scene": EXPECTED_VISION_DIM,
    "behavior": EXPECTED_TEXT_DIM,
}


class PipelineError(RuntimeError):
    """Raised when TACME multimodal feature preparation fails."""


class SpeechRequiredError(PipelineError):
    """Raised when audio-only TACME cannot obtain a usable speech transcript."""


@dataclass(
    frozen=True,
    slots=True,
)
class SceneFrameArtifacts:
    """Scene frames and per-frame CLIP features used by TACME inference."""

    frames: tuple[Image.Image, ...]
    frame_indices: tuple[int, ...]
    total_frame_count: int
    features: Tensor
