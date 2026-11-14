"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Qwen3-VL behavior prompt, default settings, and result contracts.

License: MIT License
"""

from dataclasses import dataclass
from typing import Final

DEFAULT_BEHAVIOR_MODEL: Final = "Qwen/Qwen3-VL-4B-Instruct"


DEFAULT_MAX_NEW_TOKENS: Final = 100


DEFAULT_VIDEO_FPS: Final = 8.0


DEFAULT_MAX_FRAMES: Final = 16


QWEN3_VL_IMAGE_PATCH_SIZE: Final = 16


QWEN3_VL_VIDEO_READER: Final = "torchcodec"


BEHAVIOR_PROMPT: Final = (
    "You are analyzing a short video clip of a person. "
    "Describe only observable behavior: facial movements, gaze direction, "
    "head motion, body posture, hand gestures, speaking-related movements, "
    "and relevant scene context. "
    "Note how these visible cues change over time when possible. "
    "Do not infer emotions, personality traits, hesitation, confidence, "
    "intentions, mental states, or any other hidden attributes. "
    "Write one coherent paragraph with no line breaks, no bullets, "
    "no special formatting, no longer than 100 tokens, "
    "and always end with a period."
)


@dataclass(
    frozen=True,
    slots=True,
)
class BehaviorDescription:
    """Observable Qwen3-VL behavior description retained for TACME UI."""

    text: str
    model_name: str
    video_fps: float
    max_frames: int


class BehaviorDescriptionError(RuntimeError):
    """Raised when TACME behavior description generation fails."""
