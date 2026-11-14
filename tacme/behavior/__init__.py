"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible public API for behavior description.

License: MIT License
"""

# Re-export the original public API at its historical import path.
from tacme.behavior.contracts import (
    BEHAVIOR_PROMPT as BEHAVIOR_PROMPT,
    DEFAULT_BEHAVIOR_MODEL as DEFAULT_BEHAVIOR_MODEL,
    DEFAULT_MAX_FRAMES as DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_NEW_TOKENS as DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_VIDEO_FPS as DEFAULT_VIDEO_FPS,
    QWEN3_VL_IMAGE_PATCH_SIZE as QWEN3_VL_IMAGE_PATCH_SIZE,
    QWEN3_VL_VIDEO_READER as QWEN3_VL_VIDEO_READER,
    BehaviorDescription as BehaviorDescription,
    BehaviorDescriptionError as BehaviorDescriptionError,
)
from tacme.behavior.describer import (
    QwenBehaviorDescriber as QwenBehaviorDescriber,
    _normalize_description as _normalize_description,
)
from tacme.behavior.video import (
    _load_process_vision_info as _load_process_vision_info,
    _split_video_metadata as _split_video_metadata,
    _validate_video_path as _validate_video_path,
)

# Preserve the original paths for Python object serialization and introspection.
for _legacy_class in (BehaviorDescription, BehaviorDescriptionError, QwenBehaviorDescriber):
    _legacy_class.__module__ = __name__
del _legacy_class

__all__ = [
    "BEHAVIOR_PROMPT",
    "DEFAULT_BEHAVIOR_MODEL",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_VIDEO_FPS",
    "BehaviorDescription",
    "BehaviorDescriptionError",
    "QwenBehaviorDescriber",
]
