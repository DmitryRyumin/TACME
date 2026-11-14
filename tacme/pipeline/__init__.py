"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible public interfaces for TACME feature pipeline.

License: MIT License
"""

from tacme.pipeline.aggregation import _aggregate_and_batch as _aggregate_and_batch
from tacme.pipeline.audio import _decode_whisper_waveform as _decode_whisper_waveform
from tacme.pipeline.contracts import (
    AGGREGATED_SEQUENCE_LENGTH as AGGREGATED_SEQUENCE_LENGTH,
    DEFAULT_SCENE_FRAME_COUNT as DEFAULT_SCENE_FRAME_COUNT,
    MODALITY_DIMENSIONS as MODALITY_DIMENSIONS,
    PipelineError as PipelineError,
    SceneFrameArtifacts as SceneFrameArtifacts,
    SpeechRequiredError as SpeechRequiredError,
)
from tacme.pipeline.core import TacmeFeaturePipeline as TacmeFeaturePipeline
from tacme.pipeline.frames import (
    _decode_selected_frames as _decode_selected_frames,
    _probe_video_frame_count as _probe_video_frame_count,
    _sample_scene_frames as _sample_scene_frames,
    _sample_scene_frames_with_indices as _sample_scene_frames_with_indices,
    _uniform_frame_indices as _uniform_frame_indices,
)
from tacme.pipeline.media import (
    _find_executable as _find_executable,
    _validate_media_path as _validate_media_path,
)

# Preserve historical qualified paths used by existing Python/pickle consumers.
for _legacy_class in (PipelineError, SceneFrameArtifacts, TacmeFeaturePipeline):
    _legacy_class.__module__ = __name__
del _legacy_class

__all__ = [
    "AGGREGATED_SEQUENCE_LENGTH",
    "DEFAULT_SCENE_FRAME_COUNT",
    "MODALITY_DIMENSIONS",
    "PipelineError",
    "SceneFrameArtifacts",
    "SpeechRequiredError",
    "TacmeFeaturePipeline",
]
