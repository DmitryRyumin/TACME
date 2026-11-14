"""
File: core.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: End-to-end multimodal feature preparation and artifact retention.

License: MIT License
"""

from pathlib import Path

from PIL import Image
import torch
from torch import Tensor

from tacme.behavior import BehaviorDescription, QwenBehaviorDescriber
from tacme.extractors import (
    EXPECTED_AUDIO_DIM,
    EXPECTED_TEXT_DIM,
    EXPECTED_VISION_DIM,
    CLIPSceneExtractor,
    EmoRoBERTaExtractor,
    EmoWav2Vec2Extractor,
)
from tacme.models.common import FeatureBatch, resolve_device
from tacme.performance import ProgressCallback, timed_stage
from tacme.pipeline.aggregation import _aggregate_and_batch
from tacme.pipeline.audio import _decode_whisper_waveform
from tacme.pipeline.contracts import DEFAULT_SCENE_FRAME_COUNT, PipelineError, SceneFrameArtifacts, SpeechRequiredError
from tacme.pipeline.frames import _sample_scene_frames_with_indices
from tacme.pipeline.media import _resolve_pipeline_stream_mode, _validate_media_path
from tacme.transcription import WhisperTranscriber, WhisperTranscription


class TacmeFeaturePipeline:
    """Prepare TACME features from audiovisual or opted-in audio-only media."""

    def __init__(
        self,
        *,
        device: str | torch.device = "auto",
        scene_frame_count: int = DEFAULT_SCENE_FRAME_COUNT,
    ) -> None:
        """Load the shared TACME pretrained feature stack."""

        if scene_frame_count <= 0:
            msg = f"scene_frame_count must be positive, got {scene_frame_count}."
            raise ValueError(msg)

        self.device = resolve_device(
            device,
        )

        self.scene_frame_count = scene_frame_count

        self.scene_extractor = CLIPSceneExtractor(
            device=self.device,
        )

        self.audio_extractor = EmoWav2Vec2Extractor(
            device=self.device,
        )

        self.text_extractor = EmoRoBERTaExtractor(
            device=self.device,
        )

        self.transcriber = WhisperTranscriber(
            device=self.device,
        )

        self.behavior_describer = QwenBehaviorDescriber(
            device=self.device,
        )

    @torch.inference_mode()
    def _process_internal(
        self,
        media_path: str | Path,
        *,
        retain_scene_artifacts: bool,
        retain_transcription_metadata: bool,
        retain_behavior_metadata: bool,
        allow_audio_only: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[
        FeatureBatch,
        SceneFrameArtifacts | None,
        WhisperTranscription | None,
        BehaviorDescription | None,
    ]:
        """Prepare TACME features and optionally retain inference artifacts."""

        with timed_stage("media_validation", progress_callback, self.device):
            path = _validate_media_path(media_path)
            stream_mode = _resolve_pipeline_stream_mode(
                path,
                allow_audio_only=allow_audio_only,
            )

        scene_frames: list[Image.Image] | None = None
        scene_frame_indices: list[int] | None = None
        total_frame_count: int | None = None
        scene_raw: Tensor | None = None

        if stream_mode == "audiovisual":
            with timed_stage("frame_sampling", progress_callback, self.device):
                scene_frames, scene_frame_indices, total_frame_count = _sample_scene_frames_with_indices(
                    path,
                    self.scene_frame_count,
                )

            with timed_stage("clip_scene", progress_callback, self.device):
                scene_raw = self.scene_extractor.extract(scene_frames)

        with timed_stage("wav2vec_audio", progress_callback, self.device):
            audio_raw = self.audio_extractor.extract(path)

        with timed_stage("whisper_decoding", progress_callback, self.device):
            whisper_waveform = _decode_whisper_waveform(
                path,
                target_sample_rate=self.transcriber.sample_rate,
            )

        if stream_mode == "audio_only" and not bool(torch.count_nonzero(whisper_waveform).item()):
            raise SpeechRequiredError("Audio-only input is digitally silent; a usable speech transcript is required.")

        with timed_stage("whisper_transcription", progress_callback, self.device):
            transcription: WhisperTranscription | None = None
            if retain_transcription_metadata:
                transcription = self.transcriber.transcribe_waveform_with_metadata(
                    whisper_waveform,
                    sample_rate=self.transcriber.sample_rate,
                )
                transcript = transcription.text
            else:
                transcript = self.transcriber.transcribe_waveform(
                    whisper_waveform,
                    sample_rate=self.transcriber.sample_rate,
                )
            if not transcript.strip():
                if stream_mode == "audio_only":
                    raise SpeechRequiredError(
                        "Whisper produced no usable transcript for audio-only input; speech transcription is required."
                    )
                raise PipelineError("Whisper returned an empty transcription.")

        with timed_stage("roberta_transcription", progress_callback, self.device):
            text_raw = self.text_extractor.extract(transcript)

        behavior: BehaviorDescription | None = None
        behavior_raw: Tensor | None = None

        if stream_mode == "audiovisual":
            with timed_stage("qwen_behavior", progress_callback, self.device):
                if retain_behavior_metadata:
                    behavior = self.behavior_describer.describe_with_metadata(path)
                    behavior_description = behavior.text
                else:
                    behavior_description = self.behavior_describer.describe(path)
                if not behavior_description.strip():
                    raise PipelineError("Qwen3-VL returned an empty behavioral description.")

            with timed_stage("roberta_behavior", progress_callback, self.device):
                behavior_raw = self.text_extractor.extract(behavior_description)

        with timed_stage("feature_aggregation", progress_callback, self.device):
            features: dict[str, Tensor] = {
                "audio": _aggregate_and_batch("audio", audio_raw, EXPECTED_AUDIO_DIM),
                "text": _aggregate_and_batch("text", text_raw, EXPECTED_TEXT_DIM),
            }
            if stream_mode == "audiovisual":
                if scene_raw is None or behavior_raw is None:
                    raise PipelineError("Visual features were not extracted for audiovisual media.")
                features["scene"] = _aggregate_and_batch("scene", scene_raw, EXPECTED_VISION_DIM)
                features["behavior"] = _aggregate_and_batch("behavior", behavior_raw, EXPECTED_TEXT_DIM)

        masks: dict[str, Tensor] = {
            modality: torch.ones(
                feature.shape[:2],
                dtype=torch.bool,
                device=feature.device,
            )
            for modality, feature in features.items()
        }

        batch: FeatureBatch = {
            "features": features,
            "feature_masks": masks,
        }

        scene_artifacts: SceneFrameArtifacts | None = None

        if retain_scene_artifacts and stream_mode == "audiovisual":
            if scene_frames is None or scene_frame_indices is None or total_frame_count is None or scene_raw is None:
                raise PipelineError("Scene evidence is unavailable for audiovisual media.")
            scene_artifacts = SceneFrameArtifacts(
                frames=tuple(
                    scene_frames,
                ),
                frame_indices=tuple(
                    scene_frame_indices,
                ),
                total_frame_count=total_frame_count,
                features=(
                    scene_raw.detach()
                    .to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    .contiguous()
                ),
            )

        return (
            batch,
            scene_artifacts,
            transcription,
            behavior,
        )

    @torch.inference_mode()
    def process(
        self,
        media_path: str | Path,
        *,
        allow_audio_only: bool = False,
    ) -> FeatureBatch:
        """Convert media to a FeatureBatch; audio-only requires explicit opt-in."""

        batch, _, _, _ = self._process_internal(
            media_path,
            retain_scene_artifacts=False,
            retain_transcription_metadata=False,
            retain_behavior_metadata=False,
            allow_audio_only=allow_audio_only,
        )

        return batch

    @torch.inference_mode()
    def process_with_artifacts(
        self,
        media_path: str | Path,
    ) -> tuple[
        FeatureBatch,
        SceneFrameArtifacts,
    ]:
        """Prepare TACME features and retain the exact scene evidence used."""

        batch, scene_artifacts, _, _ = self._process_internal(
            media_path,
            retain_scene_artifacts=True,
            retain_transcription_metadata=False,
            retain_behavior_metadata=False,
        )

        if scene_artifacts is None:
            raise PipelineError("Scene artifacts were not retained during TACME processing.")

        return (
            batch,
            scene_artifacts,
        )

    @torch.inference_mode()
    def process_with_ui_artifacts(
        self,
        media_path: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[
        FeatureBatch,
        SceneFrameArtifacts,
        WhisperTranscription,
        BehaviorDescription,
    ]:
        """Prepare TACME features and retain scene, speech, and behavior UI artifacts."""

        batch, scene_artifacts, transcription, behavior = self._process_internal(
            media_path,
            retain_scene_artifacts=True,
            retain_transcription_metadata=True,
            retain_behavior_metadata=True,
            progress_callback=progress_callback,
        )

        if scene_artifacts is None:
            raise PipelineError("Scene artifacts were not retained during TACME processing.")

        if transcription is None:
            raise PipelineError("Transcription metadata was not retained during TACME processing.")

        if behavior is None:
            raise PipelineError("Behavior metadata was not retained during TACME processing.")

        return (
            batch,
            scene_artifacts,
            transcription,
            behavior,
        )
