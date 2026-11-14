"""
File: engine.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: End-to-end multimodal feature extraction, prediction, and attribution.

License: MIT License
"""

from pathlib import Path

import torch

from tacme.explainability.attribution import run_tacme_feature_attribution
from tacme.inference.adaptation import adapt_tacme_outputs
from tacme.inference.contracts import (
    TacmeInferenceError,
    TacmeInferenceResult,
    TacmeInferenceRun,
)
from tacme.models import MultiModalFusionModel_v2, load_tacme_model
from tacme.performance import ProgressCallback, timed_stage
from tacme.pipeline import TacmeFeaturePipeline


class TacmeInferenceEngine:
    """Run the complete TACME multimodal inference stack."""

    def __init__(
        self,
        *,
        device: str | torch.device = "auto",
    ) -> None:
        """Load the multimodal feature pipeline and TACME model."""

        try:
            self.pipeline = TacmeFeaturePipeline(
                device=device,
            )

            self.model: MultiModalFusionModel_v2 = load_tacme_model(
                device=self.pipeline.device,
            )

        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            msg = f"Unable to initialize TACME inference: {error}"
            raise TacmeInferenceError(msg) from error

        self.device = self.pipeline.device

    @torch.inference_mode()
    def predict(
        self,
        media_path: str | Path,
    ) -> TacmeInferenceResult:
        """Run TACME inference for one input video."""

        try:
            batch = self.pipeline.process(
                media_path,
            )

            outputs = self.model(
                batch,
            )

        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            msg = f"TACME inference failed for {media_path}: {error}"
            raise TacmeInferenceError(msg) from error

        return adapt_tacme_outputs(
            outputs,
        )

    def predict_with_artifacts(
        self,
        media_path: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> TacmeInferenceRun:
        """Run TACME inference and retain multimodal UI and attribution artifacts."""

        try:
            (
                batch,
                scene_artifacts,
                transcription,
                behavior_description,
            ) = self.pipeline.process_with_ui_artifacts(
                media_path,
                progress_callback=progress_callback,
            )

            with timed_stage("tacme_attribution", progress_callback, self.device):
                outputs, feature_attribution = run_tacme_feature_attribution(
                    self.model,
                    batch,
                )

        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            msg = f"TACME inference failed for {media_path}: {error}"
            raise TacmeInferenceError(msg) from error

        with timed_stage("output_adaptation", progress_callback, self.device):
            result = adapt_tacme_outputs(outputs)

        return TacmeInferenceRun(
            result=result,
            scene_artifacts=scene_artifacts,
            transcription=transcription,
            behavior_description=behavior_description,
            feature_attribution=feature_attribution,
        )
