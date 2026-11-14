"""
File: vision.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: CLIP scene feature extraction for the TACME inference pipeline.

License: MIT License
"""

from collections.abc import Sequence
from typing import Final, cast

from PIL import Image
import torch
from torch import Tensor, nn
from transformers import (
    CLIPImageProcessorPil,
    CLIPVisionModelWithProjection,
)

from tacme.models.common import resolve_device

DEFAULT_VISION_MODEL: Final = "openai/clip-vit-base-patch32"

EXPECTED_VISION_DIM: Final = 512


class VisionExtractionError(RuntimeError):
    """Raised when TACME visual feature extraction fails."""


class CLIPSceneExtractor:
    """Extract projected CLIP scene features for TACME."""

    def __init__(
        self,
        model_name: str = DEFAULT_VISION_MODEL,
        *,
        device: str | torch.device = "auto",
    ) -> None:
        """Load the pretrained CLIP image processor and vision model."""

        self.model_name = model_name

        self.device = resolve_device(
            device,
        )

        try:
            self.processor = CLIPImageProcessorPil.from_pretrained(
                self.model_name,
            )

            self.model = CLIPVisionModelWithProjection.from_pretrained(
                self.model_name,
                dtype=torch.float32,
            )

        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to load CLIP vision model {self.model_name!r}: {error}"

            raise VisionExtractionError(msg) from error

        projection_dim = getattr(
            self.model.config,
            "projection_dim",
            None,
        )

        if projection_dim != EXPECTED_VISION_DIM:
            msg = f"Unexpected CLIP projection dimension: {projection_dim}; expected {EXPECTED_VISION_DIM}."

            raise VisionExtractionError(msg)

        model_module = cast(
            nn.Module,
            self.model,
        )

        model_module.to(
            device=self.device,
        )

        model_module.requires_grad_(
            False,
        )

        model_module.eval()

    def fingerprint(
        self,
    ) -> str:
        """Return the stable identifier of the visual feature extractor."""

        return f"hf-clip-vision:{self.model_name}"

    @staticmethod
    def _validate_frames(
        frames: Sequence[Image.Image],
    ) -> list[Image.Image]:
        """Validate and normalize input frames to RGB images."""

        if not frames:
            raise VisionExtractionError("No video frames were provided for CLIP extraction.")

        rgb_frames: list[Image.Image] = []

        for (
            index,
            frame,
        ) in enumerate(
            frames,
        ):
            if not isinstance(
                frame,
                Image.Image,
            ):
                msg = f"Unexpected frame type at index {index}: {type(frame).__name__}."

                raise VisionExtractionError(msg)

            if frame.width <= 0 or frame.height <= 0:
                msg = f"Invalid frame dimensions at index {index}: {frame.size}."

                raise VisionExtractionError(msg)

            rgb_frames.append(
                frame.convert(
                    "RGB",
                )
            )

        return rgb_frames

    def _prepare_inputs(
        self,
        frames: Sequence[Image.Image],
    ) -> Tensor:
        """Convert video frames into CLIP pixel values."""

        rgb_frames = self._validate_frames(
            frames,
        )

        try:
            raw_inputs = self.processor(
                images=rgb_frames,
                return_tensors="pt",
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to preprocess frames for {self.model_name!r}: {error}"

            raise VisionExtractionError(msg) from error

        pixel_values = raw_inputs.get(
            "pixel_values",
        )

        if not isinstance(
            pixel_values,
            Tensor,
        ):
            raise VisionExtractionError("CLIP image processor did not return pixel_values.")

        if pixel_values.ndim != 4:
            msg = f"Unexpected CLIP pixel-value shape: {tuple(pixel_values.shape)}; expected [T, C, H, W]."

            raise VisionExtractionError(msg)

        if pixel_values.shape[0] != len(rgb_frames):
            msg = f"CLIP image processor returned {pixel_values.shape[0]} frames; expected {len(rgb_frames)}."

            raise VisionExtractionError(msg)

        return pixel_values.to(
            device=self.device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def extract(
        self,
        frames: Sequence[Image.Image],
    ) -> Tensor:
        """Extract float32 projected CLIP features with shape [T, 512]."""

        pixel_values = self._prepare_inputs(
            frames,
        )

        try:
            outputs = self.model(
                pixel_values=pixel_values,
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"CLIP vision feature extraction failed: {error}"

            raise VisionExtractionError(msg) from error

        raw_features = getattr(
            outputs,
            "image_embeds",
            None,
        )

        if not isinstance(
            raw_features,
            Tensor,
        ):
            raise VisionExtractionError("CLIP vision model did not return image_embeds.")

        features = raw_features.to(
            dtype=torch.float32,
        )

        if features.ndim != 2:
            msg = f"Unexpected CLIP feature shape: {tuple(features.shape)}; expected [T, D]."

            raise VisionExtractionError(msg)

        if features.shape[0] != len(frames):
            msg = f"CLIP returned {features.shape[0]} feature vectors; expected {len(frames)}."

            raise VisionExtractionError(msg)

        if features.shape[1] != EXPECTED_VISION_DIM:
            msg = f"Unexpected CLIP feature dimension: {features.shape[1]}; expected {EXPECTED_VISION_DIM}."

            raise VisionExtractionError(msg)

        if not bool(
            torch.isfinite(
                features,
            )
            .all()
            .item()
        ):
            raise VisionExtractionError("CLIP features contain NaN or infinite values.")

        return features.contiguous()


__all__ = [
    "DEFAULT_VISION_MODEL",
    "EXPECTED_VISION_DIM",
    "CLIPSceneExtractor",
    "VisionExtractionError",
]
