"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Unmodified Qwen3-VL describer implementation and output normalization.

License: MIT License
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from tacme.behavior.contracts import (
    BEHAVIOR_PROMPT,
    DEFAULT_BEHAVIOR_MODEL,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_VIDEO_FPS,
    QWEN3_VL_IMAGE_PATCH_SIZE,
    BehaviorDescription,
    BehaviorDescriptionError,
)
from tacme.behavior.video import (
    _load_process_vision_info,
    _split_video_metadata,
    _validate_video_path,
)
from tacme.models.common import resolve_device


def _normalize_description(
    text: str,
) -> str:
    """Normalize generated behavior text to the TACME runtime contract."""

    normalized = " ".join(
        text.replace(
            "\n",
            " ",
        ).split()
    ).strip()

    if normalized and not normalized.endswith("."):
        normalized += "."

    return normalized


class QwenBehaviorDescriber:
    """Generate observable behavior descriptions from video with Qwen3-VL."""

    def __init__(
        self,
        model_name: str = DEFAULT_BEHAVIOR_MODEL,
        *,
        device: str | torch.device = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        video_fps: float = DEFAULT_VIDEO_FPS,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> None:
        """Load the TACME Qwen3-VL behavior description model."""

        self.model_name = model_name

        self.device = resolve_device(
            device,
        )

        self.max_new_tokens = int(
            max_new_tokens,
        )

        self.video_fps = float(
            video_fps,
        )

        self.max_frames = int(
            max_frames,
        )

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        if self.video_fps <= 0.0:
            raise ValueError("video_fps must be positive.")

        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive.")

        self._process_vision_info = _load_process_vision_info()

        processor_loader = cast(
            Callable[..., object],
            AutoProcessor.from_pretrained,
        )

        model_loader = cast(
            Callable[..., object],
            Qwen3VLForConditionalGeneration.from_pretrained,
        )

        try:
            self.processor = processor_loader(
                self.model_name,
            )

            raw_model = model_loader(
                self.model_name,
                dtype="auto",
            )

        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to load Qwen3-VL model {self.model_name!r}: {error}"
            raise BehaviorDescriptionError(msg) from error

        if not isinstance(
            raw_model,
            nn.Module,
        ):
            raise BehaviorDescriptionError("Loaded Qwen3-VL object is not a PyTorch module.")

        try:
            raw_model = raw_model.to(
                self.device,
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to move Qwen3-VL model to {self.device}: {error}"
            raise BehaviorDescriptionError(msg) from error

        self.model = raw_model

        self.model.requires_grad_(
            False,
        )

        self.model.eval()

        generate = getattr(
            self.model,
            "generate",
            None,
        )

        if not callable(
            generate,
        ):
            raise BehaviorDescriptionError("Loaded Qwen3-VL model does not expose generate.")

        self._generate = cast(
            Callable[..., object],
            generate,
        )

    def fingerprint(
        self,
    ) -> str:
        """Return the stable identifier of the behavior description model."""

        return f"hf-qwen3vl:{self.model_name}"

    def _build_messages(
        self,
        video_path: Path,
    ) -> list[dict[str, object]]:
        """Build the Qwen3-VL behavior-only multimodal prompt."""

        content: list[dict[str, object]] = [
            {
                "type": "video",
                "video": str(video_path),
                "fps": self.video_fps,
                "max_frames": self.max_frames,
            },
            {
                "type": "text",
                "text": BEHAVIOR_PROMPT,
            },
        ]

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _apply_chat_template(
        self,
        messages: list[dict[str, object]],
    ) -> str:
        """Render Qwen3-VL chat messages into the model prompt."""

        function = getattr(
            self.processor,
            "apply_chat_template",
            None,
        )

        if not callable(
            function,
        ):
            raise BehaviorDescriptionError("Qwen3-VL processor does not expose apply_chat_template.")

        apply_chat_template = cast(
            Callable[..., object],
            function,
        )

        try:
            text = apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to build Qwen3-VL chat template: {error}"
            raise BehaviorDescriptionError(msg) from error

        if not isinstance(
            text,
            str,
        ):
            msg = f"Unexpected Qwen3-VL chat template type: {type(text).__name__}."
            raise BehaviorDescriptionError(msg)

        return text

    def _prepare_visual_inputs(
        self,
        messages: list[dict[str, object]],
    ) -> tuple[
        object | None,
        object | None,
        list[object] | None,
        dict[str, object],
    ]:
        """Prepare Qwen3-VL images, videos, metadata, and processor arguments."""

        try:
            result = self._process_vision_info(
                messages,
                image_patch_size=QWEN3_VL_IMAGE_PATCH_SIZE,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Qwen3-VL visual preprocessing failed: {error}"
            raise BehaviorDescriptionError(msg) from error

        if (
            not isinstance(
                result,
                tuple,
            )
            or len(result) != 3
        ):
            raise BehaviorDescriptionError("Qwen3-VL visual preprocessing returned an unexpected result.")

        image_inputs = result[0]
        raw_video_inputs = result[1]
        raw_video_kwargs = result[2]

        if not isinstance(
            raw_video_kwargs,
            Mapping,
        ):
            msg = f"Unexpected Qwen3-VL video kwargs type: {type(raw_video_kwargs).__name__}."
            raise BehaviorDescriptionError(msg)

        video_inputs, video_metadata = _split_video_metadata(
            raw_video_inputs,
        )

        video_kwargs = {str(key): value for key, value in raw_video_kwargs.items()}

        return (
            image_inputs,
            video_inputs,
            video_metadata,
            video_kwargs,
        )

    def _prepare_model_inputs(
        self,
        text: str,
        image_inputs: object | None,
        video_inputs: object | None,
        video_metadata: list[object] | None,
        video_kwargs: Mapping[str, object],
    ) -> dict[str, Any]:
        """Convert processed Qwen3-VL inputs to tensors on the model device."""

        if not callable(
            self.processor,
        ):
            raise BehaviorDescriptionError("Qwen3-VL processor is not callable.")

        processor_call = cast(
            Callable[..., object],
            self.processor,
        )

        try:
            raw_inputs = processor_call(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata,
                padding=True,
                return_tensors="pt",
                do_resize=False,
                cap_pixels_per_frame=True,
                **video_kwargs,
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to prepare Qwen3-VL model inputs: {error}"
            raise BehaviorDescriptionError(msg) from error

        if not isinstance(
            raw_inputs,
            Mapping,
        ):
            msg = f"Unexpected Qwen3-VL processor output type: {type(raw_inputs).__name__}."
            raise BehaviorDescriptionError(msg)

        model_inputs: dict[str, Any] = {}

        for key, value in raw_inputs.items():
            name = str(
                key,
            )

            if isinstance(
                value,
                Tensor,
            ):
                model_inputs[name] = value.to(
                    self.device,
                )

            else:
                model_inputs[name] = value

        input_ids = model_inputs.get(
            "input_ids",
        )

        if not isinstance(
            input_ids,
            Tensor,
        ):
            raise BehaviorDescriptionError("Qwen3-VL processor did not return input_ids.")

        return model_inputs

    def _decode_generated_text(
        self,
        generated_ids: Tensor,
        input_ids: Tensor,
    ) -> str:
        """Decode generated Qwen3-VL tokens without the input prompt."""

        if generated_ids.ndim != 2:
            msg = f"Unexpected Qwen3-VL generated token shape: {tuple(generated_ids.shape)}."
            raise BehaviorDescriptionError(msg)

        if input_ids.ndim != 2:
            msg = f"Unexpected Qwen3-VL input token shape: {tuple(input_ids.shape)}."
            raise BehaviorDescriptionError(msg)

        if generated_ids.shape[0] != input_ids.shape[0]:
            msg = (
                "Qwen3-VL generated batch size does not match input batch size: "
                f"{generated_ids.shape[0]} vs {input_ids.shape[0]}."
            )
            raise BehaviorDescriptionError(msg)

        prompt_length = input_ids.shape[-1]

        if generated_ids.shape[-1] < prompt_length:
            msg = (
                "Qwen3-VL generated sequence is shorter than the input prompt: "
                f"{generated_ids.shape[-1]} < {prompt_length}."
            )
            raise BehaviorDescriptionError(msg)

        generated_only = generated_ids[
            :,
            prompt_length:,
        ]

        function = getattr(
            self.processor,
            "batch_decode",
            None,
        )

        if not callable(
            function,
        ):
            raise BehaviorDescriptionError("Qwen3-VL processor does not expose batch_decode.")

        batch_decode = cast(
            Callable[..., object],
            function,
        )

        decoded = batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if (
            not isinstance(
                decoded,
                list,
            )
            or len(decoded) != 1
        ):
            raise BehaviorDescriptionError("Qwen3-VL decoding returned an unexpected number of descriptions.")

        description = decoded[0]

        if not isinstance(
            description,
            str,
        ):
            msg = f"Unexpected Qwen3-VL decoded output type: {type(description).__name__}."
            raise BehaviorDescriptionError(msg)

        return description

    @torch.inference_mode()
    def describe(
        self,
        video_path: str | Path,
    ) -> str:
        """Generate one normalized observable behavior description."""

        path = _validate_video_path(
            Path(
                video_path,
            ),
        )

        messages = self._build_messages(
            path,
        )

        text = self._apply_chat_template(
            messages,
        )

        (
            image_inputs,
            video_inputs,
            video_metadata,
            video_kwargs,
        ) = self._prepare_visual_inputs(
            messages,
        )

        model_inputs = self._prepare_model_inputs(
            text,
            image_inputs,
            video_inputs,
            video_metadata,
            video_kwargs,
        )

        input_ids = model_inputs.get(
            "input_ids",
        )

        if not isinstance(
            input_ids,
            Tensor,
        ):
            raise BehaviorDescriptionError("Qwen3-VL input_ids are unexpectedly unavailable.")

        try:
            generated = self._generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Qwen3-VL generation failed for {path}: {error}"
            raise BehaviorDescriptionError(msg) from error

        if not isinstance(
            generated,
            Tensor,
        ):
            msg = f"Unexpected Qwen3-VL generation output type: {type(generated).__name__}."
            raise BehaviorDescriptionError(msg)

        raw_description = self._decode_generated_text(
            generated,
            input_ids,
        )

        description = _normalize_description(
            raw_description,
        )

        if not description:
            raise BehaviorDescriptionError("Qwen3-VL returned an empty behavior description.")

        return description

    @torch.inference_mode()
    def describe_with_metadata(
        self,
        video_path: str | Path,
    ) -> BehaviorDescription:
        """Generate one description and retain the exact Qwen3-VL UI metadata."""

        description = self.describe(
            video_path,
        )

        return BehaviorDescription(
            text=description,
            model_name=self.model_name,
            video_fps=self.video_fps,
            max_frames=self.max_frames,
        )
