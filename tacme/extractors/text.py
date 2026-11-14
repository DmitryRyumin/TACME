"""
File: text.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: EmoRoBERTa text feature extraction for the TACME inference pipeline.

License: MIT License
"""

from collections.abc import Callable, Mapping
from typing import Final, Protocol, cast

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from tacme.models.common import resolve_device

DEFAULT_TEXT_MODEL: Final = "michellejieli/emotion_text_classifier"

DEFAULT_MAX_LENGTH: Final = 512
EXPECTED_TEXT_DIM: Final = 768


class TextExtractionError(RuntimeError):
    """Raised when TACME text feature extraction fails."""


class _TokenizerProtocol(Protocol):
    """Minimal tokenizer interface required by the TACME text extractor."""

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> Mapping[str, object]:
        """Tokenize one text sequence."""
        ...


class EmoRoBERTaExtractor:
    """Extract EmoRoBERTa token-level features for TACME."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        *,
        device: str | torch.device = "auto",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        """Load the pretrained EmoRoBERTa tokenizer and encoder."""

        if max_length <= 0:
            msg = f"max_length must be positive, got {max_length}."
            raise ValueError(msg)

        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_length = max_length

        try:
            tokenizer_loader = cast(
                Callable[[str], object],
                AutoTokenizer.from_pretrained,
            )

            tokenizer = tokenizer_loader(
                self.model_name,
            )

            model = AutoModel.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                add_pooling_layer=False,
            )

        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to load EmoRoBERTa model {self.model_name!r}: {error}"
            raise TextExtractionError(msg) from error

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TextExtractionError("Loaded EmoRoBERTa object is not a PyTorch module.")

        self.tokenizer = cast(
            _TokenizerProtocol,
            tokenizer,
        )

        self.model = model

        self.model.to(
            self.device,
        )

        self.model.requires_grad_(
            False,
        )

        self.model.eval()

    def fingerprint(
        self,
    ) -> str:
        """Return the stable identifier of the text feature extractor."""

        return f"hf-emoroberta:{self.model_name}"

    def _prepare_inputs(
        self,
        text: str,
    ) -> dict[str, Tensor]:
        """Tokenize one non-empty text string."""

        if not text.strip():
            raise TextExtractionError("Text input is empty.")

        try:
            raw_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to tokenize text for {self.model_name!r}: {error}"
            raise TextExtractionError(msg) from error

        inputs: dict[str, Tensor] = {}

        for key, value in raw_inputs.items():
            if not isinstance(
                value,
                Tensor,
            ):
                msg = f"Unexpected non-tensor EmoRoBERTa tokenizer output for {key!r}."
                raise TextExtractionError(msg)

            inputs[key] = value.to(
                self.device,
            )

        if "input_ids" not in inputs:
            raise TextExtractionError("EmoRoBERTa tokenizer did not return input_ids.")

        if not inputs:
            raise TextExtractionError("EmoRoBERTa tokenizer returned no model inputs.")

        return inputs

    @torch.inference_mode()
    def extract(
        self,
        text: str,
    ) -> Tensor:
        """Extract a float32 token sequence with shape [T, 768]."""

        inputs = self._prepare_inputs(
            text,
        )

        try:
            outputs = self.model(
                **inputs,
                return_dict=True,
            )

            hidden_state = getattr(
                outputs,
                "last_hidden_state",
                None,
            )

            if not isinstance(
                hidden_state,
                Tensor,
            ):
                raise TextExtractionError(
                    "EmoRoBERTa did not return last_hidden_state.",
                )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"EmoRoBERTa inference failed: {error}"
            raise TextExtractionError(msg) from error

        raw_hidden_state = getattr(
            outputs,
            "last_hidden_state",
            None,
        )

        hidden_state = (
            raw_hidden_state
            if isinstance(
                raw_hidden_state,
                Tensor,
            )
            else None
        )

        if hidden_state is None:
            raw_hidden_states = getattr(
                outputs,
                "hidden_states",
                None,
            )

            if (
                isinstance(
                    raw_hidden_states,
                    (
                        tuple,
                        list,
                    ),
                )
                and raw_hidden_states
            ):
                candidate = raw_hidden_states[-1]

                if isinstance(
                    candidate,
                    Tensor,
                ):
                    hidden_state = candidate

        if hidden_state is None:
            raise TextExtractionError("EmoRoBERTa did not return a valid hidden-state tensor.")

        if hidden_state.ndim != 3 or hidden_state.shape[0] != 1:
            msg = f"Unexpected EmoRoBERTa output shape: {tuple(hidden_state.shape)}; expected [1, T, D]."
            raise TextExtractionError(msg)

        features = hidden_state.squeeze(
            0,
        ).to(
            dtype=torch.float32,
        )

        attention_mask = inputs.get(
            "attention_mask",
        )

        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != 1 or attention_mask.shape[1] != features.shape[0]:
                msg = f"Unexpected EmoRoBERTa attention-mask shape: {tuple(attention_mask.shape)}."
                raise TextExtractionError(msg)

            valid_tokens = attention_mask.squeeze(
                0,
            ).to(
                dtype=torch.bool,
            )

            features = features[valid_tokens]

        if features.ndim != 2:
            msg = f"Unexpected EmoRoBERTa feature rank: {features.ndim}; expected 2."
            raise TextExtractionError(msg)

        if features.shape[0] == 0:
            raise TextExtractionError("EmoRoBERTa returned an empty feature sequence.")

        if features.shape[1] != EXPECTED_TEXT_DIM:
            msg = f"Unexpected EmoRoBERTa feature dimension: {features.shape[1]}; expected {EXPECTED_TEXT_DIM}."
            raise TextExtractionError(msg)

        if not bool(
            torch.isfinite(
                features,
            )
            .all()
            .item()
        ):
            raise TextExtractionError("EmoRoBERTa features contain NaN or infinite values.")

        return features.contiguous()


__all__ = [
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_TEXT_MODEL",
    "EXPECTED_TEXT_DIM",
    "EmoRoBERTaExtractor",
    "TextExtractionError",
]
