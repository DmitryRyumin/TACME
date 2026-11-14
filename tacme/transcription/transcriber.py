"""
File: transcriber.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Whisper model loading, language detection, and transcription inference.

License: MIT License
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor, nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from tacme.models.common import resolve_device
from tacme.transcription.audio import _decode_media_waveform, _validate_media_path
from tacme.transcription.contracts import (
    DEFAULT_WHISPER_MODEL,
    WHISPER_CHUNK_SECONDS,
    WhisperTranscription,
    WhisperTranscriptionError,
)
from tacme.transcription.language import _WHISPER_LANGUAGE_NAMES, _WhisperLanguageDetection


class _WhisperFeatureExtractorProtocol(Protocol):
    """Sampling-rate interface required from the Whisper feature extractor."""

    sampling_rate: int


class _WhisperProcessorProtocol(Protocol):
    """Minimal Whisper processor interface required by TACME."""

    feature_extractor: _WhisperFeatureExtractorProtocol

    def __call__(
        self,
        audio: object,
        *,
        sampling_rate: int,
        return_tensors: str,
        truncation: bool,
        padding: str,
        return_attention_mask: bool,
    ) -> Mapping[str, object]:
        """Prepare one waveform for Whisper."""
        ...

    def batch_decode(
        self,
        sequences: Tensor,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        """Decode generated Whisper token IDs."""
        ...


def _normalize_transcription(
    text: str,
) -> str:
    """Normalize Whisper output exactly as in the released TACME preprocessing."""

    normalized = " ".join(
        text.replace(
            "\n",
            " ",
        ).split()
    ).strip()

    if normalized and not normalized.endswith("."):
        normalized += "."

    return normalized


class WhisperTranscriber:
    """Generate speech transcripts with the TACME Whisper model."""

    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        *,
        device: str | torch.device = "auto",
    ) -> None:
        """Load the pretrained TACME Whisper processor and model."""

        self.model_name = model_name

        self.device = resolve_device(
            device,
        )

        try:
            processor_loader = cast(
                Callable[..., object],
                WhisperProcessor.from_pretrained,
            )

            raw_processor = processor_loader(
                self.model_name,
            )

            model_loader = cast(
                Callable[..., object],
                WhisperForConditionalGeneration.from_pretrained,
            )

            raw_model = model_loader(
                self.model_name,
                dtype=torch.float32,
            )

        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to load Whisper model {self.model_name!r}: {error}"
            raise WhisperTranscriptionError(msg) from error

        processor = cast(
            _WhisperProcessorProtocol,
            raw_processor,
        )

        if not isinstance(
            raw_model,
            nn.Module,
        ):
            raise WhisperTranscriptionError("Loaded Whisper object is not a PyTorch module.")

        generate = getattr(
            raw_model,
            "generate",
            None,
        )

        if not callable(
            generate,
        ):
            raise WhisperTranscriptionError("Loaded Whisper model does not provide a callable generate method.")

        self.processor = processor
        self.model: nn.Module = raw_model
        self._generate: Callable[..., object] = generate

        self.model.to(
            self.device,
        )

        self.model.requires_grad_(
            False,
        )

        self.model.eval()

        sample_rate = self.processor.feature_extractor.sampling_rate

        if sample_rate <= 0:
            msg = f"Invalid Whisper processor sample rate: {sample_rate}."
            raise WhisperTranscriptionError(msg)

        self.sample_rate = int(
            sample_rate,
        )

    def fingerprint(
        self,
    ) -> str:
        """Return the stable identifier of the Whisper transcriber."""

        return f"hf-whisper:{self.model_name}"

    def _prepare_inputs(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
    ) -> dict[str, Tensor]:
        """Convert one waveform into Whisper model inputs."""

        try:
            raw_inputs = self.processor(
                waveform.numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                truncation=False,
                padding="max_length",
                return_attention_mask=True,
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to preprocess Whisper waveform: {error}"
            raise WhisperTranscriptionError(msg) from error

        inputs: dict[str, Tensor] = {}

        for key, value in raw_inputs.items():
            if not isinstance(
                value,
                Tensor,
            ):
                msg = f"Unexpected non-tensor Whisper processor output for {key!r}."
                raise WhisperTranscriptionError(msg)

            if key == "input_features":
                value = value.to(
                    device=self.device,
                    dtype=torch.float32,
                )

            else:
                value = value.to(
                    self.device,
                )

            inputs[key] = value

        if "input_features" not in inputs:
            raise WhisperTranscriptionError("Whisper processor returned no input_features tensor.")

        return inputs

    @staticmethod
    def _language_code_from_token(
        token: str,
    ) -> str:
        """Convert one Whisper language token into its compact language code."""

        normalized = token.strip()

        if normalized.startswith("<|") and normalized.endswith("|>"):
            normalized = normalized[2:-2]

        return normalized.lower()

    @torch.inference_mode()
    def _detect_language(
        self,
        inputs: Mapping[str, Tensor],
    ) -> _WhisperLanguageDetection | None:
        """Estimate Whisper language and normalized confidence from encoder features."""

        generation_config = getattr(
            self.model,
            "generation_config",
            None,
        )

        raw_lang_to_id = getattr(
            generation_config,
            "lang_to_id",
            None,
        )

        if not isinstance(
            raw_lang_to_id,
            Mapping,
        ):
            return None

        language_tokens: list[tuple[str, int]] = []

        for raw_token, raw_token_id in raw_lang_to_id.items():
            if not isinstance(
                raw_token,
                str,
            ) or not isinstance(
                raw_token_id,
                int,
            ):
                continue

            code = self._language_code_from_token(
                raw_token,
            )

            if code not in _WHISPER_LANGUAGE_NAMES:
                continue

            language_tokens.append(
                (
                    raw_token,
                    raw_token_id,
                )
            )

        if not language_tokens:
            return None

        decoder_start_token_id = getattr(
            generation_config,
            "decoder_start_token_id",
            None,
        )

        if not isinstance(
            decoder_start_token_id,
            int,
        ):
            model_config = getattr(
                self.model,
                "config",
                None,
            )

            decoder_start_token_id = getattr(
                model_config,
                "decoder_start_token_id",
                None,
            )

        if not isinstance(
            decoder_start_token_id,
            int,
        ):
            return None

        input_features = inputs.get(
            "input_features",
        )

        if input_features is None:
            return None

        decoder_input_ids = torch.tensor(
            [
                [
                    decoder_start_token_id,
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )

        forward_kwargs: dict[str, object] = {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": False,
        }

        attention_mask = inputs.get(
            "attention_mask",
        )

        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask

        model_forward = cast(
            Callable[..., object],
            self.model,
        )

        try:
            raw_outputs = model_forward(
                **forward_kwargs,
            )
        except (
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return None

        logits = getattr(
            raw_outputs,
            "logits",
            None,
        )

        if (
            not isinstance(
                logits,
                Tensor,
            )
            or logits.ndim != 3
            or logits.shape[0] != 1
            or logits.shape[1] < 1
        ):
            return None

        language_token_ids = torch.tensor(
            [token_id for _, token_id in language_tokens],
            dtype=torch.long,
            device=logits.device,
        )

        language_logits = logits[0, -1].index_select(
            0,
            language_token_ids,
        )

        language_probabilities = torch.softmax(
            language_logits.to(
                dtype=torch.float32,
            ),
            dim=0,
        )

        best_index = int(
            torch.argmax(
                language_probabilities,
            )
            .detach()
            .cpu()
            .item()
        )

        token = language_tokens[best_index][0]
        code = self._language_code_from_token(
            token,
        )

        confidence = float(language_probabilities[best_index].detach().cpu().item())

        return _WhisperLanguageDetection(
            code=code,
            name=_WHISPER_LANGUAGE_NAMES.get(
                code,
                code.upper(),
            ),
            confidence=confidence,
        )

    def _validate_waveform(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
    ) -> Tensor:
        """Validate and move one Whisper waveform to contiguous CPU float32."""

        if sample_rate <= 0:
            msg = f"Invalid Whisper input sample rate: {sample_rate}."
            raise WhisperTranscriptionError(msg)

        if sample_rate != self.sample_rate:
            msg = f"Whisper input sample rate is {sample_rate} Hz; expected {self.sample_rate} Hz."
            raise WhisperTranscriptionError(msg)

        if waveform.ndim != 1:
            msg = f"Whisper waveform must have shape [N], got {tuple(waveform.shape)}."
            raise WhisperTranscriptionError(msg)

        if waveform.numel() == 0:
            raise WhisperTranscriptionError("Whisper waveform is empty.")

        prepared_waveform = (
            waveform.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )

        if not bool(
            torch.isfinite(
                prepared_waveform,
            )
            .all()
            .item()
        ):
            raise WhisperTranscriptionError("Whisper waveform contains NaN or infinite values.")

        return prepared_waveform

    @torch.inference_mode()
    def _transcribe_prepared_waveform(
        self,
        prepared_waveform: Tensor,
        *,
        sample_rate: int,
        detect_language: bool,
    ) -> WhisperTranscription:
        """Transcribe one validated waveform and optionally retain language metadata."""

        inputs = self._prepare_inputs(
            prepared_waveform,
            sample_rate=sample_rate,
        )

        long_form = prepared_waveform.numel() > (WHISPER_CHUNK_SECONDS * sample_rate)

        language_detection: _WhisperLanguageDetection | None = None

        if detect_language:
            language_inputs = inputs

            if long_form:
                language_waveform = prepared_waveform[: WHISPER_CHUNK_SECONDS * sample_rate]
                language_inputs = self._prepare_inputs(
                    language_waveform,
                    sample_rate=sample_rate,
                )

            language_detection = self._detect_language(
                language_inputs,
            )

        input_features = inputs["input_features"]
        attention_mask = inputs.get(
            "attention_mask",
        )

        generate_kwargs: dict[str, object] = {
            "input_features": input_features,
            "task": "transcribe",
            "return_timestamps": long_form,
        }

        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask

        try:
            generated_raw = self._generate(
                **generate_kwargs,
            )
        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Whisper inference failed for in-memory waveform: {error}"
            raise WhisperTranscriptionError(msg) from error

        if not isinstance(
            generated_raw,
            Tensor,
        ):
            msg = f"Unexpected Whisper generation output type: {type(generated_raw).__name__}."
            raise WhisperTranscriptionError(msg)

        try:
            decoded = self.processor.batch_decode(
                generated_raw,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to decode Whisper output tokens: {error}"
            raise WhisperTranscriptionError(msg) from error

        if len(decoded) != 1:
            msg = f"Whisper returned {len(decoded)} transcriptions; expected exactly one."
            raise WhisperTranscriptionError(msg)

        return WhisperTranscription(
            text=_normalize_transcription(
                decoded[0],
            ),
            language_code=(language_detection.code if language_detection is not None else None),
            language_name=(language_detection.name if language_detection is not None else None),
            language_confidence=(language_detection.confidence if language_detection is not None else None),
        )

    @torch.inference_mode()
    def transcribe_waveform(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
    ) -> str:
        """Transcribe one mono float32 waveform directly from memory."""

        prepared_waveform = self._validate_waveform(
            waveform,
            sample_rate=sample_rate,
        )

        return self._transcribe_prepared_waveform(
            prepared_waveform,
            sample_rate=sample_rate,
            detect_language=False,
        ).text

    @torch.inference_mode()
    def transcribe_waveform_with_metadata(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
    ) -> WhisperTranscription:
        """Transcribe one waveform and retain Whisper language metadata."""

        prepared_waveform = self._validate_waveform(
            waveform,
            sample_rate=sample_rate,
        )

        return self._transcribe_prepared_waveform(
            prepared_waveform,
            sample_rate=sample_rate,
            detect_language=True,
        )

    def transcribe(
        self,
        media_path: str | Path,
    ) -> str:
        """Transcribe speech from one audio or video file."""

        path = _validate_media_path(
            Path(
                media_path,
            ),
        )

        waveform = _decode_media_waveform(
            path,
            target_sample_rate=self.sample_rate,
        )

        return self.transcribe_waveform(
            waveform,
            sample_rate=self.sample_rate,
        )
