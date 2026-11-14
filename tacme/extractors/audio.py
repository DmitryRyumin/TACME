"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: EmoWav2Vec2 audio feature extraction for the TACME inference pipeline.

License: MIT License
"""

from collections.abc import Callable, Mapping
from pathlib import Path
import shutil
import subprocess
from typing import Final, Protocol, cast

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoProcessor

from tacme.models.common import resolve_device

DEFAULT_AUDIO_MODEL: Final = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

DEFAULT_SAMPLE_RATE: Final = 16_000
EXPECTED_AUDIO_DIM: Final = 1024
AMPLITUDE_EPSILON: Final = 1e-8


class AudioExtractionError(RuntimeError):
    """Raised when TACME audio feature extraction fails."""


class _FeatureExtractorProtocol(Protocol):
    """Required sampling-rate interface of the Hugging Face feature extractor."""

    sampling_rate: int


class _AudioProcessorProtocol(Protocol):
    """Minimal processor interface required by the TACME audio extractor."""

    feature_extractor: _FeatureExtractorProtocol

    def __call__(
        self,
        audio: Tensor,
        *,
        sampling_rate: int,
        return_tensors: str,
    ) -> Mapping[str, object]:
        """Prepare one waveform for EmoWav2Vec2."""
        ...


def _validate_audio_path(
    audio_path: Path,
) -> Path:
    """Resolve and validate an input media file path."""

    path = audio_path.expanduser().resolve()

    if not path.is_file():
        msg = f"Audio file does not exist: {path}"
        raise AudioExtractionError(msg)

    return path


def _resolve_ffmpeg() -> str:
    """Resolve the FFmpeg executable used for audio decoding."""

    executable = shutil.which(
        "ffmpeg",
    )

    if executable is None:
        raise AudioExtractionError("FFmpeg is required for audio decoding but was not found in PATH.")

    return executable


def _load_audio(
    audio_path: Path,
    *,
    target_sample_rate: int,
) -> Tensor:
    """Decode an input media file into a normalized mono float32 waveform."""

    path = _validate_audio_path(
        audio_path,
    )

    ffmpeg = _resolve_ffmpeg()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
        )

    except OSError as error:
        msg = f"Unable to execute FFmpeg for {path}: {error}"
        raise AudioExtractionError(msg) from error

    if result.returncode != 0:
        error_message = result.stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if not error_message:
            error_message = f"FFmpeg exited with code {result.returncode}."

        msg = f"Unable to decode audio file {path}: {error_message}"
        raise AudioExtractionError(msg)

    if not result.stdout:
        msg = f"Audio file contains no decodable audio samples: {path}"
        raise AudioExtractionError(msg)

    if len(result.stdout) % torch.float32.itemsize != 0:
        msg = f"FFmpeg returned an invalid float32 audio buffer with {len(result.stdout)} bytes."
        raise AudioExtractionError(msg)

    buffer = bytearray(
        result.stdout,
    )

    waveform = torch.frombuffer(
        buffer,
        dtype=torch.float32,
    ).clone()

    if waveform.numel() == 0:
        msg = f"Audio file contains no usable samples: {path}"
        raise AudioExtractionError(msg)

    if not bool(
        torch.isfinite(
            waveform,
        )
        .all()
        .item()
    ):
        raise AudioExtractionError("Decoded audio waveform contains NaN or infinite values.")

    waveform = waveform / (waveform.abs().max() + AMPLITUDE_EPSILON)

    return waveform.contiguous()


class EmoWav2Vec2Extractor:
    """Extract EmoWav2Vec2 sequence features for TACME."""

    def __init__(
        self,
        model_name: str = DEFAULT_AUDIO_MODEL,
        *,
        device: str | torch.device = "auto",
    ) -> None:
        """Load the pretrained EmoWav2Vec2 processor and encoder."""

        self.model_name = model_name

        self.device = resolve_device(
            device,
        )

        try:
            processor_loader = cast(
                Callable[[str], object],
                AutoProcessor.from_pretrained,
            )

            processor = processor_loader(
                self.model_name,
            )

            model = AutoModel.from_pretrained(
                self.model_name,
                dtype=torch.float32,
            )

        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to load EmoWav2Vec2 model {self.model_name!r}: {error}"
            raise AudioExtractionError(msg) from error

        if not isinstance(
            model,
            nn.Module,
        ):
            raise AudioExtractionError("Loaded EmoWav2Vec2 object is not a PyTorch module.")

        self.processor = cast(
            _AudioProcessorProtocol,
            processor,
        )

        self.model = model

        self.model.to(
            self.device,
        )

        self.model.requires_grad_(
            False,
        )

        self.model.eval()

        sample_rate = self.processor.feature_extractor.sampling_rate

        if sample_rate <= 0:
            msg = f"Invalid EmoWav2Vec2 sampling rate: {sample_rate}."
            raise AudioExtractionError(msg)

        self.sample_rate = int(
            sample_rate,
        )

    def fingerprint(
        self,
    ) -> str:
        """Return the stable identifier of the audio feature extractor."""

        return f"hf-w2v2:{self.model_name}"

    def _prepare_inputs(
        self,
        waveform: Tensor,
    ) -> dict[str, Tensor]:
        """Convert one waveform into model input tensors."""

        try:
            raw_inputs = self.processor(
                waveform.cpu(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"Unable to preprocess audio for {self.model_name!r}: {error}"
            raise AudioExtractionError(msg) from error

        inputs: dict[str, Tensor] = {}

        for key, value in raw_inputs.items():
            if not isinstance(
                value,
                Tensor,
            ):
                msg = f"Unexpected non-tensor EmoWav2Vec2 processor output for {key!r}."
                raise AudioExtractionError(msg)

            inputs[key] = value.to(
                self.device,
            )

        if not inputs:
            raise AudioExtractionError("EmoWav2Vec2 processor returned no model inputs.")

        return inputs

    @torch.inference_mode()
    def extract(
        self,
        audio_path: str | Path,
    ) -> Tensor:
        """Extract a float32 sequence tensor with shape [T, 1024]."""

        waveform = _load_audio(
            Path(
                audio_path,
            ),
            target_sample_rate=self.sample_rate,
        )

        inputs = self._prepare_inputs(
            waveform,
        )

        try:
            outputs = self.model(
                **inputs,
                return_dict=True,
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"EmoWav2Vec2 inference failed for {audio_path}: {error}"
            raise AudioExtractionError(msg) from error

        hidden_state = getattr(
            outputs,
            "last_hidden_state",
            None,
        )

        if not isinstance(
            hidden_state,
            Tensor,
        ):
            raise AudioExtractionError("EmoWav2Vec2 did not return last_hidden_state.")

        if hidden_state.ndim != 3 or hidden_state.shape[0] != 1:
            msg = f"Unexpected EmoWav2Vec2 output shape: {tuple(hidden_state.shape)}; expected [1, T, D]."
            raise AudioExtractionError(msg)

        features = hidden_state.squeeze(
            0,
        ).to(
            dtype=torch.float32,
        )

        if features.shape[-1] != EXPECTED_AUDIO_DIM:
            msg = f"Unexpected EmoWav2Vec2 feature dimension: {features.shape[-1]}; expected {EXPECTED_AUDIO_DIM}."
            raise AudioExtractionError(msg)

        if features.shape[0] == 0:
            raise AudioExtractionError("EmoWav2Vec2 returned an empty feature sequence.")

        if not bool(
            torch.isfinite(
                features,
            )
            .all()
            .item()
        ):
            raise AudioExtractionError("EmoWav2Vec2 features contain NaN or infinite values.")

        return features.contiguous()


__all__ = [
    "DEFAULT_AUDIO_MODEL",
    "EXPECTED_AUDIO_DIM",
    "AudioExtractionError",
    "EmoWav2Vec2Extractor",
]
