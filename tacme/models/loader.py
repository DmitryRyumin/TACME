"""
File: loader.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Safe runtime loading and validation of the TACME Safetensors model.

License: MIT License
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file
import torch
from torch import Tensor

from tacme.config import PROJECT_ROOT
from tacme.models.common import resolve_device
from tacme.models.fusion import MultiModalFusionModel_v2

DEFAULT_MODEL_PATH: Final = PROJECT_ROOT / "models" / "tacme.safetensors"

EXPECTED_PROJECT: Final = "TACME"
EXPECTED_FORMAT: Final = "pt"
EXPECTED_CONVERSION_VERSION: Final = "1"

EXPECTED_TENSOR_COUNT: Final = 496
EXPECTED_STATE_NUMEL: Final = 3_990_047


class TacmeModelLoadError(RuntimeError):
    """Raised when the TACME runtime model cannot be validated or loaded."""


def _resolve_model_path(
    path: Path,
) -> Path:
    """Resolve a model path relative to the TACME project root."""

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _read_model_metadata(
    path: Path,
) -> dict[str, str]:
    """Read and validate Safetensors metadata without loading tensor data."""

    try:
        with safe_open(
            path,
            framework="pt",
            device="cpu",
        ) as model_file:
            raw_metadata = model_file.metadata()

    except (
        OSError,
        SafetensorError,
    ) as error:
        msg = f"Unable to read TACME Safetensors metadata from {path}: {error}"
        raise TacmeModelLoadError(msg) from error

    metadata = cast(
        dict[str, str] | None,
        raw_metadata,
    )

    if metadata is None:
        msg = "TACME runtime model does not contain Safetensors metadata."
        raise TacmeModelLoadError(msg)

    return dict(
        metadata,
    )


def _validate_model_metadata(
    metadata: Mapping[str, str],
) -> None:
    """Validate TACME-specific Safetensors metadata."""

    project = metadata.get(
        "project",
    )

    if project != EXPECTED_PROJECT:
        msg = f"Invalid TACME model metadata: project={project!r}, expected {EXPECTED_PROJECT!r}."
        raise TacmeModelLoadError(msg)

    model_format = metadata.get(
        "format",
    )

    if model_format != EXPECTED_FORMAT:
        msg = f"Invalid TACME model metadata: format={model_format!r}, expected {EXPECTED_FORMAT!r}."
        raise TacmeModelLoadError(msg)

    conversion_version = metadata.get(
        "conversion_version",
    )

    if conversion_version != EXPECTED_CONVERSION_VERSION:
        msg = (
            "Unsupported TACME model conversion version: "
            f"{conversion_version!r}; "
            f"expected "
            f"{EXPECTED_CONVERSION_VERSION!r}."
        )
        raise TacmeModelLoadError(msg)

    source_sha256 = metadata.get(
        "source_sha256",
    )

    if (
        source_sha256 is None
        or len(
            source_sha256,
        )
        != 64
    ):
        msg = "TACME model metadata contains an invalid source SHA-256 digest."
        raise TacmeModelLoadError(msg)


def _state_numel(
    state_dict: Mapping[str, Tensor],
) -> int:
    """Return the total number of scalar values in a state dictionary."""

    return sum(tensor.numel() for tensor in state_dict.values())


def _validate_runtime_architecture(
    model: MultiModalFusionModel_v2,
) -> dict[str, Tensor]:
    """Ensure that the runtime architecture still matches the released model."""

    state_dict = dict(
        model.state_dict(),
    )

    tensor_count = len(
        state_dict,
    )

    if tensor_count != EXPECTED_TENSOR_COUNT:
        msg = (
            "TACME runtime architecture has changed: "
            f"{tensor_count} state tensors found, "
            f"expected {EXPECTED_TENSOR_COUNT}."
        )
        raise TacmeModelLoadError(msg)

    state_numel = _state_numel(
        state_dict,
    )

    if state_numel != EXPECTED_STATE_NUMEL:
        msg = (
            "TACME runtime architecture has changed: "
            f"{state_numel:,} state values found, "
            f"expected {EXPECTED_STATE_NUMEL:,}."
        )
        raise TacmeModelLoadError(msg)

    return state_dict


def _validate_checkpoint_state(
    checkpoint_state: Mapping[str, Tensor],
    architecture_state: Mapping[str, Tensor],
) -> None:
    """Validate checkpoint keys, shapes, and dtypes against the architecture."""

    checkpoint_keys = set(
        checkpoint_state,
    )

    architecture_keys = set(
        architecture_state,
    )

    missing_keys = sorted(
        architecture_keys - checkpoint_keys,
    )

    unexpected_keys = sorted(
        checkpoint_keys - architecture_keys,
    )

    if missing_keys or unexpected_keys:
        message_parts = [
            "TACME checkpoint is incompatible with the runtime architecture.",
        ]

        if missing_keys:
            message_parts.append(
                "Missing keys: "
                + ", ".join(
                    missing_keys,
                )
            )

        if unexpected_keys:
            message_parts.append(
                "Unexpected keys: "
                + ", ".join(
                    unexpected_keys,
                )
            )

        raise TacmeModelLoadError(
            " ".join(
                message_parts,
            )
        )

    if (
        len(
            checkpoint_state,
        )
        != EXPECTED_TENSOR_COUNT
    ):
        msg = f"TACME checkpoint contains {len(checkpoint_state)} tensors, expected {EXPECTED_TENSOR_COUNT}."
        raise TacmeModelLoadError(msg)

    checkpoint_numel = _state_numel(
        checkpoint_state,
    )

    if checkpoint_numel != EXPECTED_STATE_NUMEL:
        msg = f"TACME checkpoint contains {checkpoint_numel:,} state values, expected {EXPECTED_STATE_NUMEL:,}."
        raise TacmeModelLoadError(msg)

    for name, expected_tensor in architecture_state.items():
        checkpoint_tensor = checkpoint_state[name]

        if checkpoint_tensor.shape != expected_tensor.shape:
            msg = (
                f"TACME checkpoint shape mismatch "
                f"for {name!r}: "
                f"{tuple(checkpoint_tensor.shape)} != "
                f"{tuple(expected_tensor.shape)}."
            )
            raise TacmeModelLoadError(msg)

        if checkpoint_tensor.dtype != expected_tensor.dtype:
            msg = f"TACME checkpoint dtype mismatch for {name!r}: {checkpoint_tensor.dtype} != {expected_tensor.dtype}."
            raise TacmeModelLoadError(msg)


def _load_checkpoint_state(
    path: Path,
) -> dict[str, Tensor]:
    """Load the TACME Safetensors state dictionary on CPU."""

    try:
        state_dict = load_file(
            path,
            device="cpu",
        )

    except (
        OSError,
        SafetensorError,
    ) as error:
        msg = f"Unable to load TACME Safetensors model from {path}: {error}"
        raise TacmeModelLoadError(msg) from error

    return state_dict


def load_tacme_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    *,
    device: str | torch.device = "auto",
) -> MultiModalFusionModel_v2:
    """Load and validate the TACME runtime model."""

    path = _resolve_model_path(
        model_path,
    )

    if not path.is_file():
        msg = f"TACME runtime model does not exist: {path}. Run `just model-ensure` first."
        raise TacmeModelLoadError(msg)

    metadata = _read_model_metadata(
        path,
    )

    _validate_model_metadata(
        metadata,
    )

    resolved_device = resolve_device(
        device,
    )

    model = MultiModalFusionModel_v2(
        device="cpu",
    )

    architecture_state = _validate_runtime_architecture(
        model,
    )

    checkpoint_state = _load_checkpoint_state(
        path,
    )

    _validate_checkpoint_state(
        checkpoint_state,
        architecture_state,
    )

    try:
        model.load_state_dict(
            checkpoint_state,
            strict=True,
            assign=True,
        )

    except RuntimeError as error:
        msg = "Unable to apply TACME checkpoint to the runtime architecture."
        raise TacmeModelLoadError(msg) from error

    model.to(
        resolved_device,
    )

    model.device = resolved_device

    model.requires_grad_(
        False,
    )

    model.eval()

    return model


__all__ = [
    "DEFAULT_MODEL_PATH",
    "TacmeModelLoadError",
    "load_tacme_model",
]
