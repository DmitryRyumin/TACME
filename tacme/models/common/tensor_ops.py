"""
File: tensor_ops.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Device selection and masked tensor operations.

License: MIT License
"""

import torch
from torch import Tensor, nn


def resolve_device(
    device: str | torch.device,
) -> torch.device:
    """Resolve a requested PyTorch device."""

    if isinstance(
        device,
        torch.device,
    ):
        requested = device
    else:
        value = device.strip().lower()

        if value == "auto":
            if torch.cuda.is_available():
                return torch.device(
                    "cuda",
                )

            if torch.backends.mps.is_available():
                return torch.device(
                    "mps",
                )

            return torch.device(
                "cpu",
            )

        requested = torch.device(
            value or "cpu",
        )

    if requested.type == "cuda" and not torch.cuda.is_available():
        msg = f"CUDA was requested but is not available: {requested}."
        raise RuntimeError(msg)

    if requested.type == "mps" and not torch.backends.mps.is_available():
        msg = "MPS was requested but is not available."
        raise RuntimeError(msg)

    return requested


def module_device(
    module: nn.Module,
) -> torch.device:
    """Return the current device of a module's parameters or buffers."""

    parameter = next(
        module.parameters(),
        None,
    )

    if parameter is not None:
        return parameter.device

    buffer = next(
        module.buffers(),
        None,
    )

    if buffer is not None:
        return buffer.device

    return torch.device(
        "cpu",
    )


def masked_mean(
    x: Tensor,
    mask: Tensor | None,
) -> Tensor:
    """Mean-pool a sequence while ignoring invalid positions."""

    if mask is None:
        return x.mean(
            dim=1,
        )

    mask_f = mask.unsqueeze(
        -1,
    ).to(
        dtype=x.dtype,
    )

    denominator = mask_f.sum(
        dim=1,
    ).clamp_min(
        1.0,
    )

    return (x * mask_f).sum(
        dim=1,
    ) / denominator


def ensure_sequence_input(
    features: Tensor,
    mask: Tensor | None,
) -> tuple[
    Tensor,
    Tensor,
]:
    """Normalize modality features to [batch, time, features] form."""

    if features.ndim == 2:
        features = features.unsqueeze(
            1,
        )

    elif features.ndim != 3:
        msg = f"Expected feature tensor [B, D] or [B, T, D], got {tuple(features.shape)}."
        raise ValueError(msg)

    if mask is None:
        mask = torch.ones(
            features.shape[:2],
            dtype=torch.bool,
            device=features.device,
        )

    else:
        mask = mask.to(
            device=features.device,
            dtype=torch.bool,
        )

        if mask.shape != features.shape[:2]:
            msg = (
                "Feature mask must have shape [B, T] matching "
                "the feature sequence; "
                f"got {tuple(mask.shape)} for "
                f"features {tuple(features.shape)}."
            )
            raise ValueError(msg)

    return (
        features,
        mask,
    )
