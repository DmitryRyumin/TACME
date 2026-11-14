"""
File: ensure_model.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Idempotent preparation and verification of the TACME Safetensors runtime model.

License: MIT License
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
from pathlib import Path
import sys
from typing import Final

from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_PATH: Final = PROJECT_ROOT / "checkpoints" / "tacme.pt"

DEFAULT_DESTINATION_PATH: Final = PROJECT_ROOT / "models" / "tacme.safetensors"

CONVERSION_VERSION: Final = "1"

HASH_CHUNK_SIZE: Final = 1024 * 1024


def _parse_arguments() -> argparse.Namespace:
    """Parse TACME model preparation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Ensure that the TACME Safetensors runtime model exists and matches the source PyTorch checkpoint."
        ),
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help=(f"Source PyTorch checkpoint. Default: {DEFAULT_SOURCE_PATH}"),
    )

    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION_PATH,
        help=(f"Destination Safetensors model. Default: {DEFAULT_DESTINATION_PATH}"),
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--check",
        action="store_true",
        help=("Check whether the Safetensors model is current without modifying any files."),
    )

    mode_group.add_argument(
        "--force",
        action="store_true",
        help=("Rebuild the Safetensors model even if it is already current."),
    )

    return parser.parse_args()


def _resolve_path(
    path: Path,
) -> Path:
    """Resolve a path relative to the TACME project root."""

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _sha256(
    path: Path,
) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(
            HASH_CHUNK_SIZE,
        ):
            digest.update(
                chunk,
            )

    return digest.hexdigest()


def _format_file_size(
    size_bytes: int,
) -> str:
    """Format a byte count as MiB."""

    size_mib = size_bytes / (1024 * 1024)

    return f"{size_mib:.2f} MiB"


def _read_safetensors_metadata(
    path: Path,
) -> dict[str, str] | None:
    """Read Safetensors metadata without loading tensor data."""

    if not path.is_file():
        return None

    try:
        with safe_open(
            path,
            framework="pt",
            device="cpu",
        ) as model:
            keys = tuple(
                model.keys(),
            )

            metadata = model.metadata()

    except (
        OSError,
        SafetensorError,
    ):
        return None

    if not keys:
        return None

    if metadata is None:
        return {}

    return dict(
        metadata,
    )


def _is_model_current(
    destination: Path,
    source_sha256: str,
) -> bool:
    """Return whether the runtime model matches the source checkpoint."""

    metadata = _read_safetensors_metadata(
        destination,
    )

    if metadata is None:
        return False

    return (
        metadata.get(
            "project",
        )
        == "TACME"
        and metadata.get(
            "source_sha256",
        )
        == source_sha256
        and metadata.get(
            "conversion_version",
        )
        == CONVERSION_VERSION
    )


def _load_pytorch_state_dict(
    path: Path,
) -> dict[str, Tensor]:
    """Load a tensor-only PyTorch state dictionary safely."""

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        checkpoint,
        Mapping,
    ):
        msg = "The TACME checkpoint must contain a mapping of parameter names to tensors."

        raise TypeError(msg)

    state_dict: dict[str, Tensor] = {}

    for key, value in checkpoint.items():
        if not isinstance(
            key,
            str,
        ):
            msg = "The TACME checkpoint contains a non-string parameter key."

            raise TypeError(msg)

        if not isinstance(
            value,
            Tensor,
        ):
            msg = (
                "The TACME checkpoint must be a pure "
                "tensor state dictionary. "
                f"Key {key!r} contains "
                f"{type(value).__name__}."
            )

            raise TypeError(msg)

        state_dict[key] = value.detach().cpu().contiguous()

    if not state_dict:
        msg = "The TACME checkpoint contains no tensors."

        raise ValueError(msg)

    return state_dict


def _validate_state_dict(
    state_dict: Mapping[str, Tensor],
) -> None:
    """Validate basic TACME state-dictionary invariants."""

    parameter_names = tuple(
        state_dict,
    )

    if len(parameter_names) != len(
        set(parameter_names),
    ):
        msg = "The TACME state dictionary contains duplicate parameter names."

        raise ValueError(msg)

    for name, tensor in state_dict.items():
        if tensor.layout != torch.strided:
            msg = f"Tensor {name!r} uses unsupported layout {tensor.layout}."

            raise ValueError(msg)

        if tensor.device.type != "cpu":
            msg = f"Tensor {name!r} was not moved to CPU."

            raise ValueError(msg)

        if not tensor.is_contiguous():
            msg = f"Tensor {name!r} is not contiguous."

            raise ValueError(msg)


def _checkpoint_statistics(
    state_dict: Mapping[str, Tensor],
) -> tuple[
    int,
    int,
    tuple[str, ...],
]:
    """Return tensor count, parameter count, and dtypes."""

    tensor_count = len(
        state_dict,
    )

    parameter_count = sum(tensor.numel() for tensor in state_dict.values())

    dtypes = tuple(
        sorted(
            {
                str(
                    tensor.dtype,
                )
                for tensor in state_dict.values()
            }
        )
    )

    return (
        tensor_count,
        parameter_count,
        dtypes,
    )


def _save_safetensors_model(
    state_dict: Mapping[str, Tensor],
    destination: Path,
    source_sha256: str,
) -> None:
    """Save the TACME runtime model in Safetensors format."""

    save_file(
        dict(
            state_dict,
        ),
        destination,
        metadata={
            "format": "pt",
            "project": "TACME",
            "source": "checkpoints/tacme.pt",
            "source_sha256": source_sha256,
            "conversion_version": CONVERSION_VERSION,
        },
    )


def _verify_key_sets(
    source_state_dict: Mapping[str, Tensor],
    converted_state_dict: Mapping[str, Tensor],
) -> None:
    """Verify that both checkpoints contain identical tensor keys."""

    source_keys = set(
        source_state_dict,
    )

    converted_keys = set(
        converted_state_dict,
    )

    missing_keys = sorted(
        source_keys - converted_keys,
    )

    unexpected_keys = sorted(
        converted_keys - source_keys,
    )

    if not missing_keys and not unexpected_keys:
        return

    message_parts = [
        "Safetensors key verification failed.",
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

    raise ValueError(
        " ".join(
            message_parts,
        )
    )


def _verify_tensor_equality(
    source_state_dict: Mapping[str, Tensor],
    converted_state_dict: Mapping[str, Tensor],
) -> None:
    """Verify exact tensor equality after conversion."""

    _verify_key_sets(
        source_state_dict,
        converted_state_dict,
    )

    for name, source_tensor in source_state_dict.items():
        converted_tensor = converted_state_dict[name]

        if source_tensor.shape != converted_tensor.shape:
            msg = f"Shape mismatch for {name!r}: {tuple(source_tensor.shape)} != {tuple(converted_tensor.shape)}."

            raise ValueError(msg)

        if source_tensor.dtype != converted_tensor.dtype:
            msg = f"Dtype mismatch for {name!r}: {source_tensor.dtype} != {converted_tensor.dtype}."

            raise ValueError(msg)

        if not torch.equal(
            source_tensor,
            converted_tensor,
        ):
            msg = f"Tensor data mismatch for {name!r}."

            raise ValueError(msg)


def _convert_model(
    source: Path,
    destination: Path,
    source_sha256: str,
) -> None:
    """Convert, verify, and atomically install the TACME runtime model."""

    print("Loading TACME PyTorch checkpoint...")

    source_state_dict = _load_pytorch_state_dict(
        source,
    )

    _validate_state_dict(
        source_state_dict,
    )

    (
        tensor_count,
        parameter_count,
        dtypes,
    ) = _checkpoint_statistics(
        source_state_dict,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_destination = destination.with_name(f".{destination.name}.tmp")

    if temporary_destination.exists():
        temporary_destination.unlink()

    try:
        print("Writing temporary Safetensors model...")

        _save_safetensors_model(
            source_state_dict,
            temporary_destination,
            source_sha256,
        )

        print("Reloading converted model...")

        converted_state_dict = load_file(
            temporary_destination,
            device="cpu",
        )

        print("Verifying tensor keys, shapes, dtypes, and values...")

        _verify_tensor_equality(
            source_state_dict,
            converted_state_dict,
        )

        metadata = _read_safetensors_metadata(
            temporary_destination,
        )

        if metadata is None:
            msg = "Converted TACME model does not contain readable Safetensors metadata."

            raise ValueError(msg)

        if (
            metadata.get(
                "source_sha256",
            )
            != source_sha256
        ):
            msg = "Converted TACME model contains an incorrect source SHA-256."

            raise ValueError(msg)

        temporary_destination.replace(
            destination,
        )

    finally:
        if temporary_destination.exists():
            temporary_destination.unlink()

    source_size = source.stat().st_size

    destination_size = destination.stat().st_size

    destination_sha256 = _sha256(
        destination,
    )

    print()
    print("TACME runtime model prepared successfully.")

    print(f"Source: {source}")

    print(f"Destination: {destination}")

    print(f"Tensors: {tensor_count:,}")

    print(f"Parameters: {parameter_count:,}")

    print(f"Dtypes: {', '.join(dtypes)}")

    print(f"Source size: {_format_file_size(source_size)}")

    print(f"Destination size: {_format_file_size(destination_size)}")

    print(f"Source SHA-256: {source_sha256}")

    print(f"Destination SHA-256: {destination_sha256}")

    print("Missing keys: 0")

    print("Unexpected keys: 0")

    print("Tensor mismatches: 0")


def _ensure_model(
    source: Path,
    destination: Path,
    *,
    check: bool,
    force: bool,
) -> bool:
    """Ensure that the TACME Safetensors model is current."""

    if not source.is_file():
        msg = f"Source TACME checkpoint does not exist: {source}"

        raise FileNotFoundError(msg)

    if source == destination:
        msg = "Source and destination model paths must be different."

        raise ValueError(msg)

    source_sha256 = _sha256(
        source,
    )

    current = _is_model_current(
        destination,
        source_sha256,
    )

    if current and not force:
        print("TACME runtime model is up to date.")

        print(f"Source: {source}")

        print(f"Destination: {destination}")

        print(f"Source SHA-256: {source_sha256}")

        return True

    if check:
        reason = "runtime model is missing" if not destination.is_file() else "runtime model is out of date or invalid"

        print(
            (f"TACME model check failed: {reason}."),
            file=sys.stderr,
        )

        print(
            "Run: just model-ensure",
            file=sys.stderr,
        )

        return False

    if force:
        print("Rebuilding TACME runtime model...")

    elif destination.is_file():
        print("TACME source checkpoint changed. Rebuilding runtime model...")

    else:
        print("TACME runtime model is missing. Building it...")

    _convert_model(
        source,
        destination,
        source_sha256,
    )

    return True


def main() -> int:
    """Run TACME model preparation."""

    arguments = _parse_arguments()

    source = _resolve_path(
        arguments.source,
    )

    destination = _resolve_path(
        arguments.destination,
    )

    try:
        success = _ensure_model(
            source,
            destination,
            check=arguments.check,
            force=arguments.force,
        )

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        SafetensorError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"TACME model preparation failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
