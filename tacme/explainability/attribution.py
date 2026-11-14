"""
File: attribution.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Gradient-times-input feature attribution for TACME inference.

License: MIT License
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor
from torch.nn import functional as torch_functional

if TYPE_CHECKING:
    from tacme.models import MultiModalFusionModel_v2
    from tacme.models.common import FeatureBatch

FEATURE_ATTRIBUTION_BIN_COUNT: Final = 32

ATTRIBUTION_MODALITY_ORDER: Final[tuple[str, ...]] = (
    "text",
    "audio",
    "behavior",
    "scene",
)

ATTRIBUTION_TASK_ORDER: Final[tuple[str, ...]] = (
    "emotion",
    "personality",
    "ambivalence",
)


@dataclass(
    frozen=True,
    slots=True,
)
class FeatureAttributionTarget:
    """One TACME output target and its compact multimodal attribution map."""

    task: str
    target_index: int
    score: float
    values: tuple[tuple[float, ...], ...]
    row_magnitudes: tuple[float, ...]
    modality_magnitudes: tuple[float, ...]

    @property
    def target_id(self) -> str:
        """Return the stable UI identifier of this attribution target."""

        return f"{self.task}:{self.target_index}"


@dataclass(
    frozen=True,
    slots=True,
)
class FeatureAttributionBundle:
    """All local feature attributions computed for one TACME prediction."""

    targets: tuple[FeatureAttributionTarget, ...]
    modality_order: tuple[str, ...]
    feature_bin_count: int
    method: str = "gradient_x_input"

    def targets_for_task(
        self,
        task: str,
    ) -> tuple[FeatureAttributionTarget, ...]:
        """Return attribution targets belonging to one TACME task."""

        return tuple(target for target in self.targets if target.task == task)

    def get_target(
        self,
        target_id: str,
    ) -> FeatureAttributionTarget | None:
        """Return one attribution target by its stable identifier."""

        return next(
            (target for target in self.targets if target.target_id == target_id),
            None,
        )


def _clone_batch_for_attribution(
    batch: FeatureBatch,
) -> tuple[FeatureBatch, dict[str, Tensor]]:
    """Clone TACME feature tensors as regular gradient-enabled tensors."""

    cloned_features: dict[str, Tensor] = {}

    for modality, features in batch["features"].items():
        cloned_features[modality] = features.detach().clone().requires_grad_(True)

    cloned_masks = {
        modality: mask.detach().clone()
        for modality, mask in batch.get(
            "feature_masks",
            {},
        ).items()
    }

    attribution_batch: FeatureBatch = {
        "features": cloned_features,
        "feature_masks": cloned_masks,
    }

    return (
        attribution_batch,
        cloned_features,
    )


def _require_model_output(
    outputs: Mapping[str, Tensor | None],
    name: str,
) -> Tensor:
    """Return one required TACME output tensor for attribution."""

    output = outputs.get(
        name,
    )

    if output is None:
        msg = f"TACME returned no required attribution output {name!r}."
        raise RuntimeError(msg)

    return output


def _target_scalars(
    outputs: Mapping[str, Tensor | None],
) -> tuple[tuple[str, int, Tensor], ...]:
    """Create the seven TACME attribution targets used by the application."""

    emotion_logits = _require_model_output(
        outputs,
        "emotion_logits",
    )
    personality_scores = _require_model_output(
        outputs,
        "personality_scores",
    )
    ambivalence_logits = _require_model_output(
        outputs,
        "ah_logits",
    )

    emotion_probabilities = torch.softmax(
        emotion_logits[0],
        dim=0,
    )
    emotion_index = int(
        torch.argmax(
            emotion_probabilities,
        ).item()
    )

    ambivalence_probabilities = torch.softmax(
        ambivalence_logits[0],
        dim=0,
    )
    ambivalence_index = int(
        torch.argmax(
            ambivalence_probabilities,
        ).item()
    )

    targets: list[tuple[str, int, Tensor]] = [
        (
            "emotion",
            emotion_index,
            emotion_probabilities[emotion_index],
        ),
    ]

    targets.extend(
        (
            "personality",
            index,
            personality_scores[0, index],
        )
        for index in range(
            personality_scores.shape[-1],
        )
    )

    targets.append(
        (
            "ambivalence",
            ambivalence_index,
            ambivalence_probabilities[ambivalence_index],
        )
    )

    return tuple(
        targets,
    )


def _compact_modality_attribution(
    features: Tensor,
    gradient: Tensor | None,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, float],
]:
    """Create two 32-bin rows plus exact Mean/STD attribution magnitudes."""

    attribution = torch.zeros_like(features) if gradient is None else gradient * features

    if attribution.ndim != 3 or attribution.shape[0] != 1:
        msg = f"Expected one batched TACME attribution tensor with shape [1, T, D], got {tuple(attribution.shape)}."
        raise RuntimeError(msg)

    sample_attribution = (
        attribution[0]
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
    )

    if sample_attribution.shape[0] != 2:
        msg = (
            "TACME feature attribution expects mean+STD aggregation with two tokens, "
            f"got shape {tuple(sample_attribution.shape)}."
        )
        raise RuntimeError(msg)

    compact = torch_functional.adaptive_avg_pool1d(
        sample_attribution,
        FEATURE_ATTRIBUTION_BIN_COUNT,
    )

    compact_rows = tuple(
        tuple(
            float(
                value,
            )
            for value in row.tolist()
        )
        for row in compact
    )

    row_magnitudes = tuple(
        float(
            value,
        )
        for value in sample_attribution.abs()
        .mean(
            dim=1,
        )
        .tolist()
    )

    if len(row_magnitudes) != 2:
        raise RuntimeError("Expected exactly two TACME attribution row magnitudes.")

    return (
        compact_rows,
        (
            row_magnitudes[0],
            row_magnitudes[1],
        ),
    )


def _normalize_modality_magnitudes(
    magnitudes: list[float],
) -> tuple[float, ...]:
    """Normalize non-negative modality attribution magnitudes to unit sum."""

    total = sum(
        magnitudes,
    )

    if total <= 0.0:
        return tuple(0.0 for _ in magnitudes)

    return tuple(magnitude / total for magnitude in magnitudes)


def _detach_outputs(
    outputs: Mapping[str, Tensor | None],
) -> dict[str, Tensor | None]:
    """Detach TACME outputs from the attribution graph."""

    return {name: (tensor.detach() if tensor is not None else None) for name, tensor in outputs.items()}


def run_tacme_feature_attribution(
    model: MultiModalFusionModel_v2,
    batch: FeatureBatch,
) -> tuple[dict[str, Tensor | None], FeatureAttributionBundle]:
    """Run TACME once with gradients and compute seven local Gradient x Input maps."""

    with (
        torch.inference_mode(
            False,
        ),
        torch.enable_grad(),
    ):
        attribution_batch, attribution_inputs = _clone_batch_for_attribution(
            batch,
        )

        outputs = model(
            attribution_batch,
        )

        targets = _target_scalars(
            outputs,
        )

        input_modalities = tuple(modality for modality in ATTRIBUTION_MODALITY_ORDER if modality in attribution_inputs)

        if not input_modalities:
            raise RuntimeError("No TACME modalities are available for feature attribution.")

        input_tensors = tuple(attribution_inputs[modality] for modality in input_modalities)

        attribution_targets: list[FeatureAttributionTarget] = []

        for target_position, (
            task,
            target_index,
            target_scalar,
        ) in enumerate(
            targets,
        ):
            gradients = torch.autograd.grad(
                target_scalar,
                input_tensors,
                retain_graph=(target_position < len(targets) - 1),
                create_graph=False,
                allow_unused=True,
            )

            gradient_by_modality = dict(
                zip(
                    input_modalities,
                    gradients,
                    strict=True,
                )
            )

            rows: list[tuple[float, ...]] = []
            row_magnitudes: list[float] = []
            modality_magnitudes: list[float] = []

            for modality in ATTRIBUTION_MODALITY_ORDER:
                features = attribution_inputs.get(
                    modality,
                )

                if features is None:
                    rows.extend(
                        (
                            tuple(
                                0.0
                                for _ in range(
                                    FEATURE_ATTRIBUTION_BIN_COUNT,
                                )
                            ),
                            tuple(
                                0.0
                                for _ in range(
                                    FEATURE_ATTRIBUTION_BIN_COUNT,
                                )
                            ),
                        )
                    )
                    row_magnitudes.extend(
                        (
                            0.0,
                            0.0,
                        )
                    )
                    modality_magnitudes.append(
                        0.0,
                    )
                    continue

                compact_rows, modality_row_magnitudes = _compact_modality_attribution(
                    features,
                    gradient_by_modality.get(
                        modality,
                    ),
                )

                rows.extend(
                    compact_rows,
                )
                row_magnitudes.extend(
                    modality_row_magnitudes,
                )
                modality_magnitudes.append(
                    sum(
                        modality_row_magnitudes,
                    )
                    / len(
                        modality_row_magnitudes,
                    )
                )

            attribution_targets.append(
                FeatureAttributionTarget(
                    task=task,
                    target_index=target_index,
                    score=float(target_scalar.detach().cpu().item()),
                    values=tuple(
                        rows,
                    ),
                    row_magnitudes=tuple(
                        row_magnitudes,
                    ),
                    modality_magnitudes=_normalize_modality_magnitudes(
                        modality_magnitudes,
                    ),
                )
            )

        detached_outputs = _detach_outputs(
            outputs,
        )

    return (
        detached_outputs,
        FeatureAttributionBundle(
            targets=tuple(
                attribution_targets,
            ),
            modality_order=ATTRIBUTION_MODALITY_ORDER,
            feature_bin_count=FEATURE_ATTRIBUTION_BIN_COUNT,
        ),
    )


__all__ = [
    "ATTRIBUTION_MODALITY_ORDER",
    "ATTRIBUTION_TASK_ORDER",
    "FEATURE_ATTRIBUTION_BIN_COUNT",
    "FeatureAttributionBundle",
    "FeatureAttributionTarget",
    "run_tacme_feature_attribution",
]
