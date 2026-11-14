"""
File: selection.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Target/task/view resolution for cached feature attribution.

License: MIT License
"""

from tacme.explainability.attribution import (
    ATTRIBUTION_TASK_ORDER,
    FeatureAttributionBundle,
    FeatureAttributionTarget,
)
from tacme.localization import get_task_keys_from_labels
from tacme.visualization.attribution import (
    ATTRIBUTION_VIEW_MEAN_STD,
    ATTRIBUTION_VIEWS,
)


def _selected_task_keys(
    bundle: FeatureAttributionBundle,
    selected_task_labels: list[str] | None,
) -> tuple[str, ...]:
    """Resolve task keys that are both selected in the UI and present in the bundle."""

    selected_keys = set(
        get_task_keys_from_labels(
            selected_task_labels,
        )
    )
    available_keys = {target.task for target in bundle.targets}

    return tuple(
        task_key for task_key in ATTRIBUTION_TASK_ORDER if task_key in selected_keys and task_key in available_keys
    )


def _resolve_task(
    task_keys: tuple[str, ...],
    requested_task: str | None,
) -> str | None:
    """Resolve the active attribution task without resetting a valid selection."""

    if requested_task in task_keys:
        return requested_task

    return task_keys[0] if task_keys else None


def _resolve_target(
    bundle: FeatureAttributionBundle,
    task: str,
    requested_target: str | None,
) -> FeatureAttributionTarget | None:
    """Resolve the active target while preserving a valid selection."""

    task_targets = bundle.targets_for_task(
        task,
    )

    if not task_targets:
        return None

    if requested_target is not None:
        target = bundle.get_target(
            requested_target,
        )

        if target is not None and target.task == task:
            return target

    return task_targets[0]


def _resolve_view(
    requested_view: str | None,
) -> str:
    """Resolve one supported attribution view."""

    return requested_view if requested_view in ATTRIBUTION_VIEWS else ATTRIBUTION_VIEW_MEAN_STD
