"""
File: handlers.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Unchanged Gradio feature-attribution event handlers.

License: MIT License
"""

from typing import Any

import gradio as gr

from tacme.events.attribution.labels import (
    _create_attribution_note,
    _create_attribution_title,
    _target_choices,
    _task_choices,
    _view_choices,
)
from tacme.events.attribution.selection import (
    _resolve_target,
    _resolve_task,
    _resolve_view,
    _selected_task_keys,
)
from tacme.explainability.attribution import FeatureAttributionBundle
from tacme.localization import get_language_index, get_localized_text, get_task_label
from tacme.visualization.attribution import (
    create_feature_attribution_figure,
    create_feature_attribution_summary_html,
)


def handle_clear_feature_attribution() -> tuple[Any, ...]:
    """Clear and hide the feature-attribution section."""

    return (
        None,
        gr.update(
            visible=False,
        ),
        gr.update(),
        gr.update(choices=[], value=None, interactive=False),
        gr.update(choices=[], value=None, interactive=False),
        gr.update(choices=[], value=None),
        gr.update(
            value=None,
        ),
        gr.update(
            value="",
        ),
        gr.update(),
    )


def handle_render_feature_attribution(
    bundle: FeatureAttributionBundle | None,
    selected_task_labels: list[str] | None,
    language: str,
    current_task: str | None = None,
    current_target: str | None = None,
    current_view: str | None = None,
) -> tuple[Any, ...]:
    """Render the interactive attribution section from cached local explanations."""

    language_index = get_language_index(
        language,
    )

    if bundle is None:
        return (
            None,
            gr.update(
                visible=False,
            ),
            gr.update(
                value=_create_attribution_title(
                    language_index,
                ),
            ),
            gr.update(
                choices=[],
                value=None,
                interactive=False,
                label=get_localized_text(
                    "Labels_ATTRIBUTION_TASK",
                    language_index,
                ),
            ),
            gr.update(
                choices=[],
                value=None,
                interactive=False,
                label=get_localized_text(
                    "Labels_ATTRIBUTION_TARGET",
                    language_index,
                ),
            ),
            gr.update(
                choices=_view_choices(
                    language_index,
                ),
                value=_resolve_view(
                    current_view,
                ),
                label=get_localized_text(
                    "Labels_ATTRIBUTION_VIEW",
                    language_index,
                ),
            ),
            gr.update(
                value=None,
            ),
            gr.update(
                value="",
            ),
            gr.update(
                value=_create_attribution_note(
                    language_index,
                ),
            ),
        )

    task_keys = _selected_task_keys(
        bundle,
        selected_task_labels,
    )

    task = _resolve_task(
        task_keys,
        current_task,
    )

    if task is None:
        return (
            bundle,
            gr.update(
                visible=False,
            ),
            gr.update(
                value=_create_attribution_title(
                    language_index,
                ),
            ),
            gr.update(
                choices=[],
                value=None,
                interactive=False,
                label=get_localized_text(
                    "Labels_ATTRIBUTION_TASK",
                    language_index,
                ),
            ),
            gr.update(
                choices=[],
                value=None,
                interactive=False,
                label=get_localized_text(
                    "Labels_ATTRIBUTION_TARGET",
                    language_index,
                ),
            ),
            gr.update(
                choices=_view_choices(
                    language_index,
                ),
                value=_resolve_view(
                    current_view,
                ),
                label=get_localized_text(
                    "Labels_ATTRIBUTION_VIEW",
                    language_index,
                ),
            ),
            gr.update(
                value=None,
            ),
            gr.update(
                value="",
            ),
            gr.update(
                value=_create_attribution_note(
                    language_index,
                ),
            ),
        )

    target = _resolve_target(
        bundle,
        task,
        current_target,
    )

    if target is None:
        return handle_clear_feature_attribution()

    view = _resolve_view(
        current_view,
    )
    task_choices = _task_choices(task_keys, language_index)
    target_choices = _target_choices(bundle, task, language_index)

    return (
        bundle,
        gr.update(
            visible=True,
        ),
        gr.update(
            value=_create_attribution_title(
                language_index,
            ),
        ),
        gr.update(
            choices=task_choices,
            value=task,
            interactive=len(task_choices) > 1,
            label=get_localized_text(
                "Labels_ATTRIBUTION_TASK",
                language_index,
            ),
        ),
        gr.update(
            choices=target_choices,
            value=target.target_id,
            interactive=len(target_choices) > 1,
            label=get_task_label(task, language_index),
        ),
        gr.update(
            choices=_view_choices(
                language_index,
            ),
            value=view,
            label=get_localized_text(
                "Labels_ATTRIBUTION_VIEW",
                language_index,
            ),
        ),
        gr.update(
            value=create_feature_attribution_figure(
                bundle,
                target,
                view,
                language_index,
            ),
        ),
        gr.update(
            value=create_feature_attribution_summary_html(
                target,
                view,
                language_index,
            ),
        ),
        gr.update(
            value=_create_attribution_note(
                language_index,
            ),
        ),
    )


def handle_feature_attribution_task_change(
    bundle: FeatureAttributionBundle | None,
    task: str | None,
    view: str | None,
    language: str,
) -> tuple[Any, ...]:
    """Switch the attribution target list and heatmap to another task."""

    language_index = get_language_index(
        language,
    )

    if bundle is None or task is None:
        return (
            gr.update(
                choices=[],
                value=None,
                interactive=False,
                label=get_localized_text(
                    "Labels_ATTRIBUTION_TARGET",
                    language_index,
                ),
            ),
            gr.update(
                value=None,
            ),
            gr.update(
                value="",
            ),
        )

    target = _resolve_target(
        bundle,
        task,
        None,
    )

    if target is None:
        return (
            gr.update(
                choices=[],
                value=None,
                interactive=False,
                label=get_localized_text(
                    "Labels_ATTRIBUTION_TARGET",
                    language_index,
                ),
            ),
            gr.update(
                value=None,
            ),
            gr.update(
                value="",
            ),
        )

    resolved_view = _resolve_view(
        view,
    )
    target_choices = _target_choices(bundle, task, language_index)

    return (
        gr.update(
            choices=target_choices,
            value=target.target_id,
            interactive=len(target_choices) > 1,
            label=get_task_label(task, language_index),
        ),
        gr.update(
            value=create_feature_attribution_figure(
                bundle,
                target,
                resolved_view,
                language_index,
            ),
        ),
        gr.update(
            value=create_feature_attribution_summary_html(
                target,
                resolved_view,
                language_index,
            ),
        ),
    )


def handle_feature_attribution_target_change(
    bundle: FeatureAttributionBundle | None,
    target_id: str | None,
    view: str | None,
    language: str,
) -> tuple[Any, ...]:
    """Switch the heatmap to one cached output target."""

    if bundle is None or target_id is None:
        return (
            gr.update(
                value=None,
            ),
            gr.update(
                value="",
            ),
        )

    target = bundle.get_target(
        target_id,
    )

    if target is None:
        return (
            gr.update(
                value=None,
            ),
            gr.update(
                value="",
            ),
        )

    language_index = get_language_index(
        language,
    )
    resolved_view = _resolve_view(
        view,
    )

    return (
        gr.update(
            value=create_feature_attribution_figure(
                bundle,
                target,
                resolved_view,
                language_index,
            ),
        ),
        gr.update(
            value=create_feature_attribution_summary_html(
                target,
                resolved_view,
                language_index,
            ),
        ),
    )


def handle_feature_attribution_view_change(
    bundle: FeatureAttributionBundle | None,
    target_id: str | None,
    view: str | None,
    language: str,
) -> tuple[Any, ...]:
    """Switch Mean/STD rows without recomputing model gradients."""

    return handle_feature_attribution_target_change(
        bundle,
        target_id,
        view,
        language,
    )
