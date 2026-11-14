"""
File: results.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Localized TACME inference result rendering and Gradio result updates.

License: MIT License
"""

from html import escape
from typing import Any

import gradio as gr

from tacme.inference import AMBIVALENCE_LABELS, EMOTION_LABELS, TacmeInferenceResult
from tacme.localization import (
    get_language_index,
    get_localized_text,
    get_task_keys_from_labels,
    get_task_label,
)


def _create_predictions_title(
    language_index: int,
) -> str:
    """Create the localized predictions section title."""

    return "### " + get_localized_text(
        "Labels_PREDICTIONS",
        language_index,
    )


def _personality_trait_name(
    trait_index: int,
    language_index: int,
) -> str:
    """Return the localized long name of one apparent personality trait."""

    config_fields = (
        "Labels_PERSONALITY_OPE",
        "Labels_PERSONALITY_CON",
        "Labels_PERSONALITY_EXT",
        "Labels_PERSONALITY_AGR",
        "Labels_PERSONALITY_NNEU",
    )

    return get_localized_text(
        config_fields[trait_index],
        language_index,
    )


def _render_emotion_result(
    result: TacmeInferenceResult,
    language_index: int,
) -> str:
    """Render the TACME emotion prediction as a probability dashboard."""

    prediction = result.emotion
    title = escape(
        get_task_label(
            "emotion",
            language_index,
        )
    )
    emotion_labels = tuple(
        get_localized_text(
            f"Labels_SUMMARY_EMOTION_{class_label.upper()}",
            language_index,
        )
        for class_label in EMOTION_LABELS
    )
    label = escape(
        emotion_labels[prediction.class_index],
    )
    confidence = prediction.confidence * 100.0
    target_id = f"emotion:{prediction.class_index}"

    probability_rows = "".join(
        (
            '<div class="application-probability-row'
            + (" is-selected" if class_index == prediction.class_index else "")
            + '">'
            '<div class="application-probability-header">'
            f"<span>{escape(class_label)}</span>"
            f"<strong>{probability * 100.0:.2f}%</strong>"
            "</div>"
            '<div class="application-probability-track">'
            '<span class="application-probability-fill" '
            f'style="--tacme-result-share: {probability * 100.0:.4f}%;"></span>'
            "</div>"
            "</div>"
        )
        for class_index, (class_label, probability) in enumerate(
            zip(
                emotion_labels,
                prediction.probabilities,
                strict=True,
            )
        )
    )

    return f"""
<section class="application-result-card-content application-result-emotion-content">
    <div class="application-result-header">
        <span class="application-result-title">{title}</span>
    </div>
    <button
        type="button"
        class="application-result-primary application-attribution-link"
        data-attribution-task="emotion"
        data-attribution-target="{target_id}"
        title="{label}"
    >
        <strong>{label}</strong>
        <span>{confidence:.2f}%</span>
    </button>
    <div class="application-result-details application-probability-list">
        {probability_rows}
    </div>
</section>
""".strip()


def _render_personality_result(
    result: TacmeInferenceResult,
    language_index: int,
) -> str:
    """Render five raw apparent-personality scores on one shared numeric axis."""

    prediction = result.personality
    title = escape(
        get_task_label(
            "personality",
            language_index,
        )
    )
    score_min = min(
        0.0,
        min(
            prediction.scores,
        ),
    )
    score_max = max(
        1.0,
        max(
            prediction.scores,
        ),
    )

    span = score_max - score_min

    if score_min < 0.0 or score_max > 1.0:
        padding = max(
            span * 0.06,
            0.02,
        )
        score_min -= padding
        score_max += padding
        span = score_max - score_min

    score_rows = "".join(
        (
            '<button type="button" '
            'class="application-personality-row application-attribution-link" '
            'data-attribution-task="personality" '
            f'data-attribution-target="personality:{trait_index}">'
            '<div class="application-personality-row-header">'
            '<div class="application-personality-labels">'
            f"<strong>{escape(_personality_trait_name(trait_index, language_index))}</strong>"
            "</div>"
            f'<span class="application-personality-score">{score:.4f}</span>'
            "</div>"
            '<div class="application-personality-axis" '
            f'style="--tacme-personality-position: {(score - score_min) / span * 100.0:.4f}%; '
            f"--tacme-personality-zero: {(0.0 - score_min) / span * 100.0:.4f}%; "
            f'--tacme-personality-one: {(1.0 - score_min) / span * 100.0:.4f}%;">'
            '<span class="application-personality-zero"></span>'
            '<span class="application-personality-one"></span>'
            '<span class="application-personality-dot"></span>'
            "</div>"
            "</button>"
        )
        for trait_index, (_, score) in enumerate(
            zip(
                prediction.labels,
                prediction.scores,
                strict=True,
            )
        )
    )

    return f"""
<section class="application-result-card-content application-result-personality-content">
    <div class="application-result-header">
        <span class="application-result-title">{title}</span>
    </div>
    <div class="application-result-details application-personality-list">
        {score_rows}
    </div>
    <div class="application-personality-axis-footer">
        <span>{score_min:.2f}</span>
        <span>{score_max:.2f}</span>
    </div>
</section>
""".strip()


def _render_ambivalence_result(
    result: TacmeInferenceResult,
    language_index: int,
) -> str:
    """Render the binary ambivalence prediction as a split probability bar."""

    prediction = result.ambivalence
    title = escape(
        get_task_label(
            "ambivalence",
            language_index,
        )
    )
    ambivalence_labels = tuple(
        get_localized_text(
            f"Labels_AMBIVALENCE_{class_label.upper()}",
            language_index,
        )
        for class_label in AMBIVALENCE_LABELS
    )
    label = escape(
        ambivalence_labels[prediction.class_index],
    )
    confidence = prediction.confidence * 100.0
    absence = prediction.absence_probability * 100.0
    presence = prediction.presence_probability * 100.0
    target_id = f"ambivalence:{prediction.class_index}"

    return f"""
<section class="application-result-card-content application-result-ambivalence-content">
    <div class="application-result-header">
        <span class="application-result-title">{title}</span>
    </div>
    <button
        type="button"
        class="application-result-primary application-attribution-link"
        data-attribution-task="ambivalence"
        data-attribution-target="{target_id}"
        title="{label}"
    >
        <strong>{label}</strong>
        <span>{confidence:.2f}%</span>
    </button>
    <div class="application-ambivalence-body">
        <div class="application-ambivalence-split" aria-hidden="true">
            <span
                class="application-ambivalence-absence{" is-selected" if prediction.class_index == 0 else ""}"
                style="--tacme-result-share: {absence:.4f}%;"
            ></span>
            <span
                class="application-ambivalence-presence{" is-selected" if prediction.class_index == 1 else ""}"
                style="--tacme-result-share: {presence:.4f}%;"
            ></span>
        </div>
        <div class="application-ambivalence-labels">
            <div>
                <span>{escape(ambivalence_labels[0])}</span>
                <strong>{absence:.2f}%</strong>
            </div>
            <div>
                <span>{escape(ambivalence_labels[1])}</span>
                <strong>{presence:.2f}%</strong>
            </div>
        </div>
    </div>
</section>
""".strip()


def create_result_updates(
    result: TacmeInferenceResult | None,
    selected_tasks: list[str] | None,
    language_index: int,
) -> tuple[
    Any,
    Any,
    Any,
    Any,
]:
    """Create visibility and HTML updates for selected TACME tasks."""

    if result is None or not selected_tasks:
        return (
            gr.update(
                visible=False,
            ),
            gr.update(
                value="",
                visible=False,
            ),
            gr.update(
                value="",
                visible=False,
            ),
            gr.update(
                value="",
                visible=False,
            ),
        )

    selected_task_keys = set(
        get_task_keys_from_labels(
            selected_tasks,
        )
    )

    show_emotion = "emotion" in selected_task_keys

    show_personality = "personality" in selected_task_keys

    show_ambivalence = "ambivalence" in selected_task_keys

    has_visible_result = show_emotion or show_personality or show_ambivalence

    return (
        gr.update(
            visible=has_visible_result,
        ),
        gr.update(
            value=(
                _render_emotion_result(
                    result,
                    language_index,
                )
                if show_emotion
                else ""
            ),
            visible=show_emotion,
        ),
        gr.update(
            value=(
                _render_personality_result(
                    result,
                    language_index,
                )
                if show_personality
                else ""
            ),
            visible=show_personality,
        ),
        gr.update(
            value=(
                _render_ambivalence_result(
                    result,
                    language_index,
                )
                if show_ambivalence
                else ""
            ),
            visible=show_ambivalence,
        ),
    )


def handle_localize_predictions_title(
    language: str,
) -> Any:
    """Relocalize the predictions section title."""

    language_index = get_language_index(
        language,
    )

    return gr.update(
        value=_create_predictions_title(
            language_index,
        ),
    )
