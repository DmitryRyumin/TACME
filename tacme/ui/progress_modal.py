"""Localized live analysis-stage overlay for TACME."""

from collections.abc import Mapping
from html import escape

from tacme.localization import get_localized_text
from tacme.performance import PROGRESS_STAGE_KEYS


def _label(key: str, language_index: int) -> str:
    return escape(get_localized_text(key, language_index))


def _seconds(value: float, language_index: int) -> str:
    number = f"{max(0.0, value):.1f}"
    return number.replace(".", ",") if language_index == 1 else number


def create_progress_modal(
    language_index: int,
    stage_states: Mapping[str, str],
    durations: Mapping[str, float],
    elapsed_seconds: float,
    *,
    active_elapsed_seconds: float = 0.0,
    finished: bool = False,
) -> str:
    """Render measured progress; active timers interpolate between server events."""

    count = len(PROGRESS_STAGE_KEYS)
    completed = sum(stage_states.get(stage) == "completed" for stage in PROGRESS_STAGE_KEYS)
    percentage = round(100 * completed / count) if count else 0
    current = next((stage for stage in PROGRESS_STAGE_KEYS if stage_states.get(stage) == "started"), None)
    current_name = (
        _label("Texts_PROGRESS_STAGE_" + current.upper(), language_index)
        if current is not None
        else _label("Texts_STATUS_ANALYSIS_COMPLETED" if finished else "Texts_PROGRESS_WAITING", language_index)
    )
    seconds_unit = _label("Texts_PROGRESS_SECONDS", language_index)
    rows: list[str] = []
    for index, stage in enumerate(PROGRESS_STAGE_KEYS, start=1):
        state = stage_states.get(stage, "pending")
        if state not in ("pending", "started", "completed", "failed"):
            state = "pending"
        label = _label("Texts_PROGRESS_STAGE_" + stage.upper(), language_index)
        if state in ("completed", "failed") and stage in durations:
            detail = f"{_seconds(durations[stage], language_index)} {seconds_unit}"
        elif state == "started":
            detail = (
                '<span class="analysis-activity-dots" aria-hidden="true">'
                "<i></i><i></i><i></i></span>"
                f'<time class="analysis-live-stage" data-base="{max(0.0, active_elapsed_seconds):.4f}" '
                f'data-language="{language_index}">{_seconds(active_elapsed_seconds, language_index)} {seconds_unit}</time>'
            )
        else:
            detail = _label("Texts_PROGRESS_PENDING", language_index)
        marker = "✓" if state == "completed" else "!" if state == "failed" else str(index)
        rows.append(
            f'<li class="analysis-stage analysis-stage-is-{state}" data-stage="{escape(stage)}">'
            f'<span class="analysis-stage-index" aria-hidden="true">{marker}</span>'
            f'<span class="analysis-stage-name">{label}</span>'
            f'<span class="analysis-stage-time">{detail}</span>'
            "</li>"
        )

    title = _label("Texts_PROGRESS_TITLE", language_index)
    elapsed_label = _label("Texts_PROGRESS_ELAPSED", language_index)
    completed_label = _label("Texts_PROGRESS_COMPLETED", language_index)
    note = _label("Texts_PROGRESS_NOTE", language_index)
    finished_class = " analysis-progress-modal-is-complete" if finished else ""
    return (
        f'<div class="analysis-busy-overlay analysis-progress-modal{finished_class}" '
        f'role="dialog" aria-modal="true" aria-label="{title}" '
        f'data-progress-percent="{percentage}" data-elapsed-base="{max(0.0, elapsed_seconds):.4f}" '
        f'data-language="{language_index}" data-finished="{str(finished).lower()}" '
        f'style="--tacme-stage-progress:{percentage}%;">'
        '<div class="analysis-busy-card">'
        '<div class="analysis-busy-header">'
        f'<p class="analysis-busy-title">{title}</p>'
        f'<span class="analysis-progress-counter">{completed}/{count} · {percentage}%</span>'
        "</div>"
        f'<p class="analysis-progress-current" aria-live="polite">{current_name}</p>'
        '<div class="analysis-busy-progress"><div class="analysis-busy-progress-track" '
        f'role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percentage}">'
        '<div class="analysis-busy-progress-bar"></div></div></div>'
        '<div class="analysis-progress-facts">'
        f"<span>{completed_label}: <strong>{completed}/{count}</strong></span>"
        f'<span>{elapsed_label}: <strong><time class="analysis-live-total">'
        f"{_seconds(elapsed_seconds, language_index)} {seconds_unit}</time></strong></span>"
        "</div>"
        f'<ol class="analysis-progress-stages">{"".join(rows)}</ol>'
        f'<p class="analysis-progress-note">{note}</p>'
        "</div></div>"
    )


__all__ = ["create_progress_modal"]
