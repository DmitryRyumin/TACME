"""Localized completed-analysis report in the same visual style as live progress."""

from html import escape
from math import isfinite, log1p

from tacme.localization import get_localized_text
from tacme.performance import PROGRESS_STAGE_KEYS, AnalysisTiming


def _text(key: str, language_index: int) -> str:
    return escape(get_localized_text(key, language_index))


def _number(value: float, digits: int, language_index: int) -> str:
    number = f"{value:.{digits}f}"
    return number.replace(".", ",") if language_index == 1 else number


def _metric(label: str, value: str, detail: str = "") -> str:
    detail_html = f'<span class="analysis-details-metric-note">{detail}</span>' if detail else ""
    return (
        '<div class="analysis-details-metric">'
        f'<span class="analysis-details-metric-label">{label}</span>'
        f'<strong class="analysis-details-metric-value">{value}</strong>'
        f"{detail_html}</div>"
    )


def _css_position(percent: float, pixels: float) -> str:
    """Combine relative time-axis position with fixed-size visual spacing."""

    if abs(percent) < 1e-9:
        return f"{pixels:.4f}px"
    operator = "+" if pixels >= 0.0 else "-"
    return f"calc({percent:.6f}% {operator} {abs(pixels):.4f}px)"


def _stage_segment_styles(durations: dict[str, float], axis_total: float) -> dict[str, str]:
    """Return non-overlapping, sequential segments with visible short stages.

    Exact percentages remain numerical. Sub-2.5% stages receive a few
    screen pixels so zero/near-zero stages can be seen and distinguished.
    The remaining track space is distributed proportionally among longer
    stages (and any unallocated pipeline overhead).
    """

    short_threshold = 2.5
    small_pixels: dict[str, float] = {}
    if axis_total > 0.0:
        for stage in PROGRESS_STAGE_KEYS:
            duration = durations.get(stage)
            if duration is None:
                continue
            measured_share = 100.0 * duration / axis_total
            if measured_share < short_threshold:
                # Four pixels even when displayed as 0.0%, with modest
                # variation to separate 0.1%, 0.3%, and ~2% stages.
                small_pixels[stage] = 4.0 + 6.0 * (log1p(measured_share / 0.1) / log1p(short_threshold / 0.1))
    else:
        small_pixels = {stage: 4.0 for stage in PROGRESS_STAGE_KEYS if stage in durations}

    reserved_pixels = sum(small_pixels.values())
    # A possible difference between stage timings and total inference time
    # stays represented by the unfilled right end of the timeline.
    large_seconds = max(0.0, axis_total - sum(durations[stage] for stage in small_pixels))
    cumulative_large = 0.0
    cumulative_pixels = 0.0
    styles: dict[str, str] = {}

    for stage in PROGRESS_STAGE_KEYS:
        duration = durations.get(stage)
        if duration is None:
            continue

        fraction = cumulative_large / large_seconds if large_seconds > 0.0 else 0.0
        left = _css_position(100.0 * fraction, cumulative_pixels - reserved_pixels * fraction)
        if stage in small_pixels:
            width = f"{small_pixels[stage]:.4f}px"
            cumulative_pixels += small_pixels[stage]
        else:
            share = duration / large_seconds if large_seconds > 0.0 else 0.0
            width = _css_position(100.0 * share, -reserved_pixels * share)
            cumulative_large += duration
        styles[stage] = f"left:{left};width:{width}"

    return styles


def create_processing_details_html(timing: AnalysisTiming | None, language_index: int) -> str:
    """Show measured stage times and time shares without rerunning inference."""

    if timing is None:
        return ""

    total = timing.analysis_seconds
    durations = {
        item.stage: item.elapsed_seconds
        for item in timing.stage_timings
        if item.stage in PROGRESS_STAGE_KEYS and isfinite(item.elapsed_seconds) and item.elapsed_seconds >= 0.0
    }
    stage_count = len(PROGRESS_STAGE_KEYS)
    completed_count = sum(stage in durations for stage in PROGRESS_STAGE_KEYS)
    completion_percentage = round(100.0 * completed_count / stage_count) if stage_count else 0

    seconds_unit = _text("Texts_PROGRESS_SECONDS", language_index)
    unavailable = _text("Texts_DETAILS_UNAVAILABLE", language_index)
    video = (
        f"{_number(timing.video_duration_seconds, 2, language_index)} {seconds_unit}"
        if timing.video_duration_seconds is not None
        else unavailable
    )
    rtf = _number(timing.rtf, 3, language_index) if timing.rtf is not None else unavailable
    device = escape(timing.device) if timing.device else unavailable
    total_display = f"{_number(total, 2, language_index)} {seconds_unit}"
    metrics = "".join(
        (
            _metric(_text("Texts_DETAILS_TOTAL_TIME", language_index), total_display),
            _metric(_text("Texts_DETAILS_VIDEO_DURATION", language_index), video),
            _metric("RTF", rtf),
            _metric(_text("Texts_DETAILS_DEVICE", language_index), device),
        )
    )

    # Preserve exact measured percentages in text, but give very short
    # timeline segments distinguishable widths without overlap.
    measured_stage_total = sum(durations.values())
    axis_total = max(total, measured_stage_total)
    segment_styles = _stage_segment_styles(durations, axis_total)

    stage_rows: list[str] = []
    for index, stage in enumerate(PROGRESS_STAGE_KEYS, start=1):
        label = _text("Texts_PROGRESS_STAGE_" + stage.upper(), language_index)
        duration = durations.get(stage)
        completed = duration is not None
        if duration is None:
            time_text = unavailable
            percentage = 0.0
            percentage_text = "—"
        else:
            time_text = f"{_number(duration, 2, language_index)} {seconds_unit}"
            percentage = 100.0 * duration / axis_total if axis_total > 0.0 else 0.0
            percentage_text = f"{_number(percentage, 1, language_index)}%"
        segment_style = segment_styles.get(stage, "display:none")
        state_class = "analysis-stage-is-completed" if completed else "analysis-stage-is-pending"
        marker = "✓" if completed else str(index)
        stage_rows.append(
            f'<li class="analysis-stage {state_class} analysis-details-stage">'
            f'<span class="analysis-stage-index" aria-hidden="true">{marker}</span>'
            '<div class="analysis-details-stage-body">'
            f'<span class="analysis-stage-name">{label}</span>'
            '<div class="analysis-details-stage-track" aria-hidden="true">'
            f'<span class="analysis-details-stage-bar" '
            f'style="{segment_style}"></span>'
            "</div></div>"
            '<div class="analysis-details-stage-values">'
            f'<strong class="analysis-details-stage-duration">{time_text}</strong>'
            f'<span class="analysis-details-stage-percent">{percentage_text}</span>'
            "</div></li>"
        )

    # This is a measured bottleneck, not an estimated or causal explanation.
    longest_html = ""
    if durations:
        longest_stage = max(durations, key=durations.__getitem__)
        longest_seconds = durations[longest_stage]
        longest_share = 100.0 * longest_seconds / axis_total if axis_total > 0.0 else 0.0
        longest_html = (
            '<div class="analysis-details-longest" role="note">'
            '<span class="analysis-details-longest-caption">'
            f"{_text('Texts_DETAILS_LONGEST_STAGE', language_index)}</span>"
            '<strong class="analysis-details-longest-name">'
            f"{_text('Texts_PROGRESS_STAGE_' + longest_stage.upper(), language_index)}</strong>"
            '<span class="analysis-details-longest-values">'
            f"{_number(longest_seconds, 2, language_index)} {seconds_unit}"
            f" · {_number(longest_share, 1, language_index)}%</span>"
            "</div>"
        )

    stages_title = _text("Texts_DETAILS_STAGES", language_index)
    percentage_label = _text("Texts_DETAILS_PERCENT", language_index)
    completed_status = _text("Texts_STATUS_ANALYSIS_COMPLETED", language_index)
    note = _text("Texts_DETAILS_NOTE", language_index)

    other_duration = max(0.0, total - sum(durations.values()))
    overhead_html = ""
    if total > 0.0 and other_duration > 0.05:
        share = _number(100.0 * other_duration / total, 1, language_index)
        overhead_html = (
            '<p class="analysis-details-overhead">'
            f"{_text('Texts_DETAILS_OTHER', language_index)}: "
            f"<strong>{_number(other_duration, 2, language_index)} {seconds_unit} · {share}%</strong>"
            "</p>"
        )

    return (
        '<section class="analysis-details-report">'
        '<div class="analysis-details-progress-heading">'
        f'<span class="analysis-details-success-label">{completed_status}</span>'
        f'<strong class="analysis-progress-counter">{completed_count}/{stage_count} · {completion_percentage}%</strong>'
        "</div>"
        '<div class="analysis-details-progress-track" role="progressbar" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{completion_percentage}">'
        f'<span class="analysis-details-progress-fill" style="width:{completion_percentage}%"></span>'
        "</div>"
        f'<div class="analysis-details-metrics">{metrics}</div>'
        f"{longest_html}"
        '<div class="analysis-details-stage-section-header">'
        f"<h4>{stages_title}</h4>"
        f"<span>{percentage_label}</span>"
        "</div>"
        f'<ol class="analysis-progress-stages analysis-details-stage-list">{"".join(stage_rows)}</ol>'
        f"{overhead_html}"
        f'<p class="analysis-details-note">{note}</p>'
        "</section>"
    )


__all__ = ["create_processing_details_html"]
