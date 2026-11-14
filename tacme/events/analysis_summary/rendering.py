"""
File: rendering.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: HTML rendering for cached TACME analysis results.

License: MIT License
"""

from html import escape

from tacme.events.analysis_summary.text import (
    _create_ambivalence_text,
    _create_emotion_text,
    _create_personality_text,
)
from tacme.inference import TacmeInferenceResult
from tacme.localization import get_localized_text, get_task_keys_from_labels


def _render_summary_section(
    *,
    section_class: str,
    title: str,
    text: str,
) -> str:
    """Render one deterministic analysis-summary section."""

    return f"""
<section class="application-analysis-summary-section {escape(section_class)}">
    <div class="application-analysis-summary-section-title">{escape(title)}</div>
    <p>{escape(text)}</p>
</section>
""".strip()


def create_analysis_summary_html(
    result: TacmeInferenceResult,
    selected_tasks: list[str],
    language_index: int,
) -> str:
    """Render a localized summary from cached TACME outputs only."""

    task_keys = set(
        get_task_keys_from_labels(
            selected_tasks,
        )
    )

    section_html: list[str] = []
    copy_sections: list[str] = []
    final_sentences: list[str] = []

    if "emotion" in task_keys:
        detailed, compact = _create_emotion_text(
            result,
            language_index,
        )
        section_title = get_localized_text(
            "Labels_SUMMARY_EMOTION",
            language_index,
        )
        section_html.append(
            _render_summary_section(
                section_class="is-emotion",
                title=section_title,
                text=detailed,
            )
        )
        copy_sections.extend(
            (
                section_title,
                detailed,
            )
        )
        final_sentences.append(
            compact,
        )

    if "personality" in task_keys:
        detailed, compact = _create_personality_text(
            result,
            language_index,
        )
        section_title = get_localized_text(
            "Labels_SUMMARY_PERSONALITY",
            language_index,
        )
        section_html.append(
            _render_summary_section(
                section_class="is-personality",
                title=section_title,
                text=detailed,
            )
        )
        copy_sections.extend(
            (
                section_title,
                detailed,
            )
        )
        final_sentences.append(
            compact,
        )

    if "ambivalence" in task_keys:
        detailed, compact = _create_ambivalence_text(
            result,
            language_index,
        )
        section_title = get_localized_text(
            "Labels_SUMMARY_AMBIVALENCE",
            language_index,
        )
        section_html.append(
            _render_summary_section(
                section_class="is-ambivalence",
                title=section_title,
                text=detailed,
            )
        )
        copy_sections.extend(
            (
                section_title,
                detailed,
            )
        )
        final_sentences.append(
            compact,
        )

    summary_title = get_localized_text(
        "Labels_ANALYSIS_SUMMARY",
        language_index,
    )
    badge = get_localized_text(
        "Labels_ANALYSIS_SUMMARY_BADGE",
        language_index,
    )
    final_title = get_localized_text(
        "Labels_SUMMARY_FINAL",
        language_index,
    )
    copy_label = get_localized_text(
        "Labels_SUMMARY_COPY",
        language_index,
    )
    copied_label = get_localized_text(
        "Labels_SUMMARY_COPIED",
        language_index,
    )
    final_text = " ".join(
        final_sentences,
    )

    copy_text = "\n\n".join(
        (
            summary_title,
            *copy_sections,
            final_title,
            final_text,
        )
    )

    return f"""
<section class="application-analysis-summary">
    <header class="application-analysis-summary-header">
        <div class="application-analysis-summary-heading">
            <span class="application-analysis-summary-icon" aria-hidden="true"></span>
            <h3>{escape(summary_title)}</h3>
        </div>
        <div class="application-analysis-summary-actions">
            <span class="application-analysis-summary-badge">{escape(badge)}</span>
            <button
                type="button"
                class="application-analysis-summary-copy"
                data-copy-label="{escape(copy_label)}"
                data-copied-label="{escape(copied_label)}"
            >{escape(copy_label)}</button>
        </div>
    </header>
    <div class="application-analysis-summary-sections">
        {"".join(section_html)}
    </div>
    <section class="application-analysis-summary-final">
        <div class="application-analysis-summary-final-label">{escape(final_title)}</div>
        <p>{escape(final_text)}</p>
    </section>
    <pre class="application-analysis-summary-copy-source">{escape(copy_text)}</pre>
</section>
""".strip()
