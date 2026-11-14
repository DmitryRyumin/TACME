"""
File: about.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: About-page content for the TACME Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from html import escape
from typing import Final

import gradio as gr

from tacme.localization import get_localized_text

ABOUT_FEATURE_KEYS: Final[tuple[tuple[str, str], ...]] = (
    (
        "About_FEATURE_CROSS_MODAL_TITLE",
        "About_FEATURE_CROSS_MODAL_TEXT",
    ),
    (
        "About_FEATURE_TASK_AWARE_TITLE",
        "About_FEATURE_TASK_AWARE_TEXT",
    ),
    (
        "About_FEATURE_DISJOINT_TITLE",
        "About_FEATURE_DISJOINT_TEXT",
    ),
)


@dataclass(
    frozen=True,
    slots=True,
)
class AboutTabComponents:
    """Components created inside the TACME About tab."""

    title: gr.Markdown
    description: gr.Markdown
    content: gr.HTML


def _create_feature_cards_html(
    language_index: int,
) -> str:
    """Create TACME capability cards."""

    cards: list[str] = []

    for index, (title_key, text_key) in enumerate(
        ABOUT_FEATURE_KEYS,
        start=1,
    ):
        title = escape(
            get_localized_text(
                title_key,
                language_index,
            )
        )

        text = escape(
            get_localized_text(
                text_key,
                language_index,
            )
        )

        cards.append(f"""
            <article
                class="about-feature-card"
                style="--about-card-index: {index};"
            >
                <span
                    class="about-feature-index"
                    aria-hidden="true"
                >
                    {index:02d}
                </span>

                <h3 class="about-feature-title">
                    {title}
                </h3>

                <p class="about-feature-text">
                    {text}
                </p>
            </article>
            """)

    return "".join(cards)


def create_about_content_html(
    language_index: int,
) -> str:
    """Create the localized TACME About content."""

    badge = escape(
        get_localized_text(
            "About_BADGE",
            language_index,
        )
    )

    full_name = escape(
        get_localized_text(
            "About_FULL_NAME",
            language_index,
        )
    )

    description = escape(
        get_localized_text(
            "About_HERO_DESCRIPTION",
            language_index,
        )
    )

    flow_input = escape(
        get_localized_text(
            "About_FLOW_INPUT",
            language_index,
        )
    )

    flow_model = escape(
        get_localized_text(
            "About_FLOW_MODEL",
            language_index,
        )
    )

    flow_output = escape(
        get_localized_text(
            "About_FLOW_OUTPUT",
            language_index,
        )
    )

    return f"""
    <section class="about-shell">
        <div class="about-hero">
            <div class="about-hero-copy">
                <div class="about-badge-row">
                    <span class="about-badge">
                        {badge}
                    </span>
                </div>

                <h2 class="about-full-name">
                    {full_name}
                </h2>

                <p class="about-hero-description">
                    {description}
                </p>
            </div>

            <div class="about-flow">
                <span class="about-flow-node">
                    {flow_input}
                </span>

                <span class="about-flow-arrow">
                    →
                </span>

                <span class="about-flow-node is-model">
                    {flow_model}
                </span>

                <span class="about-flow-arrow">
                    →
                </span>

                <span class="about-flow-node">
                    {flow_output}
                </span>
            </div>
        </div>

        <div class="about-feature-grid">
            {_create_feature_cards_html(language_index)}
        </div>
    </section>
    """


def create_about_tab(
    language_index: int = 0,
) -> AboutTabComponents:
    """Create the TACME About tab."""

    title = gr.Markdown(
        f"### {get_localized_text('Texts_ABOUT_TITLE', language_index)}",
        elem_classes="about-title",
    )

    description = gr.Markdown(
        get_localized_text(
            "Texts_ABOUT_DESCRIPTION",
            language_index,
        ),
        elem_classes="about-description",
    )

    content = gr.HTML(
        value=create_about_content_html(
            language_index,
        ),
        elem_id="about-content",
        elem_classes="about-content",
    )

    return AboutTabComponents(
        title=title,
        description=description,
        content=content,
    )
