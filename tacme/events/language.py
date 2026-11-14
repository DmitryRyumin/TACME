"""
File: language.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Language switching event handlers for the TACME Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from typing import Any

import gradio as gr

from tacme.config import get_config_str_mapping
from tacme.events.application import (
    create_analysis_completed_status,
    create_result_updates,
)
from tacme.events.media_info import create_media_content_updates
from tacme.events.video_state import (
    create_video_component_label,
)
from tacme.inference import TacmeInferenceResult
from tacme.localization import (
    TASK_KEYS,
    get_language_index,
    get_localized_text,
    get_task_choices,
    get_task_keys_from_labels,
)
from tacme.media import MediaProbeResult
from tacme.performance import AnalysisTiming
from tacme.ui.about import create_about_content_html
from tacme.ui.application import (
    create_application_title_markdown,
)
from tacme.ui.authors import (
    create_authors_content_html,
)
from tacme.ui.language_selector import (
    create_language_flag_html,
)
from tacme.ui.requirements import (
    RequirementsSessionState,
    create_requirements_content_html_from_state,
)


@dataclass(
    frozen=True,
    slots=True,
)
class LanguageChangeUpdates:
    """Named UI updates produced by a TACME language change."""

    flag: Any

    tabs: dict[
        str,
        Any,
    ]

    application_title: Any
    video_input: Any
    examples_title: Any
    examples_placeholder: Any
    task_selector: Any
    run_button: Any
    clear_button: Any
    status: Any

    media_info_button_column: Any
    media_info_button: Any
    media_info_modal_title: Any
    media_overview_tab: Any
    media_video_tab: Any
    media_audio_tab: Any
    media_compatibility_tab: Any
    media_overview_content: Any
    media_video_content: Any
    media_audio_content: Any
    media_compatibility_content: Any

    results_container: Any
    emotion_result: Any
    personality_result: Any
    ambivalence_result: Any

    about_title: Any
    about_description: Any
    about_content: Any

    authors_content: Any

    requirements_title: Any
    requirements_description: Any
    requirements_content: Any


def create_tab_language_updates(
    language_index: int,
) -> dict[
    str,
    Any,
]:
    """Create localized updates for dynamically configured tabs."""

    tab_creators = get_config_str_mapping(
        "TabCreators",
    )

    return {
        tab_name: gr.update(
            label=get_localized_text(
                f"Tabs_{tab_name}",
                language_index,
            ),
        )
        for tab_name in tab_creators
    }


def _get_selected_task_keys(selected_tasks: list[str] | None) -> list[str]:
    """Preserve task identities independently of the selected UI language."""

    if selected_tasks is None:
        return list(TASK_KEYS)

    return get_task_keys_from_labels(selected_tasks)


def _get_application_status_text(
    video_path: str | None,
    selected_task_labels: list[str],
    cached_result: TacmeInferenceResult | None,
    analysis_timing: AnalysisTiming | None,
    is_analysis_running: bool,
    language_index: int,
) -> str:
    """Return localized status text for the current application state."""

    if is_analysis_running:
        return get_localized_text(
            "Texts_STATUS_ANALYSIS_RUNNING",
            language_index,
        )

    if not video_path:
        return get_localized_text(
            "Texts_STATUS_READY",
            language_index,
        )

    if not selected_task_labels:
        return get_localized_text(
            "Texts_STATUS_NO_TASKS",
            language_index,
        )

    if cached_result is not None:
        return create_analysis_completed_status(
            analysis_timing,
            language_index,
        )

    return get_localized_text(
        "Texts_STATUS_VIDEO_READY",
        language_index,
    )


def handle_language_change(
    language: str,
    requirements_state: RequirementsSessionState | None,
    selected_task_labels: list[str] | None,
    video_path: str | None,
    video_filename: str | None,
    media_info_state: MediaProbeResult | None,
    cached_result: TacmeInferenceResult | None,
    analysis_timing: AnalysisTiming | None,
    is_analysis_running: bool,
) -> LanguageChangeUpdates:
    """Create named UI updates after a TACME language change."""

    language_index = get_language_index(
        language,
    )

    selected_task_keys = _get_selected_task_keys(selected_task_labels)
    task_choices = get_task_choices(language_index)

    has_video = bool(
        video_path,
    )

    can_run = (
        has_video
        and bool(
            selected_task_keys,
        )
        and not is_analysis_running
    )

    (
        media_overview_content,
        media_video_content,
        media_audio_content,
        media_compatibility_content,
    ) = create_media_content_updates(
        media_info_state,
        language_index,
    )

    (
        results_container,
        emotion_result,
        personality_result,
        ambivalence_result,
    ) = create_result_updates(
        cached_result,
        selected_task_keys,
        language_index,
    )

    return LanguageChangeUpdates(
        flag=gr.update(
            value=create_language_flag_html(
                language_index,
            ),
        ),
        tabs=create_tab_language_updates(
            language_index,
        ),
        application_title=gr.update(
            value=create_application_title_markdown(
                language_index,
            ),
        ),
        video_input=gr.update(
            label=create_video_component_label(
                language_index,
                video_filename,
            ),
            interactive=not is_analysis_running,
        ),
        examples_title=gr.update(
            value=(
                "### "
                + get_localized_text(
                    "Labels_EXAMPLES",
                    language_index,
                )
            ),
        ),
        examples_placeholder=gr.update(
            value=get_localized_text(
                "Texts_EXAMPLES_PLACEHOLDER",
                language_index,
            ),
        ),
        task_selector=gr.update(
            choices=task_choices,
            value=selected_task_keys,
            label=get_localized_text(
                "Labels_TASKS",
                language_index,
            ),
            visible=has_video,
            interactive=(has_video and not is_analysis_running),
        ),
        run_button=gr.update(
            value=get_localized_text(
                "Labels_RUN_ANALYSIS",
                language_index,
            ),
            interactive=can_run,
        ),
        clear_button=gr.update(
            value=get_localized_text(
                "Labels_CLEAR",
                language_index,
            ),
            interactive=(has_video and not is_analysis_running),
        ),
        status=gr.update(
            value=_get_application_status_text(
                video_path=video_path,
                selected_task_labels=selected_task_keys,
                cached_result=cached_result,
                analysis_timing=analysis_timing,
                is_analysis_running=is_analysis_running,
                language_index=language_index,
            ),
        ),
        media_info_button_column=gr.update(
            visible=has_video,
        ),
        media_info_button=gr.update(
            value=get_localized_text(
                "Labels_MEDIA_INFO_TITLE",
                language_index,
            ),
            interactive=(has_video and not is_analysis_running),
        ),
        media_info_modal_title=gr.update(
            value=(
                "### "
                + get_localized_text(
                    "Labels_MEDIA_INFO_TITLE",
                    language_index,
                )
            ),
        ),
        media_overview_tab=gr.update(
            label=get_localized_text(
                "Labels_MEDIA_TAB_OVERVIEW",
                language_index,
            ),
        ),
        media_video_tab=gr.update(
            label=get_localized_text(
                "Labels_MEDIA_TAB_VIDEO",
                language_index,
            ),
        ),
        media_audio_tab=gr.update(
            label=get_localized_text(
                "Labels_MEDIA_TAB_AUDIO",
                language_index,
            ),
        ),
        media_compatibility_tab=gr.update(
            label=get_localized_text(
                "Labels_MEDIA_TAB_COMPATIBILITY",
                language_index,
            ),
        ),
        media_overview_content=media_overview_content,
        media_video_content=media_video_content,
        media_audio_content=media_audio_content,
        media_compatibility_content=media_compatibility_content,
        results_container=results_container,
        emotion_result=emotion_result,
        personality_result=personality_result,
        ambivalence_result=ambivalence_result,
        about_title=gr.update(
            value=(
                "### "
                f"{
                    get_localized_text(
                        'Texts_ABOUT_TITLE',
                        language_index,
                    )
                }"
            ),
        ),
        about_description=gr.update(
            value=get_localized_text(
                "Texts_ABOUT_DESCRIPTION",
                language_index,
            ),
        ),
        about_content=gr.update(
            value=create_about_content_html(
                language_index,
            ),
        ),
        authors_content=gr.update(
            value=create_authors_content_html(
                language_index,
            ),
        ),
        requirements_title=gr.update(
            value=(
                "### "
                f"{
                    get_localized_text(
                        'Texts_REQUIREMENTS_TITLE',
                        language_index,
                    )
                }"
            ),
        ),
        requirements_description=gr.update(
            value=get_localized_text(
                "Requirements_DESCRIPTION",
                language_index,
            ),
        ),
        requirements_content=gr.update(
            value=(
                create_requirements_content_html_from_state(
                    language_index,
                    requirements_state,
                )
            ),
        ),
    )
