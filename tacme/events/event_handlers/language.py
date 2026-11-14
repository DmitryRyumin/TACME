"""
File: language.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Language-switch callback and ordered UI outputs for event registration.

License: MIT License
"""

from collections.abc import Callable
from typing import Any

from tacme.events.language import handle_language_change
from tacme.ui.about import AboutTabComponents
from tacme.ui.application import ApplicationTabComponents
from tacme.ui.authors import AuthorsTabComponents
from tacme.ui.language_selector import LanguageSelectorComponents
from tacme.ui.requirements import RequirementsSessionState, RequirementsTabComponents
from tacme.ui.tabs import AppTabsComponents


def create_language_change_handler(
    language_selector: LanguageSelectorComponents,
    app_tabs: AppTabsComponents,
    application_content: ApplicationTabComponents,
    about_content: AboutTabComponents,
    authors_content: AuthorsTabComponents,
    requirements_content: RequirementsTabComponents,
) -> Callable[
    [str, RequirementsSessionState | None, list[str] | None, str | None, str | None, Any, Any, Any, bool],
    dict[Any, Any],
]:
    """Bind the original localization callback to the current UI components."""

    def handle_language_change_event(
        language: str,
        requirements_state: RequirementsSessionState | None,
        selected_task_labels: list[str] | None,
        video_path: str | None,
        video_filename: str | None,
        media_info_state: Any,
        cached_result: Any,
        analysis_timing: Any,
        is_analysis_running: bool,
    ) -> dict[
        Any,
        Any,
    ]:
        """Map localized values to the active TACME UI components."""

        updates = handle_language_change(
            language=language,
            requirements_state=requirements_state,
            selected_task_labels=selected_task_labels,
            video_path=video_path,
            video_filename=video_filename,
            media_info_state=media_info_state,
            cached_result=cached_result,
            analysis_timing=analysis_timing,
            is_analysis_running=is_analysis_running,
        )

        component_updates: dict[
            Any,
            Any,
        ] = {
            language_selector.flag: updates.flag,
            application_content.title: updates.application_title,
            application_content.video_input: updates.video_input,
            application_content.examples_title: (updates.examples_title),
            application_content.examples_placeholder: (updates.examples_placeholder),
            application_content.task_selector: (updates.task_selector),
            application_content.run_button: updates.run_button,
            application_content.clear_button: (updates.clear_button),
            application_content.status: updates.status,
            application_content.media_info_button_column: (updates.media_info_button_column),
            application_content.media_info_button: (updates.media_info_button),
            application_content.media_info_modal_title: (updates.media_info_modal_title),
            application_content.media_overview_tab: (updates.media_overview_tab),
            application_content.media_video_tab: (updates.media_video_tab),
            application_content.media_audio_tab: (updates.media_audio_tab),
            application_content.media_compatibility_tab: (updates.media_compatibility_tab),
            application_content.media_overview_content: (updates.media_overview_content),
            application_content.media_video_content: (updates.media_video_content),
            application_content.media_audio_content: (updates.media_audio_content),
            application_content.media_compatibility_content: (updates.media_compatibility_content),
            application_content.results_container: (updates.results_container),
            application_content.emotion_result: (updates.emotion_result),
            application_content.personality_result: (updates.personality_result),
            application_content.ambivalence_result: (updates.ambivalence_result),
            about_content.title: updates.about_title,
            about_content.description: updates.about_description,
            about_content.content: updates.about_content,
            authors_content.content: updates.authors_content,
            requirements_content.title: (updates.requirements_title),
            requirements_content.description: (updates.requirements_description),
            requirements_content.content: (updates.requirements_content),
        }

        for (
            tab_name,
            tab_update,
        ) in updates.tabs.items():
            tab_component = app_tabs.tab_components.get(
                tab_name,
            )

            if tab_component is not None:
                component_updates[tab_component] = tab_update

        return component_updates

    return handle_language_change_event
