"""
File: tabs.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Dynamic tab layout components for the TACME Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from typing import Any, Final

import gradio as gr

from tacme.config import (
    get_config_str,
    load_tab_creators,
)
from tacme.localization import (
    get_localized_values,
)
from tacme.ui.about import create_about_tab
from tacme.ui.application import create_application_tab
from tacme.ui.authors import create_authors_tab
from tacme.ui.requirements import create_requirements_tab

TAB_STATE_ENABLED: Final = "enabled"
TAB_STATE_DISABLED: Final = "disabled"
TAB_STATE_HIDDEN: Final = "hidden"

VALID_TAB_STATES: Final = frozenset(
    {
        TAB_STATE_ENABLED,
        TAB_STATE_DISABLED,
        TAB_STATE_HIDDEN,
    }
)


@dataclass(
    frozen=True,
    slots=True,
)
class AppTabsComponents:
    """Components created by the TACME tab UI."""

    tab_components: dict[str, gr.Tab]
    tab_contents: dict[str, Any]


@dataclass(
    frozen=True,
    slots=True,
)
class TabRuntimeState:
    """Resolved runtime state of one TACME tab."""

    visible: bool
    interactive: bool


def _get_tab_labels(
    tab_name: str,
) -> list[str]:
    """Return localized labels configured for a tab."""

    return get_localized_values(
        f"Tabs_{tab_name}",
    )


def _get_tab_runtime_state(
    tab_name: str,
) -> TabRuntimeState:
    """Resolve the configured runtime state of one tab."""

    state = (
        get_config_str(
            f"TabStates_{tab_name}",
            TAB_STATE_ENABLED,
        )
        .strip()
        .lower()
    )

    if state not in VALID_TAB_STATES:
        allowed_states = ", ".join(
            sorted(VALID_TAB_STATES),
        )

        msg = f"Unsupported tab state '{state}' for '{tab_name}'. Expected one of: {allowed_states}."

        raise ValueError(msg)

    if state == TAB_STATE_HIDDEN:
        return TabRuntimeState(
            visible=False,
            interactive=False,
        )

    if state == TAB_STATE_DISABLED:
        return TabRuntimeState(
            visible=True,
            interactive=False,
        )

    return TabRuntimeState(
        visible=True,
        interactive=True,
    )


def create_app_tabs(
    language_index: int = 0,
) -> AppTabsComponents:
    """Create TACME tabs dynamically from configuration."""

    available_functions = {
        "create_application_tab": create_application_tab,
        "create_about_tab": create_about_tab,
        "create_authors_tab": create_authors_tab,
        "create_requirements_tab": create_requirements_tab,
    }

    tab_creators = load_tab_creators(
        available_functions,
    )

    tab_components: dict[str, gr.Tab] = {}
    tab_contents: dict[str, Any] = {}

    with gr.Tabs(
        elem_classes="app-tabs",
    ):
        for tab_name, tab_creator in tab_creators.items():
            tab_labels = _get_tab_labels(
                tab_name,
            )

            if language_index >= len(tab_labels):
                msg = f"Language index {language_index} is not available for tab '{tab_name}'."

                raise IndexError(msg)

            tab_state = _get_tab_runtime_state(
                tab_name,
            )

            tab_id = tab_name.lower().replace(
                "_",
                "-",
            )

            with gr.Tab(
                label=tab_labels[language_index],
                id=tab_id,
                visible=tab_state.visible,
                interactive=tab_state.interactive,
                elem_id=f"tab-{tab_id}",
                elem_classes=f"{tab_id}-tab",
            ) as tab:
                tab_contents[tab_name] = tab_creator(
                    language_index,
                )

            tab_components[tab_name] = tab

    return AppTabsComponents(
        tab_components=tab_components,
        tab_contents=tab_contents,
    )
