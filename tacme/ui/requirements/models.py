"""
File: models.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: TACME Requirements tab implementation.

License: MIT License
"""

from dataclasses import dataclass
from datetime import datetime

import gradio as gr
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


@dataclass(
    frozen=True,
    slots=True,
)
class RequirementsTabComponents:
    """Components created inside the TACME Requirements tab."""

    title: gr.Markdown
    description: gr.Markdown
    content: gr.HTML
    session_state: gr.State


@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeDependency:
    """One direct runtime dependency declared by TACME."""

    requirement: Requirement
    raw_requirement: str
    installed_version: str | None

    @property
    def canonical_name(
        self,
    ) -> str:
        """Return the normalized distribution name."""

        return canonicalize_name(
            self.requirement.name,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RequirementsSessionState:
    """Per-browser-page PyPI version-check state."""

    latest_versions: tuple[
        tuple[
            str,
            str | None,
        ],
        ...,
    ]

    pending_names: frozenset[str]

    checked_at: datetime | None
    resolved_name: str | None
    completed: bool


# Keep the original import path used by Gradio state serialization and
# external callers, even though implementation classes live in models.py.
RequirementsTabComponents.__module__ = "tacme.ui.requirements"
RuntimeDependency.__module__ = "tacme.ui.requirements"
RequirementsSessionState.__module__ = "tacme.ui.requirements"
