"""
File: session.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: TACME Requirements tab implementation.

License: MIT License
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Final

from tacme.ui.requirements.dependencies import (
    LOGGER,
    _fetch_latest_pypi_version,
    get_runtime_dependencies,
)
from tacme.ui.requirements.models import RequirementsSessionState

PYPI_MAX_WORKERS: Final = 4


def _create_pending_session_state() -> RequirementsSessionState:
    """Create a fresh state for a newly loaded browser page."""

    _, dependencies = get_runtime_dependencies()

    return RequirementsSessionState(
        latest_versions=(),
        pending_names=frozenset(dependency.canonical_name for dependency in dependencies),
        checked_at=None,
        resolved_name=None,
        completed=False,
    )


def stream_requirements_session_state(
    session_state: RequirementsSessionState | None,
) -> Iterator[RequirementsSessionState]:
    """Check PyPI once for the current browser page."""

    if session_state is not None and session_state.completed:
        yield RequirementsSessionState(
            latest_versions=tuple(
                session_state.latest_versions,
            ),
            pending_names=frozenset(),
            checked_at=session_state.checked_at,
            resolved_name=None,
            completed=True,
        )

        return

    _, dependencies = get_runtime_dependencies()

    dependency_by_name = {dependency.canonical_name: dependency for dependency in dependencies}

    latest_versions = (
        dict(
            session_state.latest_versions,
        )
        if session_state is not None
        else {}
    )

    completed_names = set(
        latest_versions,
    )

    pending_names = set(
        dependency_by_name,
    ).difference(
        completed_names,
    )

    initial_state = RequirementsSessionState(
        latest_versions=tuple(
            sorted(
                latest_versions.items(),
            )
        ),
        pending_names=frozenset(
            pending_names,
        ),
        checked_at=(session_state.checked_at if session_state is not None else None),
        resolved_name=None,
        completed=not pending_names,
    )

    yield initial_state

    if not pending_names:
        return

    dependencies_to_check = [dependency_by_name[name] for name in pending_names]

    max_workers = min(
        PYPI_MAX_WORKERS,
        len(dependencies_to_check),
    )

    with ThreadPoolExecutor(
        max_workers=max_workers,
    ) as executor:
        futures = {
            executor.submit(
                _fetch_latest_pypi_version,
                dependency.requirement.name,
            ): dependency
            for dependency in dependencies_to_check
        }

        for future in as_completed(
            futures,
        ):
            dependency = futures[future]

            try:
                latest_version = future.result()

            except Exception:
                LOGGER.exception(
                    "Unexpected error while checking the latest version of '%s'.",
                    dependency.requirement.name,
                )

                latest_version = None

            latest_versions[dependency.canonical_name] = latest_version

            pending_names.discard(
                dependency.canonical_name,
            )

            completed = not pending_names

            yield RequirementsSessionState(
                latest_versions=tuple(
                    sorted(
                        latest_versions.items(),
                    )
                ),
                pending_names=frozenset(
                    pending_names,
                ),
                checked_at=(
                    datetime.now(
                        UTC,
                    )
                    if completed
                    else None
                ),
                resolved_name=(dependency.canonical_name),
                completed=completed,
            )
