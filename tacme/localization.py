"""
File: localization.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Localization helpers for the TACME Gradio application.

License: MIT License
"""

from typing import Final

from tacme.config import (
    PROJECT_ROOT,
    get_config_str,
    get_config_str_list,
)

DEFAULT_LANGUAGE_INDEX: Final = 0

TASK_KEYS: Final[tuple[str, ...]] = (
    "emotion",
    "personality",
    "ambivalence",
)

TASK_CONFIG_FIELDS: Final = {
    "emotion": "Tasks_EMOTION",
    "personality": "Tasks_PERSONALITY",
    "ambivalence": "Tasks_AMBIVALENCE",
}


def get_localized_values(
    config_field: str,
) -> list[str]:
    """Return localized values from the TACME configuration."""

    values = get_config_str_list(
        config_field,
        [],
    )

    if not values:
        msg = f"Localized configuration field {config_field!r} must be a non-empty list of strings."
        raise TypeError(msg)

    return values


def get_localized_text(
    config_field: str,
    language_index: int,
) -> str:
    """Return a localized string by language index."""

    values = get_localized_values(
        config_field,
    )

    if language_index < 0 or language_index >= len(
        values,
    ):
        language_index = DEFAULT_LANGUAGE_INDEX

    return values[language_index]


def get_language_index(
    language: str,
) -> int:
    """Return the index of a configured language."""

    languages = get_localized_values(
        "Languages_CHOICES",
    )

    try:
        return languages.index(
            language,
        )

    except ValueError:
        return DEFAULT_LANGUAGE_INDEX


def get_language_flag_path(
    language_index: int,
) -> str:
    """Return the flag image path for a configured language."""

    images_dir = get_config_str(
        "StaticPaths_IMAGES",
        "static/images",
    )

    language_images = get_localized_values(
        "Images_LANGUAGES",
    )

    if language_index < 0 or language_index >= len(
        language_images,
    ):
        language_index = DEFAULT_LANGUAGE_INDEX

    return str(PROJECT_ROOT / images_dir / language_images[language_index])


def get_task_label(
    task_key: str,
    language_index: int,
) -> str:
    """Return the localized label for one TACME task."""

    config_field = TASK_CONFIG_FIELDS.get(
        task_key,
    )

    if config_field is None:
        return task_key

    return get_localized_text(
        config_field,
        language_index,
    )


def get_task_labels(
    language_index: int,
) -> list[str]:
    """Return localized TACME task labels in canonical order."""

    return [
        get_task_label(
            task_key,
            language_index,
        )
        for task_key in TASK_KEYS
    ]


def get_task_choices(language_index: int) -> list[tuple[str, str]]:
    """Return translated CheckboxGroup labels with language-independent values."""

    return [(get_task_label(task_key, language_index), task_key) for task_key in TASK_KEYS]


def get_task_keys_from_labels(
    task_labels: list[str] | None,
) -> list[str]:
    """Resolve canonical TACME task keys from localized labels."""

    if not task_labels:
        return []

    resolved_keys: list[str] = []

    for task_key in TASK_KEYS:
        localized_labels = get_localized_values(
            TASK_CONFIG_FIELDS[task_key],
        )

        if task_key in task_labels or any(label in task_labels for label in localized_labels):
            resolved_keys.append(
                task_key,
            )

    return resolved_keys


def get_task_labels_from_keys(
    task_keys: list[str],
    language_index: int,
) -> list[str]:
    """Return localized task labels from canonical task keys."""

    return [
        get_task_label(
            task_key,
            language_index,
        )
        for task_key in task_keys
        if task_key in TASK_CONFIG_FIELDS
    ]


__all__ = [
    "DEFAULT_LANGUAGE_INDEX",
    "TASK_CONFIG_FIELDS",
    "TASK_KEYS",
    "get_language_flag_path",
    "get_language_index",
    "get_localized_text",
    "get_localized_values",
    "get_task_choices",
    "get_task_keys_from_labels",
    "get_task_label",
    "get_task_labels",
    "get_task_labels_from_keys",
]
