"""
File: ablation.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Model ablation configuration and validation.

License: MIT License
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast


def _optional_bool(
    value: object | None,
    *,
    field_name: str,
) -> bool | None:
    """Validate one optional boolean ablation value."""

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    msg = f"{field_name} must be a boolean or None, got {type(value).__name__}."
    raise TypeError(msg)


def _optional_string_tuple(
    value: object | None,
    *,
    field_name: str,
) -> tuple[str, ...] | None:
    """Validate one optional sequence of strings."""

    if value is None:
        return None

    if isinstance(
        value,
        str,
    ):
        msg = f"{field_name} must be a sequence of strings, not a single string."
        raise TypeError(msg)

    if not isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        msg = f"{field_name} must be a list or tuple of strings, got {type(value).__name__}."
        raise TypeError(msg)

    if not all(
        isinstance(
            item,
            str,
        )
        for item in value
    ):
        msg = f"{field_name} must contain only strings."
        raise TypeError(msg)

    return tuple(
        cast(
            str,
            item,
        )
        for item in value
    )


def _get_ablation_value(
    source: object,
    field_name: str,
) -> object | None:
    """Read an ablation field from a mapping or object."""

    if isinstance(
        source,
        Mapping,
    ):
        return source.get(
            field_name,
        )

    return getattr(
        source,
        field_name,
        None,
    )


@dataclass(
    frozen=True,
    slots=True,
)
class AblationCfg:
    """Optional modality and task restrictions retained for experiment compatibility."""

    use_graph: bool | None = None
    use_attention: bool | None = None
    use_guidebank: bool | None = None
    use_task_projectors: bool | None = None

    disabled_modalities: tuple[str, ...] | None = None
    active_modalities: tuple[str, ...] | None = None
    active_tasks: tuple[str, ...] | None = None

    @classmethod
    def from_obj(
        cls,
        value: object | None,
    ) -> AblationCfg:
        """Create a normalized ablation configuration."""

        if value is None:
            return cls()

        if isinstance(
            value,
            cls,
        ):
            return value

        return cls(
            use_graph=_optional_bool(
                _get_ablation_value(
                    value,
                    "use_graph",
                ),
                field_name="use_graph",
            ),
            use_attention=_optional_bool(
                _get_ablation_value(
                    value,
                    "use_attention",
                ),
                field_name="use_attention",
            ),
            use_guidebank=_optional_bool(
                _get_ablation_value(
                    value,
                    "use_guidebank",
                ),
                field_name="use_guidebank",
            ),
            use_task_projectors=_optional_bool(
                _get_ablation_value(
                    value,
                    "use_task_projectors",
                ),
                field_name=("use_task_projectors"),
            ),
            disabled_modalities=(
                _optional_string_tuple(
                    _get_ablation_value(
                        value,
                        "disabled_modalities",
                    ),
                    field_name=("disabled_modalities"),
                )
            ),
            active_modalities=(
                _optional_string_tuple(
                    _get_ablation_value(
                        value,
                        "active_modalities",
                    ),
                    field_name=("active_modalities"),
                )
            ),
            active_tasks=(
                _optional_string_tuple(
                    _get_ablation_value(
                        value,
                        "active_tasks",
                    ),
                    field_name=("active_tasks"),
                )
            ),
        )
