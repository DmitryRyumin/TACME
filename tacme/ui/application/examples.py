"""
File: examples.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Configured example-video discovery and display labels.

License: MIT License
"""

from pathlib import Path

from tacme.config import PROJECT_ROOT, get_config_str, get_config_str_list


def get_example_video_labels(
    example_video_paths: list[list[str]],
) -> list[str]:
    """Return display labels for example video files."""

    labels: list[str] = []

    for example in example_video_paths:
        stem = Path(
            example[0],
        ).stem

        parts = stem.split(
            "_",
            maxsplit=1,
        )

        if (
            len(
                parts,
            )
            == 2
        ):
            index, name = parts

            labels.append(f"{index} · {name.replace('_', ' ').title()}")

        else:
            labels.append(
                stem.replace(
                    "_",
                    " ",
                ).title()
            )

    return labels


def get_existing_example_video_paths() -> list[list[str]]:
    """Return existing example videos configured for TACME."""

    examples_dir = get_config_str(
        "StaticPaths_EXAMPLES",
        "examples",
    )

    example_files = get_config_str_list(
        "Examples_VIDEO",
        [],
    )

    existing_examples: list[list[str]] = []

    for example_file in example_files:
        path = PROJECT_ROOT / examples_dir / example_file

        if path.is_file():
            existing_examples.append(
                [
                    str(
                        path,
                    ),
                ]
            )

    return existing_examples
