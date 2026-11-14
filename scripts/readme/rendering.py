"""README generator implementation for the TACME project.

Original authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.
"""

from __future__ import annotations

import sys

from scripts.readme.authors import (
    _build_authors_block,
)
from scripts.readme.badges import (
    _build_badge_block,
)
from scripts.readme.constants import (
    AUTHORS_BLOCK_END,
    AUTHORS_BLOCK_START,
    README_BLOCK_END,
    README_BLOCK_START,
    README_PATH,
)


def _replace_generated_block(
    source: str,
    start_marker: str,
    end_marker: str,
    generated_block: str,
) -> str:
    """Replace one generated README region."""

    start_index = source.find(
        start_marker,
    )

    if start_index < 0:
        msg = f"README marker not found: {start_marker}"
        raise ValueError(msg)

    content_start = start_index + len(
        start_marker,
    )

    end_index = source.find(
        end_marker,
        content_start,
    )

    if end_index < 0:
        msg = f"README marker not found: {end_marker}"
        raise ValueError(msg)

    return source[:content_start] + "\n\n" + generated_block + "\n\n" + source[end_index:]


def _render_readme(
    source: str,
) -> str:
    """Render all generated TACME README regions."""

    rendered = _replace_generated_block(
        source,
        README_BLOCK_START,
        README_BLOCK_END,
        _build_badge_block(),
    )

    return _replace_generated_block(
        rendered,
        AUTHORS_BLOCK_START,
        AUTHORS_BLOCK_END,
        _build_authors_block(),
    )


def _update_readme(
    *,
    check: bool,
) -> bool:
    """Update README metadata or verify that it is current."""

    source = README_PATH.read_text(
        encoding="utf-8",
    )

    rendered = _render_readme(
        source,
    )

    if rendered == source:
        print("README metadata is up to date.")

        return True

    if check:
        print(
            ("README metadata is out of date. Run `just fix` to regenerate it."),
            file=sys.stderr,
        )

        return False

    README_PATH.write_text(
        rendered,
        encoding="utf-8",
    )

    print("README metadata updated.")

    return True
