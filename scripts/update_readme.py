"""README metadata and Authors generator for TACME.

Authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.

Keep this entry point for `python -m scripts.update_readme`.
"""

from pathlib import Path
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.readme import (
    Badge as Badge,
    ProjectMetadata as ProjectMetadata,
    ReadmeAuthor as ReadmeAuthor,
    ReadmeAuthorLink as ReadmeAuthorLink,
    main as main,
)

if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
