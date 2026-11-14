"""Technical metadata and README rendering utilities for TACME."""

from scripts.readme.badge_elements import Badge as Badge
from scripts.readme.cli import main as main
from scripts.readme.models import (
    ProjectMetadata as ProjectMetadata,
    ReadmeAuthor as ReadmeAuthor,
    ReadmeAuthorLink as ReadmeAuthorLink,
)

__all__ = [
    "Badge",
    "ProjectMetadata",
    "ReadmeAuthor",
    "ReadmeAuthorLink",
    "main",
]
