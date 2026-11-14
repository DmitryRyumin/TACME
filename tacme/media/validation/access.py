"""
File: validation_access.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: A1/A2.1 media file accessibility checks.

License: MIT License
"""

import logging
from pathlib import Path
from stat import S_ISREG
from typing import Literal

LOGGER = logging.getLogger("tacme.media.validation")

type VideoFileStatus = Literal[
    "empty",
    "empty_file",
    "missing",
    "unreadable",
    "invalid_container",
    "truncated_file",
    "undecodable_video",
    "corrupted_frames",
    "decoder_unavailable",
    "probe_unavailable",
    "preview_unavailable",
    "available",
]


def check_video_file_access(video_path: str | None) -> VideoFileStatus:
    """Check a file before analysis without decoding its media streams.

    A missing selection is normal. A previously selected but inaccessible
    local file is an error. A zero-byte file is rejected by A2.1.
    Malformed or unsupported media are handled by later A2/A3 checks.
    """

    if not video_path:
        return "empty"

    path = Path(video_path)
    try:
        if not S_ISREG(path.stat().st_mode):
            LOGGER.warning("Video input is not a regular file: %s", path)
            return "unreadable"
        # Open the file first, preserving A1 detection of unreadable files.
        # Both a zero length and truncation between stat and read are A2.1.
        with path.open("rb") as file:
            first_byte = file.read(1)
        if path.stat().st_size == 0 or not first_byte:
            LOGGER.warning("Video input is an empty file: %s", path)
            return "empty_file"
    except FileNotFoundError, NotADirectoryError:
        LOGGER.warning("Video input no longer exists: %s", path)
        return "missing"
    except (OSError, ValueError) as error:
        LOGGER.warning("Video input is not readable (%s): %s", path, error)
        return "unreadable"

    return "available"


__all__ = ["LOGGER", "VideoFileStatus", "check_video_file_access"]
