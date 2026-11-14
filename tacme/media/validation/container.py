"""
File: validation_container.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: A2.2/A2.3 container and truncation checks.

License: MIT License
"""

from functools import lru_cache
from pathlib import Path

from tacme.media.metadata import probe_media_file
from tacme.media.validation.access import LOGGER, VideoFileStatus


def _is_truncated_mp4(path: Path) -> bool:
    """Detect MP4/M4V/M4A atoms whose declared length exceeds the actual file.

    Only inspect atom boundaries; seek over media payload without decoding it.
    Require a real ``ftyp`` atom to avoid classifying arbitrary damaged input
    as a truncated video. A valid atom layout does not prove valid video frames.
    """

    if path.suffix.lower() not in {".m4a", ".mp4", ".m4v"}:
        return False

    try:
        with path.open("rb") as source:
            file_size = source.seek(0, 2)
            source.seek(0)
            if file_size < 8 or source.read(8)[4:] != b"ftyp":
                return False

            offset = 0
            # Avoid doing unbounded work for a malformed or highly fragmented file.
            for _ in range(4096):
                if offset == file_size:
                    return False
                remaining = file_size - offset
                if remaining < 8:
                    return True

                source.seek(offset)
                header = source.read(8)
                if len(header) != 8:
                    return True
                box_size = int.from_bytes(header[:4], "big")
                header_size = 8
                if box_size == 1:
                    extended = source.read(8)
                    if len(extended) != 8:
                        return True
                    box_size = int.from_bytes(extended, "big")
                    header_size = 16
                elif box_size == 0:
                    # ISO BMFF permits a final box extending to EOF. Its
                    # payload completeness cannot be inferred from its header.
                    box_size = remaining

                if box_size < header_size:
                    return False  # Malformed structure belongs to A2.2.
                if box_size > remaining:
                    return True
                offset += box_size

            # Unusual files with very many top-level boxes are left to ffprobe.
            return False
    except OSError, ValueError:
        # Access errors are handled by A1 or the following ffprobe call.
        return False


@lru_cache(maxsize=32)
def _probe_video_container(
    file_path: str,
    size_bytes: int,
    modified_ns: int,
) -> VideoFileStatus:
    """Cache a metadata probe for an unchanged file, avoiding repeated ffprobe calls.

    The size and modification time form part of the cache key. The file can
    still change between inspection and inference; decoder errors remain
    covered by the existing inference error handler.
    """

    result = probe_media_file(file_path)
    if result.is_available:
        # Video/audio stream requirements are checked separately in A3/A4.
        return "available"

    detail = result.error or "unknown ffprobe error"
    if detail.startswith("Unable to execute ffprobe:"):
        # A missing ffprobe binary or a timeout is an inspection problem, not
        # evidence that a user's file is damaged.
        LOGGER.warning("Unable to run the media inspector for %s", file_path)
        LOGGER.debug("ffprobe diagnostic: %s", detail)
        return "probe_unavailable"

    # Invalid user input is expected; keep verbose ffprobe diagnostics in the
    # debug log rather than flooding the application terminal.
    LOGGER.debug("Video container rejected: %s; ffprobe: %s", file_path, detail)
    return "invalid_container"


__all__ = ["_is_truncated_mp4", "_probe_video_container"]
