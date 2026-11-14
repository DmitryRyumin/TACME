"""
File: validation_decode.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: A2.4/A2.5 video frame-decoding checks.

License: MIT License
"""

from functools import lru_cache
import subprocess

from tacme.media.validation.access import LOGGER, VideoFileStatus


@lru_cache(maxsize=32)
def _check_first_decodable_frame(
    file_path: str,
    size_bytes: int,
    modified_ns: int,
) -> VideoFileStatus:
    """A2.4: verify that the video stream produces at least one decoded frame.

    ffprobe checks metadata, not actual picture decoding. FFmpeg decodes until
    one frame is found; the decoded result is scaled to just four grayscale
    bytes, so the output stays tiny even for a high-resolution video.
    Individual later corrupt frames are handled separately by A2.5.
    """

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        file_path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        "-vf",
        "scale=2:2:flags=fast_bilinear",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        # A missing or stalled decoder is an application inspection failure;
        # do not incorrectly accuse the uploaded file of being corrupted.
        LOGGER.warning("Video decoding preflight unavailable: %s", type(error).__name__)
        return "decoder_unavailable"

    if completed.returncode != 0 or len(completed.stdout) != 4:
        LOGGER.debug("Video stream has no decodable frame: %s", file_path)
        return "undecodable_video"

    return "available"


@lru_cache(maxsize=32)
def _check_full_video_decode(
    file_path: str,
    size_bytes: int,
    modified_ns: int,
) -> VideoFileStatus:
    """Reject files with decode errors after the first valid video frame.

    A first-frame-only preflight is insufficient: corruption in later H.264
    packets can leave the container and initial picture perfectly readable.
    Decode the video stream without writing output; ``-xerror`` ensures an
    error in a middle frame is not silently treated as success by FFmpeg.
    """

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-i",
        file_path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.warning("Full-frame video inspection unavailable: %s", type(error).__name__)
        return "decoder_unavailable"

    if completed.returncode != 0:
        LOGGER.debug("Video contains damaged frames: %s", file_path)
        # FFmpeg's diagnostic is intentionally retained only at DEBUG level.
        LOGGER.debug("Video decoder details: %s", completed.stderr.decode("utf-8", "replace")[:2000])
        return "corrupted_frames"

    return "available"


__all__ = ["_check_first_decodable_frame", "_check_full_video_decode"]
