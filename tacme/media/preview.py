"""Create browser-compatible playback separately from the inference source."""

import hashlib
import logging
import os
from pathlib import Path
import subprocess
import tempfile

from tacme.media.metadata import MediaProbeResult, probe_media_file

LOGGER = logging.getLogger("tacme.media.preview")


class PreviewConversionError(RuntimeError):
    """A validated video cannot be prepared for browser playback."""


def _remux_webm_duration_preview(path: Path) -> str:
    """Cache an atomic WebM playback copy with container duration metadata."""

    try:
        stat = path.stat()
        identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:webm-duration"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        preview_dir = Path(tempfile.gettempdir()) / "tacme-video-previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview = preview_dir / f"{digest}.webm"
    except OSError as error:
        raise PreviewConversionError("Unable to prepare WebM preview storage") from error

    def has_duration(candidate: Path) -> bool:
        probe = probe_media_file(candidate)
        metadata = probe.metadata
        return metadata is not None and metadata.duration_seconds is not None and metadata.duration_seconds > 0

    if preview.is_file() and preview.stat().st_size > 0 and has_duration(preview):
        return str(preview)

    try:
        with tempfile.NamedTemporaryFile(dir=preview_dir, suffix=".webm", delete=False) as output:
            temporary = Path(output.name)
    except OSError as error:
        raise PreviewConversionError("Unable to create WebM preview storage") from error

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0",
        "-c",
        "copy",
        str(temporary),
    ]
    try:
        temporary.unlink(missing_ok=True)
        completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
        if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
            LOGGER.warning(
                "WebM duration remux failed for %s: %s",
                path,
                completed.stderr.decode("utf-8", "replace")[:1200],
            )
            raise PreviewConversionError("Unable to recover WebM preview duration")
        if not has_duration(temporary):
            raise PreviewConversionError("Remuxed WebM still has no duration metadata")
        os.replace(temporary, preview)
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.warning("WebM duration preview unavailable (%s)", type(error).__name__)
        raise PreviewConversionError("WebM duration preview is unavailable") from error
    finally:
        temporary.unlink(missing_ok=True)

    return str(preview)


def prepare_video_preview(
    video_path: str,
    probe: MediaProbeResult | None,
    *,
    repair_missing_duration: bool = False,
) -> str:
    """Return browser-compatible playback, repairing recorded WebM duration if needed.

    This is strictly a presentation copy. The uploaded source remains available
    separately for TACME inference, metadata and audio extraction.
    """

    metadata = probe.metadata if probe is not None else None
    video = metadata.video if metadata is not None else None
    audio = metadata.audio if metadata is not None else None
    if video is None:
        raise PreviewConversionError("Video stream metadata is unavailable")

    path = Path(video_path)
    suffix = path.suffix.lower()
    video_codec = video.codec_name
    audio_codec = audio.codec_name if audio is not None else None
    browser_h264 = video_codec == "h264" and video.pixel_format in {"yuv420p", "yuvj420p"}

    if suffix in {".mp4", ".m4v"} and browser_h264 and audio_codec in {None, "aac", "mp3"}:
        return video_path
    if (
        suffix == ".webm"
        and video_codec in {"vp8", "vp9"}
        and audio_codec
        in {
            None,
            "opus",
            "vorbis",
        }
    ):
        if repair_missing_duration and metadata is not None and metadata.duration_seconds is None:
            return _remux_webm_duration_preview(path)
        return video_path

    # Examples may live in a read-only project directory. Keep playback files
    # in a writable cache keyed by source path, size, and modification time.
    try:
        stat = path.stat()
        identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        preview_dir = Path(tempfile.gettempdir()) / "tacme-video-previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview = preview_dir / f"{digest}.mp4"
        if preview.is_file() and preview.stat().st_size > 0:
            return str(preview)
    except OSError as error:
        raise PreviewConversionError("Unable to prepare video preview storage") from error

    try:
        with tempfile.NamedTemporaryFile(dir=preview_dir, suffix=".mp4", delete=False) as output:
            temporary = Path(output.name)
    except OSError as error:
        raise PreviewConversionError("Unable to create video preview storage") from error

    video_options = (
        ["-c:v", "copy"]
        if browser_h264
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    )
    audio_options = ["-c:a", "copy"] if audio_codec in {None, "aac", "mp3"} else ["-c:a", "aac", "-b:a", "128k"]
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-sn",
        "-dn",
        *video_options,
        *audio_options,
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        temporary.unlink(missing_ok=True)
        completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
        if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
            LOGGER.debug(
                "Browser preview conversion failed for %s: %s",
                path,
                completed.stderr.decode("utf-8", "replace")[:1200],
            )
            raise PreviewConversionError("Video preview encoding failed")
        os.replace(temporary, preview)
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.warning("Browser video preview unavailable (%s)", type(error).__name__)
        raise PreviewConversionError("Video preview conversion is unavailable") from error
    finally:
        temporary.unlink(missing_ok=True)

    return str(preview)


__all__ = ["PreviewConversionError", "prepare_video_preview"]
