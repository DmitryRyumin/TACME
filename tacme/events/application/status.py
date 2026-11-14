"""
File: status.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Localized media-file status messages and analysis-completion text.

License: MIT License
"""

from tacme.localization import get_localized_text
from tacme.media.validation import VideoFileStatus
from tacme.performance import AnalysisTiming


def _video_file_status_text(status: VideoFileStatus, language_index: int) -> str:
    """User-facing file-access status without paths or exception details."""

    if status == "empty":
        return get_localized_text("Texts_STATUS_READY", language_index)
    if status == "empty_file":
        if language_index == 1:
            return "Выбранный видеофайл пуст. Загрузите корректное видео."
        return "The selected video file is empty. Upload a valid video."
    if status == "invalid_container":
        if language_index == 1:
            return "Не удалось прочитать видеоконтейнер. Файл может быть поврежден или иметь неподдерживаемый формат."  # noqa: RUF001
        return "Unable to inspect video container. The file may be damaged or use an unsupported format."
    if status == "truncated_file":
        if language_index == 1:
            return "Видеофайл записан не полностью или обрезан. Загрузите полный файл."
        return "The video file is incomplete or truncated. Upload the complete file."
    if status == "undecodable_video":
        if language_index == 1:
            return "Не удалось декодировать видеокадры. Загрузите другое видео."  # noqa: RUF001
        return "Unable to decode video frames. Upload a different video."
    if status == "corrupted_frames":
        if language_index == 1:
            return "В видео обнаружены поврежденные кадры. Загрузите другое видео."  # noqa: RUF001
        return "Video contains corrupted frames. Upload another video."
    if status == "decoder_unavailable":
        if language_index == 1:
            return "Проверка видеокадров недоступна. Повторите попытку позже."
        return "Video frame inspection is unavailable. Try again later."
    if status == "preview_unavailable":
        if language_index == 1:
            return "Не удалось подготовить видео для воспроизведения. Попробуйте другое видео."  # noqa: RUF001
        return "Unable to prepare the video for playback. Try another video."
    if status == "probe_unavailable":
        if language_index == 1:
            return "Не удалось проверить видеофайл. Повторите попытку или обратитесь к администратору."  # noqa: RUF001
        return "Unable to inspect the video file. Try again or contact the administrator."
    if status == "missing":
        if language_index == 1:
            return "Видеофайл больше недоступен. Загрузите его повторно."  # noqa: RUF001
        return "Video file is no longer available. Upload it again."
    if status == "unreadable":
        if language_index == 1:
            return (
                "Не удалось прочитать видеофайл. "  # noqa: RUF001
                "Проверьте доступ к файлу или загрузите его повторно."  # noqa: RUF001
            )
        return "Unable to read the video file. Check access or upload it again."
    raise ValueError("A valid video has no file-access error message")


def create_analysis_completed_status(
    timing: AnalysisTiming | None,
    language_index: int,
) -> str:
    """Return the localized completion status; metrics live in the details action."""

    return get_localized_text(
        "Texts_STATUS_ANALYSIS_COMPLETED",
        language_index,
    )
