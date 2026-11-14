"""Localized upload notifications and error detection patterns."""

# ruff: noqa: RUF001

UPLOAD_ERROR_MESSAGES_JS = r"""
(language) => {
    const texts = {
        en: {
            upload: [
                "Unable to upload video",
                "The file could not be uploaded. Check file access and try again."
            ],
            unreadable: [
                "Unable to read video",
                "The selected video cannot be read. Check file access or upload it again."
            ],
            missing: [
                "Video is no longer available",
                "The selected video is no longer available. Upload it again."
            ],
            empty_file: [
                "Empty video file",
                "The selected file is empty and cannot be processed. Upload a valid video."
            ],
            invalid_container: [
                "Invalid or damaged video",
                "The video container could not be read. The file may be damaged, incomplete, or unsupported. Upload another video."
            ],
            probe_unavailable: [
                "Unable to inspect video",
                "Video inspection is temporarily unavailable. Try again or contact the administrator."
            ],
            truncated_file: [
                "Incomplete video file",
                "The video recording is incomplete or truncated. Upload the complete video."
            ],
            undecodable_video: [
                "Unable to decode video frames",
                "No usable video frames could be decoded. The video may be damaged or use an unsupported codec. Upload another video."
            ],
            corrupted_frames: [
                "Corrupted video frames",
                "Some video frames cannot be decoded correctly. Upload another video."
            ],
            playback_unavailable: [
                "Unable to play video",
                "The browser could not play this video. It may be damaged or use an unsupported codec."
            ],
            preview_unavailable: [
                "Unable to prepare video playback",
                "A browser-compatible preview could not be created. Try another video."
            ],
            decoder_unavailable: [
                "Unable to inspect video frames",
                "Video decoding could not be checked. Try again later or contact the administrator."
            ],
            validation_timeout: [
                "Video validation timed out",
                "The video could not be verified in time. Please try again."
            ],
            acknowledge: "Got it",
            close: "Close notification"
        },
        ru: {
            upload: [
                "Не удалось загрузить видео",
                "Не удалось загрузить файл. Проверьте доступ к нему и повторите попытку."
            ],
            unreadable: [
                "Не удалось прочитать видео",
                "Выбранный видеофайл недоступен для чтения. Проверьте доступ или загрузите его повторно."
            ],
            missing: [
                "Видеофайл больше недоступен",
                "Выбранный видеофайл больше недоступен. Загрузите его повторно."
            ],
            empty_file: [
                "Пустой видеофайл",
                "Выбранный файл пуст и не может быть обработан. Загрузите корректный видеофайл."
            ],
            invalid_container: [
                "Поврежденный или некорректный видеофайл",
                "Не удалось прочитать видеоконтейнер. Файл может быть поврежден, записан не полностью или иметь неподдерживаемый формат. Загрузите другое видео."
            ],
            probe_unavailable: [
                "Не удалось проверить видео",
                "Проверка видео временно недоступна. Повторите попытку или обратитесь к администратору."
            ],
            truncated_file: [
                "Неполный видеофайл",
                "Видеозапись обрезана или сохранена не полностью. Загрузите полный видеофайл."
            ],
            undecodable_video: [
                "Не удалось декодировать видеокадры",
                "Из файла не удалось получить ни одного пригодного видеокадра. Запись может быть повреждена или использовать неподдерживаемый кодек. Загрузите другое видео."
            ],
            corrupted_frames: [
                "Поврежденные видеокадры",
                "Некоторые кадры видео не удается корректно декодировать. Загрузите другое видео."
            ],
            playback_unavailable: [
                "Не удалось воспроизвести видео",
                "Браузер не может воспроизвести эту запись. Возможно, файл поврежден или используется неподдерживаемый кодек."
            ],
            preview_unavailable: [
                "Не удалось подготовить видео к воспроизведению",
                "Не удалось создать совместимый с браузером предпросмотр. Загрузите другое видео."
            ],
            decoder_unavailable: [
                "Не удалось проверить видеокадры",
                "Проверка декодирования временно недоступна. Повторите попытку позже или обратитесь к администратору."
            ],
            validation_timeout: [
                "Время проверки видео истекло",
                "Не удалось вовремя завершить проверку видео. Повторите попытку."
            ],
            acknowledge: "Понятно",
            close: "Закрыть сообщение"
        }
    };

"""

UPLOAD_ERROR_MATCHERS_JS = r"""    const serverMessages = [
        ["preview_unavailable", /Unable to prepare the video for playback|Не удалось подготовить видео для воспроизведения/i],
        ["undecodable_video", /Unable to decode video frames|Не удалось декодировать видеокадры/i],
        ["corrupted_frames", /Video contains corrupted frames|В видео обнаружены поврежденные кадры/i],
        ["decoder_unavailable", /Video frame inspection is unavailable|Проверка видеокадров недоступна/i],
        ["truncated_file", /The video file is incomplete or truncated|Видеофайл записан не полностью или обрезан/i],
        ["invalid_container", /Unable to inspect video container|Не удалось прочитать видеоконтейнер/i],
        ["probe_unavailable", /Unable to inspect the video file|Не удалось проверить видеофайл/i],
        ["empty_file", /The selected video file is empty|Выбранный видеофайл пуст/i],
        ["missing", /Video file is no longer available|Видеофайл больше недоступен/i],
        ["unreadable", /Unable to read the video file|Не удалось прочитать видеофайл/i]
    ];
    const uploadMessage = /error uploading file|error uploading|upload error|upload failed|failed to upload|unable to upload|не удалось загрузить файл/i;
    // In Gradio 6, an interrupted upload can be reported by the client as a
    // generic connection error. Treat it as an upload failure *only* if the
    // user has just selected or dropped a video.
    const uploadConnectionMessage = /connection errored out\.\s*failed to fetch/i;
    const notPlayableMessage = /video not playable|video is not playable/i;
    const nativeToastSelector = ".toast-body.error, [role='alert'], .toast, gradio-toast, gradio-alert";
    const uploadErrorWindowMs = 120000;

    const locale = () => texts[currentLanguage];
    const pendingTexts = {
        en: ["Checking video", "Verifying the video before enabling analysis."],
        ru: ["Проверка видео", "Проверяем видеозапись перед запуском анализа."]
    };
    // Use the app's existing localized Ready message; validation progress and
    // errors belong exclusively to the overlay above the video input.
    const idleStatusTexts = __TACME_IDLE_STATUS_TEXTS__;
    let keepIdleStatus = false;

"""

__all__ = ["UPLOAD_ERROR_MATCHERS_JS", "UPLOAD_ERROR_MESSAGES_JS"]
