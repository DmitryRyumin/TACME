"""Browser-side video upload UI, composed without changing runtime behavior.

The fragments are assembled in their original order. The resulting JavaScript
callback is identical to the previously working implementation.
"""

import json

from tacme.localization import get_localized_text
from tacme.ui.upload_error.dialog import UPLOAD_ERROR_DIALOG_JS
from tacme.ui.upload_error.messages import (
    UPLOAD_ERROR_MATCHERS_JS,
    UPLOAD_ERROR_MESSAGES_JS,
)
from tacme.ui.upload_error.preflight import UPLOAD_ERROR_PREFLIGHT_JS

_UPLOAD_ERROR_CONTROLLER_SETUP_JS = r"""    const existing = window.__tacmeUploadError;
    if (existing) {
        existing.setLanguage(language);
        existing.mount();
        return [];
    }

    let currentLanguage = language === "Русский" ? "ru" : "en";
    let errorKind = null;
    let previousFocus = null;
    let errorLayer = null;
    let titleNode = null;
    let messageNode = null;
    let acknowledgeButton = null;
    let closeButton = null;
    let lastServerKind = null;
    let lastBrowserError = null;
    let lastFileAttempt = 0;
    let checkScheduled = false;
    let pendingLayer = null;
    let pendingTimeout = null;
    const locked = new Map();
    const approvedInputs = new WeakSet();
    const approvedDrops = new WeakSet();
    const videoSelector = "#application-video-input";
    const columnSelector = ".application-video-column";
    // Only the server's explicit successful-validation statuses may release
    // Checking video. A stale file reference can exist before rejection lands.
    const validatedStatuses = __TACME_VALIDATED_STATUSES__;
"""

_UPLOAD_ERROR_CONTROLLER_FINISH_JS = r"""    const beginAttempt = () => {
        lastFileAttempt = Date.now();
        lastBrowserError = null;
    };

    const onFileSelected = (event) => {
        const path = event.composedPath?.() ?? [];
        const input = path.find((element) => element?.tagName === "INPUT" && element.type === "file");
        if (!input || !path.some((element) => element?.id === "application-video-input")) return;
        if (approvedInputs.has(input)) {
            approvedInputs.delete(input);
            return; // Safe file was inspected; pass the replayed change to Gradio.
        }
        const file = input.files?.[0];
        if (!file) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        if (errorKind) {
            input.value = "";
            return;
        }
        lastFileAttempt = 0;
        lastBrowserError = null;
        // Cover the video immediately, before asynchronous browser preflight.
        // Gradio sees only the replayed event after the preflight succeeds.
        setPending();
        inspectFile(file).then((kind) => {
            if (kind) {
                input.value = "";
                show(kind);
                return;
            }
            if (!input.isConnected || input.files?.[0] !== file || errorKind) return;
            beginAttempt();
            approvedInputs.add(input);
            input.dispatchEvent(new Event("change", {bubbles: true, composed: true}));
        }).catch(() => {
            input.value = "";
            show("upload");
        });
    };

    const onDrop = (event) => {
        const video = document.querySelector(videoSelector);
        if (!video || !event.composedPath?.().includes(video)) return;
        if (approvedDrops.has(event.target)) {
            approvedDrops.delete(event.target);
            return;
        }
        const file = event.dataTransfer?.files?.[0];
        if (!file) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        if (errorKind) return;
        lastFileAttempt = 0;
        lastBrowserError = null;
        const target = event.target;
        setPending();
        // Retain file data before the native drop event is released.
        inspectFile(file).then((kind) => {
            if (kind) {
                show(kind);
                return;
            }
            if (!target.isConnected || errorKind) return;
            const transfer = new DataTransfer();
            transfer.items.add(file);
            beginAttempt();
            approvedDrops.add(target);
            target.dispatchEvent(new DragEvent("drop", {
                bubbles: true, cancelable: true, composed: true, dataTransfer: transfer
            }));
        }).catch(() => show("upload"));
    };

    const checkNewUploadAlert = (node) => {
        if (!(node instanceof Element)) return;
        if (!lastFileAttempt || Date.now() - lastFileAttempt > uploadErrorWindowMs) return;
        if (node.closest(".tacme-upload-error-layer") || node.matches(".tacme-upload-error-layer")) return;

        // Gradio 6 uses `.toast-body.error`. Restrict this to a recent video
        // upload and known upload/connection-failure messages; unrelated
        // Gradio errors must remain visible for diagnosis.
        const candidates = [node, ...node.querySelectorAll(nativeToastSelector)];
        for (const candidate of candidates) {
            const nativeToast = candidate.matches(nativeToastSelector)
                ? candidate.closest(".toast-body.error") ?? candidate
                : candidate.closest(nativeToastSelector);
            if (!nativeToast || nativeToast.closest(".tacme-upload-error-layer")) continue;
            const content = (nativeToast.textContent ?? "").trim();
            if (content.length > 2000) continue;
            const notPlayable = notPlayableMessage.test(content);
            if (!uploadMessage.test(content) && !uploadConnectionMessage.test(content) && !notPlayable) continue;

            // Replace the duplicate visual notification, not the transport
            // error or its console/server logs. Do not interfere with Gradio's
            // upload request, promise handling, or global error handler.
            nativeToast.style.setProperty("display", "none", "important");
            // The raw AVI/MKV is not the final preview. Wait for the server
            // to validate it and replace it with an HTML5-compatible MP4.
            if (notPlayable && pendingLayer) return;
            if (content !== lastBrowserError) {
                lastBrowserError = content;
                // Preserve a specific server error when the browser reports
                // a late playback error for the already-rejected video.
                if (!errorKind) show(notPlayable ? "playback_unavailable" : "upload");
            }
            return;
        }
    };

    const scan = (records) => {
        if (!document.querySelector(videoSelector)) return;
        checkServerStatus();
        for (const record of records) {
            for (const node of record.addedNodes) checkNewUploadAlert(node);
            // Some Gradio notifications insert their text after the toast node.
            const element = record.target instanceof Element
                ? record.target : record.target.parentElement;
            const toast = element?.closest(nativeToastSelector);
            if (toast) checkNewUploadAlert(toast);
        }
        if (checkScheduled) return;
        checkScheduled = true;
        window.requestAnimationFrame(() => {
            checkScheduled = false;
            if (errorLayer && !errorLayer.isConnected) {
                errorLayer = null;
                if (errorKind) show(errorKind);
            }
            // Gradio may replace video controls while processing an upload.
            // Keep the opaque validation gate anchored to the current column.
            if (pendingLayer && !pendingLayer.isConnected) {
                const column = document.querySelector(columnSelector);
                if (column) column.appendChild(pendingLayer);
            }
            if (errorKind || pendingLayer) lock();
            checkServerStatus();
        });
    };

    const onKeyDown = (event) => {
        if (!errorKind || !errorLayer) return;
        if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            dismiss();
        } else if (event.key === "Tab") {
            const buttons = [closeButton, acknowledgeButton];
            const position = buttons.indexOf(document.activeElement);
            const next = event.shiftKey ? (position <= 0 ? 1 : 0) : (position === 1 ? 0 : 1);
            event.preventDefault();
            buttons[next].focus({preventScroll: true});
        }
    };

    const controller = {
        validated(status, videoPath) {
            if (!pendingLayer) return;
            // This callback runs only after the server-side .change handler.
            // Gradio may also emit an intermediate change with videoPath=null.
            // A generic Ready status is NOT proof of successful validation.
            const message = typeof status === "string" ? status : "";
            for (const [kind, matcher] of serverMessages) {
                if (matcher.test(message)) {
                    show(kind);
                    return;
                }
            }
            checkServerStatus();
            if (errorKind) return;
            // This callback runs after handle_video_change has finished.
            // In the Gradio browser, gr.Video may be a VideoData object
            // ({video: FileData, subtitles: ...}), not the Python str path.
            // Accept only an actual file reference. A rejected/cleared video
            // remains null, so it must never release the validation gate.
            const fileData = typeof videoPath === "string"
                ? videoPath : (videoPath?.video ?? videoPath);
            const hasValidatedVideo = typeof fileData === "string"
                ? fileData.length > 0
                : Boolean(fileData && typeof fileData === "object" && (
                    (typeof fileData.path === "string" && fileData.path.length > 0)
                    || (typeof fileData.url === "string" && fileData.url.length > 0)
                ));
            // Gradio can briefly expose the uploaded file and an older Ready
            // status while its server-side decode check is still resolving.
            // Never treat a non-null video path alone as validation success.
            const explicitlyValidated = validatedStatuses.en.includes(message.trim())
                || validatedStatuses.ru.includes(message.trim());
            if (hasValidatedVideo && explicitlyValidated) {
                clearPending();
                // Only suppress toasts for the original video while its
                // replacement is in flight. New requests have new attempts.
                lastFileAttempt = 0;
            }
        },
        mount() {
            checkServerStatus();
            if (errorKind) {
                if (!errorLayer?.isConnected) show(errorKind);
                lock();
            }
            syncValidationGate();
        },
        setLanguage(value) {
            currentLanguage = value === "Русский" ? "ru" : "en";
            updateText();
            if (pendingLayer) {
                const [title, description] = pendingTexts[currentLanguage];
                pendingLayer.querySelector(".tacme-upload-check-title").textContent = title;
                pendingLayer.querySelector(".tacme-upload-check-description").textContent = description;
            }
            syncValidationGate();
        },
        show,
        dismiss
    };
    window.__tacmeUploadError = controller;
    document.addEventListener("change", onFileSelected, true);
    document.addEventListener("drop", onDrop, true);
    document.addEventListener("keydown", onKeyDown, true);
    new MutationObserver(scan).observe(document.body, {
        subtree: true,
        childList: true,
        characterData: true
    });
    controller.mount();
    return [];
}
"""

UPLOAD_ERROR_UI_JS = (
    (
        UPLOAD_ERROR_MESSAGES_JS
        + _UPLOAD_ERROR_CONTROLLER_SETUP_JS
        + UPLOAD_ERROR_MATCHERS_JS
        + UPLOAD_ERROR_DIALOG_JS
        + UPLOAD_ERROR_PREFLIGHT_JS
        + _UPLOAD_ERROR_CONTROLLER_FINISH_JS
    )
    .replace(
        "__TACME_IDLE_STATUS_TEXTS__",
        json.dumps(
            {
                "en": get_localized_text("Texts_STATUS_READY", 0),
                "ru": get_localized_text("Texts_STATUS_READY", 1),
            },
            ensure_ascii=False,
        ),
    )
    .replace(
        "__TACME_VALIDATED_STATUSES__",
        json.dumps(
            {
                "en": [
                    get_localized_text("Texts_STATUS_VIDEO_READY", 0),
                    get_localized_text("Texts_STATUS_NO_TASKS", 0),
                ],
                "ru": [
                    get_localized_text("Texts_STATUS_VIDEO_READY", 1),
                    get_localized_text("Texts_STATUS_NO_TASKS", 1),
                ],
            },
            ensure_ascii=False,
        ),
    )
)

__all__ = ["UPLOAD_ERROR_UI_JS"]
