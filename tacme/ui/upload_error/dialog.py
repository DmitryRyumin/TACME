"""Validation gate, animated checking panel, and error dialog behavior."""

UPLOAD_ERROR_DIALOG_JS = r"""    // Lock controls while validating, but keep a single neutral status below.
    // After rejection, retain the neutral label even when the dialog closes.
    const syncValidationGate = () => {
        const active = Boolean(pendingLayer || errorKind);
        const showIdle = active || keepIdleStatus;
        const root = document.documentElement;
        root.classList.toggle("tacme-upload-validation-gated", active);
        root.classList.toggle("tacme-upload-status-idle", showIdle);
        if (showIdle) {
            root.style.setProperty("--tacme-upload-idle-label", JSON.stringify(idleStatusTexts[currentLanguage]));
        } else {
            root.style.removeProperty("--tacme-upload-idle-label");
        }
    };

    const clearPending = (releaseControls = true) => {
        const wasPending = Boolean(pendingLayer);
        window.clearTimeout(pendingTimeout);
        pendingTimeout = null;
        pendingLayer?.remove();
        pendingLayer = null;
        if (releaseControls && !errorKind) {
            unlock();
            // Only a successfully validated video can replace Ready below.
            if (wasPending) keepIdleStatus = false;
        }
        syncValidationGate();
    };

    const setPending = () => {
        clearPending();
        if (errorKind) return;
        const column = document.querySelector(columnSelector);
        if (!column) return;
        pendingLayer = document.createElement("div");
        pendingLayer.className = "tacme-upload-error-layer tacme-upload-check-layer";
        pendingLayer.setAttribute("aria-live", "polite");
        pendingLayer.innerHTML = `
            <div class="tacme-upload-error-panel tacme-upload-check-panel" role="status">
                <div class="tacme-upload-check-emblem" aria-hidden="true">
                    <span class="tacme-upload-check-orbit"></span>
                    <span class="tacme-upload-error-symbol"></span>
                </div>
                <div class="tacme-upload-check-copy">
                    <strong class="tacme-upload-check-title"></strong>
                    <p class="tacme-upload-check-description"></p>
                    <div class="tacme-upload-check-track" aria-hidden="true">
                        <span class="tacme-upload-check-scan"></span>
                    </div>
                </div>
            </div>`;
        column.appendChild(pendingLayer);
        syncValidationGate();
        const [title, description] = pendingTexts[currentLanguage];
        pendingLayer.querySelector(".tacme-upload-check-title").textContent = title;
        pendingLayer.querySelector(".tacme-upload-check-description").textContent = description;
        lock();
        // Fail closed on an interrupted validation. Never expose an unverified
        // Gradio video just because the validation request took too long.
        pendingTimeout = window.setTimeout(() => {
            if (pendingLayer && !errorKind) show("validation_timeout");
        }, 90000);
    };

    const updateText = () => {
        if (!errorKind || !errorLayer) return;
        const [title, message] = locale()[errorKind];
        titleNode.textContent = title;
        messageNode.textContent = message;
        acknowledgeButton.textContent = locale().acknowledge;
        closeButton.setAttribute("aria-label", locale().close);
        errorLayer.setAttribute("lang", currentLanguage);
    };

    const unlock = () => {
        for (const [element, originallyInert] of locked) {
            if (element.isConnected && !originallyInert) {
                element.removeAttribute("inert");
            }
        }
        locked.clear();
    };

    const lock = () => {
        const elements = [
            document.querySelector(videoSelector),
            document.querySelector("#application-examples"),
            document.querySelector("#run-button"),
            document.querySelector("#clear-button")
        ];
        for (const element of elements) {
            if (!element || locked.has(element)) continue;
            locked.set(element, element.hasAttribute("inert"));
            element.setAttribute("inert", "");
        }
    };

    const dismiss = () => {
        if (!errorKind) return;
        errorKind = null;
        lastFileAttempt = 0; // The rejected upload must not trigger a late playback alert.
        lastBrowserError = null;
        errorLayer?.remove();
        errorLayer = null;
        titleNode = null;
        messageNode = null;
        acknowledgeButton = null;
        closeButton = null;
        unlock();
        syncValidationGate();

        // Reuse the existing Gradio Clear event and its dependent callbacks.
        // Closing an error must also discard the rejected video, its filename,
        // media metadata, task selection and every cached analysis result.
        // The button may already be disabled when browser preflight rejected
        // a file before Gradio received it; in that case nothing was loaded.
        const clearControl = document.querySelector("#clear-button");
        const clearButton = clearControl?.matches("button")
            ? clearControl : clearControl?.querySelector("button");
        if (clearButton && !clearButton.disabled) {
            previousFocus = null;
            clearButton.click();
            return;
        }

        if (previousFocus?.isConnected && typeof previousFocus.focus === "function") {
            previousFocus.focus({preventScroll: true});
        }
        previousFocus = null;
    };

    const show = (kind) => {
        if (!texts.en[kind]) return;
        clearPending(false);
        const column = document.querySelector(columnSelector);
        if (!column) return;
        if (errorLayer && errorLayer.parentElement !== column) {
            errorLayer.remove();
            errorLayer = null;
        }
        if (!errorLayer) {
            previousFocus = document.activeElement;
            errorLayer = document.createElement("div");
            errorLayer.className = "tacme-upload-error-layer";
            errorLayer.innerHTML = `
                <div class="tacme-upload-error-panel" role="alertdialog"
                    aria-modal="true" aria-labelledby="tacme-upload-error-title"
                    aria-describedby="tacme-upload-error-description">
                    <div class="tacme-upload-error-heading">
                        <span class="tacme-upload-error-symbol" aria-hidden="true"></span>
                        <strong id="tacme-upload-error-title"></strong>
                        <button type="button" class="tacme-upload-error-close"></button>
                    </div>
                    <p id="tacme-upload-error-description"></p>
                    <div class="tacme-upload-error-actions">
                        <button type="button" class="tacme-upload-error-acknowledge"></button>
                    </div>
                </div>`;
            column.appendChild(errorLayer);
            titleNode = errorLayer.querySelector("#tacme-upload-error-title");
            messageNode = errorLayer.querySelector("#tacme-upload-error-description");
            acknowledgeButton = errorLayer.querySelector(".tacme-upload-error-acknowledge");
            closeButton = errorLayer.querySelector(".tacme-upload-error-close");
            acknowledgeButton.addEventListener("click", dismiss);
            closeButton.addEventListener("click", dismiss);
        }
        errorKind = kind;
        keepIdleStatus = true;
        // Distinct Tabler icons for empty, damaged, and access-related files.
        const symbol = errorLayer.querySelector(".tacme-upload-error-symbol");
        if (symbol) {
            const iconName = kind === "empty_file" ? "file-x"
                : (kind === "invalid_container" || kind === "probe_unavailable" || kind === "truncated_file" || kind === "validation_timeout") ? "file-info"
                : (kind === "undecodable_video" || kind === "corrupted_frames" || kind === "playback_unavailable" || kind === "preview_unavailable" || kind === "decoder_unavailable") ? "video-off"
                : null;
            if (iconName) {
                const icon = `url("/gradio_api/file=static/images/icons/${iconName}.svg")`;
                symbol.style.setProperty("mask-image", icon);
                symbol.style.setProperty("-webkit-mask-image", icon);
            } else {
                symbol.style.removeProperty("mask-image");
                symbol.style.removeProperty("-webkit-mask-image");
            }
        }
        updateText();
        lock();
        syncValidationGate();
        window.requestAnimationFrame(() => acknowledgeButton?.focus({preventScroll: true}));
    };

    const checkServerStatus = () => {
        const status = document.querySelector("div.block.application-status") ??
            document.querySelector(".application-status");
        const value = status?.textContent ?? "";
        let detected = null;
        for (const [kind, matcher] of serverMessages) {
            if (matcher.test(value)) {
                detected = kind;
                break;
            }
        }
        if (detected !== lastServerKind) {
            lastServerKind = detected;
            if (detected) show(detected);
        }
    };

"""

__all__ = ["UPLOAD_ERROR_DIALOG_JS"]
