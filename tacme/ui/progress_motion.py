"""Client-side presentation enhancements for the TACME progress dialogs."""

PROGRESS_MOTION_JS = r"""
() => {
    if (window.__tacmeProgressMotionBound) {
        return [];
    }
    window.__tacmeProgressMotionBound = true;

    let activeOverlay = null;
    let overlaySnapshotAt = 0;
    // Gradio can patch data-* values on the SAME DOM element without replacing it.
    // Re-anchor the local clock whenever a new authoritative server snapshot arrives.
    let lastElapsedBase = null;
    let lastActiveStage = null;
    let lastActiveBase = null;
    let previousStageStates = new Map();
    let wasDetailsOpen = false;
    let scrollLocked = false;
    let closingDetails = false;

    const reducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    );
    const detailsSelector = "#application-processing-details-modal";
    const closeSelector = "#application-processing-details-close";
    const buttonSelector = "#application-processing-details-button";

    const isVisible = (element) => !!(
        element &&
        !element.classList.contains("hide") &&
        !element.classList.contains("hidden") &&
        window.getComputedStyle(element).display !== "none" &&
        element.getBoundingClientRect().width > 0
    );

    const closeButton = () => {
        const root = document.querySelector(closeSelector);
        return root?.matches("button") ? root : root?.querySelector("button");
    };

    const detailsTrigger = () => {
        const root = document.querySelector(buttonSelector);
        return root?.matches("button") ? root : root?.querySelector("button");
    };

    // Use the same 360/240ms button-origin motion as Media information.
    // Center-to-center viewport coordinates remain correct after scrolling.
    const positionDetailsOrigin = (modal) => {
        const card = modal.querySelector(".application-processing-details-card");
        const trigger = detailsTrigger();
        if (!card || !isVisible(trigger)) {
            modal.style.setProperty("--tacme-media-modal-dx", "0px");
            modal.style.setProperty("--tacme-media-modal-dy", "20px");
            return;
        }
        const source = trigger.getBoundingClientRect();
        const target = card.getBoundingClientRect();
        const dx = source.left + source.width / 2 - target.left - target.width / 2;
        const dy = source.top + source.height / 2 - target.top - target.height / 2;
        modal.style.setProperty("--tacme-media-modal-dx", `${dx}px`);
        modal.style.setProperty("--tacme-media-modal-dy", `${dy}px`);
    };

    // A single viewport-positioned tooltip for genuinely truncated stage
    // names. Keeping it in document.body prevents clipping by scrollable
    // Gradio dialogs. It is also accessible from the keyboard.
    const stageLabelSelector =
        ".analysis-progress-modal .analysis-stage-name, .analysis-details-stage .analysis-stage-name";
    const stageTooltip = document.createElement("div");
    stageTooltip.className = "tacme-stage-name-tooltip";
    stageTooltip.id = "tacme-stage-name-tooltip";
    stageTooltip.setAttribute("role", "tooltip");
    stageTooltip.hidden = true;
    document.body.appendChild(stageTooltip);
    let tooltipAnchor = null;

    const dismissStageTooltip = () => {
        tooltipAnchor?.removeAttribute("aria-describedby");
        tooltipAnchor = null;
        stageTooltip.hidden = true;
    };

    const markTruncatedStageLabels = () => {
        document.querySelectorAll(stageLabelSelector).forEach((label) => {
            const clipped = label.clientWidth > 0 &&
                label.scrollWidth > label.clientWidth + 1;
            if (clipped !== label.classList.contains("tacme-name-is-truncated")) {
                label.classList.toggle("tacme-name-is-truncated", clipped);
            }
            if (clipped) {
                if (label.tabIndex !== 0) {
                    label.tabIndex = 0;
                }
            } else {
                if (label.hasAttribute("tabindex")) {
                    label.removeAttribute("tabindex");
                }
                if (tooltipAnchor === label) {
                    dismissStageTooltip();
                }
            }
        });
    };

    const showStageTooltip = (label) => {
        if (!label?.classList.contains("tacme-name-is-truncated")) {
            return;
        }
        const content = label.textContent?.trim();
        if (!content) {
            return;
        }
        if (tooltipAnchor && tooltipAnchor !== label) {
            dismissStageTooltip();
        }
        tooltipAnchor = label;
        stageTooltip.textContent = content;
        stageTooltip.hidden = false;
        label.setAttribute("aria-describedby", stageTooltip.id);

        const rect = label.getBoundingClientRect();
        const tip = stageTooltip.getBoundingClientRect();
        const gap = 11;
        const edge = 12;
        const below = rect.top < tip.height + gap + edge &&
            window.innerHeight - rect.bottom > tip.height + gap + edge;
        const left = Math.max(
            edge,
            Math.min(
                window.innerWidth - tip.width - edge,
                rect.left + rect.width / 2 - tip.width / 2
            )
        );
        const top = below ? rect.bottom + gap : rect.top - tip.height - gap;
        const arrowOffset = Math.max(
            14, Math.min(tip.width - 14, rect.left + rect.width / 2 - left)
        );
        stageTooltip.dataset.side = below ? "bottom" : "top";
        stageTooltip.style.left = `${left}px`;
        stageTooltip.style.top = `${Math.max(edge, top)}px`;
        stageTooltip.style.setProperty("--tacme-tooltip-arrow", `${arrowOffset}px`);
    };

    document.addEventListener("click", (event) => {
        // Touch users can tap a truncated name, then tap outside to dismiss.
        if (!window.matchMedia("(hover: none)").matches) {
            return;
        }
        const label = event.target instanceof Element
            ? event.target.closest(stageLabelSelector) : null;
        if (label?.classList.contains("tacme-name-is-truncated")) {
            showStageTooltip(label);
        } else {
            dismissStageTooltip();
        }
    });
    document.addEventListener("pointerover", (event) => {
        if (event.pointerType === "touch") {
            return;
        }
        const label = event.target instanceof Element
            ? event.target.closest(stageLabelSelector) : null;
        if (label) {
            showStageTooltip(label);
        }
    });
    document.addEventListener("pointerout", (event) => {
        if (event.pointerType === "touch") {
            return;
        }
        if (tooltipAnchor && event.target === tooltipAnchor &&
            !tooltipAnchor.contains(event.relatedTarget)) {
            dismissStageTooltip();
        }
    });
    document.addEventListener("focusin", (event) => {
        if (event.target instanceof Element &&
            event.target.matches(stageLabelSelector)) {
            showStageTooltip(event.target);
        }
    });
    document.addEventListener("focusout", (event) => {
        if (event.target === tooltipAnchor) {
            dismissStageTooltip();
        }
    });
    document.addEventListener("scroll", dismissStageTooltip, true);
    window.addEventListener("resize", () => {
        dismissStageTooltip();
        markTruncatedStageLabels();
    });

    const formatSeconds = (seconds, language) => {
        const number = Math.max(0, seconds).toFixed(1);
        return `${language === "1" ? number.replace(".", ",") : number} ${language === "1" ? "\u0441" : "s"}`;
    };

    // The live timer sits inside the progress track. Build it client-side so
    // the Gradio progress HTML and server-side analysis code need no changes.
    // This is safe when Gradio replaces either the entire overlay or its track.
    const ensureTrackTimer = (overlay) => {
        const track = overlay.querySelector(".analysis-busy-progress-track");
        if (!track) {
            return null;
        }
        let value = track.querySelector(".analysis-progress-timer-value");
        if (!value) {
            const clock = document.createElement("span");
            clock.className = "analysis-progress-timer-icon";
            clock.setAttribute("aria-hidden", "true");
            value = document.createElement("span");
            value.className = "analysis-progress-timer-value";
            const badge = document.createElement("span");
            badge.className = "analysis-progress-track-timer";
            badge.setAttribute("role", "timer");
            badge.setAttribute("aria-live", "off");
            badge.append(clock, value);
            track.appendChild(badge);
        }
        return value;
    };

    const updateLiveClocks = (overlay) => {
        const elapsedBase = Number(overlay.dataset.elapsedBase) || 0;
        const delta = overlay.dataset.finished === "true"
            ? 0
            : Math.max(0, (performance.now() - overlaySnapshotAt) / 1000);
        const lang = overlay.dataset.language;
        const formatted = formatSeconds(elapsedBase + delta, lang);
        // Keep the original clock (used by existing progress markup) in sync,
        // even though its redundant facts row is visually hidden.
        const total = overlay.querySelector(".analysis-live-total");
        if (total && total.textContent !== formatted) {
            total.textContent = formatted;
        }
        const trackTime = ensureTrackTimer(overlay);
        if (trackTime && trackTime.textContent !== formatted) {
            trackTime.textContent = formatted;
        }
        if (overlay.dataset.finished !== "true") {
            const stage = overlay.querySelector(".analysis-live-stage");
            if (stage) {
                const base = Number(stage.dataset.base) || 0;
                const stageText = formatSeconds(base + delta, lang);
                if (stage.textContent !== stageText) {
                    stage.textContent = stageText;
                }
            }
        }
    };

    // Reference durations from a representative MPS run, in canonical stage
    // order. These are estimates, NOT percentages of completed operations.
    // Once a full run completes, the next run uses its observed durations.
    const defaultStageSeconds = [
        0.02, 0.96, 0.35, 0.31, 0.04, 0.39,
        0.02, 8.94, 0.05, 0.02, 0.21, 0.02,
    ];
    let learnedStageSeconds = null;
    let learnedFromThisRun = false;
    let lastServerPercent = null;
    let displayedProgress = null;
    let targetProgress = 0;
    let frameHandle = null;
    let previousFrameAt = null;

    const clamp = (value, lower, upper) => Math.min(upper, Math.max(lower, value));

    const stageDuration = (stage) => {
        if (!stage.classList.contains("analysis-stage-is-completed")) {
            return null;
        }
        const durationElement = stage.querySelector(".analysis-stage-time");
        if (!durationElement) {
            return null;
        }
        const durationText = durationElement.textContent?.trim().replace(",", ".") ?? "";
        const seconds = Number.parseFloat(durationText);
        return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
    };

    const estimatePercent = (overlay) => {
        const stages = Array.from(overlay.querySelectorAll(
            ".analysis-progress-stages .analysis-stage[data-stage]"
        ));
        if (!stages.length) {
            return {percent: 0, completed: 0, count: 0, finished: false};
        }

        const completed = stages.filter((stage) =>
            stage.classList.contains("analysis-stage-is-completed")
        ).length;
        const finished = completed === stages.length;
        const observed = stages.map(stageDuration);
        const reference = learnedStageSeconds?.length === stages.length
            ? learnedStageSeconds
            : stages.map((_, index) => defaultStageSeconds[index] ?? 0.1);

        if (finished) {
            if (!learnedFromThisRun && observed.every((duration) => duration !== null)) {
                const next = observed.map((duration) => Math.max(0.02, duration));
                learnedStageSeconds = learnedStageSeconds?.length === next.length
                    ? next.map((duration, index) =>
                        0.65 * duration + 0.35 * learnedStageSeconds[index]
                    )
                    : next;
                learnedFromThisRun = true;
            }
            return {percent: 100, completed, count: stages.length, finished: true};
        }

        // Calibrate predictions only against already measured, non-trivial
        // stages. Displayed progress remains monotone even if this estimate
        // changes when a Gradio stage snapshot arrives.
        let observedTotal = 0;
        let expectedTotal = 0;
        for (let index = 0; index < stages.length; index += 1) {
            if (observed[index] !== null && reference[index] >= 0.1) {
                observedTotal += observed[index];
                expectedTotal += reference[index];
            }
        }
        const scale = expectedTotal >= 0.1
            ? clamp(observedTotal / expectedTotal, 0.7, 2.5)
            : 1;

        const activeIndex = stages.findIndex((stage) =>
            stage.classList.contains("analysis-stage-is-started")
        );
        const activeTimer = activeIndex >= 0
            ? stages[activeIndex].querySelector(".analysis-live-stage")
            : null;
        const elapsedSinceSnapshot = Math.max(
            0, (performance.now() - overlaySnapshotAt) / 1000
        );
        const activeElapsed = activeTimer
            ? (Number(activeTimer.dataset.base) || 0) + elapsedSinceSnapshot
            : 0;

        let elapsed = 0;
        let remaining = 0;
        for (let index = 0; index < stages.length; index += 1) {
            if (observed[index] !== null) {
                elapsed += observed[index];
                continue;
            }
            const expected = Math.max(0.02, reference[index] * scale);
            if (index === activeIndex) {
                elapsed += activeElapsed;
                // When a stage overruns its estimate, expand the projected
                // total rather than stopping at an artificial per-stage cap.
                // Keep a small positive remaining allowance so an unfinished
                // stage never makes the progress bar report 100%.
                remaining += Math.max(expected - activeElapsed, expected * 0.075);
            } else {
                remaining += expected;
            }
        }

        const denominator = elapsed + remaining;
        return {
            percent: denominator > 0 ? clamp(100 * elapsed / denominator, 0, 99.99) : 0,
            completed,
            count: stages.length,
            finished: false,
        };
    };

    // Draw at the browser's refresh cadence. Changing a CSS width target
    // every 100ms while CSS runs a 420ms transition caused the visible
    // stair-steps and lag in the previous version.
    const drawProgressFrame = (now) => {
        frameHandle = null;
        const overlay = activeOverlay;
        if (!overlay?.isConnected) {
            previousFrameAt = null;
            return;
        }
        const progress = estimatePercent(overlay);
        if (progress.count) {
            targetProgress = Math.max(targetProgress, progress.percent);
            if (displayedProgress === null) {
                displayedProgress = targetProgress;
            }
            const deltaSeconds = previousFrameAt === null
                ? 1 / 60
                : clamp((now - previousFrameAt) / 1000, 0, 0.06);
            const responseSeconds = progress.finished ? 0.18 : 0.21;
            const smoothing = 1 - Math.exp(-deltaSeconds / responseSeconds);
            const available = Math.max(0, targetProgress - displayedProgress);
            const speedLimit = (progress.finished ? 55 : 25) * deltaSeconds;
            displayedProgress += reducedMotion.matches
                ? available
                : Math.min(available * smoothing, speedLimit);
            if (progress.finished && targetProgress - displayedProgress < 0.015) {
                displayedProgress = 100;
            }

            const bar = overlay.querySelector(".analysis-busy-progress-bar");
            if (bar) {
                // Override only the animation of the live progress fill.
                // The rest of the existing CSS (colors, glow, shine) remains.
                if (bar.style.transition !== "none") {
                    bar.style.transition = "none";
                }
                const width = `${displayedProgress.toFixed(4)}%`;
                if (bar.style.width !== width) {
                    bar.style.width = width;
                }
            }

            const shownPercent = progress.finished
                ? Math.min(100, Math.round(displayedProgress))
                : Math.min(99, Math.round(displayedProgress));
            const status = `${progress.completed}/${progress.count} \u00b7 ${shownPercent}%`;
            const counter = overlay.querySelector(".analysis-progress-counter");
            if (counter && counter.textContent !== status) {
                counter.textContent = status;
            }
            const track = overlay.querySelector(".analysis-busy-progress-track");
            if (track) {
                track.setAttribute("role", "progressbar");
                if (track.getAttribute("aria-valuemin") !== "0") {
                    track.setAttribute("aria-valuemin", "0");
                }
                if (track.getAttribute("aria-valuemax") !== "100") {
                    track.setAttribute("aria-valuemax", "100");
                }
                if (track.getAttribute("aria-valuenow") !== String(shownPercent)) {
                    track.setAttribute("aria-valuenow", String(shownPercent));
                }
            }
        }
        previousFrameAt = now;
        frameHandle = window.requestAnimationFrame(drawProgressFrame);
    };

    const ensureProgressFrame = () => {
        if (frameHandle === null) {
            frameHandle = window.requestAnimationFrame(drawProgressFrame);
        }
    };

    const sync = () => {
        const overlay = document.querySelector(".analysis-progress-modal");
        const modal = document.querySelector(detailsSelector);
        const detailsOpen = isVisible(modal);
        const overlayChanged = overlay !== activeOverlay;

        if (overlay) {
            const serverPercent = Number(overlay.dataset.progressPercent) || 0;
            if (lastServerPercent !== null && serverPercent < lastServerPercent - 1) {
                // A new run can replace an existing overlay without an empty
                // intermediate DOM frame. Forget the preceding visual progress.
                displayedProgress = null;
                targetProgress = 0;
                previousFrameAt = null;
                learnedFromThisRun = false;
                previousStageStates = new Map();
            }
            lastServerPercent = serverPercent;
            if (overlayChanged && displayedProgress === null && !reducedMotion.matches) {
                overlay.classList.add("analysis-progress-modal-is-first");
            }

            const currentStates = new Map();
            overlay.querySelectorAll(".analysis-stage[data-stage]").forEach((stage) => {
                const name = stage.dataset.stage;
                const completed = stage.classList.contains("analysis-stage-is-completed");
                const started = stage.classList.contains("analysis-stage-is-started");
                currentStates.set(name, completed ? "completed" : started ? "started" : "pending");
                if (
                    completed && previousStageStates.get(name) === "started" &&
                    !reducedMotion.matches
                ) {
                    stage.classList.add("analysis-stage-just-completed");
                }
            });
            previousStageStates = currentStates;
            activeOverlay = overlay;

            const elapsedBase = Number(overlay.dataset.elapsedBase) || 0;
            const activeStage = overlay.querySelector(
                ".analysis-stage-is-started[data-stage]"
            );
            const activeStageName = activeStage?.dataset.stage ?? null;
            const activeTime = activeStage?.querySelector(".analysis-live-stage");
            const activeBase = activeTime ? Number(activeTime.dataset.base) || 0 : null;

            if (
                overlayChanged || elapsedBase !== lastElapsedBase ||
                activeStageName !== lastActiveStage || activeBase !== lastActiveBase
            ) {
                overlaySnapshotAt = performance.now();
                lastElapsedBase = elapsedBase;
                lastActiveStage = activeStageName;
                lastActiveBase = activeBase;
            }
            updateLiveClocks(overlay);
            if (overlayChanged && displayedProgress !== null) {
                // Gradio replaces gr.HTML nodes. Start the new fill at the
                // current displayed width instead of flashing server progress.
                const bar = overlay.querySelector(".analysis-busy-progress-bar");
                if (bar) {
                    bar.style.transition = "none";
                    bar.style.width = `${displayedProgress.toFixed(4)}%`;
                }
            }
            ensureProgressFrame();
        } else if (activeOverlay) {
            activeOverlay = null;
            lastElapsedBase = null;
            lastActiveStage = null;
            lastActiveBase = null;
            lastServerPercent = null;
            displayedProgress = null;
            targetProgress = 0;
            previousFrameAt = null;
            learnedFromThisRun = false;
            previousStageStates = new Map();
        }

        if (detailsOpen && !wasDetailsOpen) {
            // Measure the final-sized card *before* animating it from the button.
            if (!reducedMotion.matches) {
                modal.classList.remove("tacme-details-is-opening", "tacme-details-is-closing");
                positionDetailsOrigin(modal);
                modal.classList.add("tacme-details-is-opening");
            }
            window.requestAnimationFrame(() => closeButton()?.focus({preventScroll: true}));
        } else if (!detailsOpen && wasDetailsOpen) {
            closingDetails = false;
            // Gradio can keep the hidden DOM node for the next opening.
            modal?.classList.remove("tacme-details-is-opening", "tacme-details-is-closing");
            window.requestAnimationFrame(() => {
                const trigger = detailsTrigger();
                if (isVisible(trigger)) {
                    trigger.focus({preventScroll: true});
                }
            });
        }
        wasDetailsOpen = detailsOpen;
        markTruncatedStageLabels();
        if (tooltipAnchor && (
            !tooltipAnchor.isConnected ||
            !tooltipAnchor.classList.contains("tacme-name-is-truncated") ||
            (!detailsOpen && !overlay)
        )) {
            dismissStageTooltip();
        }

        const shouldLock = !!overlay || detailsOpen;
        if (shouldLock !== scrollLocked) {
            document.body.classList.toggle("tacme-modal-scroll-locked", shouldLock);
            scrollLocked = shouldLock;
        }
    };

    const closeDetails = () => {
        if (closingDetails) {
            return;
        }
        const modal = document.querySelector(detailsSelector);
        const button = closeButton();
        if (!isVisible(modal) || !button) {
            return;
        }
        if (reducedMotion.matches) {
            button.dataset.tacmeAllowClose = "true";
            button.click();
            return;
        }
        closingDetails = true;
        // Refresh the origin if the viewport moved while the report was open.
        // The closing animation then returns to the current trigger position.
        // Remove the opening animation before switching to its reverse.
        modal.classList.remove("tacme-details-is-opening");
        positionDetailsOrigin(modal);
        modal.classList.add("tacme-details-is-closing");
        window.setTimeout(() => {
            button.dataset.tacmeAllowClose = "true";
            button.click();
            // In a failed network request keep the dialog closeable.
            window.setTimeout(() => {
                button.dataset.tacmeAllowClose = "";
                closingDetails = false;
                modal.classList.remove("tacme-details-is-closing");
            }, 800);
        }, 240);
    };

    document.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        const root = target.closest(closeSelector);
        const button = closeButton();
        if (!root || !button || button.dataset.tacmeAllowClose === "true") {
            return;
        }
        // Intercept Gradio's handler to play the 240ms reverse motion first.
        event.preventDefault();
        event.stopImmediatePropagation();
        event.stopPropagation();
        closeDetails();
    }, true);

    document.addEventListener("keydown", (event) => {
        const modal = document.querySelector(detailsSelector);
        if (!isVisible(modal)) {
            return;
        }
        if (event.key === "Escape") {
            event.preventDefault();
            if (tooltipAnchor) {
                dismissStageTooltip();
            } else {
                closeDetails();
            }
        } else if (event.key === "Tab") {
            const buttons = Array.from(
                modal.querySelectorAll("button:not([disabled]), [tabindex='0']")
            ).filter((element) => isVisible(element));
            if (!buttons.length) {
                return;
            }
            const first = buttons[0];
            const last = buttons[buttons.length - 1];
            if (!modal.contains(document.activeElement)) {
                event.preventDefault();
                first.focus();
            } else if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        }
    });

    const observer = new MutationObserver((records) => {
        // Ignore writes made by our animation loop; otherwise each frame
        // triggers a redundant full DOM synchronization.
        if (records.every((record) =>
            record.type === "attributes" &&
            record.attributeName === "style" &&
            record.target instanceof Element &&
            record.target.classList.contains("analysis-busy-progress-bar")
        )) {
            return;
        }
        sync();
    });
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: [
            "class", "style", "data-elapsed-base", "data-base", "data-stage",
            "data-finished", "data-language", "data-progress-percent",
        ],
    });

    window.setInterval(() => {
        const overlay = activeOverlay;
        if (overlay?.isConnected && overlay.dataset.finished !== "true") {
            updateLiveClocks(overlay);
        }
    }, 100);

    sync();
    return [];
}
"""

__all__ = ["PROGRESS_MOTION_JS"]
