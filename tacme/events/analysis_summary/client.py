"""
File: client.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Client-side copy interaction for deterministic TACME analysis summaries.

License: MIT License
"""

from typing import Final

ANALYSIS_SUMMARY_JS: Final = r"""
() => {
    if (window.__tacmeAnalysisSummaryBound) {
        return [];
    }

    window.__tacmeAnalysisSummaryBound = true;

    document.addEventListener("click", async (event) => {
        const button = event.target?.closest?.(
            ".application-analysis-summary-copy"
        );

        if (!button) {
            return;
        }

        const root = button.closest(
            ".application-analysis-summary"
        );
        const source = root?.querySelector(
            ".application-analysis-summary-copy-source"
        );

        if (!source) {
            return;
        }

        const text = source.innerText.trim();

        if (!text) {
            return;
        }

        const copyLabel = button.dataset.copyLabel || "Copy summary";
        const copiedLabel = button.dataset.copiedLabel || "Copied";

        try {
            await navigator.clipboard.writeText(text);
            button.textContent = copiedLabel;
            button.classList.add("is-copied");

            window.setTimeout(() => {
                button.textContent = copyLabel;
                button.classList.remove("is-copied");
            }, 1400);
        } catch (_error) {
            button.textContent = copyLabel;
        }
    });

    return [];
}
"""
