"""
File: client.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Unchanged attribution links and Plotly resize JavaScript.

License: MIT License
"""

FEATURE_ATTRIBUTION_JS = r"""
() => {
    if (window.__tacmeFeatureAttributionBound) {
        return [];
    }

    window.__tacmeFeatureAttributionBound = true;

    const clickRadioValue = (rootId, value) => {
        const root = document.getElementById(rootId);

        if (!root) {
            return false;
        }

        const inputs = Array.from(
            root.querySelectorAll('input[type="radio"]')
        );

        const input = inputs.find((candidate) => (
            candidate.value === value
            || candidate.dataset?.value === value
        ));

        if (!input) {
            return false;
        }

        input.click();
        return true;
    };

    document.addEventListener("click", (event) => {
        const trigger = event.target?.closest?.(
            ".application-attribution-link"
        );

        if (!trigger) {
            return;
        }

        const task = trigger.dataset.attributionTask;
        const target = trigger.dataset.attributionTarget;

        if (!task || !target) {
            return;
        }

        event.preventDefault();

        clickRadioValue(
            "application-feature-attribution-task",
            task
        );

        let attempts = 0;

        const selectTarget = () => {
            attempts += 1;

            if (clickRadioValue(
                "application-feature-attribution-target",
                target
            )) {
                const container = document.getElementById(
                    "application-feature-attribution-container"
                );

                container?.scrollIntoView({
                    behavior: "smooth",
                    block: "start",
                });

                return;
            }

            if (attempts < 24) {
                window.setTimeout(selectTarget, 60);
            }
        };

        window.setTimeout(selectTarget, 20);
    });

    return [];
}
"""


FEATURE_ATTRIBUTION_RESIZE_JS = r"""
() => {
    const plotRootId = "application-feature-attribution-plot";

    const textColor = () => {
        const scope = document.querySelector(".gradio-container") || document.body;
        return getComputedStyle(scope).color || "rgb(244, 244, 245)";
    };

    const isDark = (color) => {
        const rgb = color.match(/rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i);
        if (rgb) {
            const luminance =
                0.2126 * Number(rgb[1]) +
                0.7152 * Number(rgb[2]) +
                0.0722 * Number(rgb[3]);
            return luminance > 128;
        }

        const srgb = color.match(/color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/i);
        if (srgb) {
            const luminance = 255 * (
                0.2126 * Number(srgb[1]) +
                0.7152 * Number(srgb[2]) +
                0.0722 * Number(srgb[3])
            );
            return luminance > 128;
        }

        return document.documentElement.classList.contains("dark") ||
            window.matchMedia("(prefers-color-scheme: dark)").matches;
    };

    const addAxisExplanation = (plot) => {
        const hint = plot?.layout?.meta?.tacme_feature_bins_hint;
        const axisTitle = plot?.querySelector(".xtitle");
        if (!hint || !axisTitle) {
            return;
        }

        let title = axisTitle.querySelector("title");
        if (!title) {
            title = document.createElementNS("http://www.w3.org/2000/svg", "title");
            axisTitle.appendChild(title);
        }
        title.textContent = hint;
        axisTitle.style.cursor = "help";
    };

    const resizePlot = (plot) => {
        if (plot && window.Plotly?.Plots?.resize) {
            window.Plotly.Plots.resize(plot);
        }
        addAxisExplanation(plot);
        window.requestAnimationFrame(() => addAxisExplanation(plot));
        window.dispatchEvent(new Event("resize"));
    };

    const synchronizePlot = () => {
        const plot = document.getElementById(plotRootId)?.querySelector(
            ".js-plotly-plot"
        );

        if (!plot) {
            return;
        }

        const foreground = textColor();
        const dark = isDark(foreground);
        const tooltipBackground = dark ? "#27272A" : "#FFFFFF";
        const colorscale = [
            [0.0, "#3B82F6"],
            [0.22, "#93C5FD"],
            [0.5, dark ? "#27272A" : "#F4F4F5"],
            [0.78, "#FDBA74"],
            [1.0, "#F97316"]
        ];

        if (!window.Plotly?.restyle || !window.Plotly?.relayout) {
            // Background and axis text remain theme-aware through CSS.
            resizePlot(plot);
            return;
        }

        const trace = plot.data?.[0];
        const needsScale = trace?.colorscale?.[2]?.[1] !== colorscale[2][1];
        const needsLayout = plot.layout?.font?.color !== foreground ||
            plot.layout?.hoverlabel?.bgcolor !== tooltipBackground;

        if (!needsScale && !needsLayout) {
            resizePlot(plot);
            return;
        }

        const plotly = window.Plotly;
        const traceUpdate = needsScale
            ? plotly.restyle(plot, {
                colorscale: [colorscale],
                "colorbar.tickfont.color": [foreground],
                "colorbar.title.font.color": [foreground]
            }, [0])
            : Promise.resolve();

        Promise.resolve(traceUpdate)
            .then(() => {
                if (!needsLayout) {
                    return undefined;
                }
                return plotly.relayout(plot, {
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    "font.color": foreground,
                    "xaxis.tickfont.color": foreground,
                    "xaxis.title.font.color": foreground,
                    "yaxis.tickfont.color": foreground,
                    "hoverlabel.bgcolor": tooltipBackground,
                    "hoverlabel.font.color": dark ? "#FAFAFA" : "#18181B"
                });
            })
            .then(() => resizePlot(plot))
            .catch((error) => console.warn("TACME Plotly theme synchronization failed:", error));
    };

    const scheduleThemeSync = () => {
        window.requestAnimationFrame(() => {
            window.requestAnimationFrame(synchronizePlot);
        });
    };

    if (!window.__tacmeAttributionThemeObserver) {
        const observer = new MutationObserver(scheduleThemeSync);
        const options = { attributes: true, attributeFilter: ["class", "style", "data-theme"] };
        observer.observe(document.documentElement, options);
        if (document.body) {
            observer.observe(document.body, options);
        }
        const gradioRoot = document.querySelector(".gradio-container");
        if (gradioRoot) {
            observer.observe(gradioRoot, options);
        }
        window.matchMedia("(prefers-color-scheme: dark)")
            .addEventListener("change", scheduleThemeSync);
        window.__tacmeAttributionThemeObserver = observer;
    }

    scheduleThemeSync();
    window.setTimeout(synchronizePlot, 80);
    window.setTimeout(synchronizePlot, 220);
    return [];
}
"""
