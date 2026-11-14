"""
File: client_scripts.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Client-side JavaScript helpers for TACME Gradio events.

License: MIT License
"""

EXAMPLES_UI_JS = """
(language) => {
    const pagesLabel =
        language === "Русский"
            ? "Страницы:"
            : "Pages:";

    const normalizeExampleName = (value) => {
        if (!value) {
            return "";
        }

        return String(value)
            .normalize("NFKC")
            .toLocaleLowerCase()
            .replace(/[^\\p{L}\\p{N}]+/gu, "");
    };

    const setupExamples = (attempt = 0) => {
        const examples = document.querySelector(
            "#application-examples"
        );

        if (!examples) {
            if (attempt < 40) {
                window.setTimeout(
                    () => setupExamples(attempt + 1),
                    50
                );
            }

            return;
        }

        const syncPaginationLabel = () => {
            const paginations =
                examples.querySelectorAll(
                    "div.paginate"
                );

            paginations.forEach(
                (pagination) => {
                    for (
                        const node
                        of pagination.childNodes
                    ) {
                        if (
                            node.nodeType ===
                                Node.TEXT_NODE &&
                            node.textContent &&
                            node.textContent
                                .trim()
                                .length > 0
                        ) {
                            const expected =
                                `${pagesLabel} `;

                            if (
                                node.textContent !==
                                expected
                            ) {
                                node.textContent =
                                    expected;
                            }

                            break;
                        }
                    }
                }
            );
        };

        const syncSelectedExample = () => {
            const selectedKey =
                window.__tacmeSelectedExampleKey ??
                "";

            const buttons =
                examples.querySelectorAll(
                    "div.gallery > button.gallery-item"
                );

            buttons.forEach((button) => {
                const buttonKey =
                    normalizeExampleName(
                        button.textContent ?? ""
                    );

                const isSelected =
                    selectedKey.length > 0 &&
                    buttonKey === selectedKey;

                button.classList.toggle(
                    "tacme-selected-example",
                    isSelected
                );

                button.setAttribute(
                    "aria-disabled",
                    String(isSelected)
                );
            });
        };

        const selectExample = (button) => {
            window.__tacmeSelectedExampleKey =
                normalizeExampleName(
                    button.textContent ?? ""
                );

            syncSelectedExample();
        };

        if (
            window.__tacmeExamplesRoot &&
            window.__tacmeExamplesRoot !==
                examples &&
            window.__tacmeExamplesClickHandler
        ) {
            window.__tacmeExamplesRoot
                .removeEventListener(
                    "click",
                    window.__tacmeExamplesClickHandler,
                    true
                );
        }

        if (
            window.__tacmeExamplesRoot !==
            examples
        ) {
            const clickHandler = (event) => {
                if (
                    !(
                        event.target
                        instanceof Element
                    )
                ) {
                    return;
                }

                const button =
                    event.target.closest(
                        "button.gallery-item"
                    );

                if (
                    !button ||
                    !examples.contains(button)
                ) {
                    return;
                }

                if (
                    button.classList.contains(
                        "tacme-selected-example"
                    )
                ) {
                    event.preventDefault();
                    event.stopPropagation();
                    event.stopImmediatePropagation();

                    return;
                }

                selectExample(button);
            };

            examples.addEventListener(
                "click",
                clickHandler,
                true
            );

            window.__tacmeExamplesRoot =
                examples;

            window.__tacmeExamplesClickHandler =
                clickHandler;
        }

        const syncExamplesUi = () => {
            syncPaginationLabel();
            syncSelectedExample();
        };

        syncExamplesUi();

        requestAnimationFrame(
            syncExamplesUi
        );

        window.setTimeout(
            syncExamplesUi,
            50
        );

        window.setTimeout(
            syncExamplesUi,
            250
        );

        if (
            window.__tacmeExamplesObserver
        ) {
            window.__tacmeExamplesObserver
                .disconnect();
        }

        const observer =
            new MutationObserver(
                syncExamplesUi
            );

        observer.observe(
            examples,
            {
                childList: true,
                subtree: true,
            }
        );

        window.__tacmeExamplesObserver =
            observer;
    };

    setupExamples();

    return [];
}
"""


CLEAR_EXAMPLE_SELECTION_JS = """
(...args) => {
    window.__tacmeSelectedExampleKey = "";

    document
        .querySelectorAll(
            "#application-examples " +
            "button.gallery-item"
        )
        .forEach((button) => {
            button.classList.remove(
                "tacme-selected-example"
            );

            button.setAttribute(
                "aria-disabled",
                "false"
            );
        });

    return args;
}
"""

# Reset the class left by an earlier close animation before Gradio makes
# the modal visible again. The normal opening animation still runs in .then().
MEDIA_INFO_MODAL_PREPARE_OPEN_JS = """
(...args) => {
    const modal = document.getElementById("media-info-modal");
    modal?.classList.remove(
        "tacme-media-modal-closing",
        "tacme-media-modal-opening"
    );
    return args;
}
"""


MEDIA_INFO_MODAL_OPEN_ANIMATION_JS = """
() => {
    const modal =
        document.getElementById(
            "media-info-modal"
        );

    const trigger =
        document.getElementById(
            "media-info-button"
        );

    const card =
        modal?.querySelector(
            ".media-info-modal-card"
        );

    if (
        !modal ||
        !trigger ||
        !card
    ) {
        return [];
    }

    const triggerRect =
        trigger.getBoundingClientRect();

    const cardRect =
        card.getBoundingClientRect();

    const triggerCenterX =
        triggerRect.left +
        triggerRect.width / 2;

    const triggerCenterY =
        triggerRect.top +
        triggerRect.height / 2;

    const cardCenterX =
        cardRect.left +
        cardRect.width / 2;

    const cardCenterY =
        cardRect.top +
        cardRect.height / 2;

    modal.style.setProperty(
        "--tacme-media-modal-dx",
        `${triggerCenterX - cardCenterX}px`
    );

    modal.style.setProperty(
        "--tacme-media-modal-dy",
        `${triggerCenterY - cardCenterY}px`
    );

    modal.classList.remove(
        "tacme-media-modal-closing"
    );

    modal.classList.remove(
        "tacme-media-modal-opening"
    );

    void modal.offsetWidth;

    modal.classList.add(
        "tacme-media-modal-opening"
    );

    window.setTimeout(
        () => {
            modal.classList.remove(
                "tacme-media-modal-opening"
            );
        },
        380
    );

    return [];
}
"""


MEDIA_INFO_MODAL_CLOSE_ANIMATION_JS = """
(...args) => {
    const modal =
        document.getElementById(
            "media-info-modal"
        );

    const trigger =
        document.getElementById(
            "media-info-button"
        );

    const card =
        modal?.querySelector(
            ".media-info-modal-card"
        );

    if (
        !modal ||
        !trigger ||
        !card
    ) {
        return args;
    }

    const triggerRect =
        trigger.getBoundingClientRect();

    const cardRect =
        card.getBoundingClientRect();

    const triggerCenterX =
        triggerRect.left +
        triggerRect.width / 2;

    const triggerCenterY =
        triggerRect.top +
        triggerRect.height / 2;

    const cardCenterX =
        cardRect.left +
        cardRect.width / 2;

    const cardCenterY =
        cardRect.top +
        cardRect.height / 2;

    modal.style.setProperty(
        "--tacme-media-modal-dx",
        `${triggerCenterX - cardCenterX}px`
    );

    modal.style.setProperty(
        "--tacme-media-modal-dy",
        `${triggerCenterY - cardCenterY}px`
    );

    modal.classList.remove(
        "tacme-media-modal-opening"
    );

    modal.classList.add(
        "tacme-media-modal-closing"
    );

    return new Promise(
        (resolve) => {
            window.setTimeout(
                () => resolve(args),
                240
            );
        }
    );
}
"""
