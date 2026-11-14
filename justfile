app_port := "7860"

css_source := "styles/app.css"

css_output := "app.css"

model_source := "checkpoints/tacme.pt"

model_output := "models/tacme.safetensors"


default:
    @just --list


_runtime executable *args:
    #!/usr/bin/env bash
    set -euo pipefail

    if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        FFMPEG_PREFIX="$(brew --prefix ffmpeg)"

        export DYLD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

        ".venv/bin/{{executable}}" {{args}}
    else
        uv run {{executable}} {{args}}
    fi


sync:
    uv sync


sync-node:
    npm ci


setup: sync sync-node images-build css-build model-ensure


lint:
    uv run ruff check .


format-python:
    uv run ruff format .


format-css:
    npm run lint:css:fix


format-web:
    npm exec -- prettier --write .


format-toml:
    npm exec -- taplo fmt


format-readme:
    uv run python -m scripts.update_readme


format: format-python format-css format-web format-toml format-readme


images-build:
    npm run build:images


images-build-check:
    npm run check:images


images-ensure:
    @if ! npm run check:images >/dev/null 2>&1; then \
        echo "Optimized images are missing or out of date. Rebuilding..."; \
        npm run build:images; \
    fi


css-build:
    npm run build:css


css-build-check:
    @tmp="$(mktemp "${TMPDIR:-/tmp}/tacme-css.XXXXXX")"; \
    trap 'rm -f "$tmp"' EXIT; \
    npm exec -- lightningcss \
        --bundle \
        --minify \
        --browserslist \
        "{{css_source}}" \
        -o "$tmp"; \
    if ! cmp -s "$tmp" "{{css_output}}"; then \
        echo "Generated {{css_output}} is out of date."; \
        echo "Run: just css-build"; \
        exit 1; \
    fi


model-build:
    uv run python -m scripts.ensure_model \
        --source "{{model_source}}" \
        --destination "{{model_output}}" \
        --force


model-build-check:
    uv run python -m scripts.ensure_model \
        --source "{{model_source}}" \
        --destination "{{model_output}}" \
        --check


model-ensure:
    uv run python -m scripts.ensure_model \
        --source "{{model_source}}" \
        --destination "{{model_output}}"


model-runtime-check:
    uv run python -m scripts.check_model


audio-extractor-check:
    uv run python -m scripts.check_audio_extractor


text-extractor-check:
    uv run python -m scripts.check_text_extractor


vision-extractor-check:
    uv run python -m scripts.check_vision_extractor


extractors-check:
    uv run python -m scripts.check_extractors


aggregation-check:
    uv run python -m scripts.check_aggregation


transcription-check:
    uv run python -m scripts.check_transcription


behavior-check:
    @just _runtime python -m scripts.check_behavior


pipeline-check:
    @just _runtime python -m scripts.check_pipeline

inference-check:
    uv run python -m scripts.check_inference

typecheck:
    uv run mypy


deptry:
    uv run deptry .


check-python:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run deptry .


check-css:
    npm run lint:css


check-web:
    npm exec -- prettier --check .


check-toml:
    npm run lint:toml
    npm exec -- taplo fmt --check


check-readme:
    uv run python -m scripts.update_readme --check


check:
    @just check-python
    @just check-css
    @just check-web
    @just check-toml
    @just check-readme
    @just images-build-check
    @just css-build-check
    @just model-build-check


runtime-check:
    @just inference-check
    @just extractors-check
    @just aggregation-check
    @just transcription-check
    @just behavior-check
    @just model-runtime-check


check-all:
    @just check
    @just runtime-check
    @just pipeline-check


fix:
    uv run ruff check . --fix
    @just format
    @just images-build
    @just css-build
    @just model-ensure


stop:
    @pid="$(lsof -tiTCP:{{app_port}} -sTCP:LISTEN 2>/dev/null || true)"; \
    if [ -n "$pid" ]; then \
        echo "Stopping process on port {{app_port}}: $pid"; \
        kill "$pid" 2>/dev/null || true; \
        for _ in 1 2 3 4 5; do \
            if ! lsof -tiTCP:{{app_port}} -sTCP:LISTEN >/dev/null 2>&1; then \
                break; \
            fi; \
            sleep 0.2; \
        done; \
        pid="$(lsof -tiTCP:{{app_port}} -sTCP:LISTEN 2>/dev/null || true)"; \
        if [ -n "$pid" ]; then \
            echo "Force stopping process on port {{app_port}}: $pid"; \
            kill -9 "$pid" 2>/dev/null || true; \
        fi; \
    fi


run: images-ensure css-build model-ensure stop
    @just _runtime python app.py


dev: images-ensure css-build model-ensure stop
    @just _runtime gradio app.py