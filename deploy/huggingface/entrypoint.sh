#!/bin/sh
set -eu

# Download only the TACME runtime checkpoint. All other pretrained encoders
# are resolved by Transformers from their upstream Hugging Face repositories.
MODEL_PATH="/app/models/tacme.safetensors"
MODEL_REPO_ID="${TACME_MODEL_REPO_ID:-DmitryRyumin/TACME}"

if [ ! -s "$MODEL_PATH" ]; then
    mkdir -p /app/models
    echo "Downloading TACME runtime model from ${MODEL_REPO_ID}..."

    # For public models no token is required. For private models, the Space
    # secret HF_TOKEN is read automatically by the Hugging Face CLI.
    hf download "$MODEL_REPO_ID" tacme.safetensors \
        --local-dir /app/models
fi

if [ ! -s "$MODEL_PATH" ]; then
    echo >&2 "TACME deployment: models/tacme.safetensors was not downloaded."
    exit 1
fi

echo "TACME checkpoint found; starting CPU-only Gradio app."
export CUDA_VISIBLE_DEVICES=-1
exec python app.py
