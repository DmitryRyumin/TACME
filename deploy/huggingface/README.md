---
title: TACME
emoji: 🎭
colorFrom: orange
colorTo: gray
sdk: docker
app_port: 7860
fullWidth: true
short_description: Multimodal emotion, personality, and ambivalence analysis.
license: mit
---

# TACME

Task-Aware Cross-Modal Experts for Disjoint-Corpus Multimodal Behavioral Understanding.

This CPU-only Docker Space is deployed automatically from the
[GitHub source repository](https://github.com/DmitryRyumin/TACME).

**Research paper:** [Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.957/)
([DOI](https://doi.org/10.18653/2026.findings-emnlp.957)).

The Space requires the original TACME checkpoint in Safetensors format.
By default, the runtime checkpoint is downloaded from the public
[TACME model repository](https://huggingface.co/DmitryRyumin/TACME)
(`DmitryRyumin/TACME`, file `tacme.safetensors`).

If the model is moved or renamed, set the optional **Space Variable**
`TACME_MODEL_REPO_ID` to the new `OWNER/REPO`. If the model becomes private,
provide a **Space Secret** `HF_TOKEN` with read permission for that model.

A separate **GitHub Actions Secret** `HF_TOKEN` with write access to the
Space is needed for automatic deployment from GitHub. Never put tokens
inside this file, the workflow YAML, or any tracked source file.

Deployment updates are published through GitHub Actions; the CPU Space does not
use ZeroGPU.
