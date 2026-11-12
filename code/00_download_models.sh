#!/bin/bash
# Fetch pinned open-weight checkpoints into the shared model cache.
set -euo pipefail
source ./local.env
export HF_HUB_ENABLE_HF_TRANSFER=0
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
specs = [
    ("Qwen/Qwen-Image-2512", None),
    ("Qwen/Qwen-Image-Edit-2511", None),
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", None),
    ("facebook/dinov2-base", ["*.json", "*.safetensors", "*.txt"]),
    ("openai/clip-vit-large-patch14", ["*.json", "*.safetensors", "*.txt"]),
    ("google/owlv2-base-patch16-ensemble", ["*.json", "*.safetensors", "*.txt"]),
]
for repo, allow in specs:
    p = snapshot_download(repo, allow_patterns=allow, max_workers=8)
    print(repo, p, flush=True)
PYEOF
