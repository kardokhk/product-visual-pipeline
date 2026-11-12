# Environment

Captured 2026-11-11 on a Linux HPC cluster: CPU steps on a login node; generation on Slurm GPU nodes with 4 x NVIDIA A100-SXM4-40GB (driver 610.57.04), one worker per GPU.

## Interpreter and key packages

- Environment: micromamba, conda-forge Python and ffmpeg, PyTorch CUDA 12.6 wheels, then pip (requirements.txt)
- Python 3.11.18
- torch 2.14.1+cu126
- diffusers 0.42.1
- transformers 5.22.2
- accelerate 1.15.0
- PIL 13.0.0
- numpy 2.4.6
- skimage 0.26.0
- scipy 1.17.1
- imageio 2.38.1
- av 18.1.0
- pandas 3.1.1
- CUDA (torch build) 12.6
- ffmpeg 9.0.2 (conda-forge, libx264)

Full pinned list: requirements.txt

## Model weights (Hugging Face, downloaded 2026-11-11)

| repository | revision | licence |
|---|---|---|
| Qwen/Qwen-Image-2512 | 25468b98e3276ca6700de15c6628e51b7de54a26 | apache-2.0 |
| Qwen/Qwen-Image-Edit-2511 | 6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 | apache-2.0 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | b8fff7315c768468a5333511427288870b2e9635 | apache-2.0 |
| openai/clip-vit-large-patch14 | 32bd64288804d66eefd0ccbe215aa642df71cc41 | MIT (github.com/openai/CLIP LICENSE; no tag on the model card) |
| facebook/dinov2-base | f9e44c814b77203eaa57a6bdbbd535f21ede1415 | apache-2.0 |
| google/owlv2-base-patch16-ensemble | cfd3195ba4ea9592eec887ded089f4c08eff231d | apache-2.0 |

## Seeds

Every generation seed is listed in code/config/campaign.yaml and recorded in the JSON sidecar of each output. Dominant-colour k-means uses seed 0.

## Fonts

Nimbus Sans (Helvetica metric clone) when FONTS_DIR points to it, otherwise DejaVu Sans; DejaVu Serif (matplotlib) for the HALVIA wordmark.
