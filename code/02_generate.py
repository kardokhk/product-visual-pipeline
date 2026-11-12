"""Run one shard of one stage (t2i, edit or video) on one GPU.

Usage:
  python code/02_generate.py --stage t2i --run-dir results/<run> --shard 0 --nshards 4

Outputs that already exist are skipped, so an interrupted run resumes. Every
output gets a JSON sidecar with the rendered prompt, seed, sampler settings,
model repository and revision, wall time and peak GPU memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import templates as T  # noqa: E402

REPOS = {
    "qwen_image_t2i": "Qwen/Qwen-Image-2512",
    "qwen_image_edit": "Qwen/Qwen-Image-Edit-2511",
    "wan_i2v": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}


def source(repo: str) -> str:
    """Local staged copy when LOCAL_MODEL_ROOT is set, otherwise the hub cache."""
    root = os.environ.get("LOCAL_MODEL_ROOT")
    if root and (Path(root) / repo / "model_index.json").exists():
        return str(Path(root) / repo)
    return repo


def revision(repo: str) -> str:
    root = os.environ.get("LOCAL_MODEL_ROOT")
    if root and (Path(root) / repo / ".revision").exists():
        return (Path(root) / repo / ".revision").read_text().strip()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, local_files_only=True)).name


def load(model: str):
    bf16 = torch.bfloat16
    if model == "qwen_image_t2i":
        from diffusers import QwenImagePipeline
        pipe = QwenImagePipeline.from_pretrained(source(REPOS[model]), torch_dtype=bf16)
    elif model == "qwen_image_edit":
        from diffusers import QwenImageEditPlusPipeline
        pipe = QwenImageEditPlusPipeline.from_pretrained(source(REPOS[model]), torch_dtype=bf16)
    elif model == "wan_i2v":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        vae = AutoencoderKLWan.from_pretrained(source(REPOS[model]), subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(source(REPOS[model]), vae=vae, torch_dtype=bf16)
    else:
        raise ValueError(model)
    if model.startswith("qwen"):
        # The 20B transformer is 41 GB in bf16, above one A100-40GB. Store its
        # weights in float8 and upcast each layer to bf16 for computation.
        pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=bf16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def fit_crop(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to cover w x h, then centre-crop."""
    s = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


def to_canvas(img: Image.Image, canvas: str | None) -> Image.Image:
    """Re-frame a landscape master for another aspect ratio before editing.

    portrait: centre square crop (keeps the product), padded top and bottom to
    9:16 with the median colour of the adjacent edge rows; the scene-swap edit
    then replaces the whole background, padding included.
    """
    if not canvas:
        return img
    if canvas != "portrait":
        raise ValueError(canvas)
    import numpy as np
    side = min(img.size)
    l = (img.width - side) // 2
    sq = img.crop((l, 0, l + side, side))
    h = round(side * 16 / 9)
    arr = np.asarray(sq)
    top = tuple(int(v) for v in np.median(arr[:8].reshape(-1, 3), axis=0))
    bot = tuple(int(v) for v in np.median(arr[-8:].reshape(-1, 3), axis=0))
    out = Image.new("RGB", (side, h), top)
    out.paste(Image.new("RGB", (side, h // 2), bot), (0, h - h // 2))
    out.paste(sq, (0, (h - side) // 2 + side // 6))
    return out


def run_job(pipe, job: dict):
    g = torch.Generator(device="cpu").manual_seed(job["seed"])
    if job["model"] == "qwen_image_t2i":
        w, h = job["size"]
        out = pipe(prompt=job["prompt"], negative_prompt=job["negative_prompt"], width=w, height=h,
                   num_inference_steps=job["steps"], true_cfg_scale=job["cfg"], generator=g).images[0]
        out.save(job["output"])
    elif job["model"] == "qwen_image_edit":
        src = to_canvas(Image.open(job["input"]).convert("RGB"), job.get("canvas"))
        out = pipe(image=[src], prompt=job["prompt"], negative_prompt=job["negative_prompt"],
                   num_inference_steps=job["steps"], true_cfg_scale=job["cfg"], guidance_scale=1.0,
                   generator=g).images[0]
        out.save(job["output"])
    elif job["model"] == "wan_i2v":
        from diffusers.utils import export_to_video
        w, h = job["size"]
        src = fit_crop(Image.open(job["input"]).convert("RGB"), w, h)
        frames = pipe(image=src, prompt=job["prompt"], negative_prompt=job["negative_prompt"],
                      height=h, width=w, num_frames=job["frames"], guidance_scale=job["cfg"],
                      num_inference_steps=job["steps"], generator=g).frames[0]
        export_to_video(frames, job["output"], fps=job["fps"], quality=10)  # near-lossless master; final cuts re-encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=T.STAGES, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cfg-dir", default=str(Path(__file__).resolve().parent / "config"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="restrict to these job ids")
    ap.add_argument("--steps", type=int, help="override steps (smoke tests only)")
    ap.add_argument("--frames", type=int, help="override frame count (smoke tests only)")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    sel_path = run_dir / "selection.yaml"
    selection = yaml.safe_load(sel_path.read_text()) if sel_path.exists() else {}
    jobs = T.expand(a.stage, Path(a.cfg_dir), run_dir, selection)
    if a.only:
        jobs = [j for j in jobs if j["id"] in a.only]
    # shard only the outstanding jobs, so that a resumed run stays balanced across GPUs
    pending = [j for j in jobs if not Path(j["output"]).exists()]
    todo = T.shard(pending, a.shard, a.nshards)
    print(f"[shard {a.shard}/{a.nshards}] {a.stage}: {len(todo)} to do here, {len(jobs) - len(pending)} already done", flush=True)
    if not todo:
        return
    (run_dir / a.stage).mkdir(parents=True, exist_ok=True)

    pipes, revs = {}, {}
    for job in todo:
        if a.steps:
            job["steps"] = a.steps
        if a.frames:
            job["frames"] = a.frames
        m = job["model"]
        if m not in pipes:
            t0 = time.time()
            pipes[m] = load(m)
            revs[m] = revision(REPOS[m])
            print(f"loaded {m} in {time.time() - t0:.0f} s", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        run_job(pipes[m], job)
        meta = dict(job, repo=REPOS[m], revision=revs[m], seconds=round(time.time() - t0, 1),
                    peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                    diffusers=__import__("diffusers").__version__,
                    host=os.uname().nodename)
        Path(job["output"]).with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"done {Path(job['output']).name} in {meta['seconds']} s, peak {meta['peak_gpu_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
