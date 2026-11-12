"""Score every generated output in a run and write <run>/qc_<stage>.csv.

Columns by stage:
  t2i    clip_score, palette_de (brand jobs only)
  edit   clip_score, product_dino and product_ssim on the product region of the
         master (scene_swap, colourway), edge_ssim against the input (transform)
  video  clip_score (middle frame), identity_mean / identity_min (DINOv2 frame
         similarity to the source still), flicker (0 to 255)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from skimage import color, metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import qc  # noqa: E402

CFG = Path(__file__).resolve().parent / "config"


def product_metrics(enc, src: Image.Image, out: Image.Image, query: str):
    """Detect the product separately in master and variant, compare the crops."""
    bs, bo = enc.detect(src, query), enc.detect(out, query)
    if bs is None or bo is None:
        return float("nan"), float("nan"), float("nan")
    a = src.crop(bs)
    b = out.crop(bo).resize(a.size, Image.LANCZOS)
    ga = color.rgb2gray(np.asarray(a) / 255.0)
    gb = color.rgb2gray(np.asarray(b) / 255.0)
    return (enc.dino_similarity(a, b), float(metrics.structural_similarity(ga, gb, data_range=1.0)),
            qc.product_colour_de(src, bs, out, bo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", required=True, choices=["t2i", "edit", "video"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--subdir", help="score <run>/<subdir> instead of <run>/<stage> (e.g. archived outputs)")
    a = ap.parse_args()
    run = Path(a.run_dir)
    brands = yaml.safe_load((CFG / "brand.yaml").read_text())
    enc = qc.Encoders(a.device)

    rows = []
    src_dir = run / (a.subdir or a.stage)
    for meta_path in sorted(src_dir.glob("*.json")):
        m = json.loads(meta_path.read_text())
        row = {k: m[k] for k in ("id", "seed", "template", "brand", "seconds", "peak_gpu_gib")}
        pal = list(brands[m["brand"]]["palette"].values()) if m.get("brand") else None
        if a.stage in ("t2i", "edit"):
            out = Image.open(m["output"]).convert("RGB")
            row["clip_score"] = enc.clip_score(out, m["qc_caption"])
            if pal:
                row["palette_de"] = qc.palette_delta_e(out, pal)
            if a.stage == "edit":
                src = Image.open(m["input"]).convert("RGB")
                if m["template"] in ("scene_swap", "scene_swap_lock", "colourway"):
                    row["product_dino"], row["product_ssim"], row["product_colour_de"] = product_metrics(enc, src, out, "a " + brands[m["brand"]]["short"])
                else:
                    row["edge_ssim"] = qc.edge_ssim(src, out)
                q = m.get("count_query")
                if q is None and m.get("brand"):
                    q = "a " + brands[m["brand"]]["short"]
                if q is None and m["id"].startswith("after_sneaker"):
                    q = "a shoe"
                if q is None and m["id"].startswith("after_personalised"):
                    q = "a serum bottle"
                if q:
                    row["count_in"], row["count_out"] = enc.count(src, q), enc.count(out, q)
                    row["dino_to_input"] = enc.dino_similarity(src, out)
        else:
            frames = iio.imread(src_dir / Path(m["output"]).name, plugin="pyav")
            src = Image.open(m["input"]).convert("RGB")
            row["clip_score"] = enc.clip_score(frames[len(frames) // 2], m["qc_caption"])
            row["identity_mean"], row["identity_min"] = enc.video_identity(frames, src)
            row["flicker"] = qc.flicker(frames)
            row["n_frames"] = len(frames)
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    name = f"qc_{a.subdir or a.stage}.csv"
    df.to_csv(run / name, index=False, float_format="%.4f")
    print(f"wrote {run / name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
