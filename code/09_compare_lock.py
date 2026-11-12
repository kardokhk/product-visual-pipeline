"""Paired comparison of scene_swap (v1) and scene_swap_lock (v2) on matched scene and seed.

Writes <run>/compare_lock.csv (per pair) and <run>/compare_lock_summary.csv.
Usage: python code/09_compare_lock.py --run-dir results/<run>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.stats import paired_diff  # noqa: E402

METRICS = [("product_dino", "higher"), ("product_colour_de", "lower"), ("palette_de", "lower"), ("product_ssim", "higher"), ("clip_score", "higher")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    q = pd.read_csv(run / "qc_edit.csv")
    v2 = q[q["template"] == "scene_swap_lock"].copy()
    v2["base"] = v2["id"].str.replace("_lock$", "", regex=True)
    v1 = q[q["template"] == "scene_swap"]
    pairs = v2.merge(v1, left_on=["base", "seed"], right_on=["id", "seed"], suffixes=("_v2", "_v1"))
    keep = ["base", "seed", "brand_v1"] + [f"{m}_{v}" for m, _ in METRICS for v in ("v1", "v2")]
    pairs[keep].to_csv(run / "compare_lock.csv", index=False, float_format="%.4f")
    rows = []
    for brand, sub in [("all", pairs)] + list(pairs.groupby("brand_v1")):
        for m, better in METRICS:
            r = paired_diff(sub[f"{m}_v2"], sub[f"{m}_v1"])
            r.update(brand=brand, metric=m, better=better,
                     v1_mean=sub[f"{m}_v1"].mean(), v2_mean=sub[f"{m}_v2"].mean())
            rows.append(r)
    # Seeds within a scene share the scene prompt, so the scene is the unit of
    # analysis: average the two seeds per scene, then compare paired scenes.
    scene = pairs.groupby(["base", "brand_v1"], as_index=False).mean(numeric_only=True)
    for m, better in METRICS:
        r = paired_diff(scene[f"{m}_v2"], scene[f"{m}_v1"])
        r.update(brand="all, scene level", metric=m, better=better,
                 v1_mean=scene[f"{m}_v1"].mean(), v2_mean=scene[f"{m}_v2"].mean())
        rows.append(r)
    out = pd.DataFrame(rows)[["brand", "metric", "better", "n", "v1_mean", "v2_mean", "mean", "ci_low", "ci_high",
                              "n_up", "t", "df", "p"]]
    out.to_csv(run / "compare_lock_summary.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
