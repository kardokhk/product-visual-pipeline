"""Workflow overview graphic, with counts and timings read from the run's JSON sidecars.

Usage: python code/07_overview.py --run-dir results/<run> --out work_samples/01_workflow_overview.png
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import font_path  # noqa: E402
W, H = 2400, 1360
BG, FG, MUTED, LINE = (250, 249, 247), (20, 20, 20), (90, 90, 90), (60, 60, 60)
ACCENT = (0, 114, 178)      # Okabe-Ito blue
GATE = (213, 94, 0)         # Okabe-Ito vermillion


def font(size, bold=False):
    return ImageFont.truetype(font_path(bold), size)


def stats(run: Path, stage: str):
    metas = [json.loads(p.read_text()) for p in (run / stage).glob("*.json")]
    if not metas:
        return 0, float("nan")
    return len(metas), statistics.median(m["seconds"] for m in metas)


def box(d, xy, title, lines, outline=LINE, title_fill=FG, width=3):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=18, outline=outline, width=width, fill=(255, 255, 255))
    d.text((x0 + 24, y0 + 20), title, font=font(32, True), fill=title_fill)
    for i, s in enumerate(lines):
        d.text((x0 + 24, y0 + 70 + i * 36), s, font=font(26), fill=MUTED)


def arrow(d, x0, y0, x1, y1):
    d.line([(x0, y0), (x1, y1)], fill=LINE, width=4)
    if x1 > x0:
        d.polygon([(x1, y1), (x1 - 18, y1 - 10), (x1 - 18, y1 + 10)], fill=LINE)
    else:
        d.polygon([(x1, y1), (x1 - 10, y1 - 18), (x1 + 10, y1 - 18)], fill=LINE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    n1, t1 = stats(run, "t2i")
    n2, t2 = stats(run, "edit")
    n3, t3 = stats(run, "video")

    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((60, 50), "Workflow overview", font=font(56, True), fill=FG)
    d.text((60, 125), "Configuration in, scored and approved assets out. Every output is reproducible from its seed and settings.",
           font=font(30), fill=MUTED)

    # inputs
    y0 = 220
    cfg = [("Brand kit", ["product description", "exact label text", "palette (hex), mood, tagline"]),
           ("Template library", ["named slots", "model and sampler settings", "negative prompt, QC caption"]),
           ("Campaign file", ["what to make", "template + brand + slots", "seeds per item"])]
    for i, (t, ls) in enumerate(cfg):
        x = 60 + i * 560
        box(d, (x, y0, x + 520, y0 + 200), t, ls, outline=ACCENT, title_fill=ACCENT)
    d.text((1760, y0 + 40), "Plain-text configuration.", font=font(28, True), fill=FG)
    d.text((1760, y0 + 82), "A new product or campaign", font=font(26), fill=MUTED)
    d.text((1760, y0 + 118), "is a few edited lines,", font=font(26), fill=MUTED)
    d.text((1760, y0 + 154), "not new prompts.", font=font(26), fill=MUTED)
    # configuration feeds every generation stage
    bus = y0 + 245
    for i in range(3):
        d.line([(60 + i * 560 + 260, y0 + 200), (60 + i * 560 + 260, bus)], fill=ACCENT, width=3)
    d.line([(60 + 260, bus), (60 + 2 * 590 + 260, bus)], fill=ACCENT, width=3)

    # stages
    y1 = y0 + 290
    sw, gap = 520, 70
    stages = [
        ("1  Master shots", ["Qwen-Image-2512, text to image", "1664 x 928, 50 steps", f"{n1} candidates, median {t1:.0f} s each"]),
        ("2  Variants by editing", ["Qwen-Image-Edit-2511", "scenes, colourways, before/after", f"{n2} candidates, median {t2:.0f} s each"]),
        ("3  Motion", ["Wan2.2 TI2V-5B, image to video", "1280 x 704, 5 s at 24 fps", f"{n3} clips, median {t3 / 60:.0f} min each" if n3 else "clips"]),
        ("4  Assembly", ["boards with scores", "promo cut with end card", "before/after reveal"]),
    ]
    for i, (t, ls) in enumerate(stages):
        x = 60 + i * (sw + gap)
        box(d, (x, y1, x + sw, y1 + 200), t, ls)
        if i < 3:
            d.line([(x + 260, y0 + 245), (x + 260, y1 - 4)], fill=ACCENT, width=3)
            d.polygon([(x + 260, y1), (x + 250, y1 - 16), (x + 270, y1 - 16)], fill=ACCENT)
        if i < 3:
            arrow(d, x + sw, y1 + 100, x + sw + gap, y1 + 100)
    # gates
    yg = y1 + 240
    for i in range(3):
        x = 60 + i * (sw + gap)
        d.rounded_rectangle((x, yg, x + sw, yg + 110), radius=14, outline=GATE, width=3, fill=(255, 255, 255))
        d.text((x + 24, yg + 16), "Quality gate", font=font(28, True), fill=GATE)
        d.text((x + 24, yg + 58), "automatic scores rank, a person approves", font=font(24), fill=MUTED)
        d.line([(x + sw / 2, y1 + 200), (x + sw / 2, yg)], fill=GATE, width=3)

    # QC table
    yt = yg + 170
    d.text((60, yt), "Automatic checks", font=font(36, True), fill=FG)
    rows = [("Prompt match", "CLIP ViT-L/14 image-text score", "every image"),
            ("Product identity", "DINOv2 similarity of the detected product (OWLv2) in master and variant", "scene and colourway edits"),
            ("Palette distance", "area-weighted CIEDE2000 from dominant colours to brand palette", "brand images"),
            ("Edge alignment", "SSIM of Canny edge maps, before vs after", "before-and-after pairs"),
            ("Identity drift, flicker", "DINOv2 frame-to-still similarity; mean frame-to-frame change", "video")]
    cols = [60, 520, 1880]
    for j, h in enumerate(["Check", "Method", "Applied to"]):
        d.text((cols[j], yt + 60), h, font=font(26, True), fill=FG)
    d.line([(60, yt + 100), (W - 60, yt + 100)], fill=LINE, width=2)
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            d.text((cols[j], yt + 115 + i * 44), c, font=font(26), fill=MUTED if j else FG)
    d.text((60, H - 80), "Models are open-weight under Apache 2.0. Each output has a JSON record of prompt, seed, settings, model revision and run time.",
           font=font(24), fill=MUTED)
    im.save(a.out, optimize=True)
    print(a.out, im.size)


if __name__ == "__main__":
    main()
