"""Assemble client-facing sample boards (JPEG) from selected outputs and QC scores.

Boards:
  <brand>_product_visuals.jpg   master, four scenes, colourways, with scores
  before_after_previews.jpg     four aligned before/after pairs, with scores

Usage: python code/05_boards.py --run-dir results/<run> --out-dir work_samples
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.templates import pick  # noqa: E402
from PIL import Image, ImageDraw, ImageFont

CFG = Path(__file__).resolve().parent / "config"
from lib.paths import font_path  # noqa: E402
W = 2400
M = 60                 # outer margin
G = 40                 # gutter
TW = (W - 2 * M - G) // 2
TH = round(TW * 9 / 16)
CAP = 92               # caption block height under a tile
BG, FG, MUTED = (250, 249, 247), (20, 20, 20), (95, 95, 95)


def font(size, bold=False):
    return ImageFont.truetype(font_path(bold), size)


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    s = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


def header(d: ImageDraw.ImageDraw, title: str, sub: str) -> int:
    d.text((M, M), title, font=font(54, True), fill=FG)
    d.text((M, M + 74), sub, font=font(30), fill=MUTED)
    return M + 150


def zoom(img: Image.Image, factor: float) -> Image.Image:
    """Centre crop by `factor` (applied identically to both images of a pair)."""
    if factor <= 1:
        return img
    w, h = round(img.width / factor), round(img.height / factor)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


def tile(canvas, d, img_path, x, y, label, metrics, w=TW, h=TH, z=1.0):
    canvas.paste(fit(zoom(Image.open(img_path).convert("RGB"), z), w, h), (x, y))
    d.text((x, y + h + 14), label, font=font(30, True), fill=FG)
    d.text((x, y + h + 52), metrics, font=font(25), fill=MUTED)


def footer(d, y, lines):
    for i, s in enumerate(lines):
        d.text((M, y + i * 36), s, font=font(24), fill=MUTED)
    return y + len(lines) * 36


def qc_table(run: Path, stage: str) -> pd.DataFrame:
    p = run / f"qc_{stage}.csv"
    return pd.read_csv(p).set_index(["id", "seed"]) if p.exists() else pd.DataFrame()


def fmt(row, keys):
    names = {"product_dino": "Identity", "clip_score": "Prompt match", "palette_de": "Palette ΔE",
             "edge_ssim": "Edge alignment", "dino_to_input": "Similarity to before",
             "product_colour_de": "Colour \u0394E"}
    out = []
    for k in keys:
        if row is not None and k in row and pd.notna(row[k]):
            out.append(f"{names[k]} {row[k]:.1f}" if k in ("clip_score", "palette_de", "product_colour_de") else f"{names[k]} {row[k]:.2f}")
    return "   ".join(out)


def product_board(run, sel, brand_key, brand, scenes, colourways, out):
    q_t2i, q_edit = qc_table(run, "t2i"), qc_table(run, "edit")
    rows = 3
    H = 150 + M + rows * (TH + CAP + G) + 170
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    y = header(d, f"{brand['brand_name']}  ·  product visual set",
               f"Fictional brand, {brand['category']}. One approved master; every variant is an instruction edit of that master.")
    cells = [("halvia_master" if brand_key == "halvia" else "kivo_master", "t2i", "Approved master (text to image)")]
    cells += [(s, "edit", lbl) for s, lbl in scenes]
    for i, (jid, stage, lbl) in enumerate(cells):
        x = M + (i % 2) * (TW + G)
        yy = y + (i // 2) * (TH + CAP + G)
        p = run / stage / ("%s_s%d.png" % pick(sel, jid))
        q = (q_t2i if stage == "t2i" else q_edit)
        r = q.loc[pick(sel, jid)] if pick(sel, jid) in q.index else None
        keys = ["clip_score", "palette_de"] if stage == "t2i" else ["product_dino", "clip_score"]
        tile(canvas, d, p, x, yy, lbl, fmt(r, keys))
    # colourways: 2 x 2 grid of full 16:9 thumbnails in the last cell, original first
    x0, y0 = M + TW + G, y + 2 * (TH + CAP + G)
    tw = (TW - 16) // 2
    th = round(tw * 9 / 16)
    master_id = "halvia_master" if brand_key == "halvia" else "kivo_master"
    thumbs = [(master_id, "t2i", "Original")] + [(jid, "edit", lbl) for jid, lbl in colourways]
    for j, (jid, stage, lbl) in enumerate(thumbs[:4]):
        x, yy = x0 + (j % 2) * (tw + 16), y0 + (j // 2) * (th + 16)
        p = run / stage / ("%s_s%d.png" % pick(sel, jid))
        canvas.paste(fit(Image.open(p).convert("RGB"), tw, th), (x, yy))
        r = q_edit.loc[pick(sel, jid)] if stage == "edit" and pick(sel, jid) in q_edit.index else None
        tag = lbl if r is None else f"{lbl}   identity {r['product_dino']:.2f}"
        f = font(22, True)
        bw = d.textbbox((0, 0), tag, font=f)[2]
        d.rectangle([x + 10, yy + th - 46, x + 10 + bw + 20, yy + th - 10], fill=(255, 255, 255))
        d.text((x + 20, yy + th - 41), tag, font=f, fill=FG)
    d.text((x0, y0 + 2 * th + 16 + 14), "Colourways", font=font(30, True), fill=FG)
    footer(d, y + rows * (TH + CAP + G) + 10, [
        "Identity: DINOv2 similarity of the product, located by an OWLv2 detector in the master and in the variant (1.00 = identical).",
        "Prompt match: CLIP ViT-L/14 image-text score against the shot description (higher is closer; typical range 20 to 35).",
        "Palette \u0394E (master): area-weighted CIEDE2000 from the image's dominant colours to the nearest brand colour (lower is more on-brand).",
    ])
    canvas.save(out, quality=90, optimize=True, progressive=True)
    print(out, f"{out.stat().st_size / 2**20:.1f} MiB")


def before_after_board(run, sel, pairs, out):
    q = qc_table(run, "edit")
    H = 150 + M + len(pairs) * (TH + CAP + G) + 130
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    y = header(d, "Before-and-after previews",
               "Each 'after' is an instruction edit of the 'before', keeping camera position, perspective and geometry.")
    for i, (bid, bstage, aid, title) in enumerate(pairs):
        yy = y + i * (TH + CAP + G)
        bp = run / bstage / ("%s_s%d.png" % pick(sel, bid))
        ap = run / "edit" / ("%s_s%d.png" % pick(sel, aid))
        r = q.loc[pick(sel, aid)] if pick(sel, aid) in q.index else None
        z = 1.8 if title.startswith("Personalised") else 1.0
        note = "  (1.8x crop)" if z > 1 else ""
        tile(canvas, d, bp, M, yy, f"{title}: before{note}", "", z=z)
        tile(canvas, d, ap, M + TW + G, yy, f"{title}: after{note}", fmt(r, ["edge_ssim", "clip_score"]), z=z)
    footer(d, y + len(pairs) * (TH + CAP + G) + 10, [
        "Edge alignment: SSIM between Canny edge maps of the before and after images (1.00 = identical structure). Prompt match: CLIP ViT-L/14 score.",
        "New surfaces and furniture add edges and lower the score, so it is used to compare candidates of the same edit (a lost doorway or sink scores lower), not as an absolute grade.",
    ])
    canvas.save(out, quality=90, optimize=True, progressive=True)
    print(out, f"{out.stat().st_size / 2**20:.1f} MiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    run, out = Path(a.run_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = yaml.safe_load((run / "selection.yaml").read_text())
    brands = yaml.safe_load((CFG / "brand.yaml").read_text())
    product_board(run, sel, "halvia", brands["halvia"],
                  [("halvia_bathroom", "Bathroom shelf"), ("halvia_botanical", "Botanical, forest floor"),
                   ("halvia_water", "Water and petals"), ("halvia_vanity", "Vanity, morning daylight")],
                  [("halvia_cw_amber", "Amber glass"), ("halvia_cw_blush", "Blush glass")],
                  out / "02_halvia_product_visuals.jpg")
    product_board(run, sel, "kivo", brands["kivo"],
                  [("kivo_street", "City pavement, sunrise"), ("kivo_track", "Running track"),
                   ("kivo_rooftop", "Rooftop, golden hour"), ("kivo_studio_coral", "Brand-colour studio")],
                  [("kivo_cw_teal", "Teal"), ("kivo_cw_lime", "Lime"), ("kivo_cw_black", "Black upper")],
                  out / "03_kivo_product_visuals.jpg")
    before_after_board(run, sel, [
        ("before_kitchen", "t2i", "after_kitchen", "Kitchen renovation"),
        ("before_livingroom", "t2i", "after_livingroom", "Virtual staging"),
        ("before_sneaker", "t2i", "after_sneaker2", "Sneaker restoration"),
        ("halvia_master", "t2i", "after_personalised2", "Personalised label"),
    ], out / "04_before_after_previews.jpg")


if __name__ == "__main__":
    main()
