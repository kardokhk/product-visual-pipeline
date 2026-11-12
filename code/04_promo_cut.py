"""Cut generated clips into a short branded promo per brand.

clip A -> 0.6 s crossfade -> clip B -> 0.6 s crossfade -> 3 s end card
(brand name, tagline, palette bar), encoded H.264, yuv420p, CRF 18, 24 fps.

Usage: python code/04_promo_cut.py --run-dir results/<run> --out-dir work_samples
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import yaml
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.templates import pick  # noqa: E402
from PIL import Image, ImageDraw, ImageFont

CFG = Path(__file__).resolve().parent / "config"
import sys as _sys2
_sys2.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib.paths import FFMPEG, font_path  # noqa: E402
SERIF = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSerif.ttf"
W, H, FPS, XF, CARD = 1280, 704, 24, 0.6, 3.0

CUTS = {
    "halvia": {"out": "05_halvia_promo", "clips": ["v_halvia_water", "v_halvia_botanical"], "serif": True, "bg": "sand", "fg": "charcoal"},
    "kivo": {"out": "06_kivo_promo", "clips": ["v_kivo_street", "v_kivo_rooftop"], "serif": False, "bg": "graphite", "fg": "offwhite"},
}


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


REELS = {
    "halvia": {"out": "08_halvia_reel_9x16", "clip": "v_halvia_reel", "serif": True, "bg": "sand", "fg": "charcoal"},
    "kivo": {"out": "09_kivo_reel_9x16", "clip": "v_kivo_reel", "serif": False, "bg": "graphite", "fg": "offwhite"},
}


def end_card(brand: dict, spec: dict, path: Path, W=W, H=H):
    pal = brand["palette"]
    im = Image.new("RGB", (W, H), hex_rgb(pal[spec["bg"]]))
    d = ImageDraw.Draw(im)
    big = ImageFont.truetype(str(SERIF if spec["serif"] else font_path(True)), min(120, W // 6))
    small = ImageFont.truetype(font_path(), 34)
    fg = hex_rgb(pal[spec["fg"]])
    d.text((W / 2, H / 2 - 40), brand["brand_name"], font=big, fill=fg, anchor="mm")
    d.text((W / 2, H / 2 + 60), brand["tagline"], font=small, fill=fg, anchor="mm")
    sw = 60
    x0 = W / 2 - sw * len(pal) / 2
    for i, hx in enumerate(pal.values()):
        d.rectangle([x0 + i * sw, H - 92, x0 + (i + 1) * sw - 8, H - 76], fill=hex_rgb(hx), outline=fg, width=1)
    im.save(path)


def duration(p: Path) -> float:
    out = subprocess.run([FFMPEG, "-i", str(p)], capture_output=True, text=True).stderr
    hh, mm, ss = out.split("Duration: ")[1].split(",")[0].split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    run, out_dir = Path(a.run_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    brands = yaml.safe_load((CFG / "brand.yaml").read_text())
    sel = yaml.safe_load((run / "selection.yaml").read_text())

    for key, spec in CUTS.items():
        clips = [run / "video" / ("%s_s%d.mp4" % pick(sel, c)) for c in spec["clips"]]
        with tempfile.TemporaryDirectory() as td:
            card_png = Path(td) / "card.png"
            end_card(brands[key], spec, card_png)
            d0, d1 = duration(clips[0]), duration(clips[1])
            off1 = d0 - XF
            off2 = off1 + d1 - XF
            fc = (
                f"[0:v]scale={W}:{H},fps={FPS},format=yuv420p,setsar=1[a];"
                f"[1:v]scale={W}:{H},fps={FPS},format=yuv420p,setsar=1[b];"
                f"[2:v]scale={W}:{H},fps={FPS},format=yuv420p,setsar=1[c];"
                f"[a][b]xfade=transition=fade:duration={XF}:offset={off1:.3f}[ab];"
                f"[ab][c]xfade=transition=fade:duration={XF}:offset={off2:.3f},fade=t=out:st={off2 + CARD - 0.5:.3f}:d=0.5[v]"
            )
            target = out_dir / f"{spec['out']}_{int(round(off2 + CARD))}s.mp4"
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(clips[0]), "-i", str(clips[1]),
                   "-loop", "1", "-t", f"{CARD}", "-framerate", str(FPS), "-i", str(card_png),
                   "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-preset", "slow",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
            subprocess.run(cmd, check=True)
            print(target, f"{target.stat().st_size / 2**20:.1f} MiB")
    reels(run, out_dir, brands, sel)


def reels(run: Path, out_dir: Path, brands: dict, sel: dict):
    """Single vertical clip + 2.5 s portrait end card, 704 x 1280."""
    rw, rh, card = 704, 1280, 2.5
    for key, spec in REELS.items():
        if spec["clip"] not in sel:
            continue
        clip = run / "video" / ("%s_s%d.mp4" % pick(sel, spec['clip']))
        with tempfile.TemporaryDirectory() as td:
            card_png = Path(td) / "card.png"
            end_card(brands[key], spec, card_png, W=rw, H=rh)
            off = duration(clip) - XF
            fc = (f"[0:v]scale={rw}:{rh},fps={FPS},format=yuv420p,setsar=1[a];"
                  f"[1:v]scale={rw}:{rh},fps={FPS},format=yuv420p,setsar=1[c];"
                  f"[a][c]xfade=transition=fade:duration={XF}:offset={off:.3f},fade=t=out:st={off + card - 0.5:.3f}:d=0.5[v]")
            target = out_dir / f"{spec['out']}.mp4"
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(clip), "-loop", "1", "-t", f"{card}",
                            "-framerate", str(FPS), "-i", str(card_png), "-filter_complex", fc, "-map", "[v]",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(target)], check=True)
            print(target, f"{target.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
