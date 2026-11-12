"""Before-to-after reveal video: for each pair, hold 'before' 1.5 s, wipe to
'after' over 1.0 s, hold 1.5 s; pairs joined by 0.5 s fades. 1280x704, 24 fps.

Usage: python code/06_reveal.py --run-dir results/<run> --out work_samples/<name>.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import yaml
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.templates import pick  # noqa: E402
from PIL import Image, ImageDraw, ImageFont

from lib.paths import FFMPEG, font_path  # noqa: E402
FONT = font_path(True)
W, H, FPS = 1280, 704, 24
HOLD, WIPE, JOIN = 1.5, 1.0, 0.5
PAIRS = [("before_kitchen", "t2i", "after_kitchen", "Kitchen renovation"),
         ("before_livingroom", "t2i", "after_livingroom", "Virtual staging"),
         ("before_sneaker", "t2i", "after_sneaker2", "Sneaker restoration"),
         ("halvia_master", "t2i", "after_personalised2", "Personalised label")]


def still(src: Path, label: str, dst: Path):
    im = Image.open(src).convert("RGB")
    s = max(W / im.width, H / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - W) // 2, (im.height - H) // 2
    im = im.crop((l, t, l + W, t + H))
    d = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT, 30)
    box = d.textbbox((0, 0), label, font=f)
    d.rectangle([24, 24, 24 + box[2] + 28, 24 + box[3] + 22], fill=(0, 0, 0, 150))
    d.text((38, 34), label, font=f, fill=(255, 255, 255))
    im.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    sel = yaml.safe_load((run / "selection.yaml").read_text())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        segs = []
        for i, (bid, bst, aid, title) in enumerate(PAIRS):
            b, af = td / f"b{i}.png", td / f"a{i}.png"
            still(run / bst / ("%s_s%d.png" % pick(sel, bid)), f"{title}: before", b)
            still(run / "edit" / ("%s_s%d.png" % pick(sel, aid)), f"{title}: after", af)
            seg = td / f"seg{i}.mp4"
            fc = (f"[0:v]fps={FPS},format=yuv420p[x];[1:v]fps={FPS},format=yuv420p[y];"
                  f"[x][y]xfade=transition=wiperight:duration={WIPE}:offset={HOLD}[v]")
            subprocess.run([FFMPEG, "-y", "-loglevel", "error",
                            "-loop", "1", "-t", f"{HOLD + WIPE}", "-framerate", str(FPS), "-i", str(b),
                            "-loop", "1", "-t", f"{HOLD + WIPE / 2 + JOIN}", "-framerate", str(FPS), "-i", str(af),
                            "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "18",
                            "-pix_fmt", "yuv420p", str(seg)], check=True)
            segs.append(seg)
        # join segments with short fades
        seg_len = HOLD + HOLD + WIPE / 2 + JOIN
        inputs, chain, last = [], "", "[0:v]"
        for s in segs:
            inputs += ["-i", str(s)]
        off = 0.0
        for i in range(1, len(segs)):
            off += seg_len - JOIN if i == 1 else seg_len - JOIN
            out = f"[j{i}]"
            chain += f"{last}[{i}:v]xfade=transition=fadeblack:duration={JOIN}:offset={off:.3f}{out};"
            last = out
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", *inputs, "-filter_complex", chain.rstrip(";"),
                        "-map", last, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", a.out], check=True)
    print(a.out, f"{Path(a.out).stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
