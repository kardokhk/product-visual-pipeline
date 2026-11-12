"""Frame strip for reviewing a clip: python code/lib/strip.py out.jpg clip1.mp4 [clip2.mp4 ...]"""
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import font_path  # noqa: E402

out, clips = sys.argv[1], sys.argv[2:]
rows = []
for c in clips:
    fr = iio.imread(c, plugin="pyav")
    idx = np.linspace(0, len(fr) - 1, 4).round().astype(int)
    ims = [Image.fromarray(fr[i]) for i in idx]
    w = 480
    ims = [i.resize((w, round(i.height * w / i.width))) for i in ims]
    row = Image.new("RGB", (w * 4 + 30, ims[0].height + 28), "white")
    ImageDraw.Draw(row).text((4, 4), f"{Path(c).name}  ({len(fr)} frames)", font=ImageFont.truetype(font_path(), 18), fill="black")
    for k, im in enumerate(ims):
        row.paste(im, (k * (w + 10), 28))
    rows.append(row)
sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
y = 0
for r in rows:
    sheet.paste(r, (0, y))
    y += r.height
sheet.save(out, quality=85)
