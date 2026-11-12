"""Quick inspection sheet: python code/lib/contact.py out.jpg img1 img2 ... (2 columns, labelled)."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out, paths = sys.argv[1], sys.argv[2:]
tw = int(sys.argv[0] and 800)
ims = [Image.open(p).convert("RGB") for p in paths]
th = round(tw * ims[0].height / ims[0].width)
cols = 2
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * (th + 30)), "white")
d = ImageDraw.Draw(sheet)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import font_path  # noqa: E402
f = ImageFont.truetype(font_path(), 20)
for i, (im, p) in enumerate(zip(ims, paths)):
    x, y = (i % cols) * tw, (i // cols) * (th + 30)
    sheet.paste(im.resize((tw, round(tw * im.height / im.width))), (x, y + 30))
    d.text((x + 6, y + 4), Path(p).name, font=f, fill="black")
sheet.save(out, quality=85)
