import importlib.util
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location("gen", Path(__file__).resolve().parents[1] / "code" / "02_generate.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def test_portrait_canvas_shape_and_content():
    im = Image.new("RGB", (1664, 928), (200, 190, 180))
    im.paste(Image.new("RGB", (100, 300), (10, 120, 10)), (782, 300))   # product at centre
    out = gen.to_canvas(im, "portrait")
    assert out.size == (928, round(928 * 16 / 9))
    assert out.getpixel((5, 5)) == (200, 190, 180)                      # padded with edge colour
    assert out.getpixel((464, (out.height - 928) // 2 + 928 // 6 + 450)) == (10, 120, 10)


def test_canvas_none_is_identity_and_unknown_raises():
    im = Image.new("RGB", (10, 10))
    assert gen.to_canvas(im, None) is im
    with pytest.raises(ValueError):
        gen.to_canvas(im, "banner")
