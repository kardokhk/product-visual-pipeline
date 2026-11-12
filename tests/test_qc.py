import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from lib import qc  # noqa: E402


def solid(rgb255, h=32, w=48):
    return np.tile(np.array(rgb255, dtype=np.uint8), (h, w, 1))


def test_hex_to_rgb01_known():
    assert np.allclose(qc.hex_to_rgb01("#FF8000"), [1.0, 128 / 255, 0.0])
    with pytest.raises(ValueError):
        qc.hex_to_rgb01("#FFF")


def test_palette_delta_e_zero_for_exact_brand_colour():
    img = solid((0x8A, 0x9A, 0x82))
    assert qc.palette_delta_e(img, ["#8A9A82", "#2B2B2B"]) == pytest.approx(0.0, abs=1e-6)


def test_palette_delta_e_matches_ciede2000_for_single_colour():
    from skimage import color
    img = solid((200, 30, 30))
    ref = color.deltaE_ciede2000(color.rgb2lab(np.array([[[200, 30, 30]]]) / 255.0)[0, 0],
                                 color.rgb2lab(np.array([[[30, 30, 200]]]) / 255.0)[0, 0])
    assert qc.palette_delta_e(img, ["#1E1EC8"]) == pytest.approx(ref, rel=1e-6)


def test_palette_delta_e_weights_by_area():
    img = solid((0x8A, 0x9A, 0x82), 40, 40)
    img[:10] = (0, 0, 0)  # 25% black, not in palette
    from skimage import color
    de_black = color.deltaE_ciede2000(color.rgb2lab(np.zeros((1, 1, 3)))[0, 0],
                                      color.rgb2lab(qc.hex_to_rgb01("#8A9A82")[None, None])[0, 0])
    assert qc.palette_delta_e(img, ["#8A9A82"], k=2) == pytest.approx(0.25 * de_black, rel=1e-3)


def test_edge_ssim_identity_and_shift():
    rng = np.random.default_rng(0)
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    for _ in range(8):
        y, x = rng.integers(5, 70, 2)
        img[y:y + 20, x:x + 20] = 255
    assert qc.edge_ssim(img, img) == pytest.approx(1.0, abs=1e-9)
    shifted = np.roll(img, 12, axis=1)
    assert qc.edge_ssim(img, shifted) < 0.9


def test_edge_ssim_resizes_second_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = 255
    big = np.kron(img, np.ones((2, 2, 1), dtype=np.uint8))
    assert qc.edge_ssim(img, big) > 0.8


def test_flicker_known_answer_and_errors():
    f = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    f[1] = 10
    f[2] = 10
    assert qc.flicker(f) == pytest.approx(5.0)  # diffs: 10 then 0
    assert qc.flicker(np.zeros((5, 2, 2, 3), dtype=np.uint8)) == 0.0
    with pytest.raises(ValueError):
        qc.flicker(np.zeros((1, 2, 2, 3)))


def test_foreground_bbox_finds_rectangle():
    img = solid((230, 220, 200), 100, 200)
    img[30:70, 80:120] = (40, 60, 50)
    l, t, r, b = qc.foreground_bbox(img, pad=0.0)
    assert (l, t, r, b) == (80, 30, 120, 70)


def test_foreground_bbox_blank_image_returns_full_frame():
    assert qc.foreground_bbox(solid((10, 10, 10), 20, 30)) == (0, 0, 30, 20)


def test_crop_rel_scales_box():
    from PIL import Image
    im = Image.new("RGB", (200, 100))
    assert qc.crop_rel(im, (10, 10, 50, 30), (100, 50)).size == (80, 40)


def test_core_colour_ignores_border_and_detects_shift():
    img = solid((255, 255, 255), 100, 100)
    img[25:75, 25:75] = (0x8A, 0x9A, 0x82)          # product core
    same = img.copy()
    same[:10] = (0, 0, 0)                            # background change only
    assert qc.product_colour_de(img, (0, 0, 100, 100), same, (0, 0, 100, 100)) == pytest.approx(0.0, abs=1e-6)
    shifted = img.copy()
    shifted[25:75, 25:75] = (0xE8, 0xDC, 0xC8)       # sage -> sand
    from skimage import color
    ref = color.deltaE_ciede2000(color.rgb2lab(qc.hex_to_rgb01("#8A9A82")[None, None])[0, 0],
                                 color.rgb2lab(qc.hex_to_rgb01("#E8DCC8")[None, None])[0, 0])
    assert qc.product_colour_de(img, (0, 0, 100, 100), shifted, (0, 0, 100, 100)) == pytest.approx(ref, rel=1e-4)
