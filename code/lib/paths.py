"""Machine-specific locations, read from the environment (see local.env.example)."""
import os
import shutil
from pathlib import Path

import matplotlib

_MPL = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONTS_DIR = Path(os.environ.get("FONTS_DIR", str(_MPL)))
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def font_path(bold: bool = False) -> str:
    """Nimbus Sans if FONTS_DIR has it, otherwise matplotlib's DejaVu Sans."""
    name = "NimbusSans-Bold.otf" if bold else "NimbusSans-Regular.otf"
    if (FONTS_DIR / name).exists():
        return str(FONTS_DIR / name)
    return str(_MPL / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
