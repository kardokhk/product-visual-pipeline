"""Screenshot a local HTML page: python code/lib/shot.py page.html out.png [width] [full]"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

page_path, out = Path(sys.argv[1]).resolve(), sys.argv[2]
width = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
full = len(sys.argv) > 4 and sys.argv[4] == "full"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": width, "height": 900})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(page_path.as_uri())
    pg.wait_for_timeout(800)
    pg.screenshot(path=out, full_page=full)
    sw = pg.evaluate("document.documentElement.scrollWidth")
    print("scrollWidth", sw, "viewport", width, "errors", errors)
    b.close()
