"""Build the static demo site (GitHub Pages) from a finished run.

Writes site/index.html plus site/assets/ (web-sized JPEGs, MP4s). Numbers on
the page are read from the run's QC tables and JSON sidecars.

Usage: python code/08_site.py --run-dir results/<run> --samples work_samples --out site
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import statistics
from pathlib import Path

import pandas as pd
import yaml
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.templates import pick  # noqa: E402
from PIL import Image

CFG = Path(__file__).resolve().parent / "config"

PRODUCTS = {
    "halvia": {"master": "halvia_master",
               "scenes": [("halvia_bathroom", "Bathroom shelf"), ("halvia_botanical", "Forest floor"),
                          ("halvia_water", "Water and petals"), ("halvia_vanity", "Vanity, morning daylight")],
               "colourways": [("halvia_cw_amber", "Amber glass"), ("halvia_cw_blush", "Blush glass")]},
    "kivo": {"master": "kivo_master",
             "scenes": [("kivo_street", "City pavement, sunrise"), ("kivo_track", "Running track"),
                        ("kivo_rooftop", "Rooftop, golden hour"), ("kivo_studio_coral", "Brand-colour studio")],
             "colourways": [("kivo_cw_teal", "Teal"), ("kivo_cw_lime", "Lime"), ("kivo_cw_black", "Black upper")]},
}
PAIRS = [("before_kitchen", "t2i", "after_kitchen", "Kitchen renovation"),
         ("before_livingroom", "t2i", "after_livingroom", "Virtual staging"),
         ("before_sneaker", "t2i", "after_sneaker2", "Sneaker restoration"),
         ("halvia_master", "t2i", "after_personalised2", "Personalised label")]

CSS = """
:root{--bg:#faf9f7;--fg:#161616;--muted:#5c5c5c;--line:#d9d6d0;--card:#fff;--accent:#0072b2}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--muted:#a8a8a8;--line:#333;--card:#1d1d1d;--accent:#56b4e9}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 "Helvetica Neue",Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:2.3rem;margin:0 0 .3rem}h2{font-size:1.45rem;margin:3.2rem 0 .4rem}h3{font-size:1.05rem;margin:.2rem 0}
p.lead{color:var(--muted);font-size:1.12rem;max-width:820px}p{max-width:820px}
.muted{color:var(--muted)}.grid{display:grid;gap:18px;grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
figure img,figure video{display:block;width:100%;height:auto}figcaption{padding:10px 14px 12px;font-size:.92rem}
figcaption b{display:block}.metrics{color:var(--muted);font-size:.85rem}
.cw{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(180px,1fr))}
.ba{position:relative;overflow:hidden;border-radius:10px;border:1px solid var(--line);aspect-ratio:16/9;touch-action:none;cursor:ew-resize;user-select:none}
.ba img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;pointer-events:none}
.ba .after{clip-path:inset(0 0 0 50%)}.ba .handle{position:absolute;top:0;bottom:0;left:50%;width:3px;background:#fff;box-shadow:0 0 6px rgba(0,0,0,.5)}
.ba .tag{position:absolute;top:10px;padding:3px 9px;background:rgba(0,0,0,.6);color:#fff;font-size:.8rem;border-radius:4px}
.ba .tag.l{left:10px}.ba .tag.r{right:10px}
.videos{display:grid;gap:18px;grid-template-columns:repeat(auto-fill,minmax(480px,1fr))}
.reels{display:grid;gap:18px;grid-template-columns:repeat(auto-fill,minmax(240px,300px))}
@media (max-width:560px){.videos{grid-template-columns:1fr}.grid{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:.92rem;margin-top:.6rem}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.tablewrap{overflow-x:auto}code{font-size:.9em}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 10px;margin:2px 4px 2px 0;font-size:.85rem;color:var(--muted)}
li{max-width:820px;margin:.2rem 0}footer{margin-top:4rem;color:var(--muted);font-size:.85rem}a{color:var(--accent)}
"""

JS = """
document.querySelectorAll('.ba').forEach(function(el){
  var after=el.querySelector('.after'),h=el.querySelector('.handle');
  function set(x){var r=el.getBoundingClientRect();var p=Math.max(0,Math.min(100,(x-r.left)/r.width*100));
    after.style.clipPath='inset(0 0 0 '+p+'%)';h.style.left=p+'%';}
  var down=false;
  el.addEventListener('pointerdown',function(e){down=true;el.setPointerCapture(e.pointerId);set(e.clientX)});
  el.addEventListener('pointermove',function(e){if(down)set(e.clientX)});
  el.addEventListener('pointerup',function(){down=false});
});
"""


def web_jpeg(src: Path, dst: Path, width: int = 1376):
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=84, optimize=True, progressive=True)
    return dst


def qc(run: Path, stage: str):
    p = run / f"qc_{stage}.csv"
    return pd.read_csv(p).set_index(["id", "seed"]) if p.exists() else pd.DataFrame()


def m(row, key, label, fmt="{:.2f}"):
    if row is None or key not in row or pd.isna(row[key]):
        return ""
    return f"{label} {fmt.format(row[key])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-url", default="")
    a = ap.parse_args()
    run, samples, out = Path(a.run_dir), Path(a.samples), Path(a.out)
    if (out / "assets").exists():
        shutil.rmtree(out / "assets")
    (out / "assets").mkdir(parents=True)
    sel = yaml.safe_load((run / "selection.yaml").read_text())
    brands = yaml.safe_load((CFG / "brand.yaml").read_text())
    q_t2i, q_edit, q_vid = qc(run, "t2i"), qc(run, "edit"), qc(run, "video")

    def row(q, jid):
        k = pick(sel, jid)
        return q.loc[k] if k in q.index else None

    def img(stage, jid):
        return web_jpeg(run / stage / ("%s_s%d.png" % pick(sel, jid)), out / "assets" / f"{jid}.jpg").name

    parts = []
    e = html.escape
    # product sections
    for key, spec in PRODUCTS.items():
        b = brands[key]
        r = row(q_t2i, spec["master"])
        cards = [f'<figure><img loading="lazy" src="assets/{img("t2i", spec["master"])}" alt="{e(b["brand_name"])} master shot">'
                 f'<figcaption><b>Approved master</b><span class="metrics">{m(r, "clip_score", "Prompt match", "{:.1f}")} &nbsp; '
                 f'{m(r, "palette_de", "Palette ΔE", "{:.1f}")}</span></figcaption></figure>']
        for jid, lbl in spec["scenes"]:
            r = row(q_edit, jid)
            cards.append(f'<figure><img loading="lazy" src="assets/{img("edit", jid)}" alt="{e(lbl)}"><figcaption><b>{e(lbl)}</b>'
                         f'<span class="metrics">{m(r, "product_dino", "Identity")} &nbsp; {m(r, "clip_score", "Prompt match", "{:.1f}")} &nbsp; '
                         f'{m(r, "product_colour_de", "Colour ΔE", "{:.1f}")}</span></figcaption></figure>')
        cw = []
        for jid, lbl in spec["colourways"]:
            r = row(q_edit, jid)
            cw.append(f'<figure><img loading="lazy" src="assets/{img("edit", jid)}" alt="{e(lbl)} colourway"><figcaption><b>{e(lbl)}</b>'
                      f'<span class="metrics">{m(r, "product_dino", "Identity")}</span></figcaption></figure>')
        parts.append(f'<h2>{e(b["brand_name"])}: {e(b["category"])}</h2><p class="muted">Fictional brand. One approved master; '
                     f'every scene and colourway below is an instruction edit of that master, so shape, label and materials carry over.</p>'
                     f'<div class="grid">{"".join(cards)}</div><h3 style="margin-top:18px">Colourways</h3><div class="cw">{"".join(cw)}</div>')

    # before / after
    ba = []
    for bid, bst, aid, title in PAIRS:
        r = row(q_edit, aid)
        bi, ai = img(bst, bid), img("edit", aid)
        ba.append(f'<figure><div class="ba"><img src="assets/{bi}" alt="{e(title)} before"><img class="after" src="assets/{ai}" alt="{e(title)} after">'
                  f'<div class="handle"></div><span class="tag l">Before</span><span class="tag r">After</span></div>'
                  f'<figcaption><b>{e(title)}</b><span class="metrics">{m(r, "edge_ssim", "Edge alignment")} &nbsp; '
                  f'{m(r, "clip_score", "Prompt match", "{:.1f}")}</span></figcaption></figure>')
    parts.append('<h2>Before-and-after previews</h2><p class="muted">Drag across each image. The edit changes surfaces, furnishings or '
                 'condition and keeps the camera, perspective and layout, so the two frames stay aligned.</p>'
                 f'<div class="grid">{"".join(ba)}</div>')

    # videos
    vids = sorted(samples.glob("0[5-6]_*promo*.mp4"))
    reels = sorted(samples.glob("*reel*.mp4"))
    reveal = sorted(samples.glob("*reveal*.mp4"))
    titles = {"05": "HALVIA promo, 12 s", "06": "KIVO promo, 12 s", "07": "Before-and-after reveal, 14.5 s",
              "08": "HALVIA reel, 9:16", "09": "KIVO reel, 9:16"}

    def poster(v: Path, t: float) -> str:
        """Still frame for the player before it is started."""
        import imageio.v3 as iio
        frames = iio.imread(v, plugin="pyav")
        name = v.stem + "_poster.jpg"
        im = Image.fromarray(frames[min(int(t * 24), len(frames) - 1)])
        im.save(out / "assets" / name, quality=82, optimize=True)
        return name

    def card(v: Path, t: float) -> str:
        shutil.copy(v, out / "assets" / v.name)
        title = titles.get(v.name[:2], v.stem)
        return (f'<figure><video controls muted playsinline preload="none" poster="assets/{poster(v, t)}" '
                f'src="assets/{v.name}"></video><figcaption><b>{e(title)}</b></figcaption></figure>')

    vcards = [card(v, 2.0) for v in vids] + [card(v, 2.0) for v in reveal]
    rcards = [card(v, 2.0) for v in reels]
    vq = ""
    if len(q_vid):
        rows_ = []
        for (jid, seed), r in q_vid.iterrows():
            keys = [k for k in sel if k.startswith("v_") and pick(sel, k) == (jid, seed)]
            if keys:
                jid = keys[0]
                rows_.append(f"<tr><td>{e(jid)}</td><td>{r['identity_mean']:.2f}</td><td>{r['identity_min']:.2f}</td>"
                             f"<td>{r['flicker']:.1f}</td><td>{r['clip_score']:.1f}</td></tr>")
        vq = ('<div class="tablewrap"><table><tr><th>Clip</th><th>Identity, mean</th><th>Identity, min</th><th>Flicker (0 to 255)</th>'
              f'<th>Prompt match</th></tr>{"".join(rows_)}</table></div>')
    parts.append('<h2>Promotional video</h2><p class="muted">Image-to-video clips generated from the approved stills (5 s, 24 fps), '
                 'cut with a crossfade and a branded end card.</p>'
                 f'<div class="videos">{"".join(vcards)}</div>'
                 + (f'<h3 style="margin-top:22px">Vertical 9:16 reels</h3><div class="reels">{"".join(rcards)}</div>' if rcards else "")
                 + vq)

    # workflow
    ov = samples / "01_workflow_overview.png"
    if ov.exists():
        web_jpeg(ov, out / "assets" / "workflow.jpg", width=2000)
    stats = {}
    for st in ("t2i", "edit", "video"):
        ms = [json.loads(p.read_text()) for p in (run / st).glob("*.json")]
        stats[st] = (len(ms), statistics.median(x["seconds"] for x in ms) if ms else float("nan"))
    repo = f' Code: <a href="{e(a.repo_url)}">{e(a.repo_url)}</a>.' if a.repo_url else ""
    parts.append('<h2>How it works</h2><p>Three plain-text files drive every run: a brand kit (product description, exact label text, '
                 'palette, mood), a template library with named slots, and a campaign file listing what to make. A run expands these into '
                 'written prompts, generates several seeded candidates per item on GPU, scores them, and waits for a person to approve '
                 f'the picks before the next stage.{repo}</p>'
                 + ('<img src="assets/workflow.jpg" alt="Workflow overview" style="width:100%;border:1px solid var(--line);border-radius:10px">' if ov.exists() else "")
                 + f'<p class="muted">This run: {stats["t2i"][0]} master and before candidates (median {stats["t2i"][1]:.0f} s each), '
                 f'{stats["edit"][0]} edit candidates (median {stats["edit"][1]:.0f} s), {stats["video"][0]} video clips '
                 f'(median {stats["video"][1] / 60:.0f} min), on single NVIDIA A100 40 GB GPUs.</p>'
                 '<p><span class="pill">Qwen-Image-2512</span><span class="pill">Qwen-Image-Edit-2511</span>'
                 '<span class="pill">Wan2.2 TI2V-5B</span><span class="pill">CLIP</span><span class="pill">DINOv2</span>'
                 '<span class="pill">OWLv2</span><span class="pill">diffusers</span><span class="pill">PyTorch</span>'
                 '<span class="pill">ffmpeg</span></p><p class="muted">All models are open-weight under Apache 2.0 '
                 '(CLIP under MIT), so outputs carry no subscription or usage restrictions.</p>')

    # quality loop: what review caught, and the measured template revision
    comp = run / "compare_lock_summary.csv"
    qrows = ""
    if comp.exists():
        c = pd.read_csv(comp)
        c = c[c["brand"] == "all, scene level"].set_index("metric")
        names = [("product_colour_de", "Product colour error, CIEDE2000 (lower is better)", "{:.1f}"),
                 ("product_dino", "Product identity, DINOv2 (higher is better)", "{:.3f}"),
                 ("clip_score", "Prompt match, CLIP", "{:.1f}")]
        for k, lbl, f in names:
            r = c.loc[k]
            qrows += (f"<tr><td>{lbl}</td><td>{f.format(r['v1_mean'])}</td><td>{f.format(r['v2_mean'])}</td>"
                      f"<td>{f.format(r['mean'])} ({f.format(r['ci_low'])} to {f.format(r['ci_high'])})</td></tr>")
        cd = c.loc["product_colour_de"]
        n_better = int(cd["n"] - cd["n_up"])
        qtxt = (f"<p>After the first pass, warm scene light was pulling product colours off-brand. The scene template was revised with an "
                f"explicit colour lock and all eight product scenes were rerun with the same seeds. Product colour error fell in "
                f"{n_better} of {int(cd['n'])} scenes (paired t({int(cd['df'])}) = {cd['t']:.2f}, <i>P</i> = {cd['p']:.2f}), with no detectable "
                "change in identity or prompt match. The scene is the unit of analysis (the two seeds per scene are averaged); the comparison "
                "was designed after reviewing the first pass and is exploratory.</p>"
                '<div class="tablewrap"><table><tr><th>Scene mean</th><th>Template v1</th><th>Template v2</th><th>Difference (95% CI)</th></tr>'
                f"{qrows}</table></div>")
    else:
        qtxt = ""
    parts.append('<h2>Quality loop</h2><p>Every candidate is scored automatically and approved by a person. In this run review caught:</p>'
                 '<ul><li>edits that duplicated the product (both first-pass personalised labels added a second bottle; the object-count '
                 'check flagged both, and a revised instruction fixed all four new candidates),</li>'
                 '<li>edits that removed elements the instruction said to keep (a sink, a doorway),</li>'
                 '<li>colour drift under warm scene light.</li></ul>' + qtxt +
                 '<p class="muted">The checks rank and flag; they do not replace review. In this run the count check missed two '
                 'overlapping duplicates (a single shoe turned into a pair, caught at a second review and fixed the same way) and flagged three wet-pavement reflections as second objects.</p>')

    # order: video first, then before/after, products, quality, how it works
    order = [3, 2, 0, 1, 5, 4]
    parts = [parts[i] for i in order]

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Product visualisation demo</title><meta name="description" content="Product visuals, before-and-after previews and promo video from one reproducible, quality-scored workflow.">
<style>{CSS}</style></head><body><main>
<h1>Product visualisation demo</h1>
<p class="lead">Product visuals, before-and-after previews and short promotional videos for two fictional brands, all produced by one
reproducible workflow with an automatic quality check on every output. Kardokh Kaka Bra, 2026.</p>
{''.join(parts)}
<footer>HALVIA and KIVO are fictional brands created for this demonstration. Images and video are generated; no real products or people are shown.</footer>
</main><script>{JS}</script></body></html>"""
    (out / "index.html").write_text(page)
    (out / ".nojekyll").write_text("")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(out / "index.html", f"{total / 2**20:.1f} MiB total")


if __name__ == "__main__":
    main()
