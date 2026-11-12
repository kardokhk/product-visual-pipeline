"""Publishable manifest of every candidate: settings and provenance without local paths or host names.

Usage: python code/10_manifest.py --run-dir results/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.templates import pick  # noqa: E402

KEEP = ["id", "seed", "stage", "template", "brand", "repo", "revision", "prompt", "negative_prompt", "steps", "cfg",
        "size", "frames", "fps", "canvas", "seconds", "peak_gpu_gib", "gpu", "torch", "diffusers"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    sel = yaml.safe_load((run / "selection.yaml").read_text())
    chosen = {pick(sel, k) for k in sel}
    items = []
    for stage in ("t2i", "edit", "video"):
        for p in sorted((run / stage).glob("*.json")):
            m = json.loads(p.read_text())
            rec = {k: m.get(k) for k in KEEP}
            rec["output"] = Path(m["output"]).name
            rec["input"] = Path(m["input"]).name if m.get("input") else None
            rec["selected"] = (m["id"], m["seed"]) in chosen
            items.append(rec)
    out = run / "manifest.json"
    out.write_text(json.dumps(items, indent=1))
    print(out, len(items), "records,", sum(r["selected"] for r in items), "selected")


if __name__ == "__main__":
    main()
