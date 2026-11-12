"""Expand campaign entries into fully rendered, reproducible generation jobs.

A job is a plain dict: id, stage, template, model, rendered prompt, negative
prompt, seed, sampler settings, input path (for edit and video stages), output
path and QC caption. Jobs are deterministic functions of the three YAML files
plus selection.yaml, so the same configuration always yields the same job list.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import yaml

STAGES = ("t2i", "edit", "video")
_SLOT = re.compile(r"\{([a-zA-Z_][\w.]*)\}")


def load_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def render(template: str, brand: dict | None, slots: dict) -> str:
    """Fill {b.field} from the brand kit and {name} from slots.

    Raises KeyError naming the slot when a value is missing, so a template
    typo fails at expansion time rather than silently reaching the model.
    """
    b = SimpleNamespace(**(brand or {}))

    def sub(m):
        key = m.group(1)
        if key.startswith("b."):
            attr = key[2:]
            if not hasattr(b, attr):
                raise KeyError(f"brand field '{attr}' missing for slot {{{key}}}")
            return _clean(getattr(b, attr))
        if key not in slots:
            raise KeyError(f"slot '{key}' missing")
        return _clean(slots[key])

    return _clean(_SLOT.sub(sub, template))


def output_path(run_dir: Path, stage: str, job_id: str, seed: int) -> Path:
    ext = "mp4" if stage == "video" else "png"
    return Path(run_dir) / stage / f"{job_id}_s{seed}.{ext}"


def pick(selection: dict, key: str) -> tuple[str, int]:
    """Selection value is either a seed of `key` itself (int) or 'other_id:seed'
    when the approved candidate came from a revised template."""
    if key not in (selection or {}):
        raise KeyError(f"no selection recorded for {key!r}; run the review gate first")
    v = selection[key]
    if isinstance(v, str) and ":" in v:
        jid, seed = v.rsplit(":", 1)
        return jid, int(seed)
    return key, int(v)


def stage_of(job_id: str, campaign: dict) -> str:
    for stage in STAGES:
        if any(e["id"] == job_id for e in campaign.get(stage, [])):
            return stage
    raise KeyError(f"{job_id!r} not defined in campaign")


def resolve_input(ref: str, campaign: dict, selection: dict, run_dir: Path) -> Path:
    """Turn 'some_id@best' into the path of the candidate chosen for some_id."""
    key, _, tag = ref.partition("@")
    if tag != "best":
        raise ValueError(f"unsupported input reference {ref!r}")
    job_id, seed = pick(selection, key)
    for stage in STAGES:
        if any(e["id"] == job_id for e in campaign.get(stage, [])):
            return output_path(run_dir, stage, job_id, seed)
    raise KeyError(f"input {job_id!r} not defined in campaign")


def expand(stage: str, cfg_dir: Path, run_dir: Path, selection: dict | None = None) -> list[dict]:
    cfg_dir = Path(cfg_dir)
    brands = load_yaml(cfg_dir / "brand.yaml")
    lib = load_yaml(cfg_dir / "prompts.yaml")
    campaign = load_yaml(cfg_dir / "campaign.yaml")
    neg_img = _clean(lib["defaults"]["negative_image"])
    neg_vid = _clean(lib["defaults"]["negative_video"])

    jobs = []
    for entry in campaign.get(stage, []):
        tpl = lib["templates"][entry["template"]]
        brand = brands[entry["brand"]] if entry.get("brand") else None
        slots = entry.get("slots", {})
        prompt = render(tpl["text"], brand, slots)
        caption = render(tpl["qc_caption"], brand, slots)
        inp = None
        if stage in ("edit", "video"):
            inp = str(resolve_input(entry["input"], campaign, selection, run_dir))
        for seed in entry["seeds"]:
            job = {
                "id": entry["id"],
                "stage": stage,
                "template": entry["template"],
                "brand": entry.get("brand"),
                "model": tpl["model"],
                "prompt": prompt,
                "negative_prompt": neg_vid if stage == "video" else neg_img,
                "seed": int(seed),
                "steps": int(tpl["steps"]),
                "cfg": float(tpl["cfg"]),
                "size": entry.get("size", tpl.get("size")),
                "canvas": entry.get("canvas"),
                "count_query": entry.get("count_query") or (f"a {brand['short']}" if brand else None),
                "frames": tpl.get("frames"),
                "fps": tpl.get("fps"),
                "input": inp,
                "qc_caption": caption,
                "output": str(output_path(run_dir, stage, entry["id"], seed)),
            }
            jobs.append(job)
    return jobs


def shard(jobs: list[dict], index: int, count: int) -> list[dict]:
    """Round-robin split so that each GPU worker gets a similar load."""
    if not 0 <= index < count:
        raise ValueError("shard index out of range")
    return jobs[index::count]
