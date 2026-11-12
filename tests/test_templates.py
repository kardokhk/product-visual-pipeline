import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from lib import templates as T  # noqa: E402

CFG = ROOT / "code" / "config"


def test_render_fills_brand_and_slots():
    out = T.render("A {b.short} on {surface}.", {"short": "serum  bottle"}, {"surface": "stone"})
    assert out == "A serum bottle on stone."


def test_render_missing_slot_raises():
    with pytest.raises(KeyError, match="surface"):
        T.render("on {surface}", {}, {})
    with pytest.raises(KeyError, match="short"):
        T.render("a {b.short}", {}, {})


def test_expand_t2i_is_deterministic_and_complete(tmp_path):
    a = T.expand("t2i", CFG, tmp_path)
    b = T.expand("t2i", CFG, tmp_path)
    assert a == b
    assert len(a) == 4 + 4 + 2 + 2 + 2
    assert all("{" not in j["prompt"] for j in a)
    assert len({j["output"] for j in a}) == len(a)


def test_expand_edit_requires_selection(tmp_path):
    with pytest.raises(KeyError):
        T.expand("edit", CFG, tmp_path, selection={})


def test_expand_edit_resolves_selected_seed(tmp_path):
    sel = {"halvia_master": 33, "kivo_master": 11, "before_kitchen": 5,
           "before_livingroom": 6, "before_sneaker": 5}
    jobs = T.expand("edit", CFG, tmp_path, selection=sel)
    j = next(x for x in jobs if x["id"] == "halvia_bathroom")
    assert j["input"].endswith("t2i/halvia_master_s33.png")
    assert all("{" not in x["prompt"] for x in jobs)


def test_shard_partitions_jobs():
    jobs = list(range(10))
    parts = [T.shard(jobs, i, 4) for i in range(4)]
    assert sorted(sum(parts, [])) == jobs
    with pytest.raises(ValueError):
        T.shard(jobs, 4, 4)


def test_expand_carries_canvas_and_size_override(tmp_path):
    sel = {"halvia_master": 33, "kivo_master": 22, "before_kitchen": 6, "before_livingroom": 6,
           "before_sneaker": 5, "halvia_reel": 1, "kivo_reel": 1, "halvia_water": 1,
           "halvia_botanical": 1, "kivo_street": 1, "kivo_rooftop": 1}
    e = {j["id"]: j for j in T.expand("edit", CFG, tmp_path, selection=sel)}
    assert e["halvia_reel"]["canvas"] == "portrait" and e["halvia_bathroom"]["canvas"] is None
    v = {j["id"]: j for j in T.expand("video", CFG, tmp_path, selection=sel)}
    assert v["v_kivo_reel"]["size"] == [704, 1280] and v["v_kivo_street"]["size"] == [1280, 704]


def test_pick_plain_and_redirected():
    sel = {"a": 3, "b": "b_lock:2"}
    assert T.pick(sel, "a") == ("a", 3)
    assert T.pick(sel, "b") == ("b_lock", 2)
    with pytest.raises(KeyError):
        T.pick(sel, "c")


def test_resolve_input_follows_redirect(tmp_path):
    camp = {"edit": [{"id": "x"}, {"id": "x_lock"}]}
    p = T.resolve_input("x@best", camp, {"x": "x_lock:2"}, tmp_path)
    assert p == tmp_path / "edit" / "x_lock_s2.png"
