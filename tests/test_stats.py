import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from lib.stats import paired_diff  # noqa: E402


def test_paired_diff_known_answer():
    # differences 1,2,3,4,5: mean 3, sd sqrt(2.5); t crit (df 4, 95%) = 2.776445
    r = paired_diff([2, 4, 6, 8, 10], [1, 2, 3, 4, 5])
    assert r["mean"] == pytest.approx(3.0)
    assert r["sd"] == pytest.approx(2.5 ** 0.5)
    half = 2.776445 * (2.5 ** 0.5) / 5 ** 0.5
    assert r["ci_low"] == pytest.approx(3 - half, rel=1e-5)
    assert r["ci_high"] == pytest.approx(3 + half, rel=1e-5)
    assert r["t"] == pytest.approx(3 / ((2.5 ** 0.5) / 5 ** 0.5))
    assert r["n_up"] == 5 and r["df"] == 4


def test_paired_diff_matches_scipy_ttest_rel():
    from scipy import stats
    a, b = [0.91, 0.88, 0.95, 0.80, 0.86], [0.85, 0.90, 0.89, 0.70, 0.84]
    r = paired_diff(a, b)
    ref = stats.ttest_rel(a, b)
    assert r["t"] == pytest.approx(ref.statistic)
    assert r["p"] == pytest.approx(ref.pvalue)


def test_paired_diff_edge_cases():
    r = paired_diff([1, 1, 1], [1, 1, 1])
    assert r["mean"] == 0 and r["p"] == 1.0
    with pytest.raises(ValueError):
        paired_diff([1], [2])
    with pytest.raises(ValueError):
        paired_diff([1, 2], [1, 2, 3])
