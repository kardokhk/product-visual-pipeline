"""Paired comparison helpers (numbers that reach the write-up)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def paired_diff(after, before, conf: float = 0.95) -> dict:
    """Mean of (after - before) with a t-based confidence interval.

    Returns n, mean, sd, ci_low, ci_high, n_up (after > before), and the paired
    t statistic with its two-sided P value.
    """
    a, b = np.asarray(after, float), np.asarray(before, float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("need two equal-length samples with n >= 2")
    d = a - b
    n = d.size
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.5 + conf / 2, n - 1))
    if sd == 0:
        t, p = (math.inf if mean else 0.0), (0.0 if mean else 1.0)
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), n - 1))
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se,
            "n_up": int((d > 0).sum()), "t": float(t), "df": n - 1, "p": p}
