"""Raking tests (ALGORITHM §3): IPF margin match, Kish ESS, ESS normalization."""

import numpy as np
import pandas as pd
import pytest

from psyche.raking import kish_ess, rake


@pytest.fixture()
def cells():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "taxonomy": rng.choice(["207Q00000X", "207R00000X", "208D00000X"], size=n,
                               p=[0.5, 0.3, 0.2]),
        "state": rng.choice(["CA", "TX", "NY"], size=n, p=[0.6, 0.25, 0.15]),
    })


@pytest.fixture()
def margins():
    return {
        "taxonomy": {"207Q00000X": 400.0, "207R00000X": 300.0, "208D00000X": 300.0},
        "state": {"CA": 500.0, "TX": 300.0, "NY": 200.0},
    }


def test_ipf_reproduces_margins(cells, margins):
    res = rake(cells, margins)
    for dim in margins:
        levels = cells[dim].astype(str).to_numpy()
        for level, target in margins[dim].items():
            assert res.w[levels == level].sum() == pytest.approx(target, rel=1e-4)


def test_ess_math(cells, margins):
    res = rake(cells, margins)
    n = len(cells)
    # Kish ESS formula and design effect
    assert res.ess == pytest.approx(res.w.sum() ** 2 / (res.w**2).sum())
    assert res.deff == pytest.approx(n / res.ess)
    # ESS-normalized weights sum to ESS, never more than n (§3.3)
    assert res.w_tilde.sum() == pytest.approx(res.ess)
    assert res.ess <= n + 1e-9


def test_uniform_weights_give_full_ess():
    cells = pd.DataFrame({"state": ["CA", "CA", "TX", "TX"]})
    margins = {"state": {"CA": 2.0, "TX": 2.0}}
    res = rake(cells, margins)
    assert res.w == pytest.approx(np.ones(4))
    assert res.ess == pytest.approx(4.0)
    assert res.w_tilde == pytest.approx(np.ones(4))


def test_extreme_weights_reduce_ess():
    w = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    assert kish_ess(w) == pytest.approx(14.0**2 / 104.0)
    assert kish_ess(w) < len(w)


def test_unknown_cells_frozen_at_median_and_flagged():
    cells = pd.DataFrame({
        "taxonomy": ["A", "A", "B", "UNKNOWN"],
        "state": ["CA", "CA", "TX", "ZZ"],
    })
    margins = {"taxonomy": {"A": 4.0, "B": 2.0}, "state": {"CA": 4.0, "TX": 2.0}}
    res = rake(cells, margins)
    assert res.weight_flags[3] == "cell_not_in_margins"
    assert res.weight_flags[0] is None
    known = res.w[:3]
    assert res.w[3] == pytest.approx(np.median(known))


def test_trimming_bounds_leverage():
    cells = pd.DataFrame({"state": ["CA"] * 99 + ["TX"]})
    margins = {"state": {"CA": 99.0, "TX": 900.0}}  # TX lead would get w=900
    res = rake(cells, margins, trim_quantile=0.99)
    assert res.w.max() <= np.quantile(np.concatenate([[1.0] * 99, [900.0]]), 0.99) + 1e-9


def test_empty_or_missing_dim_errors(cells):
    with pytest.raises(ValueError):
        rake(cells.iloc[0:0], {"state": {"CA": 1.0}})
    with pytest.raises(ValueError):
        rake(cells, {"nonexistent_dim": {"x": 1.0}})
