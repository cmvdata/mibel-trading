"""Tests for walk-forward out-of-sample evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_trading.benchmarks import BENCH, BENCHVWAP
from mibel_trading.env import MibelTradingEnv
from mibel_trading.eval import (
    walk_forward_evaluate,
    walk_forward_splits,
)
from mibel_trading.strategies import (
    NaiveBuyHoldStrategy,
    RandomStrategy,
    RuleBasedSpreadStrategy,
)


def _features(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    price = 60.0 + 20.0 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 6, n)
    df = pd.DataFrame(
        {
            "price_eur_mwh": price,
            "esios_634": 40.0 + rng.normal(0, 8, n),
            "esios_676": rng.uniform(0, 100, n),
            "esios_686": rng.normal(0, 40, n),
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


# --------------------------------------------------------------------------- #
# Splits                                                                       #
# --------------------------------------------------------------------------- #
def test_walk_forward_splits_expanding() -> None:
    folds = walk_forward_splits(100, n_folds=5, scheme="expanding", min_train_frac=0.3)
    assert len(folds) == 5
    for f in folds:
        assert f.train_start == 0  # expanding always starts at the origin
        assert f.train_end == f.test_start  # no leakage: train ends where test starts
        assert f.n_test >= 2
    # Test blocks are contiguous and cover to the end.
    for a, b in zip(folds, folds[1:], strict=False):
        assert b.test_start == a.test_end
    assert folds[-1].test_end == 100
    # Training window grows.
    assert folds[0].n_train < folds[-1].n_train


def test_walk_forward_splits_rolling() -> None:
    folds = walk_forward_splits(100, n_folds=5, scheme="rolling", min_train_frac=0.3)
    starts = [f.train_start for f in folds]
    assert starts[0] == 0
    assert all(b >= a for a, b in zip(starts, starts[1:], strict=False))
    assert starts[-1] > 0  # the rolling window has moved off the origin


def test_walk_forward_splits_too_few_rows_raises() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(10, n_folds=5, min_train_frac=0.3)


def test_walk_forward_splits_bad_args_raise() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(100, n_folds=0)
    with pytest.raises(ValueError):
        walk_forward_splits(100, min_train_frac=1.5)


# --------------------------------------------------------------------------- #
# Evaluation (non-learned strategies)                                          #
# --------------------------------------------------------------------------- #
def test_walk_forward_evaluate_non_learned() -> None:
    features = _features(240)
    providers = {
        "random": lambda tr, te: RandomStrategy(seed=0),
        "buy_hold": lambda tr, te: NaiveBuyHoldStrategy(),
        "rule_based": lambda tr, te: RuleBasedSpreadStrategy(window=24),
        "BENCH": lambda tr, te: BENCH(len(te) - 1),
        "BENCHVWAP": lambda tr, te: BENCHVWAP(len(te) - 1, window_size=5),
    }
    result = walk_forward_evaluate(features, providers, n_folds=4, scheme="expanding")

    assert len(result.folds) == 4
    assert len(result.per_fold) == len(providers) * 4
    assert set(result.summary.index) == set(providers)
    # Every metric is finite and present.
    for col in ("total_pnl", "win_rate", "sharpe_proxy", "PT", "PD"):
        assert (col, "mean") in result.summary.columns
        assert np.isfinite(result.summary[(col, "mean")]).all()
    # All test windows are usable (>= 2 rows).
    assert (result.per_fold["n_test"] >= 2).all()


def test_walk_forward_evaluate_is_out_of_sample() -> None:
    """Each fold's test window is strictly after its training window."""
    features = _features(200)
    providers = {"buy_hold": lambda tr, te: NaiveBuyHoldStrategy()}
    result = walk_forward_evaluate(features, providers, n_folds=3)
    for fold in result.folds:
        assert fold.train_end <= fold.test_start


def test_walk_forward_empty_providers_raises() -> None:
    with pytest.raises(ValueError):
        walk_forward_evaluate(_features(100), {}, n_folds=3)


# --------------------------------------------------------------------------- #
# Evaluation (learned PPO agent — tiny budget)                                 #
# --------------------------------------------------------------------------- #
def test_walk_forward_evaluate_trains_ppo_per_fold() -> None:
    """A learning provider is trained on each fold's train window and scored OOS."""
    from mibel_trading.agents import train_ppo_baseline

    features = _features(150)

    def ppo_provider(train_df: pd.DataFrame, test_df: pd.DataFrame):
        return train_ppo_baseline(
            lambda: MibelTradingEnv(train_df), total_timesteps=400, seed=0
        )

    result = walk_forward_evaluate(
        features, {"ppo": ppo_provider}, n_folds=2, min_train_frac=0.3
    )
    assert len(result.per_fold) == 2
    assert "ppo" in result.summary.index
    assert np.isfinite(result.summary[("total_pnl", "mean")]).all()
