"""Tests for the trading benchmarks (synthetic observations / env)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_trading.benchmarks import BENCH, BENCHPLUS, BENCHVWAP
from mibel_trading.env import MibelTradingEnv
from mibel_trading.eval import run_backtest

HOLD, BUY, SELL = 0, 1, 2


def _obs(price: float) -> np.ndarray:
    o = np.zeros(9, dtype=np.float64)
    o[0] = price  # dam_t
    return o


def _features(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    price = 60.0 + 20.0 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 5, n)
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
# BENCH                                                                        #
# --------------------------------------------------------------------------- #
def test_bench_buys_first_sells_last() -> None:
    strat = BENCH(horizon=10)
    actions = [strat.act(_obs(50.0 + i), {}) for i in range(10)]
    assert actions[0] == BUY
    assert actions[-1] == SELL


def test_bench_holds_in_between() -> None:
    strat = BENCH(horizon=10)
    actions = [strat.act(_obs(50.0 + i), {}) for i in range(10)]
    assert all(a == HOLD for a in actions[1:-1])
    # The captured spread is last price minus first price.
    assert strat.captured_spread == pytest.approx((50.0 + 9) - 50.0)


def test_bench_reset_restarts_schedule() -> None:
    strat = BENCH(horizon=5)
    [strat.act(_obs(10.0), {}) for _ in range(5)]
    strat.reset()
    assert strat.act(_obs(10.0), {}) == BUY  # first step again after reset


# --------------------------------------------------------------------------- #
# BENCHVWAP                                                                    #
# --------------------------------------------------------------------------- #
def test_benchvwap_distributes_buys_and_sells() -> None:
    strat = BENCHVWAP(horizon=30, window_size=5)
    actions = [strat.act(_obs(50.0), {}) for _ in range(30)]
    assert actions.count(BUY) == 5
    assert actions.count(SELL) == 5
    assert actions.count(HOLD) == 20
    # Buys come strictly before sells.
    assert all(a == BUY for a in actions[:5])
    assert all(a == SELL for a in actions[-5:])


def test_benchvwap_window_size_clamped_to_horizon_third() -> None:
    strat = BENCHVWAP(horizon=12, window_size=10)  # 2*10 > 12 -> clamp to 12//3
    assert strat.window_size == 4
    # No overlap: buys and sells stay disjoint.
    actions = [strat.act(_obs(50.0), {}) for _ in range(12)]
    assert actions.count(BUY) == 4
    assert actions.count(SELL) == 4


# --------------------------------------------------------------------------- #
# BENCHPLUS                                                                    #
# --------------------------------------------------------------------------- #
def test_benchplus_holds_in_low_vol() -> None:
    """Constant prices -> zero volatility -> never trades."""
    strat = BENCHPLUS(horizon=120, window_size=10, vol_window=24)
    actions = [strat.act(_obs(50.0), {}) for _ in range(120)]
    assert all(a == HOLD for a in actions)


def test_benchplus_trades_in_high_vol() -> None:
    """A calm warm-up then a volatile regime clears the gate and trades fire."""
    rng = np.random.default_rng(0)
    prices = np.concatenate(
        [
            50.0 + rng.normal(0, 0.1, 24),  # calm warm-up -> low threshold
            50.0 + rng.normal(0, 40, 96),  # volatile -> exceeds the gate
        ]
    )
    strat = BENCHPLUS(horizon=120, window_size=10, vol_window=24)
    actions = [strat.act(_obs(p), {}) for p in prices]
    assert any(a != HOLD for a in actions)
    assert sum(a == BUY for a in actions) >= 1


def test_benchplus_explicit_threshold_overrides_calibration() -> None:
    """A very high explicit threshold suppresses all trades."""
    rng = np.random.default_rng(1)
    prices = 50.0 + rng.normal(0, 30, 120)
    strat = BENCHPLUS(horizon=120, window_size=10, vol_window=24, vol_threshold=1e9)
    actions = [strat.act(_obs(p), {}) for p in prices]
    assert all(a == HOLD for a in actions)


# --------------------------------------------------------------------------- #
# Integration with the env                                                     #
# --------------------------------------------------------------------------- #
def test_all_benchmarks_run_through_env_without_error() -> None:
    features = _features(n=80)
    horizon = len(features) - 1
    for strat in (BENCH(horizon), BENCHVWAP(horizon), BENCHPLUS(horizon)):
        result = run_backtest(strat, MibelTradingEnv(features))
        assert "total_pnl" in result
        assert len(result["actions"]) == horizon
