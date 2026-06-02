"""Tests for the backtest engine and metrics (synthetic market data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_trading.env import MibelTradingEnv
from mibel_trading.eval import (
    compare_strategies,
    max_drawdown,
    percentage_directional,
    percentage_winning_trades,
    pnl_per_trade,
    run_backtest,
)
from mibel_trading.strategies import (
    NaiveBuyHoldStrategy,
    RandomStrategy,
    RuleBasedSpreadStrategy,
)

_EXPECTED_KEYS = {
    "total_pnl",
    "equity_curve",
    "positions",
    "actions",
    "rewards",
    "pnl_series",
    "n_trades",
    "win_rate",
    "sharpe_proxy",
    "max_drawdown",
}


def _features(n: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    price = 60.0 + 25.0 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 6, n)
    df = pd.DataFrame(
        {
            "price_eur_mwh": price,
            "esios_634": 40.0 + rng.normal(0, 8, n),
            "esios_676": rng.uniform(0, 120, n),
            "esios_686": rng.normal(0, 50, n),
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


@pytest.fixture
def features() -> pd.DataFrame:
    return _features()


def _env_factory(features: pd.DataFrame):
    return lambda: MibelTradingEnv(features)


def test_backtest_returns_expected_keys(features: pd.DataFrame) -> None:
    result = run_backtest(NaiveBuyHoldStrategy(), MibelTradingEnv(features))
    assert set(result) >= _EXPECTED_KEYS
    assert isinstance(result["total_pnl"], float)
    assert isinstance(result["n_trades"], int)


def test_backtest_equity_curve_matches_pnl_cumsum(features: pd.DataFrame) -> None:
    result = run_backtest(RuleBasedSpreadStrategy(window=24), MibelTradingEnv(features))
    np.testing.assert_allclose(result["equity_curve"], np.cumsum(result["pnl_series"]))
    assert result["total_pnl"] == pytest.approx(result["equity_curve"][-1])


def test_backtest_n_trades_count(features: pd.DataFrame) -> None:
    """Buy-and-hold trades every step, so n_trades == number of steps."""
    result = run_backtest(NaiveBuyHoldStrategy(), MibelTradingEnv(features))
    n_steps = len(features) - 1
    assert result["n_trades"] == n_steps
    assert len(result["actions"]) == n_steps


def test_backtest_holds_make_no_trades(features: pd.DataFrame) -> None:
    """A pure-HOLD policy registers zero trades and a flat equity curve."""

    class _AlwaysHold(RuleBasedSpreadStrategy):
        def act(self, obs, info):  # type: ignore[override]
            return 0

    result = run_backtest(_AlwaysHold(window=24), MibelTradingEnv(features))
    assert result["n_trades"] == 0
    assert result["win_rate"] == 0.0
    np.testing.assert_allclose(result["equity_curve"], 0.0)


def test_compare_strategies_returns_dataframe(features: pd.DataFrame) -> None:
    strategies = {
        "random": RandomStrategy(seed=0),
        "buy_hold": NaiveBuyHoldStrategy(),
        "rule_based": RuleBasedSpreadStrategy(window=24),
    }
    df = compare_strategies(strategies, _env_factory(features))
    assert isinstance(df, pd.DataFrame)
    assert set(df.index) == set(strategies)
    for col in ("total_pnl", "n_trades", "win_rate", "sharpe_proxy", "max_drawdown"):
        assert col in df.columns


def test_metrics_pt_pd_max_drawdown() -> None:
    """PT, PD and max drawdown on hand-computed synthetic series."""
    actions = np.array([1, 0, 2, 1])  # BUY, HOLD, SELL, BUY
    pnl = np.array([10.0, -5.0, -3.0, 4.0])

    # Trades at idx 0 (+10, win), 2 (-3, loss), 3 (+4, win) -> 2/3 winners.
    assert percentage_winning_trades(actions, pnl) == pytest.approx(2.0 / 3.0)
    np.testing.assert_allclose(pnl_per_trade(actions, pnl), [10.0, -3.0, 4.0])

    # Directional: BUY@0 diff +5 correct, SELL@2 diff -2 correct, BUY@3 diff -1 wrong.
    dam_diff = np.array([5.0, 0.0, -2.0, -1.0])
    assert percentage_directional(actions, dam_diff) == pytest.approx(2.0 / 3.0)

    # Equity 0->1->2->1->3: peak-to-trough drop of 1 at the dip.
    equity = np.array([0.0, 1.0, 2.0, 1.0, 3.0])
    assert max_drawdown(equity) == pytest.approx(1.0)


def test_max_drawdown_monotone_curve_is_zero() -> None:
    assert max_drawdown(np.array([0.0, 1.0, 2.0, 3.0])) == 0.0


def test_percentage_metrics_no_trades_are_zero() -> None:
    actions = np.zeros(5, dtype=int)
    assert percentage_winning_trades(actions, np.arange(5.0)) == 0.0
    assert percentage_directional(actions, np.arange(5.0)) == 0.0
