"""Tests for rule-based strategies (synthetic observations)."""

from __future__ import annotations

import numpy as np

from mibel_trading.strategies import (
    NaiveBuyHoldStrategy,
    RandomStrategy,
    RuleBasedSpreadStrategy,
)

HOLD, BUY, SELL = 0, 1, 2


def _obs(dam: float, secundaria: float) -> np.ndarray:
    """Build a 9-dim observation with the given DAM price and secondary band."""
    o = np.zeros(9, dtype=np.float64)
    o[0] = dam  # dam_t
    o[2] = secundaria  # secundaria
    return o


def test_rule_based_hold_when_no_signal() -> None:
    """During the warm-up window the strategy only holds."""
    strat = RuleBasedSpreadStrategy(window=24, threshold_std=1.0)
    for _ in range(23):  # fewer than `window` observations
        assert strat.act(_obs(50.0, 50.0), {}) == HOLD


def test_rule_based_sells_when_dam_overvalued() -> None:
    """A large positive spread spike (DAM rich) triggers a SELL."""
    rng = np.random.default_rng(0)
    strat = RuleBasedSpreadStrategy(window=10, threshold_std=1.0)
    # Fill the window with a stable spread around zero.
    for _ in range(10):
        noise = float(rng.normal(0, 0.5))
        assert strat.act(_obs(50.0 + noise, 50.0), {}) == HOLD
    # Now DAM spikes far above the secondary band -> spread very high -> SELL.
    assert strat.act(_obs(200.0, 50.0), {}) == SELL


def test_rule_based_buys_when_dam_undervalued() -> None:
    """A large negative spread (DAM cheap) triggers a BUY."""
    rng = np.random.default_rng(1)
    strat = RuleBasedSpreadStrategy(window=10, threshold_std=1.0)
    for _ in range(10):
        noise = float(rng.normal(0, 0.5))
        assert strat.act(_obs(50.0 + noise, 50.0), {}) == HOLD
    # DAM collapses far below the secondary band -> spread very low -> BUY.
    assert strat.act(_obs(50.0, 200.0), {}) == BUY


def test_rule_based_holds_within_band() -> None:
    """A spread within the threshold band keeps the strategy flat."""
    strat = RuleBasedSpreadStrategy(window=10, threshold_std=2.0)
    rng = np.random.default_rng(2)
    for _ in range(10):
        strat.act(_obs(50.0 + float(rng.normal(0, 1.0)), 50.0), {})
    # A spread one std out, with a 2-std threshold, is not enough to trade.
    assert strat.act(_obs(51.0, 50.0), {}) == HOLD


def test_naive_buy_hold_always_buys() -> None:
    strat = NaiveBuyHoldStrategy()
    rng = np.random.default_rng(3)
    for _ in range(50):
        obs = _obs(float(rng.uniform(0, 200)), float(rng.uniform(0, 200)))
        assert strat.act(obs, {}) == BUY


def test_random_strategy_distribution() -> None:
    """The random baseline emits all three actions."""
    strat = RandomStrategy(seed=0)
    actions = [strat.act(_obs(50.0, 50.0), {}) for _ in range(300)]
    assert set(actions) == {HOLD, BUY, SELL}
    # Each action appears a non-trivial number of times (roughly uniform).
    for a in (HOLD, BUY, SELL):
        assert actions.count(a) > 50


def test_random_strategy_reset_reproducible() -> None:
    strat = RandomStrategy(seed=123)
    first = [strat.act(_obs(1.0, 1.0), {}) for _ in range(20)]
    strat.reset()
    second = [strat.act(_obs(1.0, 1.0), {}) for _ in range(20)]
    assert first == second
