"""Tests for MibelTradingEnv (synthetic, reproducible market data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from mibel_trading.env import MibelTradingEnv


@pytest.fixture
def features() -> pd.DataFrame:
    """240 hours of reproducible synthetic DAM + ancillary data."""
    n = 240
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    price = 60.0 + 25.0 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 6, n)
    df = pd.DataFrame(
        {
            "price_eur_mwh": price,
            "esios_634": rng.uniform(0, 200, n),
            "esios_676": rng.uniform(0, 120, n),
            "esios_686": rng.normal(0, 50, n),
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


def test_env_passes_env_checker(features: pd.DataFrame) -> None:
    """The env satisfies the gymnasium API contract."""
    env = MibelTradingEnv(features)
    check_env(env, skip_render_check=True)


def test_env_reset_returns_obs(features: pd.DataFrame) -> None:
    env = MibelTradingEnv(features)
    obs, info = env.reset(seed=0)
    assert obs.shape == (9,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    # Fresh book: flat position, zero cash.
    assert obs[5] == pytest.approx(0.0)  # position
    assert obs[6] == pytest.approx(0.0)  # cash


def test_env_step_returns_5tuple(features: pd.DataFrame) -> None:
    env = MibelTradingEnv(features)
    env.reset(seed=0)
    result = env.step(1)  # Buy
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert {"pnl", "position", "cash"} <= set(info)


def test_env_position_constraints(features: pd.DataFrame) -> None:
    """After 11 consecutive Buys the position is capped at position_max (10)."""
    env = MibelTradingEnv(features, position_max=10.0, position_min=-10.0)
    env.reset(seed=0)
    for _ in range(11):
        obs, _, _, _, info = env.step(1)  # Buy 1 MWh
    assert info["position"] == pytest.approx(10.0)
    assert obs[5] == pytest.approx(10.0)


def test_env_position_short_constraint(features: pd.DataFrame) -> None:
    """Symmetric floor: 11 consecutive Sells cap the position at position_min."""
    env = MibelTradingEnv(features, position_max=10.0, position_min=-10.0)
    env.reset(seed=0)
    info: dict = {}
    for _ in range(11):
        _, _, _, _, info = env.step(2)  # Sell 1 MWh
    assert info["position"] == pytest.approx(-10.0)


def test_env_reward_bounded(features: pd.DataFrame) -> None:
    """Reward stays in [-2, 0] for every step under random actions."""
    env = MibelTradingEnv(features)
    env.reset(seed=0)
    rng = np.random.default_rng(7)
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(rng.integers(0, 3))
        _, reward, terminated, truncated, _ = env.step(action)
        assert -2.0 <= reward <= 0.0


def test_env_runs_full_episode(features: pd.DataFrame) -> None:
    """A Hold-only episode terminates exactly at the last price index."""
    env = MibelTradingEnv(features)
    env.reset(seed=0)
    steps = 0
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(0)  # Hold
        steps += 1
    assert steps == len(features) - 1


def test_env_requires_price_column() -> None:
    bad = pd.DataFrame({"not_price": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        MibelTradingEnv(bad)


def test_env_requires_two_rows() -> None:
    one = pd.DataFrame({"price_eur_mwh": [50.0]})
    with pytest.raises(ValueError):
        MibelTradingEnv(one)
