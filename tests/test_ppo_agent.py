"""Tests for the PPO agent and behaviour-cloning pipeline.

Training budgets are deliberately tiny (a few hundred to ~1000 timesteps on a
200-row synthetic env) so the whole module runs in well under a minute on CI.
These tests check the *plumbing* — interfaces, shapes, convergence of BC,
end-to-end training without crashes — not trading performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_trading.agents import PPOAgent, train_ppo_baseline, train_ppo_with_bc
from mibel_trading.agents.behaviour_cloning import collect_demonstrations, train_bc
from mibel_trading.env import MibelTradingEnv
from mibel_trading.eval import run_backtest
from mibel_trading.strategies import BaseStrategy, RuleBasedSpreadStrategy


def _features(n: int = 200, seed: int = 0) -> pd.DataFrame:
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


FEATURES = _features()


def env_factory() -> MibelTradingEnv:
    return MibelTradingEnv(FEATURES)


@pytest.fixture(scope="module")
def trained_baseline() -> PPOAgent:
    """A PPO trained once (no BC) and reused across tests."""
    return train_ppo_baseline(env_factory, total_timesteps=1000, seed=0)


# --------------------------------------------------------------------------- #
# Behaviour cloning                                                            #
# --------------------------------------------------------------------------- #
def test_collect_demonstrations_shapes() -> None:
    obs, acts = collect_demonstrations(
        env_factory(), RuleBasedSpreadStrategy(window=24), n_episodes=2, seed=0
    )
    n_steps = len(FEATURES) - 1
    assert obs.shape == (2 * n_steps, 9)
    assert acts.shape == (2 * n_steps,)
    assert obs.dtype == np.float32
    assert set(np.unique(acts)).issubset({0, 1, 2})


def test_train_bc_loss_decreases() -> None:
    obs, acts = collect_demonstrations(
        env_factory(), RuleBasedSpreadStrategy(window=24), n_episodes=3, seed=0
    )
    bc = train_bc(obs, acts, n_epochs=5, seed=42)
    assert hasattr(bc, "train_losses_")
    assert len(bc.train_losses_) == 5
    assert bc.train_losses_[-1] < bc.train_losses_[0]


# --------------------------------------------------------------------------- #
# Agent interface                                                              #
# --------------------------------------------------------------------------- #
def test_ppo_agent_implements_base_strategy_interface() -> None:
    agent = PPOAgent(env_factory())
    assert isinstance(agent, BaseStrategy)
    assert hasattr(agent, "act")
    assert hasattr(agent, "reset")
    assert agent.reset() is None


def test_ppo_agent_act_returns_valid_action(trained_baseline: PPOAgent) -> None:
    obs, _ = env_factory().reset(seed=0)
    action = trained_baseline.act(obs, {})
    assert action in (0, 1, 2)
    assert isinstance(action, int)


# --------------------------------------------------------------------------- #
# Training pipelines                                                           #
# --------------------------------------------------------------------------- #
def test_train_ppo_baseline_runs(trained_baseline: PPOAgent) -> None:
    assert isinstance(trained_baseline, PPOAgent)
    obs, _ = env_factory().reset(seed=1)
    assert trained_baseline.act(obs, {}) in (0, 1, 2)


def test_train_ppo_with_bc_runs_end_to_end() -> None:
    agent = train_ppo_with_bc(
        env_factory,
        RuleBasedSpreadStrategy(window=24),
        total_timesteps=1000,
        n_bc_episodes=2,
        bc_epochs=5,
        seed=0,
    )
    assert isinstance(agent, PPOAgent)
    obs, _ = env_factory().reset(seed=0)
    assert agent.act(obs, {}) in (0, 1, 2)


def test_ppo_agent_works_with_run_backtest(trained_baseline: PPOAgent) -> None:
    result = run_backtest(trained_baseline, MibelTradingEnv(FEATURES))
    assert "total_pnl" in result
    assert isinstance(result["total_pnl"], float)
    assert len(result["actions"]) == len(FEATURES) - 1
