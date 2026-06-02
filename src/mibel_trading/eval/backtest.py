"""Backtest engine for MIBEL trading strategies.

Drives a :class:`mibel_trading.env.MibelTradingEnv` with a
:class:`mibel_trading.strategies.rule_based.BaseStrategy`, replaying full
episodes and collecting the per-step series and summary statistics needed to
compare policies (PnL, trade count, win rate, an annualised Sharpe proxy, and
drawdown via :mod:`mibel_trading.eval.metrics`).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from mibel_trading.eval.metrics import max_drawdown
from mibel_trading.strategies.rule_based import BaseStrategy

__all__ = ["compare_strategies", "run_backtest"]

#: Hours per year — annualisation factor for the Sharpe proxy.
HOURS_PER_YEAR = 8760


def run_backtest(
    strategy: BaseStrategy,
    env: object,
    n_episodes: int = 1,
    seed: int = 42,
) -> dict:
    """Replay ``strategy`` on ``env`` and return per-step series + statistics.

    Parameters
    ----------
    strategy : BaseStrategy
        Policy under test. Reset at the start of each episode.
    env : MibelTradingEnv
        A fresh trading environment (gymnasium API).
    n_episodes : int, default 1
        Episodes to run; their step series are concatenated.
    seed : int, default 42
        Base seed; episode ``k`` uses ``seed + k`` for reproducibility.

    Returns
    -------
    dict
        Keys: ``total_pnl``, ``equity_curve``, ``positions``, ``actions``,
        ``rewards``, ``pnl_series``, ``n_trades``, ``win_rate``,
        ``sharpe_proxy``, ``max_drawdown``.
    """
    if n_episodes < 1:
        raise ValueError("`n_episodes` must be >= 1.")

    actions: list[int] = []
    positions: list[float] = []
    rewards: list[float] = []
    pnls: list[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        strategy.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(strategy.act(obs, info))
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            positions.append(float(info["position"]))
            rewards.append(float(reward))
            pnls.append(float(info["pnl"]))

    actions_arr = np.asarray(actions, dtype=int)
    pnl_arr = np.asarray(pnls, dtype=np.float64)
    equity = np.cumsum(pnl_arr) if pnl_arr.size else np.asarray([], dtype=np.float64)

    traded = actions_arr != 0
    n_trades = int(traded.sum())
    win_rate = float(np.mean(pnl_arr[traded] > 0.0)) if n_trades > 0 else 0.0

    if pnl_arr.size > 1 and pnl_arr.std() > 0.0:
        sharpe_proxy = float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(HOURS_PER_YEAR))
    else:
        sharpe_proxy = 0.0

    return {
        "total_pnl": float(pnl_arr.sum()),
        "equity_curve": equity,
        "positions": np.asarray(positions, dtype=np.float64),
        "actions": actions_arr,
        "rewards": np.asarray(rewards, dtype=np.float64),
        "pnl_series": pnl_arr,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "sharpe_proxy": sharpe_proxy,
        "max_drawdown": max_drawdown(equity),
    }


def compare_strategies(
    strategies: dict[str, BaseStrategy],
    env_factory: Callable[[], object],
    *,
    n_episodes: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Backtest several strategies on fresh envs and tabulate their metrics.

    Parameters
    ----------
    strategies : dict[str, BaseStrategy]
        Named strategies to compare.
    env_factory : callable
        Zero-argument callable returning a *fresh* env, so every strategy faces
        the same market from an identical reset.
    n_episodes, seed
        Forwarded to :func:`run_backtest`.

    Returns
    -------
    pandas.DataFrame
        One row per strategy, columns ``total_pnl``, ``n_trades``, ``win_rate``,
        ``sharpe_proxy``, ``max_drawdown``.
    """
    rows: dict[str, dict] = {}
    for name, strategy in strategies.items():
        result = run_backtest(strategy, env_factory(), n_episodes=n_episodes, seed=seed)
        rows[name] = {
            "total_pnl": result["total_pnl"],
            "n_trades": result["n_trades"],
            "win_rate": result["win_rate"],
            "sharpe_proxy": result["sharpe_proxy"],
            "max_drawdown": result["max_drawdown"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def directional_diff_from_positions(
    pnl_series: npt.ArrayLike, positions: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Recover the per-step DAM move from PnL and position (helper for PD).

    Since ``pnl_t = position_t * (price_{t+1} - price_t)``, the price move is
    ``pnl_t / position_t`` where the position is non-zero (and ``0`` otherwise,
    where direction is undefined).
    """
    pnl = np.asarray(pnl_series, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    diff = np.zeros_like(pnl)
    nz = pos != 0.0
    diff[nz] = pnl[nz] / pos[nz]
    return diff
