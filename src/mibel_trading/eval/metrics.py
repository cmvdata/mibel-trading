"""Trading metrics for MIBEL strategy evaluation.

Implements the headline diagnostics of Demir (2023), §11.3.6 — percentage of
winning trades (PT) and percentage of correct directional calls (PD) — plus
per-trade PnL and maximum drawdown. All functions are pure NumPy and take plain
arrays so they compose with the backtest engine and with synthetic test data.

Action convention (matching the env / strategies): ``0=HOLD``, ``1=BUY``,
``2=SELL``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "max_drawdown",
    "percentage_directional",
    "percentage_winning_trades",
    "pnl_per_trade",
]

HOLD = 0
BUY = 1
SELL = 2


def pnl_per_trade(
    actions: npt.ArrayLike, pnl_series: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """PnL of the steps on which a trade (action != HOLD) was taken.

    Parameters
    ----------
    actions : array-like of int
        Action taken at each step.
    pnl_series : array-like of float
        Realised PnL attributed to each step.

    Returns
    -------
    numpy.ndarray
        The subset of ``pnl_series`` where ``actions != 0`` (empty if no trades).
    """
    a = np.asarray(actions)
    pnl = np.asarray(pnl_series, dtype=np.float64)
    if a.shape != pnl.shape:
        raise ValueError("`actions` and `pnl_series` must have the same shape.")
    return pnl[a != HOLD]


def percentage_winning_trades(
    actions: npt.ArrayLike, pnl_series: npt.ArrayLike
) -> float:
    """PT (Demir §11.3.6): fraction of trades with strictly positive PnL.

    Returns ``0.0`` when no trade was taken.
    """
    trade_pnl = pnl_per_trade(actions, pnl_series)
    if trade_pnl.size == 0:
        return 0.0
    return float(np.mean(trade_pnl > 0.0))


def percentage_directional(
    actions: npt.ArrayLike, dam_diff: npt.ArrayLike
) -> float:
    """PD (Demir §11.3.6): fraction of trades that called the price direction.

    A BUY is correct when the subsequent DAM move ``dam_diff`` is positive; a
    SELL is correct when it is negative. HOLD steps are ignored. Returns ``0.0``
    when no directional trade was taken.

    Parameters
    ----------
    actions : array-like of int
        Action at each step.
    dam_diff : array-like of float
        Directional DAM move attributed to each step (``price_{t+1} - price_t``).
    """
    a = np.asarray(actions)
    diff = np.asarray(dam_diff, dtype=np.float64)
    if a.shape != diff.shape:
        raise ValueError("`actions` and `dam_diff` must have the same shape.")

    is_buy = a == BUY
    is_sell = a == SELL
    n_dir = int(is_buy.sum() + is_sell.sum())
    if n_dir == 0:
        return 0.0
    correct = int(np.sum(diff[is_buy] > 0.0) + np.sum(diff[is_sell] < 0.0))
    return correct / n_dir


def max_drawdown(equity_curve: npt.ArrayLike) -> float:
    """Maximum peak-to-trough decline of an equity curve.

    Returned as a non-negative magnitude (``0.0`` for a never-declining curve).

    Parameters
    ----------
    equity_curve : array-like of float
        Cumulative PnL (or net asset value) per step.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    drawdown = running_max - eq
    return float(np.max(drawdown))
