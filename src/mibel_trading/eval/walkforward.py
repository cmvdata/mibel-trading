"""Walk-forward (rolling-origin) out-of-sample evaluation.

The single-pass backtest in :mod:`mibel_trading.eval.backtest` is in-sample for a
learned agent: it scores the policy on the very data it trained on. Walk-forward
evaluation fixes that by splitting the timeline into ordered folds and, for each
fold, **training on the past and scoring on the held-out future**. Non-learned
policies (rule-based, benchmarks) ignore the training window and are simply
scored out-of-sample, so every strategy is compared on the same footing.

A strategy is supplied as a *provider* — a callable
``provider(train_df, test_df) -> BaseStrategy`` — which lets each fold (re)build
or (re)train the strategy from its own training window. This keeps the harness
agnostic to whether a strategy learns: a benchmark provider ignores ``train_df``;
a PPO provider trains on it with whatever budget the caller passes (tiny for CI,
large on a GPU pod for the real run).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mibel_trading.env import MibelTradingEnv
from mibel_trading.eval.backtest import run_backtest
from mibel_trading.eval.metrics import (
    percentage_directional,
    percentage_winning_trades,
)
from mibel_trading.strategies.rule_based import BaseStrategy

__all__ = [
    "WalkForwardFold",
    "WalkForwardResult",
    "walk_forward_evaluate",
    "walk_forward_splits",
]

#: ``provider(train_features, test_features) -> BaseStrategy``.
StrategyProvider = Callable[[pd.DataFrame, pd.DataFrame], BaseStrategy]

_METRIC_COLS = [
    "total_pnl",
    "n_trades",
    "win_rate",
    "sharpe_proxy",
    "max_drawdown",
    "PT",
    "PD",
]


@dataclass(frozen=True)
class WalkForwardFold:
    """One rolling-origin split (all bounds are half-open row indices)."""

    index: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start


def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    scheme: str = "expanding",
    min_train_frac: float = 0.3,
) -> list[WalkForwardFold]:
    """Build ordered train/test folds over ``n`` time-ordered rows.

    Parameters
    ----------
    n : int
        Number of rows in the (time-sorted) dataset.
    n_folds : int, default 5
        Number of out-of-sample test folds.
    scheme : {"expanding", "rolling"}, default "expanding"
        ``"expanding"`` trains on everything before the test block;
        ``"rolling"`` trains on a fixed-length window (``min_train_frac * n``)
        immediately preceding it.
    min_train_frac : float, default 0.3
        Fraction of ``n`` reserved for the initial training window (and the
        rolling-window length).

    Returns
    -------
    list[WalkForwardFold]
        Contiguous, non-overlapping test blocks; the last absorbs any remainder.
    """
    if n_folds < 1:
        raise ValueError("`n_folds` must be >= 1.")
    if not 0.0 < min_train_frac < 1.0:
        raise ValueError("`min_train_frac` must lie in (0, 1).")

    initial_train = max(int(round(min_train_frac * n)), 2)
    remaining = n - initial_train
    if remaining < 2 * n_folds:
        raise ValueError(
            f"not enough rows: need >= {2 * n_folds} after the initial "
            f"{initial_train}-row train window, have {remaining}."
        )

    test_size = remaining // n_folds
    folds: list[WalkForwardFold] = []
    for k in range(n_folds):
        test_start = initial_train + k * test_size
        test_end = n if k == n_folds - 1 else test_start + test_size
        train_start = 0 if scheme == "expanding" else max(0, test_start - initial_train)
        if scheme not in ("expanding", "rolling"):
            raise ValueError("`scheme` must be 'expanding' or 'rolling'.")
        folds.append(
            WalkForwardFold(
                index=k,
                train_start=train_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


@dataclass(frozen=True)
class WalkForwardResult:
    """Outcome of a walk-forward run.

    Attributes
    ----------
    per_fold : pandas.DataFrame
        One row per ``(strategy, fold)`` with the out-of-sample metrics.
    summary : pandas.DataFrame
        Per-strategy mean and std of each metric across folds (the headline
        out-of-sample comparison).
    folds : list[WalkForwardFold]
        The splits used.
    """

    per_fold: pd.DataFrame
    summary: pd.DataFrame
    folds: list[WalkForwardFold]


def walk_forward_evaluate(
    features: pd.DataFrame,
    providers: Mapping[str, StrategyProvider],
    n_folds: int = 5,
    scheme: str = "expanding",
    min_train_frac: float = 0.3,
    seed: int = 42,
) -> WalkForwardResult:
    """Score each strategy provider out-of-sample across walk-forward folds.

    Parameters
    ----------
    features : pandas.DataFrame
        Time-sorted hourly features (``price_eur_mwh`` + ancillary columns), as
        produced by :func:`mibel_trading.data.loaders.load_features`.
    providers : mapping of name -> provider
        Each ``provider(train_df, test_df)`` returns the strategy to score on the
        fold's test window (training on ``train_df`` if it learns).
    n_folds, scheme, min_train_frac
        Forwarded to :func:`walk_forward_splits`.
    seed : int
        Backtest seed (per fold).

    Returns
    -------
    WalkForwardResult
    """
    if len(providers) == 0:
        raise ValueError("`providers` must contain at least one strategy.")
    folds = walk_forward_splits(len(features), n_folds, scheme, min_train_frac)

    records: list[dict] = []
    for fold in folds:
        train_df = features.iloc[fold.train_start : fold.train_end]
        test_df = features.iloc[fold.test_start : fold.test_end]
        dam_diff = np.diff(test_df["price_eur_mwh"].to_numpy())
        for name, provider in providers.items():
            strategy = provider(train_df, test_df)
            result = run_backtest(strategy, MibelTradingEnv(test_df), seed=seed)
            actions, pnl = result["actions"], result["pnl_series"]
            records.append(
                {
                    "strategy": name,
                    "fold": fold.index,
                    "n_train": fold.n_train,
                    "n_test": fold.n_test,
                    "total_pnl": result["total_pnl"],
                    "n_trades": result["n_trades"],
                    "win_rate": result["win_rate"],
                    "sharpe_proxy": result["sharpe_proxy"],
                    "max_drawdown": result["max_drawdown"],
                    "PT": percentage_winning_trades(actions, pnl),
                    "PD": percentage_directional(actions, dam_diff[: len(actions)]),
                }
            )

    per_fold = pd.DataFrame.from_records(records)
    summary = per_fold.groupby("strategy")[_METRIC_COLS].agg(["mean", "std"])
    return WalkForwardResult(per_fold=per_fold, summary=summary, folds=folds)
