"""Builder for notebooks/03_benchmarks_comparison.ipynb (run once, strip outputs)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))  # noqa: E731
co = lambda s: cells.append(new_code_cell(s))  # noqa: E731

md(
    """# 03 · Benchmark comparison — the full MIBEL strategy stack

Compares the **six** policies of the trading stack on the `MibelTradingEnv`:

| Group | Strategy |
|-------|----------|
| baseline | `RandomStrategy`, `NaiveBuyHoldStrategy` |
| signal | `RuleBasedSpreadStrategy` (Pieza 2) |
| benchmarks | `BENCH`, `BENCHVWAP`, `BENCHPLUS` (Demir pp. 138-139) |

The benchmarks are the bar the rule-based signal — and later the PPO agent — must
clear. Uses real OMIE+ESIOS features if reachable, otherwise a reproducible
synthetic series.
"""
)

co(
    '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mibel_trading.env import MibelTradingEnv
from mibel_trading.strategies import (
    NaiveBuyHoldStrategy,
    RandomStrategy,
    RuleBasedSpreadStrategy,
)
from mibel_trading.benchmarks import BENCH, BENCHVWAP, BENCHPLUS
from mibel_trading.eval import compare_strategies, run_backtest
from mibel_trading.eval.metrics import percentage_directional, percentage_winning_trades


def synthetic_features(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    dam = 65 + 30 * np.sin(t * 2 * np.pi / 24) + 8 * np.sin(t * 2 * np.pi / 168)
    dam += rng.normal(0, 7, n)
    gap = np.zeros(n)
    for i in range(1, n):
        gap[i] = 0.9 * gap[i - 1] + rng.normal(0, 4)
    df = pd.DataFrame(
        {
            "price_eur_mwh": dam,
            "esios_634": dam - gap,
            "esios_676": rng.uniform(0, 120, n),
            "esios_686": rng.normal(0, 50, n),
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


def load_real_or_synthetic():
    try:
        from mibel_trading.data.loaders import load_features

        feats = load_features("2023-01-01", "2023-03-31")
        if len(feats) < 200:
            raise ValueError("too few real rows")
        return feats, "real (OMIE+ESIOS)"
    except Exception as exc:  # noqa: BLE001 - any failure -> synthetic fallback
        print(f"real features unavailable ({type(exc).__name__}); using synthetic")
        return synthetic_features(), "synthetic"


features, source = load_real_or_synthetic()
horizon = len(features) - 1  # env runs len-1 steps; benchmarks need the horizon
print(f"features: {len(features)} hourly rows | source: {source} | horizon: {horizon}")'''
)

md("## 1 · Compare the six strategies")

co(
    '''def env_factory():
    return MibelTradingEnv(features)


strategies = {
    "random": RandomStrategy(seed=0),
    "buy_hold": NaiveBuyHoldStrategy(),
    "rule_based": RuleBasedSpreadStrategy(window=24, threshold_std=1.0),
    "BENCH": BENCH(horizon),
    "BENCHVWAP": BENCHVWAP(horizon, window_size=10),
    "BENCHPLUS": BENCHPLUS(horizon, window_size=10, vol_window=24),
}

table = compare_strategies(strategies, env_factory)
print(table.to_string(float_format=lambda v: f"{v:,.4f}"))'''
)

md("## 2 · Equity curves (six overlaid)")

co(
    '''results = {name: run_backtest(strat, env_factory(), seed=42)
           for name, strat in strategies.items()}

fig, ax = plt.subplots(figsize=(12, 4.5))
for name, res in results.items():
    ax.plot(res["equity_curve"], label=name, lw=1.2)
ax.axhline(0, color="0.6", lw=0.8)
ax.set_title("Cumulative PnL — six strategies")
ax.set_xlabel("step (hour)"); ax.set_ylabel("EUR")
ax.legend(ncol=3, fontsize=8)
plt.tight_layout(); plt.show()'''
)

md("## 3 · Full metrics table (incl. PT / PD)")

co(
    '''dam = features["price_eur_mwh"].to_numpy()
dam_diff = np.diff(dam)

rows = {}
for name, res in results.items():
    actions, pnl = res["actions"], res["pnl_series"]
    rows[name] = {
        "total_pnl": res["total_pnl"],
        "n_trades": res["n_trades"],
        "win_rate": res["win_rate"],
        "sharpe_proxy": res["sharpe_proxy"],
        "max_drawdown": res["max_drawdown"],
        "PT": percentage_winning_trades(actions, pnl),
        "PD": percentage_directional(actions, dam_diff[: len(actions)]),
    }
metrics = pd.DataFrame.from_dict(rows, orient="index")
print(metrics.to_string(float_format=lambda v: f"{v:,.4f}"))'''
)

md(
    """## 4 · Honest read

Among the benchmarks, **BENCHVWAP** is *usually* expected to dominate **BENCH**:
spreading the round trip over a window diversifies execution timing, so it is less
hostage to the single first/last print that BENCH bets everything on. But this is
a tendency, not a law — the ranking is data-dependent. On the synthetic series
here, for instance, BENCH's one round trip happens to land positive while VWAP's
distributed legs catch adverse moves, so BENCH actually edges VWAP on total PnL
even though VWAP wins on win-rate / PT / PD. **BENCHPLUS** only pulls ahead when
its volatility gate is informative — i.e. when high-volatility windows really do
offer a better entry/exit; if the gate fires on noise it simply trades like
BENCHVWAP for no gain, and can even lag it (as it does here).

Whether the **rule-based** signal beats the benchmarks is the empirical crux: on a
mean-reverting synthetic spread it can, but that is the hypothesis it encodes, not
a guarantee on real OMIE+ESIOS data. The benchmarks here are precisely the bar the
PPO agent of Pieza 4 has to clear to justify the added complexity.
"""
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
out = Path("notebooks/03_benchmarks_comparison.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
