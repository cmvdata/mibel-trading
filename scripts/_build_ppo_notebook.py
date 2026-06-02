"""Builder for notebooks/04_ppo_agent.ipynb (run once, then strip outputs)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))  # noqa: E731
co = lambda s: cells.append(new_code_cell(s))  # noqa: E731

md(
    """# 04 · PPO agent + behaviour cloning — full stack comparison

Trains the RL agent of Pieza 4 and pits it against the whole stack:

* `ppo_baseline` — PPO from scratch.
* `ppo_with_bc` — PPO warm-started by **behaviour cloning** the
  `RuleBasedSpreadStrategy` expert (Demir-style bootstrap).

…compared with the six earlier policies (random, buy-hold, rule-based, BENCH,
BENCHVWAP, BENCHPLUS) on the same `MibelTradingEnv`.

> Training budgets here are modest (10k timesteps on synthetic data) — enough to
> exercise the pipeline, **not** to draw performance conclusions. The real test
> is long training (>100k steps) on real OMIE+ESIOS data, which is out of scope
> for a notebook / CI.
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
from mibel_trading.agents import train_ppo_baseline, train_ppo_with_bc
from mibel_trading.eval import compare_strategies, run_backtest
from mibel_trading.eval.metrics import percentage_directional, percentage_winning_trades


def synthetic_features(n=1500, seed=7):
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


features = synthetic_features()
horizon = len(features) - 1


def env_factory():
    return MibelTradingEnv(features)


print(f"features: {len(features)} rows | horizon: {horizon}")'''
)

md("## 1 · Train the PPO agents (10k timesteps each)")

co(
    '''TOTAL_TIMESTEPS = 10_000

ppo_baseline = train_ppo_baseline(env_factory, total_timesteps=TOTAL_TIMESTEPS, seed=42)
print("ppo_baseline trained")

ppo_with_bc = train_ppo_with_bc(
    env_factory,
    RuleBasedSpreadStrategy(window=24, threshold_std=1.0),
    total_timesteps=TOTAL_TIMESTEPS,
    n_bc_episodes=5,
    bc_epochs=20,
    seed=42,
)
print("ppo_with_bc trained")'''
)

md("## 2 · Compare the eight strategies")

co(
    '''strategies = {
    "random": RandomStrategy(seed=0),
    "buy_hold": NaiveBuyHoldStrategy(),
    "rule_based": RuleBasedSpreadStrategy(window=24, threshold_std=1.0),
    "BENCH": BENCH(horizon),
    "BENCHVWAP": BENCHVWAP(horizon, window_size=10),
    "BENCHPLUS": BENCHPLUS(horizon, window_size=10, vol_window=24),
    "ppo_baseline": ppo_baseline,
    "ppo_with_bc": ppo_with_bc,
}

table = compare_strategies(strategies, env_factory)
print(table.to_string(float_format=lambda v: f"{v:,.4f}"))'''
)

md("## 3 · Equity curves (eight overlaid)")

co(
    '''results = {name: run_backtest(strat, env_factory(), seed=42)
           for name, strat in strategies.items()}

fig, ax = plt.subplots(figsize=(12, 5))
for name, res in results.items():
    style = "--" if name.startswith("ppo") else "-"
    lw = 2.0 if name.startswith("ppo") else 1.0
    ax.plot(res["equity_curve"], style, label=name, lw=lw)
ax.axhline(0, color="0.6", lw=0.8)
ax.set_title("Cumulative PnL — eight strategies (PPO dashed)")
ax.set_xlabel("step (hour)"); ax.set_ylabel("EUR")
ax.legend(ncol=4, fontsize=8)
plt.tight_layout(); plt.show()'''
)

md("## 4 · Full metrics table (incl. PT / PD)")

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
    """## 5 · Honest read

**Behaviour cloning can help or hurt** — it is not guaranteed to beat the
from-scratch PPO. On a short synthetic run the warm-start can give `ppo_with_bc`
a head start (it begins near the rule-based prior), but it can equally **overfit
the expert** and inherit its blind spots, so on some seeds/horizons it lands
*below* `ppo_baseline`. Likewise, with only 10k timesteps neither PPO is expected
to reliably clear the benchmarks — PPO needs far more experience to shape a useful
value function.

The honest conclusion for Pieza 4 is again methodological: the full pipeline
(expert demos → BC → weight transfer → PPO → evaluation as a drop-in
`BaseStrategy`) works end-to-end and is comparable on equal footing with every
other policy. Settling whether PPO (with or without BC) actually *wins* requires
long training (>100k steps) on real OMIE+ESIOS data — deliberately out of scope
for CI, and the subject of the next phase.
"""
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
out = Path("notebooks/04_ppo_agent.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
