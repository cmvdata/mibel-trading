"""Builder for notebooks/02_rule_based_backtest.ipynb (run once, strip outputs)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))  # noqa: E731
co = lambda s: cells.append(new_code_cell(s))  # noqa: E731

md(
    """# 02 · Rule-based DAM→ajuste strategy — backtest

Backtest of the rule-based strategy of **Demir (2023), Ch. 12** adapted to MIBEL:
a rolling **z-score on the DAM vs secondary-band spread**. We compare it against a
random baseline and a naive buy-and-hold, on the `MibelTradingEnv` from Pieza 1.

| Strategy | Logic |
|----------|-------|
| `RandomStrategy` | uniform action — baseline |
| `NaiveBuyHoldStrategy` | always buy — sanity check |
| `RuleBasedSpreadStrategy` | sell when DAM rich vs ajuste, buy when cheap |

> Uses real OMIE+ESIOS features if reachable on disk, otherwise a reproducible
> synthetic series so the notebook always runs.
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
from mibel_trading.eval import compare_strategies, run_backtest
from mibel_trading.eval.metrics import (
    max_drawdown,
    percentage_directional,
    percentage_winning_trades,
)


def synthetic_features(n=2000, seed=7):
    """Reproducible DAM + ancillary series with a mean-reverting spread."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    dam = 65 + 30 * np.sin(t * 2 * np.pi / 24) + 8 * np.sin(t * 2 * np.pi / 168)
    dam += rng.normal(0, 7, n)
    # Secondary band tracks DAM with a mean-reverting gap -> a tradeable spread.
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
print(f"features: {len(features)} hourly rows | source: {source}")
features.head()'''
)

md("## 1 · Compare strategies")

co(
    '''def env_factory():
    return MibelTradingEnv(features)


strategies = {
    "random": RandomStrategy(seed=0),
    "buy_hold": NaiveBuyHoldStrategy(),
    "rule_based": RuleBasedSpreadStrategy(window=24, threshold_std=1.0),
}

table = compare_strategies(strategies, env_factory)
print(table.to_string())'''
)

md("## 2 · Equity curves")

co(
    '''results = {name: run_backtest(strat, env_factory(), seed=42)
           for name, strat in strategies.items()}

fig, ax = plt.subplots(figsize=(11, 4))
for name, res in results.items():
    ax.plot(res["equity_curve"], label=name, lw=1.3)
ax.axhline(0, color="0.6", lw=0.8)
ax.set_title("Cumulative PnL (equity curve)")
ax.set_xlabel("step (hour)"); ax.set_ylabel("EUR")
ax.legend()
plt.tight_layout(); plt.show()'''
)

md("## 3 · Full metrics table (PnL / trades / win rate / Sharpe / MaxDD / PT / PD)")

co(
    '''dam = features["price_eur_mwh"].to_numpy()
dam_diff = np.diff(dam)  # per-step DAM move, aligned with the backtest steps

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

The rule-based strategy *should* beat the random baseline when the DAM-vs-ajuste
spread genuinely mean-reverts — that is the hypothesis it encodes. But this is
**not guaranteed**: if the spread is closer to a random walk (or the synthetic
generator above is mis-specified relative to the real market), the z-score signal
adds no edge and the strategy can underperform both baselines after the implicit
cost of trading every signal.

The honest conclusion for Pieza 2 is methodological: the engine, metrics
(PnL / win-rate / Sharpe proxy / MaxDD / PT / PD) and the comparison harness are
in place and validated. Whether the *specific* spread signal is profitable is an
empirical question to settle on real OMIE+ESIOS data, and is exactly the
benchmark the PPO agent of Pieza 4 must clear.
"""
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
out = Path("notebooks/02_rule_based_backtest.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
