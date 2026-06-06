"""Builder for notebooks/05_walkforward_evaluation.ipynb (run once, strip outputs)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))  # noqa: E731
co = lambda s: cells.append(new_code_cell(s))  # noqa: E731

md(
    """# 05 · Walk-forward out-of-sample evaluation

The single-pass backtest is **in-sample** for a learned agent — it scores PPO on
the data it trained on. This notebook closes Pieza 5 with a **walk-forward**
(rolling-origin) evaluation: the timeline is split into ordered folds and every
strategy is trained on the past and scored on the held-out future, so all eight
policies are compared out-of-sample on equal footing.

> Budgets here are small (synthetic data, short PPO training) so the notebook
> runs quickly. The **real** production run — long PPO training on real
> OMIE+ESIOS features — is `scripts/train_ppo_production.py`, designed to run on a
> rented GPU pod (see the report `reports/walkforward.md`).
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
from mibel_trading.eval import walk_forward_evaluate


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


def load_real_or_synthetic():
    try:
        from mibel_trading.data.loaders import load_features

        feats = load_features("2023-01-01", "2023-03-31")
        if len(feats) < 500:
            raise ValueError("too few real rows")
        return feats, "real (OMIE+ESIOS)"
    except Exception as exc:  # noqa: BLE001 - any failure -> synthetic fallback
        print(f"real features unavailable ({type(exc).__name__}); using synthetic")
        return synthetic_features(), "synthetic"


features, source = load_real_or_synthetic()
print(f"features: {len(features)} rows | source: {source}")'''
)

md("## 1 · Providers and walk-forward run")

co(
    '''PPO_BUDGET = 3000  # tiny — illustration only; the pod run uses >100k

providers = {
    "random": lambda tr, te: RandomStrategy(seed=0),
    "buy_hold": lambda tr, te: NaiveBuyHoldStrategy(),
    "rule_based": lambda tr, te: RuleBasedSpreadStrategy(window=24, threshold_std=1.0),
    "BENCH": lambda tr, te: BENCH(len(te) - 1),
    "BENCHVWAP": lambda tr, te: BENCHVWAP(len(te) - 1, window_size=10),
    "BENCHPLUS": lambda tr, te: BENCHPLUS(len(te) - 1, window_size=10),
    "ppo_baseline": lambda tr, te: train_ppo_baseline(
        lambda: MibelTradingEnv(tr), total_timesteps=PPO_BUDGET, seed=42
    ),
    "ppo_with_bc": lambda tr, te: train_ppo_with_bc(
        lambda: MibelTradingEnv(tr),
        RuleBasedSpreadStrategy(window=24),
        total_timesteps=PPO_BUDGET,
        n_bc_episodes=3,
        bc_epochs=10,
        seed=42,
    ),
}

result = walk_forward_evaluate(features, providers, n_folds=4, scheme="expanding")
print(f"folds: {len(result.folds)} | rows evaluated: {len(result.per_fold)}")
for f in result.folds:
    print(f"  fold {f.index}: train[{f.train_start}:{f.train_end}] "
          f"test[{f.test_start}:{f.test_end}] (n_test={f.n_test})")'''
)

md("## 2 · Out-of-sample summary (mean across folds)")

co(
    '''means = result.summary.xs("mean", axis=1, level=1)
means = means.sort_values("total_pnl", ascending=False)
print(means.to_string(float_format=lambda v: f"{v:,.4f}"))'''
)

md("## 3 · Per-fold out-of-sample PnL")

co(
    '''pivot = result.per_fold.pivot(index="fold", columns="strategy", values="total_pnl")
ax = pivot.plot(kind="bar", figsize=(12, 4.5))
ax.axhline(0, color="0.5", lw=0.8)
ax.set_title("Out-of-sample total PnL per fold")
ax.set_xlabel("fold"); ax.set_ylabel("EUR")
ax.legend(ncol=4, fontsize=8)
plt.tight_layout(); plt.show()'''
)

md(
    """## 4 · Honest read

Out-of-sample is a tougher, fairer test than the single-pass backtest, and the
ranking typically **compresses**: strategies that looked strong in-sample give
back some edge on unseen folds. With only a few-thousand-timestep budget, the PPO
agents are **not expected to beat the rule-based expert or the benchmarks** here —
PPO needs far more experience to shape a useful value function, and the synthetic
generator is a friendly stand-in for the real market.

That is exactly the point of Pieza 5: it provides the *apparatus* to settle the
question fairly, not the verdict. The verdict requires the **production run** —
`scripts/train_ppo_production.py` with `--timesteps 500000` on real OMIE+ESIOS
features — which is compute-heavy and meant for a rented GPU pod, the same way the
Schwartz-Smith production fit was produced in `mibel-derivatives`. Its artefacts
(model `.zip`, `walkforward_summary.csv`, `metadata.json`) drop straight back into
the repo.
"""
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
out = Path("notebooks/05_walkforward_evaluation.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
