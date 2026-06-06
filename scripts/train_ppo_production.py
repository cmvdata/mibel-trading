"""Production PPO training entrypoint — designed to run on a rented GPU pod.

CI exercises the agent only with tiny budgets on synthetic data; the *real*
training (>100k timesteps on real OMIE + ESIOS features) is compute-heavy and is
meant to run on a rented pod (e.g. RunPod), mirroring how
``mibel-derivatives/scripts/run_production_fit.py`` produced the Schwartz-Smith
production fit. This script is that turnkey entrypoint.

It (1) loads real features (or synthetic, with ``--synthetic``), (2) trains a PPO
agent (baseline or behaviour-cloning warm-started), (3) runs a walk-forward
out-of-sample evaluation against the rule-based and benchmark policies, and
(4) writes the model plus metrics to ``--out``.

Examples
--------
On a pod with ESIOS_TOKEN set::

    python scripts/train_ppo_production.py --start 2019-01-01 --end 2024-12-31 \
        --timesteps 500000 --bc --out artifacts/ppo_production

Smoke test locally (no token, no GPU)::

    python scripts/train_ppo_production.py --synthetic --timesteps 5000 \
        --out artifacts/ppo_smoke
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from mibel_trading.agents import train_ppo_baseline, train_ppo_with_bc
from mibel_trading.benchmarks import BENCH, BENCHPLUS, BENCHVWAP
from mibel_trading.env import MibelTradingEnv
from mibel_trading.eval import walk_forward_evaluate
from mibel_trading.strategies import (
    NaiveBuyHoldStrategy,
    RandomStrategy,
    RuleBasedSpreadStrategy,
)


def _synthetic_features(n: int = 8760, seed: int = 7) -> pd.DataFrame:
    """One synthetic year with a mean-reverting DAM/ajuste spread."""
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


def load_features(start: str, end: str, synthetic: bool) -> tuple[pd.DataFrame, str]:
    """Real OMIE+ESIOS features, or synthetic when requested / unavailable."""
    if synthetic:
        return _synthetic_features(), "synthetic"
    from mibel_trading.data.loaders import load_features as _load

    feats = _load(start, end)
    if len(feats) < 500:
        raise SystemExit(
            f"only {len(feats)} real rows in [{start}, {end}] — too few to train. "
            "Check ESIOS_TOKEN / date range, or pass --synthetic."
        )
    return feats, f"real OMIE+ESIOS [{start}, {end}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--bc", action="store_true", help="behaviour-cloning warm-start")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("artifacts/ppo_production"))
    args = parser.parse_args()

    features, source = load_features(args.start, args.end, args.synthetic)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"features: {len(features)} rows | source: {source}")

    def env_factory() -> MibelTradingEnv:
        return MibelTradingEnv(features)

    # --- Train the production agent ---------------------------------------- #
    expert = RuleBasedSpreadStrategy(window=24, threshold_std=1.0)
    if args.bc:
        print(f"training PPO+BC for {args.timesteps:,} timesteps ...")
        agent = train_ppo_with_bc(
            env_factory, expert, total_timesteps=args.timesteps, seed=args.seed
        )
        tag = "ppo_with_bc"
    else:
        print(f"training PPO baseline for {args.timesteps:,} timesteps ...")
        agent = train_ppo_baseline(env_factory, total_timesteps=args.timesteps, seed=args.seed)
        tag = "ppo_baseline"

    model_path = args.out / f"{tag}.zip"
    agent.save(str(model_path))
    print(f"saved model -> {model_path}")

    # --- Walk-forward out-of-sample evaluation vs the stack ----------------- #
    print(f"walk-forward evaluation ({args.n_folds} folds) ...")
    bc_budget = max(args.timesteps // args.n_folds, 1000)
    providers = {
        "random": lambda tr, te: RandomStrategy(seed=args.seed),
        "buy_hold": lambda tr, te: NaiveBuyHoldStrategy(),
        "rule_based": lambda tr, te: RuleBasedSpreadStrategy(window=24),
        "BENCH": lambda tr, te: BENCH(len(te) - 1),
        "BENCHVWAP": lambda tr, te: BENCHVWAP(len(te) - 1, window_size=10),
        "BENCHPLUS": lambda tr, te: BENCHPLUS(len(te) - 1, window_size=10),
        tag: lambda tr, te: train_ppo_baseline(
            lambda: MibelTradingEnv(tr), total_timesteps=bc_budget, seed=args.seed
        ),
    }
    wf = walk_forward_evaluate(features, providers, n_folds=args.n_folds, seed=args.seed)

    wf.per_fold.to_csv(args.out / "walkforward_per_fold.csv", index=False)
    wf.summary.to_csv(args.out / "walkforward_summary.csv")
    metadata = {
        "source": source,
        "n_rows": int(len(features)),
        "timesteps": int(args.timesteps),
        "behaviour_cloning": bool(args.bc),
        "n_folds": int(args.n_folds),
        "seed": int(args.seed),
        "platform": platform.platform(),
        "oos_total_pnl_mean": {
            str(k): float(v) for k, v in wf.summary[("total_pnl", "mean")].items()
        },
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved metrics -> {args.out}")
    print("\nOut-of-sample mean total PnL by strategy:")
    print(wf.summary[("total_pnl", "mean")].sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
