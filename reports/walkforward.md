# Walk-forward out-of-sample evaluation (Pieza 5)

**Module:** `mibel_trading.eval.walkforward` · **Notebook:** `notebooks/05_walkforward_evaluation.ipynb`
**Production entrypoint:** `scripts/train_ppo_production.py`

The single-pass backtest (Pieza 2) is *in-sample* for a learned agent: it scores
PPO on the very data it trained on. Pieza 5 closes the stack with a **walk-forward
(rolling-origin)** evaluation so every policy — rule-based, the three benchmarks,
and the PPO agents — is trained on the past and scored on the held-out future,
out-of-sample and on equal footing.

---

## 1 · What was built

| File | Role |
|------|------|
| `src/mibel_trading/eval/walkforward.py` | `walk_forward_splits` (expanding / rolling folds) and `walk_forward_evaluate` (train-on-past, score-on-future per fold), returning `WalkForwardResult` (`per_fold` + `summary`). |
| `tests/test_walkforward.py` | 8 tests: split geometry (no leakage, contiguity, expanding vs rolling), non-learned evaluation, out-of-sample ordering, PPO-trained-per-fold, validation. |
| `notebooks/05_walkforward_evaluation.ipynb` | Walk-forward over all eight strategies with an out-of-sample summary and per-fold PnL. |
| `scripts/train_ppo_production.py` | Turnkey **GPU-pod** training entrypoint (real data, long budget) producing model + metrics artefacts. |

### Design — strategy *providers*
A strategy is supplied as `provider(train_df, test_df) -> BaseStrategy`. This keeps
the harness agnostic to whether a strategy learns: a benchmark provider ignores
`train_df`; a PPO provider trains on it with whatever budget the caller passes
(tiny for CI, large on a pod). No leakage: each fold's `train_end == test_start`.

---

## 2 · Representative result (synthetic, 4 folds — illustrative only)

CI / notebook budgets are deliberately tiny (1 500 synthetic rows, 3 000-timestep
PPO), so this is an apparatus check, **not** a performance verdict.

Out-of-sample mean across folds:

| Strategy | OOS total_pnl | win_rate | Sharpe | PD |
|----------|--------------:|---------:|-------:|---:|
| BENCHVWAP | 404.98 | 0.538 | 1.363 | 0.538 |
| rule_based | 262.96 | 0.457 | 1.697 | 0.515 |
| ppo_baseline | 210.06 | 0.425 | 2.930 | 0.553 |
| BENCHPLUS | 195.33 | 0.442 | 0.672 | 0.454 |
| buy_hold | 194.70 | 0.504 | 0.646 | 0.504 |
| random | 159.20 | 0.414 | 0.998 | 0.472 |
| BENCH | 21.61 | 0.500 | 0.722 | 0.750 |
| ppo_with_bc | −132.42 | 0.367 | −1.184 | 0.494 |

**Reading.** Out-of-sample the ranking **compresses** relative to the single-pass
backtest — the fair-test penalty. With a ~3k-timestep budget the PPO agents do
**not** beat the simpler policies (`ppo_with_bc` even overfits the expert and lags),
exactly as expected: PPO needs far more experience, and the synthetic generator is
a friendly stand-in for the real market. The positive PnL across most strategies is
a synthetic artefact (a mild upward drift in the test windows rewards long
exposure), which is why the verdict must come from real data.

---

## 3 · The production run (on a rented pod)

The real verdict is compute-heavy and out of CI scope — it runs on a rented GPU
pod, mirroring how `mibel-derivatives` produced its Schwartz-Smith production fit:

```bash
# On the pod, with ESIOS_TOKEN set:
python scripts/train_ppo_production.py \
    --start 2019-01-01 --end 2024-12-31 \
    --timesteps 500000 --bc --n-folds 5 \
    --out artifacts/ppo_production
```

Outputs (gitignored locally; promoted deliberately like the derivatives `.pkl`):

- `ppo_with_bc.zip` / `ppo_baseline.zip` — the trained stable-baselines3 model.
- `walkforward_summary.csv`, `walkforward_per_fold.csv` — out-of-sample metrics.
- `metadata.json` — source, rows, budget, platform, OOS PnL by strategy.

A local smoke test (`--synthetic --timesteps 2000`) exercises the whole path
end-to-end without a token or GPU.

---

## 4 · Decisions taken autonomously

1. **Provider abstraction** (`(train, test) -> strategy`) so the harness handles
   learned and non-learned policies uniformly and trains the agent *per fold* on
   only its own past — the correct out-of-sample protocol.
2. **Expanding window default** (rolling available) — standard for non-stationary
   energy prices where more history usually helps.
3. **Pod boundary made explicit**: CI/notebook stay tiny and synthetic; the heavy
   real-data run is a separate turnkey script, so renting a pod is plug-and-play.
4. **Artefacts gitignored** (`artifacts/`, `*.zip`) — training outputs are not
   committed by default; the production artefacts are promoted intentionally.
