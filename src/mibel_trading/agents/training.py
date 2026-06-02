"""Training pipelines: PPO with behaviour-cloning warm-start, and a baseline.

Both build the PPO on a ``DummyVecEnv`` (single env; we deliberately avoid
``SubprocVecEnv`` to keep things portable and CI-friendly).
"""

from __future__ import annotations

from collections.abc import Callable

from stable_baselines3.common.vec_env import DummyVecEnv

from mibel_trading.agents.behaviour_cloning import collect_demonstrations, train_bc
from mibel_trading.agents.ppo_agent import PPOAgent

__all__ = ["train_ppo_baseline", "train_ppo_with_bc"]


def train_ppo_with_bc(
    env_factory: Callable[[], object],
    expert_strategy: object,
    total_timesteps: int = 10000,
    n_bc_episodes: int = 5,
    bc_epochs: int = 20,
    seed: int = 42,
    verbose: int = 0,
) -> PPOAgent:
    """Behaviour-clone an expert, warm-start PPO with it, then train PPO.

    Parameters
    ----------
    env_factory : callable
        Zero-argument callable returning a fresh ``MibelTradingEnv``.
    expert_strategy : BaseStrategy
        Expert to imitate (e.g. ``RuleBasedSpreadStrategy``).
    total_timesteps : int
        PPO training budget.
    n_bc_episodes, bc_epochs : int
        Demonstration rollouts and behaviour-cloning epochs.
    seed, verbose
        Reproducibility / verbosity.

    Returns
    -------
    PPOAgent
        Trained agent, ready for ``act()`` / backtesting.
    """
    # 1. Collect expert demonstrations on a fresh raw env.
    obs_arr, act_arr = collect_demonstrations(
        env_factory(), expert_strategy, n_episodes=n_bc_episodes, seed=seed
    )
    # 2. Behaviour-clone the expert.
    bc_policy = train_bc(obs_arr, act_arr, n_epochs=bc_epochs, seed=seed)
    # 3. Build PPO on a vectorised env and warm-start from BC.
    vec_env = DummyVecEnv([lambda: env_factory()])
    agent = PPOAgent(vec_env, seed=seed, verbose=verbose)
    agent.set_policy_weights_from_bc(bc_policy)
    # 4. Train PPO.
    agent.train(total_timesteps)
    return agent


def train_ppo_baseline(
    env_factory: Callable[[], object],
    total_timesteps: int = 10000,
    seed: int = 42,
    verbose: int = 0,
) -> PPOAgent:
    """Train a plain PPO (no behaviour cloning) for comparison.

    Parameters
    ----------
    env_factory : callable
        Zero-argument callable returning a fresh ``MibelTradingEnv``.
    total_timesteps, seed, verbose
        PPO budget / reproducibility / verbosity.

    Returns
    -------
    PPOAgent
    """
    vec_env = DummyVecEnv([lambda: env_factory()])
    agent = PPOAgent(vec_env, seed=seed, verbose=verbose)
    agent.train(total_timesteps)
    return agent
