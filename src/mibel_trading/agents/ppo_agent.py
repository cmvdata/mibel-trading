"""PPO trading agent wrapping stable-baselines3 as a ``BaseStrategy``.

The agent exposes the same ``act(obs, info) -> int`` interface as the rule-based
strategies and benchmarks, so :func:`mibel_trading.eval.run_backtest` and
:func:`mibel_trading.eval.compare_strategies` treat the learned policy exactly
like any hand-coded one. Internally it holds a stable-baselines3 ``PPO`` with an
``MlpPolicy`` (``net_arch=[64, 64]``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from stable_baselines3 import PPO

from mibel_trading.strategies.rule_based import BaseStrategy


class PPOAgent(BaseStrategy):
    """Proximal Policy Optimization agent, usable as a ``BaseStrategy``.

    Parameters
    ----------
    env : gymnasium.Env or VecEnv
        Training environment (a single ``MibelTradingEnv`` or a vectorised one;
        stable-baselines3 wraps a raw env automatically).
    learning_rate, n_steps, batch_size, n_epochs, gamma : PPO hyper-parameters.
    seed : int
        Seed for PPO.
    verbose : int
        stable-baselines3 verbosity.
    """

    def __init__(
        self,
        env: object,
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            seed=seed,
            verbose=verbose,
            policy_kwargs={"net_arch": [64, 64]},
        )

    def train(self, total_timesteps: int) -> PPOAgent:
        """Run PPO for ``total_timesteps`` and return self (for chaining)."""
        self.model.learn(total_timesteps=total_timesteps)
        return self

    def act(self, obs: npt.NDArray[np.float64], info: dict) -> int:
        """Greedy action from the trained policy (``0`` HOLD, ``1`` BUY, ``2`` SELL)."""
        action, _ = self.model.predict(
            np.asarray(obs, dtype=np.float32), deterministic=True
        )
        return int(np.asarray(action).reshape(-1)[0])

    def reset(self) -> None:
        """No episodic state to reset (the policy is stateless across steps)."""
        return None

    def save(self, path: str) -> None:
        """Persist the underlying PPO model (stable-baselines3 ``.zip``)."""
        self.model.save(path)

    def load(self, path: str) -> PPOAgent:
        """Load PPO weights from ``path`` into this agent's model."""
        self.model = PPO.load(path)
        return self

    def set_policy_weights_from_bc(self, bc_policy: object) -> int:
        """Warm-start the PPO policy from a behaviour-cloning MLP.

        Copies the shared feature extractor (and the action head when shapes
        match) from ``bc_policy`` into ``self.model.policy``. Returns the number
        of extractor layers copied. See
        :func:`mibel_trading.agents.behaviour_cloning.bc_policy_to_ppo`.
        """
        # Lazy import avoids a module-load cycle (behaviour_cloning is heavier).
        from mibel_trading.agents.behaviour_cloning import bc_policy_to_ppo

        return bc_policy_to_ppo(bc_policy, self)
