"""Behaviour cloning: bootstrap the PPO policy from an expert strategy.

Following the Demir (2023) recipe of warm-starting the RL agent, we (1) roll out
an expert policy (typically :class:`RuleBasedSpreadStrategy`) to collect
``(observation, action)`` demonstrations, (2) fit a small MLP classifier to
imitate the expert, and (3) copy its weights into the stable-baselines3 PPO
policy network so PPO starts from a sensible prior rather than random.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

__all__ = [
    "BCPolicy",
    "bc_policy_to_ppo",
    "collect_demonstrations",
    "train_bc",
]

#: The env's action space is Discrete(3): HOLD / BUY / SELL.
N_ACTIONS = 3


def collect_demonstrations(
    env: object,
    expert_strategy: object,
    n_episodes: int = 5,
    seed: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Roll out ``expert_strategy`` on ``env`` and return ``(obs, actions)``.

    Parameters
    ----------
    env : MibelTradingEnv
        A raw (non-vectorised) environment, gymnasium API.
    expert_strategy : BaseStrategy
        Policy whose behaviour is to be cloned.
    n_episodes : int
        Episodes to roll out (each uses seed ``seed + episode``).
    seed : int
        Base seed.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``obs`` of shape ``(N, obs_dim)`` (float32) and ``actions`` of shape
        ``(N,)`` (int64), concatenated across episodes.
    """
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(expert_strategy, "reset"):
            expert_strategy.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(expert_strategy.act(obs, info))
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(action)
            obs, _, terminated, truncated, info = env.step(action)
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(act_list, dtype=np.int64),
    )


class BCPolicy(nn.Module):
    """Small MLP classifier: ``obs_dim -> 64 -> 64 -> n_actions`` (Tanh).

    Split into a feature extractor (``net``) and an action ``head`` so the
    weights map cleanly onto the stable-baselines3 ``mlp_extractor.policy_net``
    and ``action_net``.
    """

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(obs))


def train_bc(
    obs_arr: npt.ArrayLike,
    action_arr: npt.ArrayLike,
    n_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> BCPolicy:
    """Fit a :class:`BCPolicy` to imitate the expert by cross-entropy.

    The per-epoch mean loss is stored on the returned policy as
    ``train_losses_`` (a list), which makes convergence checkable.

    Parameters
    ----------
    obs_arr, action_arr
        Demonstrations from :func:`collect_demonstrations`.
    n_epochs, lr, batch_size, device, seed
        Optimisation settings (Adam + cross-entropy).

    Returns
    -------
    BCPolicy
        The trained policy, with ``train_losses_`` attached.
    """
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(obs_arr, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(action_arr, dtype=np.int64), device=device)
    obs_dim = int(x.shape[1])

    policy = BCPolicy(obs_dim, N_ACTIONS).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = int(x.shape[0])
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for _ in range(n_epochs):
        perm = torch.randperm(n, generator=generator)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimiser.zero_grad()
            logits = policy(x[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    policy.train_losses_ = losses  # type: ignore[attr-defined]
    return policy


def bc_policy_to_ppo(bc_policy: BCPolicy, ppo_agent: object) -> int:
    """Copy BC weights into the PPO policy network (partial warm-start).

    Copies the two hidden layers of ``bc_policy.net`` into
    ``policy.mlp_extractor.policy_net`` and, when shapes match, the
    ``bc_policy.head`` into ``policy.action_net``. The value network is left at
    its initialisation — this is a deliberate *partial* warm-start: PPO refines
    the value head during training while starting from the expert's action prior.

    Returns
    -------
    int
        Number of extractor layers successfully copied.
    """
    policy = ppo_agent.model.policy
    bc_linears = [m for m in bc_policy.net if isinstance(m, nn.Linear)]
    pi_linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]

    copied = 0
    with torch.no_grad():
        for src, dst in zip(bc_linears, pi_linears, strict=False):
            if src.weight.shape == dst.weight.shape and src.bias.shape == dst.bias.shape:
                dst.weight.copy_(src.weight.to(dst.weight.device))
                dst.bias.copy_(src.bias.to(dst.bias.device))
                copied += 1
        # Best-effort action-head warm-start (skip silently if geometry differs).
        action_net = getattr(policy, "action_net", None)
        if (
            action_net is not None
            and bc_policy.head.weight.shape == action_net.weight.shape
        ):
            action_net.weight.copy_(bc_policy.head.weight.to(action_net.weight.device))
            action_net.bias.copy_(bc_policy.head.bias.to(action_net.bias.device))
    return copied
