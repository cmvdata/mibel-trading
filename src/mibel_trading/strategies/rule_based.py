"""Rule-based strategies for the MIBEL DAM -> servicios de ajuste spread.

Replicates the rule-based (RB) approach of Demir (2023), Ch. 12: a transparent,
non-learned policy the RL agent must beat. The flagship strategy here trades the
**DAM vs secondary-band spread** on a rolling z-score — selling DAM when it looks
rich relative to the ajuste signal (betting on convergence) and buying when it
looks cheap. Two trivial policies (always-buy, random) provide sanity and
baseline references.

All strategies consume the :class:`mibel_trading.env.MibelTradingEnv`
observation, whose layout is fixed::

    [dam_t, dam_{t-1}, secundaria, terciaria, desvio, position, cash, hour, dow]

and emit a discrete action ``0=HOLD``, ``1=BUY 1 MWh``, ``2=SELL 1 MWh``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import numpy.typing as npt

__all__ = [
    "BaseStrategy",
    "NaiveBuyHoldStrategy",
    "RandomStrategy",
    "RuleBasedSpreadStrategy",
]

# Observation indices (must match mibel_trading.env.mibel_env).
OBS_DAM_T = 0
OBS_SECUNDARIA = 2

# Action codes.
HOLD = 0
BUY = 1
SELL = 2

_EPS = 1e-8


class BaseStrategy(ABC):
    """Abstract trading policy: map an observation to a discrete action."""

    @abstractmethod
    def act(self, obs: npt.NDArray[np.float64], info: dict) -> int:
        """Return the action (``0`` HOLD, ``1`` BUY, ``2`` SELL) for ``obs``."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state between episodes. Default: no-op."""
        return None


class RuleBasedSpreadStrategy(BaseStrategy):
    """Rolling z-score mean-reversion on the DAM vs secondary-band spread.

    The spread ``s_t = dam_t - secundaria_t`` is tracked in a rolling window. When
    the current spread is more than ``threshold_std`` standard deviations above
    its recent mean, DAM is judged rich relative to the ajuste signal and the
    strategy **sells** DAM (action 2), expecting convergence; symmetrically it
    **buys** (action 1) when the spread is unusually low. Otherwise it holds.

    Parameters
    ----------
    window : int, default 24
        Number of past spreads in the rolling statistics (one day at hourly
        resolution). No signal is emitted until the window is full.
    threshold_std : float, default 1.0
        Z-score magnitude that triggers a trade.
    """

    def __init__(self, window: int = 24, threshold_std: float = 1.0) -> None:
        if window < 2:
            raise ValueError("`window` must be at least 2.")
        if threshold_std <= 0.0:
            raise ValueError("`threshold_std` must be positive.")
        self.window = int(window)
        self.threshold_std = float(threshold_std)
        self._buffer: deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._buffer.clear()

    def act(self, obs: npt.NDArray[np.float64], info: dict) -> int:
        spread = float(obs[OBS_DAM_T]) - float(obs[OBS_SECUNDARIA])

        # Warm-up: not enough history yet -> hold and accumulate.
        if len(self._buffer) < self.window:
            self._buffer.append(spread)
            return HOLD

        history = np.fromiter(self._buffer, dtype=np.float64)
        mean = float(history.mean())
        std = float(history.std())
        z = (spread - mean) / (std + _EPS)

        # Slide the window forward to include the current spread.
        self._buffer.append(spread)

        if z > self.threshold_std:
            return SELL  # DAM rich vs ajuste -> sell, expect convergence
        if z < -self.threshold_std:
            return BUY  # DAM cheap vs ajuste -> buy
        return HOLD


class NaiveBuyHoldStrategy(BaseStrategy):
    """Always buy one MWh. Trivial sanity-check policy."""

    def act(self, obs: npt.NDArray[np.float64], info: dict) -> int:
        return BUY


class RandomStrategy(BaseStrategy):
    """Uniformly random action. Baseline reference.

    Parameters
    ----------
    seed : int, optional
        Seed for reproducibility; re-applied on :meth:`reset`.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, obs: npt.NDArray[np.float64], info: dict) -> int:
        return int(self._rng.integers(0, 3))
