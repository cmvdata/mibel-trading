"""Trading benchmarks adapted from Demir (2023), pp. 138-139.

Demir's benchmarks are simple, non-learned execution policies the RL agent must
beat. Adapted here to the MVP env (``Discrete(3)``: Hold / Buy 1 MWh / Sell
1 MWh, one action per hourly step):

* :class:`BENCH` — the most trivial round trip: buy at the first step, hold, sell
  at the last step (capture the whole-horizon price move in one position).
* :class:`BENCHVWAP` — spread the round trip in time: buy across the first
  ``window_size`` steps and sell across the last ``window_size`` steps,
  approximating "buy at the early-window VWAP, sell at the late-window VWAP".
* :class:`BENCHPLUS` — :class:`BENCHVWAP` gated by volatility: only execute the
  scheduled trades when rolling price volatility is high enough to justify the
  execution risk; otherwise stay flat.

All three subclass :class:`mibel_trading.strategies.rule_based.BaseStrategy` and
run through the Pieza 2 backtest engine unchanged.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

from mibel_trading.strategies.rule_based import (
    BUY,
    HOLD,
    OBS_DAM_T,
    SELL,
    BaseStrategy,
)

__all__ = ["BENCH", "BENCHPLUS", "BENCHVWAP"]


class BENCH(BaseStrategy):
    """Buy at the first step, hold, sell at the last step.

    Parameters
    ----------
    horizon : int
        Total number of steps (env actions) in the episode. The sell fires at
        step ``horizon - 1``.
    """

    def __init__(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("`horizon` must be >= 1.")
        self.horizon = int(horizon)
        self._step = 0
        self._first_price = 0.0
        #: Price move captured over the round trip (set once the sell fires).
        self.captured_spread = 0.0

    def reset(self) -> None:
        self._step = 0
        self._first_price = 0.0
        self.captured_spread = 0.0

    def act(self, obs: npt.NDArray[np.float64], info: dict) -> int:
        step = self._step
        self._step += 1
        price = float(obs[OBS_DAM_T])
        if step == 0:
            self._first_price = price
            return BUY
        if step >= self.horizon - 1:
            self.captured_spread = price - self._first_price
            return SELL
        return HOLD


class BENCHVWAP(BaseStrategy):
    """Time-distributed round trip: buy early window, sell late window.

    Parameters
    ----------
    horizon : int
        Total number of steps in the episode.
    window_size : int, default 10
        Steps over which to spread the buys (and, symmetrically, the sells). If
        ``2 * window_size > horizon`` the window is clamped to ``horizon // 3``
        so the buy and sell legs cannot overlap.
    """

    def __init__(self, horizon: int, window_size: int = 10) -> None:
        if horizon < 1:
            raise ValueError("`horizon` must be >= 1.")
        if window_size < 1:
            raise ValueError("`window_size` must be >= 1.")
        self.horizon = int(horizon)
        if 2 * window_size > self.horizon:
            window_size = self.horizon // 3
        self.window_size = int(window_size)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, obs: npt.NDArray[np.float64], info: dict) -> int:
        step = self._step
        self._step += 1
        w = self.window_size
        if w < 1:
            return HOLD
        if step < w:
            return BUY
        if step >= self.horizon - w:
            return SELL
        return HOLD


class BENCHPLUS(BaseStrategy):
    """:class:`BENCHVWAP` gated by rolling volatility.

    After a ``vol_window`` warm-up (held flat to seed the volatility estimate),
    the strategy follows a VWAP-style schedule over the remaining horizon but
    only *executes* a scheduled buy/sell when the rolling price volatility
    exceeds ``vol_threshold`` — "trade only when the execution risk is worth it".

    Parameters
    ----------
    horizon : int
        Total number of steps in the episode.
    window_size : int, default 10
        VWAP buy/sell window over the post-warm-up region (clamped to a third of
        that region if too large).
    vol_window : int, default 24
        Rolling window (in steps) for the volatility estimate and warm-up length.
    vol_threshold : float, optional
        Volatility gate. If ``None`` it is calibrated to the 75th percentile of
        the rolling volatility observed during the warm-up.
    """

    def __init__(
        self,
        horizon: int,
        window_size: int = 10,
        vol_window: int = 24,
        vol_threshold: float | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("`horizon` must be >= 1.")
        if window_size < 1 or vol_window < 1:
            raise ValueError("`window_size` and `vol_window` must be >= 1.")
        self.horizon = int(horizon)
        self.window_size = int(window_size)
        self.vol_window = int(vol_window)
        self.vol_threshold = vol_threshold
        self._step = 0
        self._prices: deque[float] = deque(maxlen=self.vol_window)
        self._warmup_vols: list[float] = []
        self._threshold: float | None = vol_threshold

    def reset(self) -> None:
        self._step = 0
        self._prices.clear()
        self._warmup_vols = []
        self._threshold = self.vol_threshold

    def _effective_window(self, eff_horizon: int) -> int:
        w = self.window_size
        if 2 * w > eff_horizon:
            w = max(eff_horizon // 3, 0)
        return w

    def act(self, obs: npt.NDArray[np.float64], info: dict) -> int:
        price = float(obs[OBS_DAM_T])
        step = self._step
        self._step += 1
        self._prices.append(price)
        vol = float(np.std(self._prices)) if len(self._prices) >= 2 else 0.0

        # Warm-up: stay flat, accumulate volatility samples for calibration.
        if step < self.vol_window:
            self._warmup_vols.append(vol)
            return HOLD

        # Calibrate the gate once, when the warm-up ends.
        if self._threshold is None:
            self._threshold = (
                float(np.percentile(self._warmup_vols, 75)) if self._warmup_vols else 0.0
            )

        eff_step = step - self.vol_window
        eff_horizon = self.horizon - self.vol_window
        w = self._effective_window(eff_horizon)
        if w < 1:
            return HOLD

        if eff_step < w:
            scheduled = BUY
        elif eff_step >= eff_horizon - w:
            scheduled = SELL
        else:
            scheduled = HOLD

        if scheduled != HOLD and vol > self._threshold:
            return scheduled
        return HOLD
