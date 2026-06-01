"""Gymnasium environment for MIBEL DAM + servicios de ajuste trading (MVP).

A single-asset trading environment over the OMIE day-ahead price, augmented with
servicios de ajuste signals (secondary band, tertiary, deviation). The agent
holds an inventory of power in ``[position_min, position_max]`` MWh and trades it
one MWh at a time. The reward is the **bounded PnL** of Demir (2023, §11.2.3):
adverse mark-to-market is penalised down to ``-2`` while gains are clipped to
``0``, which stabilises PPO; the raw PnL is reported in ``info`` for evaluation.

Observation (9-dim ``Box``)::

    [price_dam_t, price_dam_{t-1}, banda_secundaria, terciaria, desvio,
     position, cash, hour_of_day, day_of_week]

Action (``Discrete(3)``): 0 = Hold, 1 = Buy 1 MWh, 2 = Sell 1 MWh.

This is an MVP; richer action sizing and a multi-product ajuste book come later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # gymnasium is a hard dependency, but keep the import error actionable.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - environment misconfiguration
    raise ImportError("gymnasium is required for mibel_trading.env") from exc

_ACTION_DELTA = {0: 0.0, 1: 1.0, 2: -1.0}  # Hold / Buy 1 MWh / Sell 1 MWh

# Generous finite Box bounds (prices in EUR/MWh, cash in EUR). Kept wide so the
# observation never leaves the space under any action sequence.
_PRICE_BOUND = 1.0e4
_ANCILLARY_BOUND = 1.0e6
_CASH_BOUND = 1.0e9


class MibelTradingEnv(gym.Env):
    """MIBEL DAM + servicios de ajuste trading environment.

    Parameters
    ----------
    data : pandas.DataFrame
        Hourly market data. Must contain ``price_col``; the ancillary columns
        are optional (missing ones are treated as zeros). A ``DatetimeIndex``
        supplies hour-of-day and day-of-week; otherwise both are zero.
    price_col, secundaria_col, terciaria_col, desvio_col : str
        Column names for the DAM price and the three ancillary signals. Defaults
        match :func:`mibel_trading.data.loaders.load_features`.
    position_min, position_max : float
        Inventory bounds in MWh (default ``[-10, 10]``).
    reward_scale : float
        PnL normaliser before the ``[-2, 0]`` clip.
    render_mode : str or None
        ``"human"`` prints each step.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        price_col: str = "price_eur_mwh",
        secundaria_col: str = "esios_634",
        terciaria_col: str = "esios_676",
        desvio_col: str = "esios_686",
        position_min: float = -10.0,
        position_max: float = 10.0,
        reward_scale: float = 100.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if price_col not in data.columns:
            raise ValueError(f"`data` must contain the price column {price_col!r}.")
        if len(data) < 2:
            raise ValueError("`data` must have at least two rows (need t and t+1).")
        if position_max <= position_min:
            raise ValueError("`position_max` must exceed `position_min`.")
        if reward_scale <= 0.0:
            raise ValueError("`reward_scale` must be positive.")

        self.position_min = float(position_min)
        self.position_max = float(position_max)
        self.reward_scale = float(reward_scale)
        self.render_mode = render_mode

        # Materialise the series the env steps over.
        def _col(name: str) -> np.ndarray:
            if name in data.columns:
                return data[name].to_numpy(dtype=np.float64)
            return np.zeros(len(data), dtype=np.float64)

        self._price = data[price_col].to_numpy(dtype=np.float64)
        self._secundaria = _col(secundaria_col)
        self._terciaria = _col(terciaria_col)
        self._desvio = _col(desvio_col)
        if isinstance(data.index, pd.DatetimeIndex):
            self._hour = data.index.hour.to_numpy(dtype=np.float64)
            self._dow = data.index.dayofweek.to_numpy(dtype=np.float64)
        else:
            self._hour = np.zeros(len(data), dtype=np.float64)
            self._dow = np.zeros(len(data), dtype=np.float64)

        self._n = len(data)
        self._max_index = self._n - 1

        low = np.array(
            [-_PRICE_BOUND, -_PRICE_BOUND, -_ANCILLARY_BOUND, -_ANCILLARY_BOUND,
             -_ANCILLARY_BOUND, self.position_min, -_CASH_BOUND, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [_PRICE_BOUND, _PRICE_BOUND, _ANCILLARY_BOUND, _ANCILLARY_BOUND,
             _ANCILLARY_BOUND, self.position_max, _CASH_BOUND, 23.0, 6.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._t = 0
        self._position = 0.0
        self._cash = 0.0

    # ------------------------------------------------------------------ #
    def _observe(self) -> np.ndarray:
        t = self._t
        price_prev = self._price[t - 1] if t > 0 else self._price[t]
        obs = np.array(
            [
                self._price[t],
                price_prev,
                self._secundaria[t],
                self._terciaria[t],
                self._desvio[t],
                self._position,
                self._cash,
                self._hour[t],
                self._dow[t],
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._t = 0
        self._position = 0.0
        self._cash = 0.0
        info = {"position": self._position, "cash": self._cash, "pnl": 0.0}
        return self._observe(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if action not in _ACTION_DELTA:
            raise ValueError(f"invalid action {action!r}; expected 0, 1 or 2.")

        price_now = float(self._price[self._t])
        delta = _ACTION_DELTA[action]
        new_position = float(
            np.clip(self._position + delta, self.position_min, self.position_max)
        )
        traded = new_position - self._position  # 0 when a bound binds
        self._cash -= traded * price_now  # buy spends cash, sell raises it
        self._position = new_position

        # Advance one hour and mark to market on the price move.
        self._t += 1
        price_next = float(self._price[self._t])
        pnl = self._position * (price_next - price_now)
        reward = float(np.clip(pnl / self.reward_scale, -2.0, 0.0))

        terminated = self._t >= self._max_index
        truncated = False
        info = {
            "pnl": pnl,
            "position": self._position,
            "cash": self._cash,
            "price": price_next,
            "nav": self._cash + self._position * price_next,
        }
        if self.render_mode == "human":
            self.render()
        return self._observe(), reward, terminated, truncated, info

    def render(self) -> None:
        print(
            f"t={self._t:4d} | price={self._price[self._t]:8.2f} "
            f"| pos={self._position:+6.2f} MWh | cash={self._cash:12.2f} EUR"
        )

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass
