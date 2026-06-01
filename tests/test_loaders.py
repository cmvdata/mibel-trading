"""Tests for the OMIE/ESIOS data loaders.

The parquet-backed loader is exercised only when the curated file is reachable
(skipped otherwise, so CI stays green without the data). The alignment logic is
tested with synthetic in-memory frames and needs neither parquets nor a token.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_trading.data.loaders import (
    _align_features,
    _resolve_omie_parquet,
    load_omie_dam,
)


def _synthetic_dam(n: int = 72, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")
    price = 50.0 + 15.0 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 4, n)
    return pd.DataFrame({"datetime": idx, "price_eur_mwh": price})


def _synthetic_ancillary(idx: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "esios_634": rng.uniform(0, 200, len(idx)),
            "esios_676": rng.uniform(0, 100, len(idx)),
            "esios_686": rng.normal(0, 40, len(idx)),
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


def test_load_omie_dam_returns_dataframe() -> None:
    """Loader returns [datetime, price_eur_mwh] when the curated parquet exists."""
    try:
        _resolve_omie_parquet()
    except FileNotFoundError:
        pytest.skip("curated OMIE parquet not available in this environment")

    df = load_omie_dam("2023-01-01", "2023-01-07")
    assert list(df.columns) == ["datetime", "price_eur_mwh"]
    assert len(df) > 0
    assert str(df["datetime"].dt.tz) == "UTC"
    assert df["datetime"].is_monotonic_increasing
    assert df["datetime"].min() >= pd.Timestamp("2023-01-01", tz="UTC")
    assert df["datetime"].max() <= pd.Timestamp("2023-01-07 23:59:59", tz="UTC")


def test_load_features_aligned() -> None:
    """_align_features inner-joins DAM and ancillary on the hourly UTC stamp."""
    dam = _synthetic_dam(n=72)
    # Ancillary covers an overlapping but offset window (drop first 12h, add 12h).
    anc_idx = pd.date_range("2023-06-01 12:00", periods=72, freq="h", tz="UTC")
    ancillary = _synthetic_ancillary(anc_idx)

    out = _align_features(dam, ancillary)

    # Index is a sorted, unique, hourly UTC DatetimeIndex.
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing
    assert out.index.is_unique
    assert (out.index.minute == 0).all()

    # Only the overlapping hours survive the inner join.
    expected = dam.set_index("datetime").index.intersection(ancillary.index)
    assert len(out) == len(expected)
    assert out.index.equals(expected.sort_values())

    # All columns present and fully populated (no misalignment-induced NaNs).
    assert "price_eur_mwh" in out.columns
    for col in ("esios_634", "esios_676", "esios_686"):
        assert col in out.columns
    assert not out.isna().any().any()


def test_align_features_disjoint_gives_empty() -> None:
    """Non-overlapping windows yield an empty aligned frame, not an error."""
    dam = _synthetic_dam(n=24)
    far_idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    out = _align_features(dam, _synthetic_ancillary(far_idx))
    assert len(out) == 0
