"""Data loaders for MIBEL DAM (OMIE) and servicios de ajuste (ESIOS).

Per the project rule, these loaders **reuse** data already curated elsewhere in
the MIBEL stack rather than re-downloading it:

* **DAM** — the curated OMIE Spain spot parquet from ``mibel-derivatives``
  (``data/curated/omie_spot_es_2019_2024.parquet``), with a fallback search in
  ``mibel-forecasting/data``.
* **Servicios de ajuste** — pulled through the cached ESIOS client
  (:mod:`mibel_trading.data.esios_client`), which hits a local monthly Parquet
  cache before ever touching the network.

The three public functions return tidy, hourly, UTC-aligned frames; tests drive
the alignment logic with synthetic frames so CI never needs the parquets or a
network token.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mibel_trading.data.esios_client import pull_indicator

#: Default ancillary-service indicators (Demir-style servicios de ajuste):
#: 634 secondary band, 682/683 secondary energy up/down, 676/677 tertiary
#: up/down, 686/687 deviation up/down.
DEFAULT_ANCILLARY_INDICATORS: tuple[int, ...] = (634, 682, 683, 676, 677, 686, 687)

#: Candidate locations for the curated OMIE DAM parquet (first existing wins).
_OMIE_PARQUET_CANDIDATES: tuple[Path, ...] = (
    Path(r"C:\Users\Carlo\Desktop\Projects\Mibel_derivatives\data\curated\omie_spot_es_2019_2024.parquet"),
)
_OMIE_FALLBACK_DIRS: tuple[Path, ...] = (
    Path(r"C:\Users\Carlo\Desktop\Projects\Mibel_forecasting\data"),
)


def _resolve_omie_parquet() -> Path:
    """Find the curated OMIE parquet, honouring ``OMIE_DAM_PARQUET`` if set."""
    override = os.environ.get("OMIE_DAM_PARQUET")
    if override and Path(override).exists():
        return Path(override)
    for cand in _OMIE_PARQUET_CANDIDATES:
        if cand.exists():
            return cand
    for base in _OMIE_FALLBACK_DIRS:
        if base.exists():
            matches = sorted(base.rglob("omie*.parquet"))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "Curated OMIE DAM parquet not found. Looked in mibel-derivatives "
        "data/curated and mibel-forecasting/data. Set OMIE_DAM_PARQUET to point "
        "at it explicitly."
    )


def _to_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def load_omie_dam(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Load OMIE Spain day-ahead hourly price, filtered to ``[start, end]``.

    Parameters
    ----------
    start, end
        Inclusive date bounds (tz-naive interpreted as UTC).

    Returns
    -------
    pandas.DataFrame
        Columns ``[datetime, price_eur_mwh]`` sorted by ``datetime`` (UTC).
    """
    path = _resolve_omie_parquet()
    df = pd.read_parquet(path)
    if "datetime_utc" in df.columns:
        df = df.rename(columns={"datetime_utc": "datetime"})
    elif "datetime" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[["datetime", "price_eur_mwh"]].sort_values("datetime").reset_index(drop=True)

    lo, hi = _to_utc(start), _to_utc(end)
    mask = (df["datetime"] >= lo) & (df["datetime"] <= hi)
    return df.loc[mask].reset_index(drop=True)


def load_esios_ancillary(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    indicators: Sequence[int] = DEFAULT_ANCILLARY_INDICATORS,
    *,
    geo_id: int = 3,
) -> pd.DataFrame:
    """Load servicios de ajuste indicators from the cached ESIOS client.

    Parameters
    ----------
    start, end
        Inclusive date bounds.
    indicators
        ESIOS indicator ids; one column ``esios_<id>`` per indicator.
    geo_id
        ESIOS geo id (3 = España).

    Returns
    -------
    pandas.DataFrame
        UTC ``DatetimeIndex`` (hourly), one ``esios_<id>`` column per indicator.
    """
    series: dict[str, pd.Series] = {}
    for ind in indicators:
        s = pull_indicator(int(ind), geo_id, start=start, end=end)
        series[f"esios_{int(ind)}"] = s
    if not series:
        return pd.DataFrame()
    out = pd.concat(series, axis=1)
    out.index.name = "datetime"
    return out.sort_index()


def _align_features(
    dam: pd.DataFrame, ancillary: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join DAM and ancillary frames on the hourly UTC timestamp.

    ``dam`` must have a ``datetime`` column and ``price_eur_mwh``; ``ancillary``
    is indexed by ``datetime``. The result is indexed by ``datetime`` (UTC,
    hourly) with ``price_eur_mwh`` plus the ancillary columns, keeping only the
    hours present in both.
    """
    left = dam.copy()
    left["datetime"] = pd.to_datetime(left["datetime"], utc=True).dt.floor("h")
    left = left.set_index("datetime")[["price_eur_mwh"]]
    left = left[~left.index.duplicated(keep="first")]

    right = ancillary.copy()
    if not isinstance(right.index, pd.DatetimeIndex):
        right.index = pd.to_datetime(right.index, utc=True)
    right.index = right.index.tz_convert("UTC") if right.index.tz else right.index.tz_localize("UTC")
    right.index = right.index.floor("h")
    right = right[~right.index.duplicated(keep="first")]
    right.index.name = "datetime"

    out = left.join(right, how="inner").sort_index()
    return out


def load_features(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    indicators: Sequence[int] = DEFAULT_ANCILLARY_INDICATORS,
) -> pd.DataFrame:
    """Combined DAM + servicios de ajuste features, hourly-aligned on UTC.

    Parameters
    ----------
    start, end
        Inclusive date bounds.
    indicators
        ESIOS ancillary indicators to attach.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``datetime`` (UTC, hourly), columns ``price_eur_mwh`` plus
        one ``esios_<id>`` per indicator, restricted to hours present in both
        sources.
    """
    dam = load_omie_dam(start, end)
    ancillary = load_esios_ancillary(start, end, indicators)
    return _align_features(dam, ancillary)
