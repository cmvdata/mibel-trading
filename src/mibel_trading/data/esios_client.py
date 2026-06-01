"""ESIOS API client with monthly Parquet cache.

Adapted from ``mibel_forecasting.data.esios`` (reused, not re-downloaded): the
cache stores one Parquet file per ``(indicator_id, geo_id, year, month)`` under
``data/cache/esios/``. Cache hits avoid network calls entirely; misses fetch
from ``https://api.esios.ree.es`` and persist before returning. The cache is
safe to delete at any time — the next call re-populates the needed months.

Token resolution honours both ``ESIOS_TOKEN`` (the variable named in the
mibel-trading brief) and ``ESIOS_API_TOKEN`` (the name used across the rest of
the MIBEL stack), read from the environment or a local ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ESIOS_BASE_URL = "https://api.esios.ree.es"


def _default_cache_dir() -> Path:
    """Resolve the default cache directory.

    Honours ``MIBEL_CACHE_DIR`` if set; otherwise anchors the cache at
    ``<repo_root>/data/cache/esios`` relative to this file so the same cache is
    hit no matter what cwd the caller runs from (notebooks, tests, scripts).
    """
    env = os.environ.get("MIBEL_CACHE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data" / "cache" / "esios"


DEFAULT_CACHE_DIR = _default_cache_dir()


class ESIOSConfigError(RuntimeError):
    """Raised when no ESIOS token is found in the environment."""


def _resolve_token() -> str:
    token = os.environ.get("ESIOS_TOKEN") or os.environ.get("ESIOS_API_TOKEN")
    if not token:
        raise ESIOSConfigError(
            "No ESIOS token set. Define ESIOS_TOKEN (or ESIOS_API_TOKEN) in the "
            "environment or a .env file; request one at https://api.esios.ree.es."
        )
    return token


def _cache_path(
    cache_dir: Path, indicator_id: int, geo_id: int, year: int, month: int
) -> Path:
    return cache_dir / f"i{indicator_id}_geo{geo_id}_{year:04d}_{month:02d}.parquet"


def _fetch_month(
    indicator_id: int,
    geo_id: int,
    year: int,
    month: int,
    *,
    timeout: float = 120.0,
) -> pd.Series:
    """Pull a single month for one indicator/geo. UTC-indexed hourly series."""
    last_day = (
        pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
    ).strftime("%Y-%m-%d")
    start = f"{year}-{month:02d}-01T00:00:00Z"
    end = f"{last_day}T23:59:59Z"
    token = _resolve_token()
    r = requests.get(
        f"{ESIOS_BASE_URL}/indicators/{indicator_id}",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": token,
        },
        params=[
            ("start_date", start),
            ("end_date", end),
            ("geo_ids[]", str(geo_id)),
        ],
        timeout=timeout,
    )
    r.raise_for_status()
    rows = r.json()["indicator"]["values"]
    if not rows:
        return pd.Series(dtype=float, name="value")
    df = pd.DataFrame(rows)
    df["dt_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    series = df.set_index("dt_utc")["value"].astype(float)
    # Some ESIOS series are 15-min; collapse to hourly mean for consistency.
    series = series.groupby(series.index.floor("h")).mean()
    series.name = "value"
    return series


def pull_indicator(
    indicator_id: int,
    geo_id: int = 3,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.Series:
    """Pull an hourly series for one indicator+geo, with monthly Parquet cache.

    Parameters
    ----------
    indicator_id
        ESIOS indicator id (e.g. ``634`` for secondary-reserve band).
    geo_id
        ESIOS geo id (3 = España by default).
    start, end
        Inclusive bounds. Strings or ``pd.Timestamp``. Tz-naive is interpreted
        as UTC; tz-aware is converted to UTC.
    cache_dir
        Override the default ``data/cache/esios/`` location.
    refresh
        If ``True``, re-fetch every required month even on cache hit.

    Returns
    -------
    pandas.Series
        UTC-indexed hourly series, named ``value``, sliced to ``[start, end]``.
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    def _to_utc(x: str | pd.Timestamp, *, kind: str) -> pd.Timestamp:
        # Date-only strings ("YYYY-MM-DD") follow pandas partial-string-indexing:
        # start -> 00:00 of the day, end -> 23:59:59.999... of the day.
        is_date_only = isinstance(x, str) and "T" not in x and ":" not in x and len(x) <= 10
        ts = pd.Timestamp(x)
        if is_date_only and kind == "end":
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts

    start_ts = _to_utc(start, kind="start")
    end_ts = _to_utc(end, kind="end")
    if end_ts < start_ts:
        raise ValueError(f"end ({end_ts}) is before start ({start_ts})")

    first_month = start_ts.tz_convert(None).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    last_month = end_ts.tz_convert(None).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    months = pd.date_range(first_month, last_month, freq="MS")

    chunks: list[pd.Series] = []
    for m in months:
        path = _cache_path(cache, indicator_id, geo_id, int(m.year), int(m.month))
        if path.exists() and not refresh:
            chunks.append(pd.read_parquet(path)["value"])
            continue
        chunk = _fetch_month(indicator_id, geo_id, int(m.year), int(m.month))
        # Persist even an empty result so we don't loop on empty months.
        chunk.to_frame("value").to_parquet(path)
        chunks.append(chunk)

    if not chunks:
        return pd.Series(dtype=float, name="value")
    out = pd.concat(chunks)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    # Cached parquet may have lost tz on round-trip; ensure UTC.
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out.loc[start_ts:end_ts]
