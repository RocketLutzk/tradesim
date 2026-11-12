"""Завантаження годинних свічок: Binance (крипта) з запасним варіантом Yahoo Finance."""
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .config import ASSETS

BINANCE_URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
HOUR_MS = 3_600_000


def _binance(symbol: str, days: int) -> pd.DataFrame:
    end = int(time.time() * 1000)
    start = end - days * 24 * HOUR_MS
    last_err = None
    for url in BINANCE_URLS:
        rows, cur = [], start
        try:
            while cur < end:
                r = requests.get(url, params={"symbol": symbol, "interval": "1h",
                                              "startTime": cur, "limit": 1000}, timeout=20)
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                rows += batch
                cur = batch[-1][0] + HOUR_MS
                time.sleep(0.15)
            if rows:
                df = pd.DataFrame(rows).iloc[:, :6]
                df.columns = ["ts", "open", "high", "low", "close", "volume"]
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                df = df.set_index("ts").astype(float)
                return df
        except Exception as e:  # пробуємо наступну адресу
            last_err = e
    raise RuntimeError(f"Binance недоступний для {symbol}: {last_err}")


def _yahoo(ticker: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    days = min(days, 729)  # Yahoo дає годинні дані лише за ~730 днів
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=f"{days}d", interval="1h",
                             progress=False, auto_adjust=True, threads=False)
            if df is not None and len(df):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                df.index = pd.to_datetime(df.index, utc=True)
                df.index.name = "ts"
                return df.dropna(subset=["close"]).astype(float)
        except Exception as e:
            print(f"  Yahoo {ticker}: спроба {attempt + 1} невдала: {e}")
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Yahoo не віддав дані для {ticker}")


def _synthetic(key: str, days: int) -> pd.DataFrame:
    """Випадкове блукання — лише для перевірки коду без інтернету."""
    rng = np.random.default_rng(abs(hash(key)) % 2**32)
    n = days * 24
    idx = pd.date_range(end=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
                        periods=n, freq="h")
    close = 100 * np.exp(np.cumsum(rng.normal(0.00002, 0.006, n)))
    noise = np.abs(rng.normal(0, 0.003, n))
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close * (1 + noise),
                         "low": close * (1 - noise), "close": close, "volume": 1.0}, index=idx)


def _raw(key: str, days: int) -> pd.DataFrame:
    cls, _, binance_symbol, yahoo_ticker = ASSETS[key]
    if os.environ.get("TRADESIM_SYNTHETIC"):
        df = _synthetic(key, days)
    elif binance_symbol:
        try:
            df = _binance(binance_symbol, days)
        except Exception as e:
            print(f"  {e} → беру Yahoo")
            df = _yahoo(yahoo_ticker, days)
    else:
        df = _yahoo(yahoo_ticker, days)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[df["close"] > 0]


def _us_market_open(now: pd.Timestamp) -> bool:
    ny = now.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    return ny.weekday() < 5 and 9 * 60 + 30 <= minutes < 16 * 60


def load_live(key: str, days: int = 60):
    """(закриті свічки, поточна ціна, чи можна торгувати просто зараз).

    Остання незакрита свічка відкидається з історії (щоб не підглядати в майбутнє),
    але її ціна — це поточна ринкова ціна, за якою реально можна купити чи продати.
    """
    raw = _raw(key, days)
    now = pd.Timestamp.now(tz="UTC")
    forming = bool(len(raw)) and raw.index[-1] + pd.Timedelta(hours=1) > now
    closed = raw.iloc[:-1] if forming else raw
    live = float(raw["close"].iloc[-1]) if len(raw) else float("nan")
    tradable = forming
    if ASSETS[key][0] == "stocks":
        tradable = tradable and _us_market_open(now)
    return closed, live, tradable


def load(key: str, days: int = 730) -> pd.DataFrame:
    """Годинні свічки для активу без останньої незакритої."""
    return load_live(key, days)[0]
