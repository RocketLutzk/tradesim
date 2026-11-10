"""Торгові стратегії. Кожна повертає позицію 0/1 (поза ринком / купили) на закритті кожної свічки.

Тільки лонг (купити → продати), без плеча — так, як торгуватиме реальний бот на старті.
Усі індикатори рахуються лише з минулих даних, тож підглядання в майбутнє немає.
"""
from itertools import product

import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Перетворює сигнали входу/виходу на позицію: 1 після входу, 0 після виходу."""
    pos = pd.Series(np.nan, index=entry.index)
    pos[entry] = 1.0
    pos[exit_] = 0.0          # якщо обидва сигнали одночасно — виходимо
    return pos.ffill().fillna(0.0)


def ema_cross(df, fast, slow):
    c = df["close"]
    return (_ema(c, fast) > _ema(c, slow)).astype(float)


def macd(df, fast, slow, signal):
    c = df["close"]
    line = _ema(c, fast) - _ema(c, slow)
    return (line > _ema(line, signal)).astype(float)


def rsi_reversion(df, low, high):
    r = _rsi(df["close"], 14)
    return _state(r < low, r > high)


def bollinger_reversion(df, period, k):
    c = df["close"]
    mid = c.rolling(period).mean()
    sd = c.rolling(period).std()
    return _state(c < mid - k * sd, c > mid)


def donchian_breakout(df, enter_n, exit_n):
    c = df["close"]
    hi = df["high"].rolling(enter_n).max().shift(1)
    lo = df["low"].rolling(exit_n).min().shift(1)
    return _state(c > hi, c < lo)


def _grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in product(*kw.values())]


STRATEGIES = {
    "ema_cross": {
        "name": "EMA (тренд)",
        "fn": ema_cross,
        "grid": [p for p in _grid(fast=[8, 12, 20, 30], slow=[26, 50, 100, 200]) if p["fast"] < p["slow"]],
    },
    "macd": {
        "name": "MACD (тренд)",
        "fn": macd,
        "grid": _grid(fast=[8, 12], slow=[21, 26, 50], signal=[9, 18]),
    },
    "donchian_breakout": {
        "name": "Пробій (Дончіан)",
        "fn": donchian_breakout,
        "grid": _grid(enter_n=[24, 48, 96, 168], exit_n=[12, 24, 48]),
    },
    "rsi_reversion": {
        "name": "RSI (відскок)",
        "fn": rsi_reversion,
        "grid": _grid(low=[20, 25, 30, 35], high=[50, 60, 70]),
    },
    "bollinger_reversion": {
        "name": "Боллінджер (відскок)",
        "fn": bollinger_reversion,
        "grid": _grid(period=[20, 48, 96], k=[1.5, 2.0, 2.5]),
    },
}


def signal(key, df, params):
    return STRATEGIES[key]["fn"](df, **params)
