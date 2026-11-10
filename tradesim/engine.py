"""Симуляція: позиція → дохідність з урахуванням витрат → метрики."""
import numpy as np
import pandas as pd


def simulate(df: pd.DataFrame, pos: pd.Series, cost: float) -> pd.DataFrame:
    """Рішення на закритті свічки t діє протягом свічки t+1. Витрати — при кожній зміні позиції."""
    close = df["close"]
    nxt = close.pct_change().shift(-1).fillna(0.0)
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    ret = pos * nxt - turn * cost
    return pd.DataFrame({"close": close, "pos": pos, "ret": ret, "equity": (1 + ret).cumprod()})


def trades(sim: pd.DataFrame, cost: float):
    """Список закритих угод: вхід, вихід, результат з витратами."""
    pos = sim["pos"].values
    close = sim["close"].values
    idx = sim.index
    out, entry = [], None
    for i in range(len(pos)):
        if pos[i] == 1 and (i == 0 or pos[i - 1] == 0):
            entry = i
        elif pos[i] == 0 and i > 0 and pos[i - 1] == 1 and entry is not None:
            r = close[i] / close[entry] * (1 - cost) ** 2 - 1
            out.append((idx[entry], idx[i], r))
            entry = None
    return out


def metrics(sim: pd.DataFrame, cost: float) -> dict:
    eq = sim["equity"]
    ret = sim["ret"]
    span_days = max((sim.index[-1] - sim.index[0]).total_seconds() / 86400, 1)
    bars_per_year = len(sim) / span_days * 365
    sd = ret.std()
    sharpe = float(ret.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    tr = trades(sim, cost)
    wins = [t for t in tr if t[2] > 0]
    total = float(eq.iloc[-1] - 1)
    years = span_days / 365
    cagr = float((1 + total) ** (1 / years) - 1) if total > -1 and years > 0 else -1.0
    return {
        "return": round(total, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(dd, 4),
        "sharpe": round(sharpe, 2),
        "trades": len(tr),
        "trades_per_day": round(len(tr) / span_days, 2),
        "win_rate": round(len(wins) / len(tr), 3) if tr else None,
        "avg_trade": round(float(np.mean([t[2] for t in tr])), 4) if tr else None,
        "exposure": round(float(sim["pos"].mean()), 3),
        "days": round(span_days, 1),
    }


def curve(sim: pd.DataFrame, points: int = 250):
    """Стиснута крива капіталу для графіка: [[unix_ms, equity], ...]."""
    eq = sim["equity"]
    step = max(1, len(eq) // points)
    eq = pd.concat([eq.iloc[::step], eq.iloc[-1:]])
    eq = eq[~eq.index.duplicated(keep="last")]
    return [[int(ts.timestamp() * 1000), round(float(v), 4)] for ts, v in eq.items()]
