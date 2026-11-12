"""Портфель «один депозит» — так, як торгував би реальний бот.

* Один віртуальний депозит DEPOSIT, а не окремий рахунок на кожну стратегію.
* Торгує лише групами-кандидатами з бектесту («клас × стратегія», що працює на більшості
  активів класу і пройшла тест на везіння).
* Угоди — за поточною ринковою ціною на момент запуску і лише коли ринок відкритий.
* Ризик на угоду RISK від депозиту: розмір позиції рахується від відстані до стоп-лосу
  (STOP_ATR × середній годинний розмах ціни), але не більше MAX_POS депозиту.
* Стоп-лос на кожній позиції. Після стопу новий вхід — лише за новим сигналом.
* Денний ліміт збитку DAY_LIMIT: усе закривається, до наступного дня (UTC) нових угод немає.
* Входить лише за свіжим сигналом, а не в угоду, що вже триває.
"""
import csv
import json
from pathlib import Path

import pandas as pd

from .config import ASSETS, CLASS_NAMES, COSTS
from .strategies import STRATEGIES, signal

STATE = Path("docs/data/portfolio.json")
TRADES_CSV = Path("data/portfolio_trades.csv")
DEPOSIT = 10_000.0
RISK = 0.01
ATR_N = 24
STOP_ATR = 2.5
MAX_POS = 0.15
DAY_LIMIT = 0.03
MIN_ORDER = 20.0
HISTORY_MAX = 3000
RECENT_TRADES = 300
CSV_FIELDS = ["time", "asset", "strategy", "side", "reason", "price", "value", "pnl", "pnl_usd"]
REASONS = {"signal": "сигнал", "stop": "стоп-лос", "limit": "денний ліміт", "dropped": "стратегію знято"}


def _atr(df: pd.DataFrame) -> float:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(ATR_N).mean().iloc[-1])


def _log(trade):
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not TRADES_CSV.exists()
    with TRADES_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(trade)


def step(market: dict, bt: dict, now):
    """market: {asset: (закриті свічки, поточна ціна, чи можна торгувати)}."""
    when = now.isoformat(timespec="seconds")
    if STATE.exists():
        st = json.loads(STATE.read_text())
    else:
        st = {"started": when, "deposit": DEPOSIT, "cash": DEPOSIT, "positions": {}, "armed": {},
              "trades": [], "history": [], "fees": 0.0, "closed": 0, "wins": 0, "day": None}

    selected = {f"{a['asset']}|{s['key']}": s["live_params"]
                for a in bt.get("assets", []) for s in a["strategies"] if s.get("selected")}

    def price(asset):
        return market[asset][1]

    def equity():
        return st["cash"] + sum(p["units"] * (price(p["asset"]) if p["asset"] in market else p["last_price"])
                                for p in st["positions"].values())

    def sell(pid, px, reason):
        p = st["positions"].pop(pid)
        gross = p["units"] * px
        fee = gross * COSTS[ASSETS[p["asset"]][0]]
        st["cash"] += gross - fee
        st["fees"] += fee
        pnl_usd = gross - fee - p["cost_basis"]
        st["closed"] += 1
        st["wins"] += pnl_usd > 0
        t = {"time": when, "asset": p["asset"], "strategy": p["strategy"], "side": "SELL",
             "reason": REASONS[reason], "price": px, "value": round(gross - fee, 2),
             "pnl": round(pnl_usd / p["cost_basis"], 4), "pnl_usd": round(pnl_usd, 2)}
        st["trades"].append(t); _log(t)

    # 1. Стоп-лоси: чи торкалась ціна стопа у свічках, що почались після входу
    for pid, p in list(st["positions"].items()):
        if p["asset"] not in market:
            continue
        df, live, tradable = market[p["asset"]]
        since = max(pd.Timestamp(p["entry_time"]), pd.Timestamp(p["checked"]))
        hit = None
        for ts, bar in df[df.index > since].iterrows():
            if bar["low"] <= p["stop"]:
                hit = min(p["stop"], float(bar["open"]))   # розрив униз → гірша ціна
                break
        if hit is None and tradable and live <= p["stop"]:
            hit = live
        p["checked"] = df.index[-1].isoformat()
        p["last_price"] = live
        if hit is not None:
            sell(pid, hit, "stop")
            st["armed"][pid] = False

    # 2. Денний ліміт збитку
    eq = equity()
    today = now.date().isoformat()
    if not st["day"] or st["day"]["date"] != today:
        st["day"] = {"date": today, "start_equity": round(eq, 2), "halted": False}
    if not st["day"]["halted"] and eq <= st["day"]["start_equity"] * (1 - DAY_LIMIT):
        st["day"]["halted"] = True
    halted = st["day"]["halted"]

    # 3. Сигнали
    for pid in sorted(set(selected) | set(st["positions"])):
        asset, key = pid.split("|")
        if asset not in market:
            continue
        df, live, tradable = market[asset]
        sig = float(signal(key, df, selected[pid]).iloc[-1]) if pid in selected else 0.0
        if sig == 0:
            st["armed"][pid] = True
        if pid in st["positions"]:
            if tradable and (sig == 0 or halted):
                sell(pid, live, "limit" if halted and sig == 1 else "signal" if pid in selected else "dropped")
        elif sig == 1 and st["armed"].get(pid) and not halted and tradable:
            atr = _atr(df)
            stop_dist = STOP_ATR * atr / live
            if not (0 < stop_dist < 1):
                continue
            eq = equity()
            size = min(eq * RISK / stop_dist, eq * MAX_POS, st["cash"])
            if size < MIN_ORDER:
                continue
            fee = size * COSTS[ASSETS[asset][0]]
            st["cash"] -= size
            st["fees"] += fee
            st["positions"][pid] = {
                "asset": asset, "strategy": key, "units": (size - fee) / live, "entry_price": live,
                "entry_time": when, "checked": when, "stop": live * (1 - stop_dist),
                "cost_basis": size, "last_price": live}
            st["armed"][pid] = False
            t = {"time": when, "asset": asset, "strategy": key, "side": "BUY", "reason": REASONS["signal"],
                 "price": live, "value": round(size, 2), "pnl": "", "pnl_usd": ""}
            st["trades"].append(t); _log(t)

    # 4. Підсумок
    eq = equity()
    for p in st["positions"].values():
        p["value"] = round(p["units"] * p["last_price"], 2)
        p["pnl"] = round(p["units"] * p["last_price"] / p["cost_basis"] - 1, 4)
    st.update(
        last_run=when, equity=round(eq, 2), **{"return": round(eq / st["deposit"] - 1, 4)},
        selected=[{"asset": pid.split("|")[0], "strategy": pid.split("|")[1],
                   "name": STRATEGIES[pid.split("|")[1]]["name"],
                   "class_name": CLASS_NAMES[ASSETS[pid.split("|")[0]][0]]} for pid in sorted(selected)],
        rules={"risk": RISK, "stop_atr": STOP_ATR, "atr_n": ATR_N, "max_pos": MAX_POS, "day_limit": DAY_LIMIT},
    )
    st["history"].append([int(now.timestamp() * 1000), st["equity"]])
    del st["history"][:-HISTORY_MAX]
    st["trades"] = st["trades"][-RECENT_TRADES:]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, separators=(",", ":")))
    print(f"Один депозит: ${st['equity']:.2f} ({st['return']:+.2%}), позицій {len(st['positions'])}, "
          f"стратегій у роботі {len(selected)}{', ДЕННИЙ ЛІМІТ' if halted else ''}")
