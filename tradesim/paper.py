"""Паперова (віртуальна) торгівля наживо.

Запускається щогодини. Для кожної пари «актив × стратегія» є окремий віртуальний рахунок
на START_CAPITAL доларів. Бот бере останню закриту свічку, рахує сигнал і, якщо треба,
«купує» або «продає» за ціною закриття, сплачуючи ті самі витрати, що й у бектесті.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from . import data
from .config import ASSETS, CLASS_NAMES, COSTS, START_CAPITAL
from .strategies import STRATEGIES, signal

STATE = Path("docs/data/paper.json")
TRADES_CSV = Path("data/trades.csv")
BACKTEST = Path("docs/data/backtest.json")
HISTORY_EVERY_H = 4      # точка на графіку раз на 4 години
HISTORY_MAX = 1000       # ≈ 5–6 місяців
RECENT_TRADES = 400


def _live_params():
    if not BACKTEST.exists():
        return {}
    bt = json.loads(BACKTEST.read_text())
    return {(a["asset"], s["key"]): s["live_params"] for a in bt["assets"] for s in a["strategies"]}


def _load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "capital": START_CAPITAL, "accounts": {}, "assets": {}, "trades": []}


def _log(trade):
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not TRADES_CSV.exists()
    with TRADES_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trade))
        if new:
            w.writeheader()
        w.writerow(trade)


def main():
    st = _load_state()
    params_map = _live_params()
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    errors, n_trades = {}, 0

    for asset, (cls, name, _, _) in ASSETS.items():
        cost = COSTS[cls]
        try:
            df = data.load(asset, days=60)
        except Exception as e:
            errors[asset] = str(e)
            print(f"✗ {asset}: {e}")
            continue
        if len(df) < 300:
            errors[asset] = f"замало даних ({len(df)})"
            continue
        price = float(df["close"].iloc[-1])
        bar = df.index[-1].isoformat()
        a = st["assets"].setdefault(asset, {"name": name, "class": cls, "class_name": CLASS_NAMES[cls],
                                            "start_price": price, "start_bar": bar})
        a.update(price=price, bar=bar, buy_hold=round(price / a["start_price"] - 1, 4))

        for key, spec in STRATEGIES.items():
            params = params_map.get((asset, key)) or spec["grid"][0]
            acc_id = f"{asset}|{key}"
            acc = st["accounts"].setdefault(acc_id, {
                "asset": asset, "strategy": key, "strategy_name": spec["name"],
                "cash": START_CAPITAL, "units": 0.0, "entry_price": None, "entry_time": None,
                "last_bar": None, "trades": 0, "wins": 0, "fees": 0.0, "history": []})
            acc["params"] = params

            if acc["last_bar"] != bar:          # нова закрита свічка → рахуємо сигнал
                want = float(signal(key, df, params).iloc[-1])
                if want == 1 and acc["units"] == 0:
                    fee = acc["cash"] * cost
                    acc["units"] = (acc["cash"] - fee) / price
                    acc["cost_basis"] = acc["cash"]
                    acc["cash"] = 0.0
                    acc["fees"] += fee
                    acc["entry_price"], acc["entry_time"] = price, bar
                    t = {"time": bar, "asset": asset, "strategy": key, "side": "BUY", "price": price,
                         "value": round(acc["cost_basis"], 2), "pnl": ""}
                    st["trades"].append(t); _log(t); n_trades += 1
                elif want == 0 and acc["units"] > 0:
                    gross = acc["units"] * price
                    fee = gross * cost
                    acc["cash"] = gross - fee
                    acc["fees"] += fee
                    pnl = acc["cash"] / acc.get("cost_basis", acc["cash"]) - 1
                    acc["trades"] += 1
                    acc["wins"] += pnl > 0
                    t = {"time": bar, "asset": asset, "strategy": key, "side": "SELL", "price": price,
                         "value": round(acc["cash"], 2), "pnl": round(pnl, 4)}
                    st["trades"].append(t); _log(t); n_trades += 1
                    acc["units"], acc["entry_price"], acc["entry_time"] = 0.0, None, None
                acc["last_bar"] = bar

            acc["equity"] = round(acc["cash"] + acc["units"] * price, 2)
            acc["return"] = round(acc["equity"] / START_CAPITAL - 1, 4)
            acc["in_position"] = acc["units"] > 0
            h = acc["history"]
            if not h or now_ms - h[-1][0] >= HISTORY_EVERY_H * 3_600_000 - 300_000:
                h.append([now_ms, acc["equity"]])
            else:
                h[-1][1] = acc["equity"]        # оновлюємо останню точку
            del h[:-HISTORY_MAX]
        print(f"✓ {asset} {price:g}")

    st["trades"] = st["trades"][-RECENT_TRADES:]
    st["last_run"] = now.isoformat(timespec="seconds")
    st["errors"] = errors
    accs = list(st["accounts"].values())
    st["total"] = {
        "accounts": len(accs),
        "equity": round(sum(x["equity"] for x in accs), 2),
        "capital": START_CAPITAL * len(accs),
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, separators=(",", ":")))
    print(f"Угод за цей запуск: {n_trades}. Помилки: {list(errors) or 'немає'}")
