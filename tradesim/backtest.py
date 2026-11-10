"""Бектест з чесною перевіркою.

Для кожного активу історія ділиться на дві частини:
  * навчальна (перші 70%) — тут для кожної стратегії підбираємо найкращі параметри;
  * тестова (останні 30%) — стратегія з ВЖЕ вибраними параметрами торгує на даних,
    яких вона «не бачила». Саме цей результат і є чесною оцінкою.
Для паперової торгівлі параметри перепідбираються на всій історії.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import data
from .config import ASSETS, CLASS_NAMES, COSTS, TRAIN_SHARE
from .engine import curve, metrics, simulate
from .strategies import STRATEGIES, signal

OUT = Path("docs/data/backtest.json")
MIN_TRADES = 8


def _pick(df, key, cost):
    """Найкращі параметри за коефіцієнтом Шарпа (з мінімальною кількістю угод)."""
    best, best_score = None, -1e9
    for p in STRATEGIES[key]["grid"]:
        m = metrics(simulate(df, signal(key, df, p), cost), cost)
        score = m["sharpe"] if m["trades"] >= MIN_TRADES else m["sharpe"] - 10
        if score > best_score:
            best, best_score = p, score
    return best


def run_asset(asset: str) -> dict:
    cls, name, _, _ = ASSETS[asset]
    cost = COSTS[cls]
    df = data.load(asset)
    if len(df) < 1500:
        raise RuntimeError(f"замало даних ({len(df)} свічок)")
    cut = int(len(df) * TRAIN_SHARE)
    train, test = df.iloc[:cut], df.iloc[cut:]

    def bh(part):
        pos = pd.Series(1.0, index=part.index)
        return simulate(part, pos, cost)

    bh_test = bh(test)
    res = {
        "asset": asset, "name": name, "class": cls, "class_name": CLASS_NAMES[cls], "cost": cost,
        "bars": len(df),
        "from": df.index[0].isoformat(), "split": test.index[0].isoformat(), "to": df.index[-1].isoformat(),
        "buy_hold": {"train": metrics(bh(train), cost), "test": metrics(bh_test, cost),
                     "curve": curve(bh_test)},
        "strategies": [],
    }
    for key, spec in STRATEGIES.items():
        params = _pick(train, key, cost)
        # Сигнал рахуємо на всій історії (індикатори «розігріті»), а оцінюємо окремо по частинах
        pos = signal(key, df, params)
        s_train = simulate(train, pos.loc[train.index], cost)
        s_test = simulate(test, pos.loc[test.index], cost)
        m_test = metrics(s_test, cost)
        res["strategies"].append({
            "key": key, "name": spec["name"], "params": params,
            "live_params": _pick(df, key, cost),
            "train": metrics(s_train, cost), "test": m_test,
            "beats_buy_hold": m_test["return"] > res["buy_hold"]["test"]["return"],
            "curve": curve(s_test),
        })
    return res


def main():
    results, errors = [], {}
    for asset in ASSETS:
        t = time.time()
        print(f"▶ {asset}")
        try:
            r = run_asset(asset)
            results.append(r)
            best = max(r["strategies"], key=lambda s: s["test"]["return"])
            print(f"  {r['bars']} свічок, тест купив-і-тримав {r['buy_hold']['test']['return']:+.1%}, "
                  f"найкраща на тесті {best['key']} {best['test']['return']:+.1%} ({time.time() - t:.0f}с)")
        except Exception as e:
            errors[asset] = str(e)
            print(f"  ✗ {e}")

    combos = [s for r in results for s in r["strategies"]]
    summary = {
        "combos": len(combos),
        "profitable": sum(s["test"]["return"] > 0 for s in combos),
        "beat_buy_hold": sum(s["beats_buy_hold"] for s in combos),
        "beat_and_profitable": sum(s["beats_buy_hold"] and s["test"]["return"] > 0 for s in combos),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_share": TRAIN_SHARE, "summary": summary, "errors": errors, "assets": results,
    }, ensure_ascii=False, separators=(",", ":")))
    print(f"\nГотово: {summary}. Помилки: {list(errors) or 'немає'}")
