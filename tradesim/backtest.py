"""Бектест з чесною перевіркою.

1. Walk-forward. Перші 40% історії — лише для навчання. Решта 60% ділиться на 4 відрізки.
   Перед кожним відрізком параметри стратегії підбираються на ВСІЙ попередній історії,
   а потім стратегія торгує на відрізку, якого вона не бачила. Чотири відрізки склеюються
   в один результат «поза вибіркою».
2. Тест на везіння. Угоди стратегії (ті самі тривалості, та сама кількість) розкидаються
   по тому самому періоду у випадковому порядку 1000 разів. Показник «краща за X% випадкових»
   каже, чи вміє стратегія вибирати момент, чи їй просто пощастило / ринок ріс.
3. Групи «клас активів × стратегія»: кандидат (працює на більшості активів класу) або
   мертва (не заробила на жодному). Кандидатами торгує портфель «один депозит».
Для паперової торгівлі параметри перепідбираються на всій історії.
"""
import json
import math
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .config import ASSETS, CLASS_NAMES, COSTS
from .engine import curve, metrics, simulate
from .strategies import STRATEGIES, signal

OUT = Path("docs/data/backtest.json")
MIN_TRADES = 8
TRAIN0 = 0.40            # частка історії лише для навчання
FOLDS = 4                # відрізків поза вибіркою
LUCK_SIMS = 1000
CANDIDATE_LUCK = 0.80    # група-кандидат: у середньому краща за 80% випадкових
LUCKY = 0.95             # окрема комбінація «пройшла тест на везіння»


def _pick(df, key, cost):
    """Найкращі параметри за коефіцієнтом Шарпа (з мінімальною кількістю угод)."""
    best, best_score = None, -1e9
    for p in STRATEGIES[key]["grid"]:
        m = metrics(simulate(df, signal(key, df, p), cost), cost)
        score = m["sharpe"] if m["trades"] >= MIN_TRADES else m["sharpe"] - 10
        if score > best_score:
            best, best_score = p, score
    return best


def luck(sim: pd.DataFrame, cost: float, seed: int):
    """Частка випадкових стратегій (ті самі угоди в іншому порядку), які стратегія обігнала."""
    pos = sim["pos"].to_numpy()
    nxt = sim["close"].pct_change().shift(-1).fillna(0.0).to_numpy()
    edges = np.flatnonzero(np.diff(pos)) + 1
    lengths = np.diff(np.r_[0, edges, len(pos)])
    values = pos[np.r_[0, edges]]
    ones, zeros = lengths[values == 1], lengths[values == 0]
    if len(ones) < 3 or len(zeros) < 2:
        return None
    real = float(sim["equity"].iloc[-1] - 1)
    rng = np.random.default_rng(seed)
    first = values[0]
    a, b = (ones, zeros) if first == 1 else (zeros, ones)
    worse = 0
    for _ in range(LUCK_SIMS):
        pa, pb = rng.permutation(a), rng.permutation(b)
        ln = np.empty(len(lengths), dtype=np.int64)
        ln[0::2], ln[1::2] = pa, pb
        vals = np.empty(len(lengths))
        vals[0::2], vals[1::2] = first, 1 - first
        p = np.repeat(vals, ln)
        turn = np.abs(np.diff(p, prepend=0.0))
        total = np.exp(np.log1p(p * nxt - turn * cost).sum()) - 1
        worse += total < real
    return round(int(worse) / LUCK_SIMS, 3)


def run_asset(asset: str) -> dict:
    cls, name, _, _ = ASSETS[asset]
    cost = COSTS[cls]
    df = data.load(asset)
    if len(df) < 1500:
        raise RuntimeError(f"замало даних ({len(df)} свічок)")
    n = len(df)
    bounds = [int(n * (TRAIN0 + (1 - TRAIN0) * k / FOLDS)) for k in range(FOLDS + 1)]
    bounds[-1] = n
    oos = df.iloc[bounds[0]:]
    bh = simulate(oos, pd.Series(1.0, index=oos.index), cost)

    def fold_ret(sim, k):
        part = sim["ret"].iloc[bounds[k] - bounds[0]:bounds[k + 1] - bounds[0]]
        return float((1 + part).prod() - 1)

    bh_folds = [fold_ret(bh, k) for k in range(FOLDS)]
    res = {
        "asset": asset, "name": name, "class": cls, "class_name": CLASS_NAMES[cls], "cost": cost,
        "bars": n,
        "from": df.index[0].isoformat(), "split": oos.index[0].isoformat(), "to": df.index[-1].isoformat(),
        "fold_starts": [df.index[b].isoformat() for b in bounds[:-1]],
        "buy_hold": {"test": metrics(bh, cost), "folds": [round(x, 4) for x in bh_folds], "curve": curve(bh)},
        "strategies": [],
    }
    for key, spec in STRATEGIES.items():
        pos = pd.Series(0.0, index=df.index)
        for k in range(FOLDS):
            params = _pick(df.iloc[:bounds[k]], key, cost)
            # Індикатори рахуються на всій історії («розігріті»), але беруть лише минулі дані
            pos.iloc[bounds[k]:bounds[k + 1]] = signal(key, df, params).iloc[bounds[k]:bounds[k + 1]]
        sim = simulate(oos, pos.loc[oos.index], cost)
        m = metrics(sim, cost)
        folds = [fold_ret(sim, k) for k in range(FOLDS)]
        res["strategies"].append({
            "key": key, "name": spec["name"],
            "live_params": _pick(df, key, cost),
            "test": m,
            "folds": [round(x, 4) for x in folds],
            "folds_pos": sum(x > 0 for x in folds),
            "folds_beat": sum(x > y for x, y in zip(folds, bh_folds)),
            "luck": luck(sim, cost, zlib.crc32(f"{asset}|{key}".encode())),
            "beats_buy_hold": m["return"] > res["buy_hold"]["test"]["return"],
            "curve": curve(sim),
        })
    return res


def _groups(results):
    """Зведення «клас × стратегія» і позначки dead / selected на кожній комбінації."""
    by = {}
    for r in results:
        for s in r["strategies"]:
            by.setdefault((r["class"], s["key"]), []).append(s)
    groups = []
    for (cls, key), xs in by.items():
        lucks = [x["luck"] for x in xs if x["luck"] is not None]
        profitable = sum(x["test"]["return"] > 0 for x in xs)
        avg_luck = float(np.mean(lucks)) if lucks else None
        if profitable == 0:
            status = "dead"
        elif (len(xs) >= 2 and profitable >= math.ceil(len(xs) * 2 / 3)
              and avg_luck is not None and avg_luck >= CANDIDATE_LUCK):
            status = "candidate"
        else:
            status = "neutral"
        for x in xs:
            x["dead"] = status == "dead"
            x["selected"] = status == "candidate"
        groups.append({
            "class": cls, "class_name": CLASS_NAMES[cls], "strategy": key, "name": STRATEGIES[key]["name"],
            "n": len(xs), "profitable": profitable, "beat": sum(x["beats_buy_hold"] for x in xs),
            "avg_return": round(float(np.mean([x["test"]["return"] for x in xs])), 4),
            "avg_luck": round(avg_luck, 3) if avg_luck is not None else None,
            "status": status,
        })
    return groups


def main():
    results, errors = [], {}
    for asset in ASSETS:
        t = time.time()
        print(f"▶ {asset}")
        try:
            r = run_asset(asset)
            results.append(r)
            best = max(r["strategies"], key=lambda s: s["test"]["return"])
            print(f"  {r['bars']} свічок, поза вибіркою: купив-і-тримав {r['buy_hold']['test']['return']:+.1%}, "
                  f"найкраща {best['key']} {best['test']['return']:+.1%} "
                  f"(краща за {best['luck']} випадкових) ({time.time() - t:.0f}с)")
        except Exception as e:
            errors[asset] = str(e)
            print(f"  ✗ {e}")

    groups = _groups(results)
    combos = [s for r in results for s in r["strategies"]]
    summary = {
        "combos": len(combos),
        "profitable": sum(s["test"]["return"] > 0 for s in combos),
        "beat_buy_hold": sum(s["beats_buy_hold"] for s in combos),
        "lucky": int(sum((s["luck"] or 0) >= LUCKY for s in combos)),
        "lucky_expected": round(len(combos) * (1 - LUCKY), 1),
        "candidates": sum(g["status"] == "candidate" for g in groups),
        "dead": sum(g["status"] == "dead" for g in groups),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train0": TRAIN0, "folds": FOLDS, "luck_sims": LUCK_SIMS, "lucky": LUCKY,
        "candidate_luck": CANDIDATE_LUCK,
        "summary": summary, "groups": groups, "errors": errors, "assets": results,
    }, ensure_ascii=False, separators=(",", ":"), default=float))
    print(f"\nГотово: {summary}. Помилки: {list(errors) or 'немає'}")
    for g in groups:
        if g["status"] != "neutral":
            print(f"  {g['status']:9} {g['class']:11} {g['strategy']:20} у плюсі {g['profitable']}/{g['n']}, "
                  f"сер. {g['avg_return']:+.1%}, везіння {g['avg_luck']}")
