#!/usr/bin/env python3
"""Generate deterministic 1-year 4h fixture for SPY with Triple Screen pullbacks.

Generates tests/fixtures/spy-1y-4h.csv with 2190 4h candles and at least 5
pullbacks matching TripleScreenStrategy entry conditions (QuantAgent-8u8.1).
"""

import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import random

from quantagent.strategy.triple_screen_strategy import TripleScreenStrategy

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "tests" / "fixtures" / "spy-1y-4h.csv"

SYMBOL, TIMEFRAME = "SPY", "4h"
START_DATE = datetime(2024, 1, 1, 0, 0, 0)
NUM_CANDLES = 2190  # 365 days * 6 candles/day
BASE_VOLUME = 100_000
PULLBACK_INDICES = [349, 649, 949, 1249, 1549, 1849]


def generate_candles(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    candles = []
    price = 400.0

    for i in range(NUM_CANDLES):
        ts = (START_DATE + timedelta(hours=4 * i)).isoformat()
        price += rng.uniform(0.04, 0.08)
        candles.append({
            "symbol": SYMBOL, "timeframe": TIMEFRAME, "timestamp": ts,
            "open": round(price - 0.2, 2), "high": round(price + 0.3, 2),
            "low": round(price - 0.3, 2), "close": round(price, 2),
            "volume": BASE_VOLUME,
        })

    for idx in PULLBACK_INDICES:
        base = candles[idx - 5]["close"]
        candles[idx - 4].update({"open": round(base + 0.5, 2), "high": round(base + 7.0, 2), "low": round(base, 2), "close": round(base + 3.0, 2)})
        candles[idx - 3].update({"open": round(base + 2.5, 2), "high": round(base + 2.5, 2), "low": round(base + 1.0, 2), "close": round(base + 1.2, 2)})
        candles[idx - 2].update({"open": round(base + 1.2, 2), "high": round(base + 1.5, 2), "low": round(base + 0.2, 2), "close": round(base + 0.3, 2)})
        candles[idx - 1].update({"open": round(base + 0.3, 2), "high": round(base + 0.6, 2), "low": round(base - 0.5, 2), "close": round(base - 0.2, 2)})
        candles[idx].update({"open": round(base + 0.2, 2), "high": round(base + 1.5, 2), "low": round(base - 0.1, 2), "close": round(base + 0.8, 2)})

        # Follow-through rally to hit take-profit (+4% TP in TripleScreen)
        entry_p = candles[idx]["close"]
        target_p = entry_p * 1.045
        for k in range(1, 7):
            p = round(entry_p + (target_p - entry_p) * (k / 6.0), 2)
            candles[idx + k].update({"open": round(p - 0.2, 2), "high": round(p + 0.4, 2), "low": round(p - 0.3, 2), "close": p})

    return candles


def find_pullbacks(candles: list[dict], window: int | None = 80) -> list[tuple[int, str, float, float, float]]:
    strategy = TripleScreenStrategy()
    pullbacks = []
    for i in range(len(candles)):
        if i < 2 or candles[i]["close"] <= candles[i - 1]["high"]:
            continue
        slice_data = candles[max(0, i + 1 - window) : i + 1] if window else candles[: i + 1]
        if len(slice_data) < strategy._min_candles:
            continue
        sig = strategy.generate_signal(slice_data, SYMBOL, TIMEFRAME, candles[i]["close"])
        if sig and sig.decision == "LONG":
            highs = [candles[j]["high"] for j in range(i - 4, i + 1)]
            lows = [candles[j]["low"] for j in range(i - 4, i + 1)]
            stoch = 100.0 * (candles[i]["close"] - min(lows)) / (max(highs) - min(lows) + 1e-10)
            pullbacks.append((i, candles[i]["timestamp"], candles[i]["close"], round(stoch, 2), candles[i - 1]["high"]))
    return pullbacks


def write_fixture(candles: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(candles)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    candles = generate_candles()
    pullbacks = find_pullbacks(candles, window=80)
    sha256 = write_fixture(candles, OUTPUT_PATH)

    print(f"Generated {len(candles)} candles to {OUTPUT_PATH}")
    print(f"sha256: {sha256}")
    print(f"Found {len(pullbacks)} pullbacks (window=80):")
    for idx, ts, price, stoch, prior_high in pullbacks:
        print(f"  - Index {idx:4d} | Date: {ts} | Close: {price:.2f} | Stoch: {stoch:.2f} | Prior High: {prior_high:.2f}")

    default_bars = TripleScreenStrategy().required_history_bars
    pullbacks_default = find_pullbacks(candles, window=default_bars)
    print(f"Signals with default required_history_bars ({default_bars}): {len(pullbacks_default)}")

    if len(pullbacks) < 5:
        raise ValueError(f"Expected >= 5 pullbacks, found {len(pullbacks)}")


if __name__ == "__main__":
    main()
