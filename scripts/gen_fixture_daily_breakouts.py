#!/usr/bin/env python3
"""Generate deterministic 2-year daily fixture for SPY with 52-week high breakouts.

Generates tests/fixtures/spy-2y-1d.csv with ~520 daily candles and at least 5
breakouts matching FiftyTwoWeekHighStrategy entry conditions (QuantAgent-46h.1).
"""

import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import random

from quantagent.strategy.fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "tests" / "fixtures" / "spy-2y-1d.csv"

SYMBOL, TIMEFRAME = "SPY", "1d"
START_DATE = datetime(2024, 1, 1)
NUM_CANDLES = 520
BASE_VOLUME, BREAKOUT_VOLUME = 1_000_000, 2_500_000


def generate_candles(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    candles = []
    breakout_indices = {320, 355, 390, 425, 460, 495}
    current_price, highest_seen = 400.0, 400.0

    for i in range(NUM_CANDLES):
        timestamp = (START_DATE + timedelta(days=i)).isoformat()
        if i in breakout_indices:
            current_price = highest_seen + 5.0 + rng.uniform(0.5, 2.0)
            open_p = highest_seen - rng.uniform(0.5, 1.5)
            high_p = current_price + rng.uniform(0.5, 1.5)
            low_p = min(open_p, current_price) - rng.uniform(0.5, 1.5)
            close_p, volume = current_price, BREAKOUT_VOLUME
            highest_seen = high_p
        else:
            max_allowed = highest_seen - 2.0
            drift = rng.uniform(-1.0, 1.0)
            current_price = max(350.0, min(max_allowed, current_price + drift))
            open_p = max(350.0, min(max_allowed, current_price + rng.uniform(-0.5, 0.5)))
            high_p = max(open_p, current_price) + rng.uniform(0.2, 0.8)
            low_p = min(open_p, current_price) - rng.uniform(0.2, 0.8)
            close_p = current_price
            volume = int(BASE_VOLUME + rng.uniform(-50_000, 50_000))

        candles.append({
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "timestamp": timestamp,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": volume,
        })
    return candles


def find_breakouts(candles: list[dict]) -> list[tuple[int, str, float]]:
    strategy = FiftyTwoWeekHighStrategy()
    breakouts = []
    for i in range(len(candles)):
        sig = strategy.generate_signal(candles[: i + 1], SYMBOL, TIMEFRAME, candles[i]["close"])
        if sig and sig.decision == "LONG":
            breakouts.append((i, candles[i]["timestamp"], candles[i]["close"]))
    return breakouts


def write_fixture(candles: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candles)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    candles = generate_candles()
    breakouts = find_breakouts(candles)
    sha256 = write_fixture(candles, OUTPUT_PATH)

    print(f"Generated {len(candles)} candles to {OUTPUT_PATH}")
    print(f"sha256: {sha256}")
    print(f"Found {len(breakouts)} breakouts:")
    for idx, ts, price in breakouts:
        print(f"  - Index {idx:3d} | Date: {ts} | Price: {price:.2f}")

    if len(breakouts) < 5:
        raise ValueError(f"Expected >= 5 breakouts, found {len(breakouts)}")


if __name__ == "__main__":
    main()
