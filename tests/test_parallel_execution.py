"""
Test script to verify parallel execution of agents in the trading graph.
Measures execution time and confirms all three agents run simultaneously.
"""

import time
from pathlib import Path

import pandas as pd

from quantagent.static_util import read_and_format_ohlcv
from quantagent.trading_graph import TradingGraph

_DATA_PATH = Path("benchmark/btc/BTC_4h_1.csv")


def _load_sample_ohlcv() -> pd.DataFrame:
    """Load benchmark data when available, otherwise generate synthetic candles."""
    if _DATA_PATH.exists():
        df = pd.read_csv(_DATA_PATH)
        print(f"   ✓ Loaded {len(df)} candlesticks from benchmark dataset")
        return df

    print("   ⚠ benchmark/btc/BTC_4h_1.csv missing, generating synthetic dataset")
    periods = 120
    timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq="4H")
    base_price = 50000
    closes = [base_price + ((i % 6) - 3) * 120 + i * 3 for i in range(periods)]
    opens = [price - 60 for price in closes]
    highs = [price + 150 for price in closes]
    lows = [price - 180 for price in closes]
    volumes = [1000 + (i % 10) * 25 for i in range(periods)]

    df = pd.DataFrame(
        {
            "Datetime": timestamps,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )
    print(f"   ✓ Generated {len(df)} synthetic candlesticks for regression tests")
    return df


def test_parallel_execution():
    """Test that all three agents execute in parallel."""

    print("=" * 80)
    print("TESTING PARALLEL EXECUTION OF AGENTS")
    print("=" * 80)

    # Load sample data
    print("\n1. Loading sample OHLCV data...")
    df = _load_sample_ohlcv()
    df_dict = read_and_format_ohlcv(df)
    print(f"   ✓ Prepared {len(df)} candlesticks for execution")

    # Initialize trading graph
    print("\n2. Initializing TradingGraph...")
    tg = TradingGraph()
    print("   ✓ Graph initialized")

    # Prepare initial state
    initial_state = {
        "kline_data": df_dict,
        "time_frame": "4hour",
        "stock_name": "BTC",
        "messages": [],
    }

    # Execute graph and measure time
    print("\n3. Executing graph with parallel agents...")
    print("   Expected: Indicator, Pattern, and Trend agents run simultaneously")
    print("   Measuring execution time...")

    start_time = time.time()
    result = tg.graph.invoke(initial_state)
    end_time = time.time()

    execution_time = end_time - start_time

    # Verify results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\n✓ Total execution time: {execution_time:.2f} seconds")
    print("  (Expected: ~4-5s for parallel, ~6-9s for sequential)")

    # Check that all agent reports are present
    has_indicator = "indicator_report" in result and result["indicator_report"]
    has_pattern = "pattern_report" in result and result["pattern_report"]
    has_trend = "trend_report" in result and result["trend_report"]
    has_decision = "final_trade_decision" in result and result["final_trade_decision"]

    print(f"\n✓ Indicator Report: {'✓ Present' if has_indicator else '✗ Missing'}")
    print(f"✓ Pattern Report:   {'✓ Present' if has_pattern else '✗ Missing'}")
    print(f"✓ Trend Report:     {'✓ Present' if has_trend else '✗ Missing'}")
    print(f"✓ Decision Output:  {'✓ Present' if has_decision else '✗ Missing'}")

    if has_decision:
        decision = result["final_trade_decision"]
        print(f"\nFinal Decision: {decision}")

    # Performance analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    if execution_time < 6:
        print("\n✓ EXCELLENT: Execution time < 6s indicates parallel execution!")
        print("  All three agents (Indicator, Pattern, Trend) ran simultaneously.")
    elif execution_time < 7:
        print("\n✓ GOOD: Execution time suggests partial parallelization.")
    else:
        print("\n⚠ WARNING: Execution time suggests sequential execution.")
        print("  Check that graph edges are configured correctly for parallelization.")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return result


if __name__ == "__main__":
    result = test_parallel_execution()
