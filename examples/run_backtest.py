"""
Example: Running a backtest with QuantAgent.

This script demonstrates how to:
1. Create a Backtest instance
2. Run a backtest on historical data
3. View performance metrics
4. Export equity curve

Usage:
    python examples/run_backtest.py
"""

from datetime import datetime, timedelta
from quantagent.backtesting.backtest import Backtest
from quantagent.database import SessionLocal

def main():
    """Run example backtest."""

    # Configuration
    config = {
        'base_position_pct': 0.05,  # 5% of portfolio per trade
        'max_daily_loss_pct': 0.05,  # 5% max daily loss
        'max_position_pct': 0.10,  # 10% max position size
        'slippage_pct': 0.01,  # 1% slippage simulation
        'agent_llm_provider': 'openai',
        'agent_llm_model': 'gpt-4o-mini',
        'agent_llm_temperature': 0.1
    }

    # Date range for backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months

    # Create backtest instance
    backtest = Backtest(
        start_date=start_date,
        end_date=end_date,
        assets=['BTC', 'SPX'],
        timeframe='4h',
        initial_capital=100000.0,
        config=config,
        use_checkpointing=True  # Enable state persistence
    )

    print(f"Running backtest from {start_date.date()} to {end_date.date()}")
    print(f"Assets: {backtest.assets}")
    print(f"Timeframe: {backtest.timeframe}")
    print(f"Initial capital: ${backtest.initial_capital:,.2f}")
    print("-" * 60)

    # Run backtest
    metrics = backtest.run(name="90-Day BTC/SPX Backtest")

    # Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Trades:      {metrics.total_trades}")
    print(f"Winning Trades:    {metrics.winning_trades}")
    print(f"Losing Trades:     {metrics.losing_trades}")
    print(f"Win Rate:          {metrics.win_rate:.2%}")
    print(f"Profit Factor:     {metrics.profit_factor:.2f}")
    print(f"Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown:      {metrics.max_drawdown:.2%}")
    print(f"Total P&L:         ${metrics.total_pnl:,.2f}")
    print(f"Total Return:      {metrics.total_return_pct:.2f}%")
    print(f"Average Win:       ${metrics.avg_win:,.2f}")
    print(f"Average Loss:      ${metrics.avg_loss:,.2f}")
    print(f"Largest Win:       ${metrics.largest_win:,.2f}")
    print(f"Largest Loss:      ${metrics.largest_loss:,.2f}")
    print("=" * 60)

    # Export equity curve
    equity_df = backtest.get_equity_curve()
    print(f"\nEquity curve data points: {len(equity_df)}")

    # Optional: Save to CSV
    # equity_df.to_csv('equity_curve.csv', index=False)
    # print("Equity curve saved to equity_curve.csv")

    # Viability assessment
    print("\nStrategy Assessment:")
    if metrics.win_rate >= 0.4 and metrics.sharpe_ratio >= 1.0 and metrics.max_drawdown <= 0.15:
        print("✅ Strategy meets viability criteria (Win Rate ≥40%, Sharpe ≥1.0, Max DD ≤15%)")
    else:
        print("⚠️  Strategy does not meet all viability criteria")
        if metrics.win_rate < 0.4:
            print(f"   - Win rate too low: {metrics.win_rate:.2%} (need ≥40%)")
        if metrics.sharpe_ratio < 1.0:
            print(f"   - Sharpe ratio too low: {metrics.sharpe_ratio:.2f} (need ≥1.0)")
        if metrics.max_drawdown > 0.15:
            print(f"   - Max drawdown too high: {metrics.max_drawdown:.2%} (need ≤15%)")


if __name__ == "__main__":
    main()
