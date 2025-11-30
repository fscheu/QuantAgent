import json
import os
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request, send_file
from openai import OpenAI

import quantagent.static_util as static_util
from quantagent import settings
from quantagent.trading_graph import TradingGraph

app = Flask(__name__)


class WebTradingAnalyzer:
    def __init__(self):
        """Initialize the web trading analyzer."""
        # TradingGraph now loads config from .env automatically
        self.trading_graph = TradingGraph(use_checkpointing=True)
        self.data_dir = Path("data")

        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Available assets and their display names
        self.asset_mapping = {
            "SPX": "S&P 500",
            "BTC": "Bitcoin",
            "GC": "Gold Futures",
            "NQ": "Nasdaq Futures",
            "CL": "Crude Oil",
            "ES": "E-mini S&P 500",
            "DJI": "Dow Jones",
            "QQQ": "Invesco QQQ Trust",
            "VIX": "Volatility Index",
            "DXY": "US Dollar Index",
            "AAPL": "Apple Inc.",  # New asset
            "TSLA": "Tesla Inc.",  # New asset
        }

        # Yahoo Finance symbol mapping
        self.yfinance_symbols = {
            "SPX": "^GSPC",  # S&P 500
            "BTC": "BTC-USD",  # Bitcoin
            "GC": "GC=F",  # Gold Futures
            "NQ": "NQ=F",  # Nasdaq Futures
            "CL": "CL=F",  # Crude Oil
            "ES": "ES=F",  # E-mini S&P 500
            "DJI": "^DJI",  # Dow Jones
            "QQQ": "QQQ",  # Invesco QQQ Trust
            "VIX": "^VIX",  # Volatility Index
            "DXY": "DX-Y.NYB",  # US Dollar Index
        }

        # Yahoo Finance interval mapping
        self.yfinance_intervals = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",  # yfinance supports 4h natively!
            "1d": "1d",
            "1w": "1wk",
            "1mo": "1mo",
        }

        # Load persisted custom assets
        self.custom_assets_file = self.data_dir / "custom_assets.json"
        self.custom_assets = self.load_custom_assets()

    def fetch_yfinance_data(
        self, symbol: str, interval: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            yf_symbol = self.yfinance_symbols.get(symbol, symbol)
            yf_interval = self.yfinance_intervals.get(interval, interval)

            df = yf.download(
                tickers=yf_symbol, start=start_date, end=end_date, interval=yf_interval
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # Ensure df is a DataFrame, not a Series
            if isinstance(df, pd.Series):
                df = df.to_frame()

            # Reset index to ensure we have a clean DataFrame
            df = df.reset_index()

            # Ensure we have a DataFrame
            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()

            # Handle potential MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns if needed
            column_mapping = {
                "Date": "Datetime",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }

            # Only rename columns that exist
            existing_columns = {
                old: new for old, new in column_mapping.items() if old in df.columns
            }
            df = df.rename(columns=existing_columns)

            # Ensure we have the required columns
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: Missing columns. Available: {list(df.columns)}")
                return pd.DataFrame()

            # Select only the required columns
            df = df[required_columns]
            df["Datetime"] = pd.to_datetime(df["Datetime"])

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_yfinance_data_with_datetime(
        self,
        symbol: str,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance using datetime objects for exact time precision."""
        try:
            yf_symbol = self.yfinance_symbols.get(symbol, symbol)
            yf_interval = self.yfinance_intervals.get(interval, interval)

            print(
                f"Fetching {yf_symbol} from {start_datetime} to {end_datetime} with interval {yf_interval}"
            )

            # Use datetime objects directly for yfinance
            df = yf.download(
                tickers=yf_symbol,
                start=start_datetime,
                end=end_datetime,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )

            if df is None or df.empty:
                print(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Ensure df is a DataFrame, not a Series
            if isinstance(df, pd.Series):
                df = df.to_frame()

            # Reset index to ensure we have a clean DataFrame
            df = df.reset_index()

            # Ensure we have a DataFrame
            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()

            # Handle potential MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns if needed
            column_mapping = {
                "Date": "Datetime",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }

            # Only rename columns that exist
            existing_columns = {
                old: new for old, new in column_mapping.items() if old in df.columns
            }
            df = df.rename(columns=existing_columns)

            # Ensure we have the required columns
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: Missing columns. Available: {list(df.columns)}")
                return pd.DataFrame()

            # Select only the required columns
            df = df[required_columns]
            df["Datetime"] = pd.to_datetime(df["Datetime"])

            print(f"Successfully fetched {len(df)} data points for {symbol}")
            print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_available_assets(self) -> list:
        """Get list of available assets from the asset mapping dictionary."""
        return sorted(list(self.asset_mapping.keys()))

    def get_available_files(self, asset: str, timeframe: str) -> list:
        """Get available data files for a specific asset and timeframe."""
        asset_dir = self.data_dir / asset.lower()
        if not asset_dir.exists():
            return []

        pattern = f"{asset}_{timeframe}_*.csv"
        files = list(asset_dir.glob(pattern))
        return sorted(files)

    def run_analysis(
        self, df: pd.DataFrame, asset_name: str, timeframe: str
    ) -> Dict[str, Any]:
        """Run the trading analysis on the provided DataFrame."""
        try:
            # Debug: Check DataFrame structure
            print(f"DataFrame columns: {df.columns}")
            print(f"DataFrame index: {type(df.index)}")
            print(f"DataFrame shape: {df.shape}")

            # Format OHLCV data using centralized utility function
            try:
                df_slice_dict = static_util.read_and_format_ohlcv(df)
            except ValueError as e:
                return {
                    "success": False,
                    "error": str(e),
                }

            # Debug: Check the resulting dictionary
            print(f"Dictionary keys: {list(df_slice_dict.keys())}")
            print(f"Dictionary key types: {[type(k) for k in df_slice_dict.keys()]}")

            # Format timeframe for display
            display_timeframe = timeframe
            if timeframe.endswith("h"):
                display_timeframe += "our"
            elif timeframe.endswith("m"):
                display_timeframe += "in"
            elif timeframe.endswith("d"):
                display_timeframe += "ay"
            elif timeframe == "1w":
                display_timeframe = "1 week"
            elif timeframe == "1mo":
                display_timeframe = "1 month"

            p_image = static_util.generate_kline_image(df_slice_dict)
            t_image = static_util.generate_trend_image(df_slice_dict)

            # Create initial state
            initial_state = {
                "kline_data": df_slice_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": display_timeframe,
                "stock_name": asset_name,
                "pattern_image": p_image["pattern_image"],
                "trend_image": t_image["trend_image"],
            }

            # Run the trading graph
            thread_id = f"{asset_name}:{timeframe}"
            final_state = self.trading_graph.graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )


            return {
                "success": True,
                "final_state": final_state,
                "asset_name": asset_name,
                "timeframe": display_timeframe,
                "data_length": len(df_slice_dict),
            }

        except Exception as e:
            error_msg = str(e)

            # Get current provider from config module
            provider = settings.AGENT_LLM_PROVIDER
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            else:
                provider_name = "Qwen"

            # Check for specific API key authentication errors
            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "success": False,
                    "error": f"❌ Invalid API Key: The {provider_name} API key you provided is invalid or has expired. Please check your API key in the Settings section and try again.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "success": False,
                    "error": f"⚠️ Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"💳 Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"🌐 Network Error: Unable to connect to {provider_name} servers. Please check your internet connection and try again.",
                }
            else:
                return {"success": False, "error": f"❌ Analysis Error: {error_msg}"}

    def extract_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format analysis results for web display.

        Handles both string and Pydantic model outputs from agents.
        """
        if not results["success"]:
            return {"error": results["error"]}

        final_state = results["final_state"]

        # Extract analysis results from state fields
        indicator_report = final_state.get("indicator_report", "")
        pattern_report = final_state.get("pattern_report", "")
        trend_report = final_state.get("trend_report", "")
        final_decision_raw = final_state.get("final_trade_decision", "")

        # Extract chart data if available
        pattern_chart = final_state.get("pattern_image", "")
        trend_chart = final_state.get("trend_image", "")
        pattern_image_filename = final_state.get("pattern_image_filename", "")
        trend_image_filename = final_state.get("trend_image_filename", "")

        # Convert Pydantic models to readable format
        def format_report(report):
            """Convert Pydantic model or string to readable text."""
            if hasattr(report, "model_dump"):
                # It's a Pydantic model
                data = report.model_dump()
                return self._format_report_dict(data)
            elif hasattr(report, "__dict__"):
                # It's an object with attributes
                return self._format_report_dict(report.__dict__)
            else:
                # It's already a string
                return str(report)

        technical_indicators = format_report(indicator_report)
        pattern_analysis = format_report(pattern_report)
        trend_analysis = format_report(trend_report)

        # Parse final decision
        final_decision = self._parse_trading_decision(final_decision_raw)

        return {
            "success": True,
            "asset_name": results["asset_name"],
            "timeframe": results["timeframe"],
            "data_length": results["data_length"],
            "technical_indicators": technical_indicators,
            "pattern_analysis": pattern_analysis,
            "trend_analysis": trend_analysis,
            "pattern_chart": pattern_chart,
            "trend_chart": trend_chart,
            "pattern_image_filename": pattern_image_filename,
            "trend_image_filename": trend_image_filename,
            "final_decision": final_decision,
        }

    def _format_report_dict(self, data: Dict[str, Any]) -> str:
        """Format a report dictionary into readable text."""
        lines = []

        # Define readable keys and their order
        key_labels = {
            "reasoning": "Analysis",
            "rsi": "RSI",
            "rsi_level": "RSI Level",
            "macd": "MACD",
            "macd_signal": "MACD Signal",
            "macd_histogram": "MACD Histogram",
            "roc": "Rate of Change",
            "stochastic": "Stochastic",
            "willr": "Williams %R",
            "trend_direction": "Trend Direction",
            "confidence": "Confidence",
            "patterns_detected": "Patterns Detected",
            "primary_pattern": "Primary Pattern",
            "breakout_probability": "Breakout Probability",
            "support_level": "Support Level",
            "resistance_level": "Resistance Level",
            "trend_strength": "Trend Strength",
        }

        # Format each key-value pair
        for key, label in key_labels.items():
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                elif isinstance(value, float):
                    value = f"{value:.2f}"
                lines.append(f"{label}: {value}")

        return "\n".join(lines)

    def _parse_trading_decision(self, decision_data) -> Dict[str, Any]:
        """Parse trading decision from Pydantic model or string."""
        if hasattr(decision_data, "model_dump"):
            # It's a TradingDecision Pydantic model
            data = decision_data.model_dump()
            return {
                "decision": data.get("decision", "N/A"),
                "confidence": data.get("confidence", 0.0),
                "risk_level": data.get("risk_level", "N/A"),
                "reasoning": data.get("reasoning", "N/A"),
                "entry_price": data.get("entry_price"),
                "stop_loss": data.get("stop_loss"),
                "take_profit": data.get("take_profit"),
            }
        elif isinstance(decision_data, str) and decision_data:
            # It's a string, try to parse as JSON
            try:
                start = decision_data.find("{")
                end = decision_data.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = decision_data[start:end]
                    decision_obj = json.loads(json_str)
                    return {
                        "decision": decision_obj.get("decision", "N/A"),
                        "confidence": decision_obj.get("confidence", 0.0),
                        "risk_level": decision_obj.get("risk_level", "N/A"),
                        "reasoning": decision_obj.get("reasoning", "N/A"),
                        "entry_price": decision_obj.get("entry_price"),
                        "stop_loss": decision_obj.get("stop_loss"),
                        "take_profit": decision_obj.get("take_profit"),
                    }
            except (json.JSONDecodeError, AttributeError):
                pass
            return {"raw": decision_data}
        else:
            return {"raw": "No decision data"}

    def get_timeframe_date_limits(self, timeframe: str) -> Dict[str, Any]:
        """Get valid date range limits for a given timeframe."""
        limits = {
            "1m": {"max_days": 7, "description": "1 minute data: max 7 days"},
            "2m": {"max_days": 60, "description": "2 minute data: max 60 days"},
            "5m": {"max_days": 60, "description": "5 minute data: max 60 days"},
            "15m": {"max_days": 60, "description": "15 minute data: max 60 days"},
            "30m": {"max_days": 60, "description": "30 minute data: max 60 days"},
            "60m": {"max_days": 730, "description": "1 hour data: max 730 days"},
            "90m": {"max_days": 60, "description": "90 minute data: max 60 days"},
            "1h": {"max_days": 730, "description": "1 hour data: max 730 days"},
            "4h": {"max_days": 730, "description": "4 hour data: max 730 days"},
            "1d": {"max_days": 730, "description": "1 day data: max 730 days"},
            "5d": {"max_days": 60, "description": "5 day data: max 60 days"},
            "1w": {"max_days": 730, "description": "1 week data: max 730 days"},
            "1wk": {"max_days": 730, "description": "1 week data: max 730 days"},
            "1mo": {"max_days": 730, "description": "1 month data: max 730 days"},
            "3mo": {"max_days": 730, "description": "3 month data: max 730 days"},
        }

        return limits.get(
            timeframe, {"max_days": 730, "description": "Default: max 730 days"}
        )

    def validate_date_range(
        self,
        start_date: str,
        end_date: str,
        timeframe: str,
        start_time: str = "00:00",
        end_time: str = "23:59",
    ) -> Dict[str, Any]:
        """Validate date and time range for the given timeframe."""
        try:
            # Create datetime objects with time
            start_datetime_str = f"{start_date} {start_time}"
            end_datetime_str = f"{end_date} {end_time}"

            start = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            end = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")

            if start >= end:
                return {
                    "valid": False,
                    "error": "Start date/time must be before end date/time",
                }

            # Get timeframe limits
            limits = self.get_timeframe_date_limits(timeframe)
            max_days = limits["max_days"]

            # Calculate time difference in days (including fractional days)
            time_diff = end - start
            days_diff = time_diff.total_seconds() / (24 * 3600)  # Convert to days

            if days_diff > max_days:
                return {
                    "valid": False,
                    "error": f"Time range too large. {limits['description']}. Please select a smaller range.",
                    "max_days": max_days,
                    "current_days": round(days_diff, 2),
                }

            return {"valid": True, "days": round(days_diff, 2)}

        except ValueError as e:
            return {"valid": False, "error": f"Invalid date/time format: {str(e)}"}

    def validate_api_key(self, provider: str = None) -> Dict[str, Any]:
        """Validate the current API key by making a simple test call."""
        try:
            # Get provider from config module if not provided
            if provider is None:
                provider = settings.AGENT_LLM_PROVIDER

            if provider == "openai":
                from openai import OpenAI

                client = OpenAI()

                # Make a simple test call
                _ = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )

                provider_name = "OpenAI"
            elif provider == "anthropic":
                from anthropic import Anthropic

                api_key = settings.ANTHROPIC_API_KEY
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Invalid API Key: The Anthropic API key is not set. Please update it in the Settings section.",
                    }

                client = Anthropic(api_key=api_key)

                # Make a simple test call
                _ = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}],
                )

                provider_name = "Anthropic"
            else:  # qwen
                from langchain_qwq import ChatQwen

                api_key = settings.DASHSCOPE_API_KEY
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Invalid API Key: The Qwen API key is not set. Please update it in the Settings section.",
                    }

                # Make a simple test call using LangChain
                llm = ChatQwen(model="qwen-flash", api_key=api_key)
                _ = llm.invoke([("user", "Hello")])

                provider_name = "Qwen"
            return {"valid": True, "message": f"{provider_name} API key is valid"}

        except Exception as e:
            error_msg = str(e)

            # Determine provider name for error messages
            if provider is None:
                provider = settings.AGENT_LLM_PROVIDER
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            else:
                provider_name = "Qwen"

            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "valid": False,
                    "error": f"❌ Invalid API Key: The {provider_name} API key is invalid or has expired. Please update it in the Settings section.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "valid": False,
                    "error": f"⚠️ Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"💳 Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"🌐 Network Error: Unable to connect to {provider_name} servers. Please check your internet connection.",
                }
            else:
                return {"valid": False, "error": f"❌ API Key Error: {error_msg}"}

    def load_custom_assets(self) -> list:
        """Load custom assets from persistent JSON file."""
        try:
            if self.custom_assets_file.exists():
                with open(self.custom_assets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            return []
        except Exception as e:
            print(f"Error loading custom assets: {e}")
            return []

    def save_custom_asset(self, symbol: str) -> bool:
        """Save a custom asset symbol persistently (avoid duplicates)."""
        try:
            symbol = symbol.strip()
            if not symbol:
                return False
            if symbol in self.custom_assets:
                return True  # already present
            self.custom_assets.append(symbol)
            # write to file
            with open(self.custom_assets_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_assets, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving custom asset '{symbol}': {e}")
            return False


# Initialize the analyzer
analyzer = WebTradingAnalyzer()


@app.route("/")
def index():
    """Main landing page - redirect to demo."""
    return render_template("demo_new.html")


@app.route("/demo")
def demo():
    """Demo page with new interface."""
    return render_template("demo_new.html")


@app.route("/output")
def output():
    """Output page with analysis results."""
    # Get results from session or query parameters
    results = request.args.get("results")
    if results:
        try:
            # Handle URL-encoded results
            results = urllib.parse.unquote(results)
            results_data = json.loads(results)
            return render_template("output.html", results=results_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing results: {e}")
            # Fall back to default results

    # Default results if none provided
    default_results = {
        "asset_name": "BTC",
        "timeframe": "1h",
        "data_length": 1247,
        "technical_indicators": "RSI (14): 65.4 - Neutral to bullish momentum\nMACD: Bullish crossover with increasing histogram\nMoving Averages: Price above 50-day and 200-day MA\nBollinger Bands: Price in upper band, showing strength\nVolume: Above average volume supporting price action",
        "pattern_analysis": "Bull Flag Pattern: Consolidation after strong upward move\nGolden Cross: 50-day MA crossing above 200-day MA\nHigher Highs & Higher Lows: Uptrend confirmation\nVolume Pattern: Increasing volume on price advances",
        "trend_analysis": "Primary Trend: Bullish (Long-term)\nSecondary Trend: Bullish (Medium-term)\nShort-term Trend: Consolidating with bullish bias\nADX: 28.5 - Moderate trend strength\nPrice Action: Higher highs and higher lows maintained\nMomentum: Positive divergence on RSI",
        "pattern_chart": "",
        "trend_chart": "",
        "pattern_image_filename": "",
        "trend_image_filename": "",
        "final_decision": {
            "decision": "LONG",
            "risk_reward_ratio": "1:2.5",
            "forecast_horizon": "24-48 hours",
            "justification": "Based on comprehensive analysis of technical indicators, pattern recognition, and trend analysis, the system recommends a LONG position on BTC. The analysis shows strong bullish momentum with key support levels holding, and multiple technical indicators confirming upward movement.",
        },
    }

    return render_template("output.html", results=default_results)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        data_source = data.get("data_source")
        asset = data.get("asset")
        timeframe = data.get("timeframe")
        redirect_to_output = data.get("redirect_to_output", False)

        if data_source != "live":
            return jsonify({"error": "Only live Yahoo Finance data is supported."})

        # Live Yahoo Finance data only
        start_date = data.get("start_date")
        start_time = data.get("start_time", "00:00")
        end_date = data.get("end_date")
        end_time = data.get("end_time", "23:59")
        use_current_time = data.get("use_current_time", False)

        # Create datetime objects for validation
        if start_date:
            start_datetime_str = f"{start_date} {start_time}"
            try:
                start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            except ValueError:
                return jsonify({"error": "Invalid start date/time format."})

            if start_dt > datetime.now():
                return jsonify({"error": "Start date/time cannot be in the future."})

        if end_date:
            if use_current_time:
                end_dt = datetime.now()
            else:
                end_datetime_str = f"{end_date} {end_time}"
                try:
                    end_dt = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    return jsonify({"error": "Invalid end date/time format."})

                if end_dt > datetime.now():
                    return jsonify({"error": "End date/time cannot be in the future."})

            if start_date and start_dt and end_dt and end_dt < start_dt:
                return jsonify(
                    {"error": "End date/time cannot be earlier than start date/time."}
                )

        # Fetch data with datetime objects
        df = analyzer.fetch_yfinance_data_with_datetime(
            asset, timeframe, start_dt, end_dt
        )
        if df.empty:
            return jsonify({"error": "No data available for the specified parameters"})

        display_name = analyzer.asset_mapping.get(asset, asset)
        if display_name is None:
            display_name = asset
        results = analyzer.run_analysis(df, display_name, timeframe)
        formatted_results = analyzer.extract_analysis_results(results)

        # If redirect is requested, return redirect URL with results
        if redirect_to_output:
            if formatted_results.get("success", False):
                # Create a version without base64 images for URL encoding
                # Base64 images are too large for URL parameters
                url_safe_results = formatted_results.copy()
                url_safe_results["pattern_chart"] = ""  # Remove base64 data
                url_safe_results["trend_chart"] = ""  # Remove base64 data

                # Encode results for URL
                results_json = json.dumps(url_safe_results)
                encoded_results = urllib.parse.quote(results_json)
                redirect_url = f"/output?results={encoded_results}"

                # Store full results (with images) in session or temporary storage
                # For now, we'll pass them back in the response for the frontend to handle
                return jsonify(
                    {
                        "redirect": redirect_url,
                        "full_results": formatted_results,  # Include images in response body
                    }
                )
            else:
                return jsonify(
                    {"error": formatted_results.get("error", "Analysis failed")}
                )

        return jsonify(formatted_results)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/files/<asset>/<timeframe>")
def get_files(asset, timeframe):
    """API endpoint to get available files for an asset/timeframe."""
    try:
        files = analyzer.get_available_files(asset, timeframe)
        file_list = []

        for i, file_path in enumerate(files):
            match = re.search(r"_(\d+)\.csv$", file_path.name)
            file_number = match.group(1) if match else "N/A"
            file_list.append(
                {"index": i, "number": file_number, "name": file_path.name}
            )

        return jsonify({"files": file_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/save-custom-asset", methods=["POST"])
def save_custom_asset():
    """Save a custom asset symbol server-side for persistence."""
    try:
        data = request.get_json()
        symbol = (data.get("symbol") or "").strip()
        if not symbol:
            return jsonify({"success": False, "error": "Symbol required"}), 400

        ok = analyzer.save_custom_asset(symbol)
        if not ok:
            return jsonify({"success": False, "error": "Failed to save symbol"}), 500

        return jsonify({"success": True, "symbol": symbol})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/custom-assets", methods=["GET"])
def custom_assets():
    """Return server-persisted custom assets."""
    try:
        return jsonify({"custom_assets": analyzer.custom_assets or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assets")
def get_assets():
    """API endpoint to get available assets."""
    try:
        assets = analyzer.get_available_assets()
        asset_list = []

        for asset in assets:
            asset_list.append(
                {"code": asset, "name": analyzer.asset_mapping.get(asset, asset)}
            )

        # Include server-persisted custom assets at the end
        for custom in analyzer.custom_assets:
            asset_list.append({"code": custom, "name": custom})

        return jsonify({"assets": asset_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/timeframe-limits/<timeframe>")
def get_timeframe_limits(timeframe):
    """API endpoint to get date range limits for a timeframe."""
    try:
        limits = analyzer.get_timeframe_date_limits(timeframe)
        return jsonify(limits)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-date-range", methods=["POST"])
def validate_date_range():
    """API endpoint to validate date and time range for a timeframe."""
    try:
        data = request.get_json()
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        timeframe = data.get("timeframe")
        start_time = data.get("start_time", "00:00")
        end_time = data.get("end_time", "23:59")

        if not all([start_date, end_date, timeframe]):
            return jsonify({"error": "Missing required parameters"})

        validation = analyzer.validate_date_range(
            start_date, end_date, timeframe, start_time, end_time
        )
        return jsonify(validation)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/update-provider", methods=["POST"])
def update_provider():
    """API endpoint to update LLM provider."""
    try:
        data = request.get_json()
        provider = data.get("provider", "openai")

        if provider not in ["openai", "anthropic", "qwen"]:
            return jsonify(
                {"error": "Provider must be 'openai', 'anthropic', or 'qwen'"}
            )

        print(f"Updating provider to: {provider}")

        # Update config module
        settings.AGENT_LLM_PROVIDER = provider
        settings.GRAPH_LLM_PROVIDER = provider

        # Update model names if switching providers
        from quantagent.default_config import DEFAULT_MODELS

        if provider == "anthropic":
            # Set default Claude models if not already set to Anthropic models
            if not settings.AGENT_LLM_MODEL.startswith("claude"):
                settings.AGENT_LLM_MODEL = DEFAULT_MODELS["anthropic"]["agent"]
            if not settings.GRAPH_LLM_MODEL.startswith("claude"):
                settings.GRAPH_LLM_MODEL = DEFAULT_MODELS["anthropic"]["graph"]
        elif provider == "qwen":
            # Set default Qwen models if not already set to Qwen models
            if not settings.AGENT_LLM_MODEL.startswith("qwen"):
                settings.AGENT_LLM_MODEL = DEFAULT_MODELS["qwen"]["agent"]
            if not settings.GRAPH_LLM_MODEL.startswith("qwen"):
                settings.GRAPH_LLM_MODEL = DEFAULT_MODELS["qwen"]["graph"]
        else:
            # Set default OpenAI models if not already set to OpenAI models
            if settings.AGENT_LLM_MODEL.startswith(("claude", "qwen")):
                settings.AGENT_LLM_MODEL = DEFAULT_MODELS["openai"]["agent"]
            if settings.GRAPH_LLM_MODEL.startswith(("claude", "qwen")):
                settings.GRAPH_LLM_MODEL = DEFAULT_MODELS["openai"]["graph"]

        # Persist to .env file
        settings.update_env_file("AGENT_LLM_PROVIDER", provider)
        settings.update_env_file("GRAPH_LLM_PROVIDER", provider)
        settings.update_env_file("AGENT_LLM_MODEL", settings.AGENT_LLM_MODEL)
        settings.update_env_file("GRAPH_LLM_MODEL", settings.GRAPH_LLM_MODEL)

        # Refresh the trading graph with new provider
        analyzer.trading_graph.refresh_llms()

        print(f"Provider updated to {provider} successfully")
        print(f"graph_llm_model updated to {settings.GRAPH_LLM_MODEL} successfully")
        print(f"agent_llm_model updated to {settings.AGENT_LLM_MODEL} successfully")
        return jsonify({"success": True, "message": f"Provider updated to {provider}"})

    except Exception as e:
        print(f"Error in update_provider: {str(e)}")
        return jsonify({"error": str(e)})


@app.route("/api/update-api-key", methods=["POST"])
def update_api_key():
    """API endpoint to update API key and persist to .env."""
    try:
        data = request.get_json()
        new_api_key = data.get("api_key")
        provider = data.get("provider", "openai")

        if not new_api_key:
            return jsonify({"error": "API key is required"})

        if provider not in ["openai", "anthropic", "qwen"]:
            return jsonify(
                {"error": "Provider must be 'openai', 'anthropic', or 'qwen'"}
            )

        print(f"Updating {provider} API key to: {new_api_key[:8]}...{new_api_key[-4:]}")

        # Update API key (persists to .env and updates runtime)
        analyzer.trading_graph.update_api_key(new_api_key, provider=provider)

        print(f"{provider} API key updated successfully and persisted to .env")
        return jsonify(
            {
                "success": True,
                "message": f"{provider.capitalize()} API key updated and persisted to .env",
            }
        )

    except Exception as e:
        print(f"Error in update_api_key: {str(e)}")
        return jsonify({"error": str(e)})


@app.route("/api/get-api-key-status")
def get_api_key_status():
    """API endpoint to check if API key is set."""
    try:
        provider = request.args.get("provider", "openai")

        # Get API key from config module
        if provider == "openai":
            api_key = settings.OPENAI_API_KEY
        elif provider == "anthropic":
            api_key = settings.ANTHROPIC_API_KEY
        elif provider == "qwen":
            api_key = settings.DASHSCOPE_API_KEY
        else:
            return jsonify({"has_key": False})

        if api_key:
            # Return masked version for security
            masked_key = (
                api_key[:3] + "..." + api_key[-3:] if len(api_key) > 12 else "***"
            )
            return jsonify({"has_key": True, "masked_key": masked_key})
        else:
            return jsonify({"has_key": False})
    except Exception as e:
        print(f"Error in get_api_key_status: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "has_key": False})


@app.route("/api/images/<image_type>")
def get_image(image_type):
    """API endpoint to serve generated images."""
    try:
        if image_type == "pattern":
            image_path = "kline_chart.png"
        elif image_type == "trend":
            image_path = "trend_graph.png"
        elif image_type == "pattern_chart":
            image_path = "pattern_chart.png"
        elif image_type == "trend_chart":
            image_path = "trend_chart.png"
        else:
            return jsonify({"error": "Invalid image type"})

        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"})

        return send_file(image_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-api-key", methods=["POST"])
def validate_api_key_endpoint():
    """API endpoint to validate the current API key."""
    try:
        data = request.get_json() or {}
        provider = data.get("provider") or settings.AGENT_LLM_PROVIDER
        validation = analyzer.validate_api_key(provider=provider)
        return jsonify(validation)
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve static assets from the assets folder."""
    try:
        return send_file(f"apps/flask/assets/{filename}")
    except FileNotFoundError:
        return jsonify({"error": "Asset not found"}), 404


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    templates_dir = Path("apps/flask/templates")
    templates_dir.mkdir(exist_ok=True)

    # Create static directory if it doesn't exist
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    app.run(debug=True, host="127.0.0.1", port=5000)
