"""
Centralized configuration management.
Loads environment variables from .env file and provides typed access to all settings.
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env files (idempotent - only loads once)
# Load .env first (shared defaults)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Load .env.local second (worktree-specific overrides)
# This allows each worktree to have its own database without changing .env
env_local_path = Path(__file__).parent.parent / ".env.local"
if env_local_path.exists():
    load_dotenv(dotenv_path=env_local_path, override=True)

# Database Configuration
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# LLM API Keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")  # Qwen

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# LLM Provider Configuration
AGENT_LLM_PROVIDER: str = os.getenv("AGENT_LLM_PROVIDER", "openai")
GRAPH_LLM_PROVIDER: str = os.getenv("GRAPH_LLM_PROVIDER", "openai")


# LLM Model Configuration (defaults based on provider)
def get_default_model(provider: str, is_agent: bool = True) -> str:
    """Get default model based on provider."""
    defaults = {
        "openai": {"agent": "gpt-4o-mini", "graph": "gpt-4o"},
        "anthropic": {
            "agent": "claude-haiku-4-5-20251001",
            "graph": "claude-haiku-4-5-20251001",
        },
        "qwen": {"agent": "qwen3-max", "graph": "qwen3-vl-plus"},
        "azure": {"agent": "", "graph": ""},
    }
    model_type = "agent" if is_agent else "graph"
    return defaults.get(provider, defaults["openai"])[model_type]


AGENT_LLM_MODEL: str = os.getenv(
    "AGENT_LLM_MODEL", get_default_model(AGENT_LLM_PROVIDER, True)
)
GRAPH_LLM_MODEL: str = os.getenv(
    "GRAPH_LLM_MODEL", get_default_model(GRAPH_LLM_PROVIDER, False)
)

# LLM Temperature Configuration
AGENT_LLM_TEMPERATURE: float = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.1"))
GRAPH_LLM_TEMPERATURE: float = float(os.getenv("GRAPH_LLM_TEMPERATURE", "0.1"))

def require(name: str) -> str:
    """Return the value of a setting, raising ValueError only when actually needed (lazy validation).

    Use this instead of accessing module-level variables directly when the value
    is required at runtime (e.g., creating an API client or DB connection).
    This allows modules to be imported without blowing up in environments that
    lack certain config (CI, tests, etc.).
    """
    value = globals().get(name, "") or os.getenv(name, "")
    if not value:
        raise ValueError(
            f"{name} not configured. "
            "Set it in your .env file or via the web interface."
        )
    return value


# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_CONSOLE: bool = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
LOG_TO_DB: bool = os.getenv("LOG_TO_DB", "true").lower() == "true"


DEFAULT_SCHEDULER_ASSETS = ["BTC", "SPX"]


def _parse_bool_env(var_name: str, default: bool) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_env(var_name: str, default: float) -> float:
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_assets_env(value: Optional[str]) -> List[str]:
    if not value:
        return DEFAULT_SCHEDULER_ASSETS.copy()
    assets = [asset.strip().upper() for asset in value.split(",") if asset.strip()]
    return assets or DEFAULT_SCHEDULER_ASSETS.copy()


@dataclass
class SchedulerSettings:
    enabled: bool = False
    interval_hours: float = 1.0
    assets: List[str] = field(default_factory=lambda: DEFAULT_SCHEDULER_ASSETS.copy())
    environment: str = "paper"
    timeframe: str = "1h"
    lookback_hours: float = 168.0

    def __post_init__(self) -> None:
        self.interval_hours = float(self.interval_hours)
        self.lookback_hours = float(self.lookback_hours)
        if self.interval_hours <= 0:
            raise ValueError("interval_hours must be > 0")
        if self.lookback_hours <= 0:
            raise ValueError("lookback_hours must be > 0")
        normalized_assets = [
            asset.strip().upper() for asset in self.assets if asset.strip()
        ]
        if not normalized_assets:
            raise ValueError("assets list cannot be empty")
        self.assets = normalized_assets
        env_value = self.environment.strip().lower()
        if env_value not in {"backtest", "paper", "prod"}:
            raise ValueError("environment must be one of: backtest, paper, prod")
        self.environment = env_value
        timeframe_value = self.timeframe.strip().lower()
        if not timeframe_value:
            raise ValueError("timeframe cannot be empty")
        self.timeframe = timeframe_value

    @classmethod
    def from_env(cls) -> "SchedulerSettings":
        return cls(
            enabled=_parse_bool_env("TRADING_SCHEDULER_ENABLED", False),
            interval_hours=_parse_float_env("TRADING_SCHEDULER_INTERVAL_HOURS", 1.0),
            assets=_parse_assets_env(os.getenv("TRADING_SCHEDULER_ASSETS")),
            environment=os.getenv("TRADING_SCHEDULER_ENVIRONMENT", "paper"),
            timeframe=os.getenv("TRADING_SCHEDULER_TIMEFRAME", "1h"),
            lookback_hours=_parse_float_env("TRADING_SCHEDULER_LOOKBACK_HOURS", 168.0),
        )

    def with_overrides(self, **overrides) -> "SchedulerSettings":
        return replace(self, **overrides)


# Scheduler configuration instance accessible as settings.scheduler
scheduler = SchedulerSettings.from_env()


def update_env_file(key: str, value: str) -> None:
    """
    Update or add a key-value pair in the .env file.
    Used by web_interface.py to persist API key changes.
    """
    env_file = Path(__file__).parent.parent / ".env"

    # Read existing content
    if env_file.exists():
        with open(env_file, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    # Update or append the key
    key_found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_found = True
            break

    if not key_found:
        lines.append(f"{key}={value}\n")

    # Write back
    with open(env_file, "w") as f:
        f.writelines(lines)

    # Update os.environ for runtime
    os.environ[key] = value
