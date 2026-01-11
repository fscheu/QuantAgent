"""
Default configuration values for LLM models.
API keys and provider selection should be set via environment variables (.env file).
"""

# Default model configuration by provider
DEFAULT_MODELS = {
    "openai": {"agent": "gpt-4o-mini", "graph": "gpt-4o"},
    "anthropic": {
        "agent": "claude-haiku-4-5-20251001",
        "graph": "claude-haiku-4-5-20251001",
    },
    "qwen": {"agent": "qwen3-max", "graph": "qwen3-vl-plus"},
}

# Default temperature (professional, deterministic outputs)
DEFAULT_TEMPERATURE = 0.1

# Retry configuration for LLM API calls
RETRY_CONFIG = {
    "max_retries": 5,
    "base_wait": 2.0,
    "max_wait": 60.0,
    "exponential_base": 2,
    "jitter": True,
    "jitter_factor": 0.5,
}

# Market hours filtering for backtests
MARKET_HOURS_CONFIG = {
    "enabled": True,
    "fallback_to_24_7": True,
    "include_extended_hours": False,
}
