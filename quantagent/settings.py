"""
Centralized configuration management.
Loads environment variables from .env file and provides typed access to all settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (idempotent - only loads once)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Database Configuration
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# LLM API Keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")  # Qwen

# LLM Provider Configuration
AGENT_LLM_PROVIDER: str = os.getenv("AGENT_LLM_PROVIDER", "openai")
GRAPH_LLM_PROVIDER: str = os.getenv("GRAPH_LLM_PROVIDER", "openai")

# LLM Model Configuration (defaults based on provider)
def get_default_model(provider: str, is_agent: bool = True) -> str:
    """Get default model based on provider."""
    defaults = {
        "openai": {"agent": "gpt-4o-mini", "graph": "gpt-4o"},
        "anthropic": {"agent": "claude-haiku-4-5-20251001", "graph": "claude-haiku-4-5-20251001"},
        "qwen": {"agent": "qwen3-max", "graph": "qwen3-vl-plus"}
    }
    model_type = "agent" if is_agent else "graph"
    return defaults.get(provider, defaults["openai"])[model_type]


AGENT_LLM_MODEL: str = os.getenv("AGENT_LLM_MODEL", get_default_model(AGENT_LLM_PROVIDER, True))
GRAPH_LLM_MODEL: str = os.getenv("GRAPH_LLM_MODEL", get_default_model(GRAPH_LLM_PROVIDER, False))

# LLM Temperature Configuration
AGENT_LLM_TEMPERATURE: float = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.1"))
GRAPH_LLM_TEMPERATURE: float = float(os.getenv("GRAPH_LLM_TEMPERATURE", "0.1"))


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
