from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ProviderCapability:
    provider: str
    supports_vision: bool
    supports_structured_output: bool
    cost_tier: str
    default_reasoning_model: str
    default_lite_model: str
    default_vision_model: str
    capability_tags: List[str] = field(default_factory=list)


PROVIDER_REGISTRY: Dict[str, ProviderCapability] = {
    "anthropic": ProviderCapability(
        provider="anthropic",
        supports_vision=True,
        supports_structured_output=True,
        cost_tier="expensive",
        default_reasoning_model="claude-haiku-4-5-20251001",
        default_lite_model="claude-haiku-4-5-20251001",
        default_vision_model="claude-haiku-4-5-20251001",
        capability_tags=["reasoning", "vision"],
    ),
    "azure": ProviderCapability(
        provider="azure",
        supports_vision=True,
        supports_structured_output=True,
        cost_tier="mid",
        default_reasoning_model="",
        default_lite_model="",
        default_vision_model="",
        capability_tags=["reasoning", "vision", "enterprise"],
    ),
    "openai": ProviderCapability(
        provider="openai",
        supports_vision=True,
        supports_structured_output=True,
        cost_tier="mid",
        default_reasoning_model="gpt-4o",
        default_lite_model="gpt-4o-mini",
        default_vision_model="gpt-4o",
        capability_tags=["reasoning", "cheap", "vision"],
    ),
    "qwen": ProviderCapability(
        provider="qwen",
        supports_vision=True,
        supports_structured_output=True,
        cost_tier="mid",
        default_reasoning_model="qwen3-max",
        default_lite_model="qwen3-max",
        default_vision_model="qwen3-vl-plus",
        capability_tags=["reasoning", "vision"],
    ),
}


def supported_providers() -> List[str]:
    return sorted(PROVIDER_REGISTRY)


def get_capability(provider: str) -> ProviderCapability:
    try:
        return PROVIDER_REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc
