from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quantagent.llm.roles import ProviderRoleConfig

ROLE_DEEP_REASONING = "deep_reasoning"
ROLE_LITE = "lite"
ROLE_IMAGE = "image"


class ProviderRoleNotConfiguredError(ValueError):
    """Raised when a required provider role cannot be resolved."""


@dataclass(eq=True)
class ProviderRoutingPolicy:
    deep_reasoning: Optional[ProviderRoleConfig] = None
    lite: Optional[ProviderRoleConfig] = None
    image: Optional[ProviderRoleConfig] = None

    def resolve(self, role: str) -> ProviderRoleConfig:
        resolved = self.resolve_or_none(role)
        if resolved is None:
            raise ProviderRoleNotConfiguredError(
                f"No provider role configured for '{role}'"
            )
        return resolved

    def resolve_or_none(self, role: str) -> Optional[ProviderRoleConfig]:
        if role == ROLE_DEEP_REASONING:
            return self.deep_reasoning
        if role == ROLE_LITE:
            return self.lite or self.deep_reasoning
        if role == ROLE_IMAGE:
            return self.image or self.deep_reasoning or self.lite
        raise ValueError(f"Unsupported provider role: {role}")

    def to_dict(self) -> dict:
        return {
            ROLE_DEEP_REASONING: self.deep_reasoning.to_dict() if self.deep_reasoning else None,
            ROLE_LITE: self.lite.to_dict() if self.lite else None,
            ROLE_IMAGE: self.image.to_dict() if self.image else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderRoutingPolicy":
        return cls(
            deep_reasoning=_role_from_dict(data.get(ROLE_DEEP_REASONING)),
            lite=_role_from_dict(data.get(ROLE_LITE)),
            image=_role_from_dict(data.get(ROLE_IMAGE)),
        )

    @classmethod
    def from_legacy_settings(cls) -> "ProviderRoutingPolicy":
        from quantagent import settings

        return cls(
            deep_reasoning=ProviderRoleConfig(
                provider=settings.GRAPH_LLM_PROVIDER,
                model_name=settings.GRAPH_LLM_MODEL,
                temperature=settings.GRAPH_LLM_TEMPERATURE,
                capability_tags=["reasoning"],
            ),
            lite=ProviderRoleConfig(
                provider=settings.AGENT_LLM_PROVIDER,
                model_name=settings.AGENT_LLM_MODEL,
                temperature=settings.AGENT_LLM_TEMPERATURE,
                capability_tags=["cheap"],
            ),
            image=None,
        )


def _role_from_dict(data: Optional[dict]) -> Optional[ProviderRoleConfig]:
    if data is None:
        return None
    return ProviderRoleConfig.from_dict(data)
