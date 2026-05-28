from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from quantagent.llm.registry import get_capability


@dataclass(eq=True)
class ProviderRoleConfig:
    provider: str
    model_name: str
    temperature: float = 0.1
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    capability_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "capability_tags": list(self.capability_tags),
        }
        if self.timeout_seconds is not None:
            data["timeout_seconds"] = self.timeout_seconds
        if self.max_retries is not None:
            data["max_retries"] = self.max_retries
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderRoleConfig":
        return cls(
            provider=str(data["provider"]),
            model_name=str(data["model_name"]),
            temperature=float(data.get("temperature", 0.1)),
            timeout_seconds=data.get("timeout_seconds"),
            max_retries=data.get("max_retries"),
            capability_tags=list(data.get("capability_tags") or []),
        )

    def validate_against_registry(self) -> None:
        get_capability(self.provider)
