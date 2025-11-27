"""Configuration Manager for persisting and loading trading profiles."""

from typing import Dict, Optional, List
from sqlalchemy.orm import Session

from quantagent.models import StrategyConfig
from quantagent.database import SessionLocal


class ConfigManager:
    """Manages persisted configuration profiles for Portfolio and Risk Management.

    Profiles allow users to save named configurations that can be:
    - Reused across multiple backtests/paper trading runs
    - Overridden hierarchically (default → sector → symbol)
    - Snapshots in database for reproducibility
    """

    def __init__(self, db: Optional[Session] = None):
        """Initialize config manager.

        Args:
            db: SQLAlchemy session (uses SessionLocal if not provided)
        """
        self.db = db or SessionLocal()

    def save_profile(
        self, name: str, kind: str, config: Dict, version: int = 1
    ) -> StrategyConfig:
        """Save or update a configuration profile.

        Args:
            name: Unique profile name (e.g., "moderate_equities")
            kind: Profile type ("portfolio", "risk", or "combined")
            config: Configuration dict
            version: Optional version number for tracking changes

        Returns:
            StrategyConfig object (persisted)
        """
        existing = self.db.query(StrategyConfig).filter(
            StrategyConfig.name == name
        ).first()

        if existing:
            # Update existing
            existing.kind = kind
            existing.json_config = config
            existing.version = version
            self.db.commit()
            return existing
        else:
            # Create new
            db_config = StrategyConfig(
                name=name,
                kind=kind,
                json_config=config,
                version=version,
            )
            self.db.add(db_config)
            self.db.commit()
            self.db.refresh(db_config)
            return db_config

    def load_profile(self, name: str) -> Optional[Dict]:
        """Load a configuration profile by name.

        Args:
            name: Profile name to load

        Returns:
            Configuration dict or None if not found
        """
        db_config = self.db.query(StrategyConfig).filter(
            StrategyConfig.name == name
        ).first()

        return db_config.json_config if db_config else None

    def get_profile(self, name: str) -> Optional[StrategyConfig]:
        """Get full StrategyConfig object (including metadata).

        Args:
            name: Profile name

        Returns:
            StrategyConfig object or None if not found
        """
        return self.db.query(StrategyConfig).filter(
            StrategyConfig.name == name
        ).first()

    def list_profiles(self, kind: Optional[str] = None) -> List[Dict]:
        """List all saved profiles, optionally filtered by kind.

        Args:
            kind: Optional filter ("portfolio", "risk", "combined")

        Returns:
            List of profile dicts with metadata
        """
        query = self.db.query(StrategyConfig)

        if kind:
            query = query.filter(StrategyConfig.kind == kind)

        profiles = query.all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "kind": p.kind,
                "version": p.version,
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat(),
            }
            for p in profiles
        ]

    def delete_profile(self, name: str) -> bool:
        """Delete a profile.

        Args:
            name: Profile name to delete

        Returns:
            True if deleted, False if not found
        """
        result = self.db.query(StrategyConfig).filter(
            StrategyConfig.name == name
        ).delete()

        self.db.commit()
        return result > 0

    def resolve_config(
        self,
        base_profile: str,
        overrides: Optional[Dict] = None,
    ) -> Dict:
        """Resolve configuration with hierarchical overrides.

        This allows loading a base profile and applying overrides,
        creating a final resolved config snapshot for reproducibility.

        Args:
            base_profile: Name of base profile to load
            overrides: Optional overrides dict to merge

        Returns:
            Resolved configuration dict
        """
        base_config = self.load_profile(base_profile)

        if not base_config:
            raise ValueError(f"Profile '{base_profile}' not found")

        resolved = base_config.copy()

        if overrides:
            resolved.update(overrides)

        return resolved

    def create_snapshot(
        self, base_profile: str, overrides: Optional[Dict] = None
    ) -> Dict:
        """Create an immutable config snapshot for reproducibility.

        Used when starting a backtest/run to capture exact configuration
        at that moment, allowing later replays with the same settings.

        Args:
            base_profile: Name of base profile
            overrides: Optional overrides

        Returns:
            Snapshot dict containing resolved config + metadata
        """
        resolved_config = self.resolve_config(base_profile, overrides)

        snapshot = {
            "base_profile": base_profile,
            "resolved_config": resolved_config,
            "overrides": overrides or {},
            "timestamp": None,  # Will be set by caller to execution time
        }

        return snapshot

    def close(self) -> None:
        """Close database session."""
        if self.db:
            self.db.close()
