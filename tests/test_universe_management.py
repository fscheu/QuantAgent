"""
Unit tests for Universe management in Configuration UI (QuantAgent-ia2).

Following acceptance criteria:
- Multi-select widget for symbols in Portfolio profile editor
- Supported symbols: BTC, SPX, CL, DAX, ES, NQ, QQQ, GC, VIX, DXY (from DataProvider)
- Universe saved in profile json_config
- Preview shows resolved universe before saving
- Backtest can use universe from profile when assets not specified
"""

import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.data.provider import DataProvider
from quantagent.models import BacktestRun, StrategyConfig

# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
def test_db():
    """Create in-memory SQLite database for testing."""
    from sqlalchemy import MetaData
    engine = create_engine("sqlite:///:memory:")

    # Only create the tables we need for universe tests (avoid JSONB compatibility issues)
    metadata = MetaData()

    # Create only StrategyConfig and BacktestRun tables
    StrategyConfig.__table__.create(engine, checkfirst=True)
    BacktestRun.__table__.create(engine, checkfirst=True)

    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    yield db
    db.close()


# ============================================================================
# DataProvider Fixtures
# ============================================================================


@pytest.fixture
def data_provider(test_db):
    """Create DataProvider instance with test database."""
    return DataProvider(test_db)


# ============================================================================
# Tests for SYMBOL_MAPPING and Supported Universe
# ============================================================================


class TestSupportedUniverse:
    """Test suite for supported universe symbols."""

    def test_symbol_mapping_exists(self):
        """Verify SYMBOL_MAPPING is defined in DataProvider."""
        assert hasattr(DataProvider, 'SYMBOL_MAPPING')
        assert isinstance(DataProvider.SYMBOL_MAPPING, dict)

    def test_symbol_mapping_contains_required_symbols(self):
        """Verify all required symbols are in SYMBOL_MAPPING."""
        required_symbols = {"BTC", "SPX", "CL", "DAX", "ES", "NQ", "QQQ", "GC", "VIX", "DXY"}
        actual_symbols = set(DataProvider.SYMBOL_MAPPING.keys())
        assert required_symbols.issubset(actual_symbols), \
            f"Missing symbols: {required_symbols - actual_symbols}"

    def test_symbol_mapping_values_are_valid_yfinance_symbols(self):
        """Verify all symbol mappings have valid yfinance symbol values."""
        for symbol, yf_symbol in DataProvider.SYMBOL_MAPPING.items():
            assert isinstance(yf_symbol, str), f"Symbol {symbol} maps to non-string: {yf_symbol}"
            assert len(yf_symbol) > 0, f"Symbol {symbol} maps to empty string"

    def test_required_symbols_exact_list(self):
        """Verify required symbols match exactly (at minimum)."""
        required_symbols = {"BTC", "SPX", "CL", "DAX", "ES", "NQ", "QQQ", "GC", "VIX", "DXY"}
        actual_symbols = set(DataProvider.SYMBOL_MAPPING.keys())
        # At minimum, we should have all required symbols
        assert required_symbols.issubset(actual_symbols)


# ============================================================================
# Tests for Profile Persistence with Universe
# ============================================================================


class TestProfileUniversePersistence:
    """Test suite for universe management in strategy configuration profiles."""

    def test_create_portfolio_profile_with_universe(self, test_db):
        """Test creating and persisting a portfolio profile with universe."""
        profile = StrategyConfig(
            name="test_portfolio_btc_spx",
            kind="portfolio",
            json_config={
                "universe": ["BTC", "SPX"],
                "base_position_pct": 0.05,
                "max_position_pct": 0.1,
                "max_daily_loss_pct": 0.05,
            }
        )
        test_db.add(profile)
        test_db.commit()

        # Retrieve and verify
        retrieved = test_db.query(StrategyConfig).filter_by(name="test_portfolio_btc_spx").one()
        assert retrieved.kind == "portfolio"
        assert retrieved.json_config["universe"] == ["BTC", "SPX"]
        assert retrieved.version == 1

    def test_create_multiple_universe_symbols(self, test_db):
        """Test creating profile with all 10 supported symbols."""
        all_symbols = list(DataProvider.SYMBOL_MAPPING.keys())
        profile = StrategyConfig(
            name="test_portfolio_all_symbols",
            kind="portfolio",
            json_config={
                "universe": all_symbols,
                "base_position_pct": 0.01,  # Smaller position for diversification
                "max_position_pct": 0.05,
                "max_daily_loss_pct": 0.02,
            }
        )
        test_db.add(profile)
        test_db.commit()

        retrieved = test_db.query(StrategyConfig).filter_by(name="test_portfolio_all_symbols").one()
        assert len(retrieved.json_config["universe"]) == len(all_symbols)
        assert set(retrieved.json_config["universe"]) == set(all_symbols)

    def test_create_portfolio_with_empty_universe(self, test_db):
        """Test creating profile with empty universe (valid edge case)."""
        profile = StrategyConfig(
            name="test_portfolio_empty",
            kind="portfolio",
            json_config={
                "universe": [],
                "base_position_pct": 0.05,
                "max_position_pct": 0.1,
                "max_daily_loss_pct": 0.05,
            }
        )
        test_db.add(profile)
        test_db.commit()

        retrieved = test_db.query(StrategyConfig).filter_by(name="test_portfolio_empty").one()
        assert retrieved.json_config["universe"] == []

    def test_risk_profile_does_not_have_universe(self, test_db):
        """Test risk profiles are allowed without universe field."""
        profile = StrategyConfig(
            name="test_risk_conservative",
            kind="risk",
            json_config={
                "max_daily_loss_pct": 0.02,
                "max_position_pct": 0.05,
                "stop_loss_pct": 0.03,
            }
        )
        test_db.add(profile)
        test_db.commit()

        retrieved = test_db.query(StrategyConfig).filter_by(name="test_risk_conservative").one()
        assert "universe" not in retrieved.json_config

    def test_combined_profile_can_have_universe(self, test_db):
        """Test combined profiles can include universe (for convenience)."""
        profile = StrategyConfig(
            name="test_combined_profile",
            kind="combined",
            json_config={
                "universe": ["BTC", "SPX", "CL"],
                "base_position_pct": 0.05,
                "max_position_pct": 0.1,
                "max_daily_loss_pct": 0.05,
                "stop_loss_pct": 0.03,
            }
        )
        test_db.add(profile)
        test_db.commit()

        retrieved = test_db.query(StrategyConfig).filter_by(name="test_combined_profile").one()
        assert retrieved.json_config.get("universe") == ["BTC", "SPX", "CL"]

    def test_update_profile_with_new_universe(self, test_db):
        """Test updating a profile's universe."""
        # Create initial profile
        profile = StrategyConfig(
            name="test_update_universe",
            kind="portfolio",
            json_config={"universe": ["BTC"]},
            version=1
        )
        test_db.add(profile)
        test_db.commit()

        # Update universe - must reassign the dict to trigger SQLAlchemy change detection
        retrieved = test_db.query(StrategyConfig).filter_by(name="test_update_universe").one()
        new_config = retrieved.json_config.copy()
        new_config["universe"] = ["BTC", "SPX", "CL"]
        retrieved.json_config = new_config
        retrieved.version = 2
        test_db.commit()

        # Verify update
        final = test_db.query(StrategyConfig).filter_by(name="test_update_universe").one()
        assert final.json_config["universe"] == ["BTC", "SPX", "CL"]
        assert final.version == 2

    def test_profile_list_with_universe_info(self, test_db):
        """Test querying multiple profiles with universe info."""
        profiles = [
            StrategyConfig(name="p1", kind="portfolio", json_config={"universe": ["BTC"]}),
            StrategyConfig(name="p2", kind="portfolio", json_config={"universe": ["BTC", "SPX"]}),
            StrategyConfig(name="p3", kind="risk", json_config={"max_loss": 0.05}),
        ]
        test_db.add_all(profiles)
        test_db.commit()

        # Query portfolio profiles
        portfolio_profiles = test_db.query(StrategyConfig).filter_by(kind="portfolio").all()
        assert len(portfolio_profiles) == 2
        universes = [p.json_config.get("universe", []) for p in portfolio_profiles]
        assert ["BTC"] in universes
        assert ["BTC", "SPX"] in universes


# ============================================================================
# Tests for Universe Filtering and Validation
# ============================================================================


class TestUniverseFiltering:
    """Test suite for filtering and validating universe symbols."""

    def test_filter_supported_symbols(self):
        """Test filtering of supported vs unsupported symbols."""
        supported = set(DataProvider.SYMBOL_MAPPING.keys())
        test_symbols = ["BTC", "SPX", "INVALID", "CL", "UNKNOWN"]

        filtered = [s for s in test_symbols if s in supported]
        unsupported = [s for s in test_symbols if s not in supported]

        assert "BTC" in filtered
        assert "SPX" in filtered
        assert "CL" in filtered
        assert "INVALID" in unsupported
        assert "UNKNOWN" in unsupported

    def test_universe_with_unsupported_symbols(self, test_db):
        """Test that profile can be created with unsupported symbols (for import scenarios)."""
        # This tests the ability to preserve unsupported symbols in config
        profile = StrategyConfig(
            name="test_mixed_universe",
            kind="portfolio",
            json_config={
                "universe": ["BTC", "SPX", "UNSUPPORTED", "CL"],
            }
        )
        test_db.add(profile)
        test_db.commit()

        retrieved = test_db.query(StrategyConfig).filter_by(name="test_mixed_universe").one()
        assert "UNSUPPORTED" in retrieved.json_config["universe"]

    def test_supported_symbols_constant_matches_provider(self):
        """Verify that supported symbols list is consistent with DataProvider."""
        supported_from_provider = list(DataProvider.SYMBOL_MAPPING.keys())

        # Test configuration file should reference the same symbols
        required_symbols = {"BTC", "SPX", "CL", "DAX", "ES", "NQ", "QQQ", "GC", "VIX", "DXY"}
        assert required_symbols.issubset(set(supported_from_provider))


# ============================================================================
# Tests for JSON Serialization and Preview
# ============================================================================


class TestUniverseJSONSerialization:
    """Test suite for universe JSON serialization and preview."""

    def test_universe_json_serialization(self):
        """Test that universe can be properly serialized to JSON."""
        data = {
            "universe": ["BTC", "SPX", "CL"],
            "base_position_pct": 0.05,
            "max_position_pct": 0.1,
        }

        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed["universe"] == ["BTC", "SPX", "CL"]
        assert isinstance(parsed["universe"], list)

    def test_universe_roundtrip_in_profile(self, test_db):
        """Test universe survives JSON roundtrip in profile."""
        original_universe = ["BTC", "SPX", "CL", "DAX", "ES"]

        # Save
        profile = StrategyConfig(
            name="test_roundtrip",
            kind="portfolio",
            json_config={"universe": original_universe}
        )
        test_db.add(profile)
        test_db.commit()

        # Retrieve
        retrieved = test_db.query(StrategyConfig).filter_by(name="test_roundtrip").one()
        retrieved_universe = retrieved.json_config["universe"]

        assert retrieved_universe == original_universe
        assert len(retrieved_universe) == len(original_universe)

    def test_universe_preview_dataframe_compatible(self):
        """Test that universe data is compatible with DataFrame display."""
        universe = ["BTC", "SPX", "CL", "DAX", "ES"]

        # Simulate what the UI does for preview
        preview_data = {"symbol": universe}

        # This should be valid for a pandas DataFrame
        assert isinstance(preview_data["symbol"], list)
        assert all(isinstance(s, str) for s in preview_data["symbol"])

        # Verify each symbol is a valid string
        for symbol in universe:
            assert len(symbol) > 0
            assert symbol.isalnum()  # Basic validation


# ============================================================================
# Tests for Backtest Integration with Universe
# ============================================================================


class TestBacktestUniverseIntegration:
    """Test suite for backtest using universe from profile."""

    def test_backtest_run_can_use_profile_universe(self, test_db):
        """Test that BacktestRun can be created with universe from profile."""
        # Create a profile with universe
        profile = StrategyConfig(
            name="test_bt_profile",
            kind="portfolio",
            json_config={"universe": ["BTC", "SPX", "CL"]}
        )
        test_db.add(profile)
        test_db.commit()

        # Retrieve profile
        retrieved_profile = test_db.query(StrategyConfig).filter_by(
            name="test_bt_profile"
        ).one()
        universe_from_profile = retrieved_profile.json_config["universe"]

        # Create backtest using that universe
        backtest = BacktestRun(
            name="test_backtest_with_universe",
            timeframe="1h",
            assets=universe_from_profile,  # Use universe from profile
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            config_snapshot={"profile": "test_bt_profile", "universe": universe_from_profile}
        )
        test_db.add(backtest)
        test_db.commit()

        # Verify backtest has correct assets
        retrieved_bt = test_db.query(BacktestRun).filter_by(
            name="test_backtest_with_universe"
        ).one()
        assert retrieved_bt.assets == ["BTC", "SPX", "CL"]

    def test_backtest_with_explicit_assets_overrides_universe(self, test_db):
        """Test that explicit assets in backtest override profile universe."""
        profile = StrategyConfig(
            name="test_override_profile",
            kind="portfolio",
            json_config={"universe": ["BTC", "SPX", "CL"]}
        )
        test_db.add(profile)
        test_db.commit()

        # Create backtest with different assets (explicit override)
        explicit_assets = ["BTC", "GC", "VIX"]
        backtest = BacktestRun(
            name="test_backtest_override",
            timeframe="1h",
            assets=explicit_assets,  # Different from profile
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            config_snapshot={"explicit_override": True}
        )
        test_db.add(backtest)
        test_db.commit()

        retrieved_bt = test_db.query(BacktestRun).filter_by(
            name="test_backtest_override"
        ).one()
        assert retrieved_bt.assets == ["BTC", "GC", "VIX"]

    def test_backtest_empty_assets_list(self, test_db):
        """Test that backtest can be created with empty assets list."""
        backtest = BacktestRun(
            name="test_empty_assets",
            timeframe="1h",
            assets=[],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            config_snapshot={}
        )
        test_db.add(backtest)
        test_db.commit()

        retrieved = test_db.query(BacktestRun).filter_by(name="test_empty_assets").one()
        assert retrieved.assets == []

    def test_backtest_snapshot_preserves_universe_info(self, test_db):
        """Test that config snapshot preserves universe information."""
        universe = ["BTC", "SPX", "CL", "DAX", "ES"]
        config_snapshot = {
            "profile_name": "test_profile",
            "profile_kind": "portfolio",
            "universe": universe,
            "base_position_pct": 0.05,
        }

        backtest = BacktestRun(
            name="test_snapshot_preservation",
            timeframe="1h",
            assets=universe,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            config_snapshot=config_snapshot
        )
        test_db.add(backtest)
        test_db.commit()

        retrieved = test_db.query(BacktestRun).filter_by(
            name="test_snapshot_preservation"
        ).one()
        assert retrieved.config_snapshot["universe"] == universe
        assert retrieved.config_snapshot["profile_name"] == "test_profile"


# ============================================================================
# Tests for Edge Cases and Error Scenarios
# ============================================================================


class TestUniverseEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_unicode_symbols_rejected(self):
        """Test that non-ASCII symbols are rejected."""
        supported = set(DataProvider.SYMBOL_MAPPING.keys())
        test_symbols = ["BTC", "SPX", "BTC€", "₹SPX"]  # EUR and INR symbols

        filtered = [s for s in test_symbols if s in supported]
        assert "BTC€" not in filtered
        assert "₹SPX" not in filtered

    def test_whitespace_in_symbols(self):
        """Test that symbols with whitespace are handled."""
        supported = set(DataProvider.SYMBOL_MAPPING.keys())
        test_symbols = ["BTC", " BTC", "BTC ", "BT C"]

        filtered = [s for s in test_symbols if s in supported]
        # Only exact matches should be included
        assert "BTC" in filtered
        assert " BTC" not in filtered
        assert "BTC " not in filtered
        assert "BT C" not in filtered

    def test_duplicate_symbols_in_universe(self, test_db):
        """Test handling of duplicate symbols in universe list."""
        profile = StrategyConfig(
            name="test_duplicates",
            kind="portfolio",
            json_config={"universe": ["BTC", "SPX", "BTC", "CL", "SPX"]}
        )
        test_db.add(profile)
        test_db.commit()

        retrieved = test_db.query(StrategyConfig).filter_by(name="test_duplicates").one()
        # The database should preserve the list as-is (dedup is UI responsibility)
        assert retrieved.json_config["universe"] == ["BTC", "SPX", "BTC", "CL", "SPX"]

    def test_case_sensitivity_of_symbols(self):
        """Test that symbols are case-sensitive."""
        supported = set(DataProvider.SYMBOL_MAPPING.keys())

        # All supported symbols should be uppercase
        for symbol in supported:
            assert symbol == symbol.upper()

        # Lowercase variants should not be supported
        lowercase_variants = ["btc", "spx", "cl"]
        filtered = [s for s in lowercase_variants if s in supported]
        assert len(filtered) == 0

    def test_profile_version_tracking(self, test_db):
        """Test that profile version increments on update."""
        # Create initial profile
        profile = StrategyConfig(
            name="test_versioning",
            kind="portfolio",
            json_config={"universe": ["BTC"]},
            version=1
        )
        test_db.add(profile)
        test_db.commit()

        # Update multiple times
        symbols_to_add = ["SPX", "CL", "DAX", "ES"]
        for i, symbol in enumerate(symbols_to_add, start=2):
            retrieved = test_db.query(StrategyConfig).filter_by(
                name="test_versioning"
            ).one()
            new_config = retrieved.json_config.copy()
            new_config["universe"] = new_config["universe"] + [symbol]
            retrieved.json_config = new_config
            retrieved.version = i
            test_db.commit()

        final = test_db.query(StrategyConfig).filter_by(name="test_versioning").one()
        assert final.version == 5
        assert len(final.json_config["universe"]) == 5


# ============================================================================
# Integration Tests
# ============================================================================


class TestUniverseManagementIntegration:
    """Integration tests for complete universe management workflow."""

    def test_complete_portfolio_creation_workflow(self, test_db):
        """Test complete workflow: create profile → save → retrieve → use in backtest."""
        # 1. Create portfolio profile
        profile = StrategyConfig(
            name="integration_test_portfolio",
            kind="portfolio",
            json_config={
                "universe": ["BTC", "SPX", "CL"],
                "base_position_pct": 0.05,
                "max_position_pct": 0.1,
                "max_daily_loss_pct": 0.05,
            }
        )
        test_db.add(profile)
        test_db.commit()

        # 2. Retrieve profile
        retrieved_profile = test_db.query(StrategyConfig).filter_by(
            name="integration_test_portfolio"
        ).one()
        assert retrieved_profile.kind == "portfolio"

        # 3. Extract universe
        universe = retrieved_profile.json_config["universe"]
        assert len(universe) == 3

        # 4. Create backtest with that universe
        backtest = BacktestRun(
            name="integration_test_backtest",
            timeframe="1h",
            assets=universe,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            config_snapshot={"profile_name": "integration_test_portfolio"}
        )
        test_db.add(backtest)
        test_db.commit()

        # 5. Verify complete chain
        final_bt = test_db.query(BacktestRun).filter_by(
            name="integration_test_backtest"
        ).one()
        assert final_bt.assets == ["BTC", "SPX", "CL"]
        assert final_bt.config_snapshot["profile_name"] == "integration_test_portfolio"

    def test_multiple_profiles_with_different_universes(self, test_db):
        """Test managing multiple profiles with different universes."""
        profiles_config = [
            {"name": "aggressive", "universe": ["BTC", "NQ", "ES", "VIX"]},
            {"name": "conservative", "universe": ["BTC", "SPX"]},
            {"name": "mixed", "universe": ["BTC", "SPX", "CL", "DAX", "GC"]},
        ]

        for cfg in profiles_config:
            profile = StrategyConfig(
                name=cfg["name"],
                kind="portfolio",
                json_config={"universe": cfg["universe"]}
            )
            test_db.add(profile)
        test_db.commit()

        # Verify all profiles are stored correctly
        profiles = test_db.query(StrategyConfig).filter_by(kind="portfolio").all()
        assert len(profiles) == 3

        # Check each profile
        for profile in profiles:
            if profile.name == "aggressive":
                assert len(profile.json_config["universe"]) == 4
            elif profile.name == "conservative":
                assert len(profile.json_config["universe"]) == 2
            elif profile.name == "mixed":
                assert len(profile.json_config["universe"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
