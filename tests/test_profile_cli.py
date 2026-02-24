"""
Comprehensive tests for Profile CLI feature (QuantAgent-ayy).

Tests cover all acceptance criteria (AC1-AC6) by executing real CLI commands
and verifying database state, not using tautological mocks.
"""

import json
import os

import pytest
from click.testing import CliRunner
from quantagent.cli.profile import profile_group
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.models import StrategyConfig


@pytest.fixture
def cli_runner(monkeypatch, tmp_path):
    """Provide CliRunner with isolated test database (SQLite)."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    # Initialize database - only create StrategyConfig table
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    StrategyConfig.__table__.create(engine, checkfirst=True)

    runner = CliRunner()
    return runner


def get_db_session(db_url):
    """Helper to get database session for verification."""
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


class TestAC1ListProfiles:
    """AC1: List Profiles - display table with columns, filtering, JSON output"""

    def test_list_empty_database(self, cli_runner):
        """Empty database returns 'No profiles found', exit 0"""
        result = cli_runner.invoke(profile_group, ["list"])

        assert result.exit_code == 0
        assert "No profiles found" in result.output

    def test_list_single_profile_table(self, cli_runner):
        """Single profile displays in table format with all columns"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="test-portfolio",
            kind="portfolio",
            json_config={"universe": ["BTC"], "base_position_pct": 0.05},
            version=1
        )
        session.add(profile)
        session.commit()
        session.close()

        result = cli_runner.invoke(profile_group, ["list"])

        assert result.exit_code == 0
        assert "test-portfolio" in result.output
        assert "portfolio" in result.output
        # Check for expected column headers
        assert any(col in result.output for col in ["id", "name", "kind", "version"])

    def test_list_multiple_profiles(self, cli_runner):
        """Multiple profiles all displayed"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profiles = [
            StrategyConfig(name="p1", kind="portfolio", json_config={"test": 1}),
            StrategyConfig(name="p2", kind="risk", json_config={"test": 2}),
            StrategyConfig(name="p3", kind="combined", json_config={"test": 3}),
        ]
        session.add_all(profiles)
        session.commit()
        session.close()

        result = cli_runner.invoke(profile_group, ["list"])

        assert result.exit_code == 0
        assert "p1" in result.output
        assert "p2" in result.output
        assert "p3" in result.output

    def test_list_filter_by_kind(self, cli_runner):
        """--kind portfolio filters correctly"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profiles = [
            StrategyConfig(name="p1", kind="portfolio", json_config={"test": 1}),
            StrategyConfig(name="p2", kind="risk", json_config={"test": 2}),
            StrategyConfig(name="p3", kind="portfolio", json_config={"test": 3}),
        ]
        session.add_all(profiles)
        session.commit()
        session.close()

        result = cli_runner.invoke(profile_group, ["list", "--kind", "portfolio"])

        assert result.exit_code == 0
        assert "p1" in result.output
        assert "p3" in result.output
        assert "p2" not in result.output

    def test_list_json_output(self, cli_runner):
        """--json outputs valid JSON array"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="test",
            kind="portfolio",
            json_config={"universe": ["BTC"]},
            version=1
        )
        session.add(profile)
        session.commit()
        session.close()

        result = cli_runner.invoke(profile_group, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)  # Will fail if not valid JSON
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["name"] == "test"
        assert data[0]["kind"] == "portfolio"


class TestAC2ShowProfile:
    """AC2: Show Profile - by name, by ID, not found, JSON output"""

    def test_show_by_name(self, cli_runner):
        """Show profile by name displays all fields"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="default-portfolio",
            kind="portfolio",
            json_config={"universe": ["BTC", "ETH"], "base_position_pct": 0.05},
            version=2
        )
        session.add(profile)
        session.commit()
        session.close()

        result = cli_runner.invoke(profile_group, ["show", "default-portfolio"])

        assert result.exit_code == 0
        assert "default-portfolio" in result.output
        assert "portfolio" in result.output
        assert "2" in result.output  # version

    def test_show_by_id(self, cli_runner):
        """Show profile by --id"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="test-profile",
            kind="risk",
            json_config={"max_daily_loss_pct": 0.05}
        )
        session.add(profile)
        session.commit()
        profile_id = profile.id
        session.close()

        result = cli_runner.invoke(profile_group, ["show", "--id", str(profile_id)])

        assert result.exit_code == 0
        assert "test-profile" in result.output

    def test_show_not_found(self, cli_runner):
        """Non-existent profile returns error with exit 1"""
        result = cli_runner.invoke(profile_group, ["show", "nonexistent"])

        assert result.exit_code == 1
        assert "Profile not found" in result.output or "nonexistent" in result.output

    def test_show_json_output(self, cli_runner):
        """--json outputs valid JSON object"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="test",
            kind="portfolio",
            json_config={"universe": ["BTC"]},
            version=1
        )
        session.add(profile)
        session.commit()
        session.close()

        result = cli_runner.invoke(profile_group, ["show", "test", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)  # Will fail if not valid JSON
        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["kind"] == "portfolio"


class TestAC3CreateProfile:
    """AC3: Create Profile - config flag, stdin, validation, duplicate detection"""

    def test_create_with_config_flag(self, cli_runner):
        """Create via --config flag creates in database with v1"""
        config = {
            "name": "new-portfolio",
            "kind": "portfolio",
            "json_config": {"universe": ["BTC"], "base_position_pct": 0.05}
        }

        result = cli_runner.invoke(
            profile_group,
            ["create", "--config", json.dumps(config)]
        )

        assert result.exit_code == 0
        assert "new-portfolio" in result.output

        # Verify in database
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)
        db_profile = session.query(StrategyConfig).filter_by(name="new-portfolio").first()
        session.close()
        assert db_profile is not None
        assert db_profile.kind == "portfolio"
        assert db_profile.version == 1

    def test_create_with_stdin(self, cli_runner):
        """Create via stdin"""
        config = {
            "name": "stdin-profile",
            "kind": "risk",
            "json_config": {"max_daily_loss_pct": 0.05}
        }

        result = cli_runner.invoke(
            profile_group,
            ["create"],
            input=json.dumps(config)
        )

        assert result.exit_code == 0

        # Verify in database
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)
        db_profile = session.query(StrategyConfig).filter_by(name="stdin-profile").first()
        session.close()
        assert db_profile is not None

    def test_create_duplicate_name_fails(self, cli_runner):
        """Duplicate name returns error with exit 1"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)
        profile = StrategyConfig(
            name="duplicate-test",
            kind="portfolio",
            json_config={"test": 1}
        )
        session.add(profile)
        session.commit()
        session.close()

        config = {
            "name": "duplicate-test",
            "kind": "risk",
            "json_config": {"test": 2}
        }

        result = cli_runner.invoke(
            profile_group,
            ["create", "--config", json.dumps(config)]
        )

        assert result.exit_code == 1
        assert "already exists" in result.output or "duplicate" in result.output.lower()

    def test_create_invalid_json_fails(self, cli_runner):
        """Invalid JSON returns error with exit 1"""
        result = cli_runner.invoke(
            profile_group,
            ["create", "--config", "{invalid json}"]
        )

        assert result.exit_code == 1
        assert "JSON" in result.output or "parse" in result.output.lower()

    def test_create_missing_required_fields_fails(self, cli_runner):
        """Missing required fields returns error with exit 1"""
        config = {
            "name": "incomplete",
            # Missing: kind, json_config
        }

        result = cli_runner.invoke(
            profile_group,
            ["create", "--config", json.dumps(config)]
        )

        assert result.exit_code == 1
        assert "required" in result.output.lower() or "missing" in result.output.lower()


class TestAC4UpdateProfile:
    """AC4: Update Profile - merge mode, replace mode, version tracking"""

    def test_update_merge_mode(self, cli_runner):
        """Update (default) merges json_config, increments version"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="update-test",
            kind="portfolio",
            json_config={"universe": ["BTC"], "base_position_pct": 0.05},
            version=1
        )
        session.add(profile)
        session.commit()
        session.close()

        # Update with partial config
        update_config = {
            "json_config": {"slippage_pct": 0.02}
        }

        result = cli_runner.invoke(
            profile_group,
            ["update", "update-test", "--config", json.dumps(update_config)]
        )

        assert result.exit_code == 0

        # Verify merge behavior
        session = get_db_session(db_url)
        db_profile = session.query(StrategyConfig).filter_by(name="update-test").first()
        session.close()
        assert db_profile.json_config["universe"] == ["BTC"]  # Original preserved
        assert db_profile.json_config["slippage_pct"] == 0.02  # New field added
        assert db_profile.version == 2  # Version incremented

    def test_update_replace_mode(self, cli_runner):
        """Update --replace replaces entire json_config, increments version"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="replace-test",
            kind="portfolio",
            json_config={"universe": ["BTC"], "base_position_pct": 0.05},
            version=1
        )
        session.add(profile)
        session.commit()
        session.close()

        # Update with --replace
        update_config = {
            "json_config": {"universe": ["ETH"]}
        }

        result = cli_runner.invoke(
            profile_group,
            ["update", "replace-test", "--replace", "--config", json.dumps(update_config)]
        )

        assert result.exit_code == 0

        # Verify replacement
        session = get_db_session(db_url)
        db_profile = session.query(StrategyConfig).filter_by(name="replace-test").first()
        session.close()
        assert db_profile.json_config == {"universe": ["ETH"]}
        assert "base_position_pct" not in db_profile.json_config
        assert db_profile.version == 2

    def test_update_keep_version(self, cli_runner):
        """Update --keep-version preserves version"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="keep-version-test",
            kind="portfolio",
            json_config={"test": 1},
            version=3
        )
        session.add(profile)
        session.commit()
        session.close()

        update_config = {
            "json_config": {"test": 2}
        }

        result = cli_runner.invoke(
            profile_group,
            ["update", "keep-version-test", "--keep-version", "--config", json.dumps(update_config)]
        )

        assert result.exit_code == 0

        # Verify version unchanged
        session = get_db_session(db_url)
        db_profile = session.query(StrategyConfig).filter_by(name="keep-version-test").first()
        session.close()
        assert db_profile.version == 3

    def test_update_not_found(self, cli_runner):
        """Update non-existent profile returns error with exit 1"""
        config = {"json_config": {"test": 1}}

        result = cli_runner.invoke(
            profile_group,
            ["update", "nonexistent", "--config", json.dumps(config)]
        )

        assert result.exit_code == 1
        assert "Profile not found" in result.output


class TestAC5DeleteProfile:
    """AC5: Delete Profile - confirmation, force flag, not found"""

    def test_delete_with_confirmation(self, cli_runner):
        """Delete with confirmation prompt (y) deletes profile"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="delete-confirm-test",
            kind="portfolio",
            json_config={"test": 1}
        )
        session.add(profile)
        session.commit()
        session.close()

        result = cli_runner.invoke(
            profile_group,
            ["delete", "delete-confirm-test"],
            input="y\n"
        )

        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

        # Verify deletion
        session = get_db_session(db_url)
        db_profile = session.query(StrategyConfig).filter_by(name="delete-confirm-test").first()
        session.close()
        assert db_profile is None

    def test_delete_cancel_confirmation(self, cli_runner):
        """Delete with confirmation prompt (n) cancels deletion"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="delete-cancel-test",
            kind="portfolio",
            json_config={"test": 1}
        )
        session.add(profile)
        session.commit()
        session.close()

        result = cli_runner.invoke(
            profile_group,
            ["delete", "delete-cancel-test"],
            input="n\n"
        )

        assert result.exit_code == 0

        # Verify profile still exists
        session = get_db_session(db_url)
        db_profile = session.query(StrategyConfig).filter_by(name="delete-cancel-test").first()
        session.close()
        assert db_profile is not None

    def test_delete_force_flag(self, cli_runner):
        """Delete --force skips confirmation, deletes profile"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="delete-force-test",
            kind="portfolio",
            json_config={"test": 1}
        )
        session.add(profile)
        session.commit()
        session.close()

        result = cli_runner.invoke(
            profile_group,
            ["delete", "delete-force-test", "--force"]
        )

        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

        # Verify deletion
        session = get_db_session(db_url)
        db_profile = session.query(StrategyConfig).filter_by(name="delete-force-test").first()
        session.close()
        assert db_profile is None

    def test_delete_not_found(self, cli_runner):
        """Delete non-existent profile returns error with exit 1"""
        result = cli_runner.invoke(
            profile_group,
            ["delete", "nonexistent", "--force"]
        )

        assert result.exit_code == 1
        assert "Profile not found" in result.output


class TestAC6CLIUsability:
    """AC6: CLI Usability - help text, command documentation"""

    def test_help_text(self, cli_runner):
        """profile --help shows available commands"""
        result = cli_runner.invoke(profile_group, ["--help"])

        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "create" in result.output
        assert "update" in result.output
        assert "delete" in result.output

    def test_command_help_text(self, cli_runner):
        """Each command --help shows usage and options"""
        for cmd in ["list", "show", "create", "update", "delete"]:
            result = cli_runner.invoke(profile_group, [cmd, "--help"])
            assert result.exit_code == 0
            assert "Usage" in result.output or "Options" in result.output


class TestVersionTracking:
    """Additional: Verify version tracking behavior"""

    def test_version_increments_on_multiple_updates(self, cli_runner):
        """Version increments with each update"""
        db_url = os.getenv("DATABASE_URL")
        session = get_db_session(db_url)

        profile = StrategyConfig(
            name="version-test",
            kind="portfolio",
            json_config={"v": 1},
            version=1
        )
        session.add(profile)
        session.commit()
        session.close()

        # First update
        cli_runner.invoke(
            profile_group,
            ["update", "version-test", "--config", json.dumps({"json_config": {"v": 2}})]
        )

        session = get_db_session(db_url)
        profile = session.query(StrategyConfig).filter_by(name="version-test").first()
        session.close()
        assert profile.version == 2

        # Second update
        cli_runner.invoke(
            profile_group,
            ["update", "version-test", "--config", json.dumps({"json_config": {"v": 3}})]
        )

        session = get_db_session(db_url)
        profile = session.query(StrategyConfig).filter_by(name="version-test").first()
        session.close()
        assert profile.version == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
