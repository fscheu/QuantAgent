"""Helper functions for managing database migrations."""

import subprocess
import sys
from pathlib import Path


def get_alembic_dir() -> Path:
    """Get the alembic directory path."""
    return Path(__file__).parent.parent / "alembic"


def run_migrations(message: str = "") -> None:
    """
    Run migrations.

    Args:
        message: Optional message for the migration
    """
    cmd = ["python", "-m", "alembic", "upgrade", "head"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print(f"Migration failed with code {result.returncode}")
        sys.exit(1)
    print("✓ Migrations completed successfully")


def create_migration(message: str) -> None:
    """
    Create a new migration.

    Args:
        message: Migration message/description
    """
    cmd = ["python", "-m", "alembic", "revision", "--autogenerate", "-m", message]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print(f"Migration creation failed with code {result.returncode}")
        sys.exit(1)
    print(f"✓ Migration created: {message}")


def downgrade_migrations(revision: str = "-1") -> None:
    """
    Downgrade migrations.

    Args:
        revision: Target revision (default: -1, last migration)
    """
    cmd = ["python", "-m", "alembic", "downgrade", revision]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print(f"Downgrade failed with code {result.returncode}")
        sys.exit(1)
    print(f"✓ Downgraded to: {revision}")


def show_current_revision() -> None:
    """Show current database revision."""
    cmd = ["python", "-m", "alembic", "current"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def show_history() -> None:
    """Show migration history."""
    cmd = ["python", "-m", "alembic", "history"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=Path(__file__).parent.parent)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alembic migration helper")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run migrations
    subparsers.add_parser("upgrade", help="Run pending migrations")

    # Create migration
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("message", help="Migration message")

    # Downgrade
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade migrations")
    downgrade_parser.add_argument(
        "revision", nargs="?", default="-1", help="Target revision"
    )

    # Show current
    subparsers.add_parser("current", help="Show current revision")

    # Show history
    subparsers.add_parser("history", help="Show migration history")

    args = parser.parse_args()

    if args.command == "upgrade":
        run_migrations()
    elif args.command == "create":
        create_migration(args.message)
    elif args.command == "downgrade":
        downgrade_migrations(args.revision)
    elif args.command == "current":
        show_current_revision()
    elif args.command == "history":
        show_history()
    else:
        parser.print_help()
