"""Script to configure PostgreSQL database connection."""

import os
from pathlib import Path


def setup_postgres_config():
    """Interactive setup for PostgreSQL configuration."""
    print("=" * 70)
    print("PostgreSQL Configuration Setup")
    print("=" * 70)

    # Get PostgreSQL connection details manually
    print("\nEnter your PostgreSQL connection details:")
    host = input("  Host (default: localhost): ").strip() or "localhost"
    port = input("  Port (default: 5432): ").strip() or "5432"
    user = input("  User (default: postgres): ").strip() or "postgres"
    password = input("  Password: ").strip()
    database = input("  Database Name (default: quantagent_dev): ").strip() or "quantagent_dev"

    # Build connection string
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    # Update .env file
    env_file = Path(__file__).parent / ".env"

    # Read existing .env or create new one
    env_content = ""
    if env_file.exists():
        with open(env_file, "r") as f:
            env_content = f.read()

    # Update or add DATABASE_URL
    lines = env_content.split("\n")
    updated_lines = []
    db_url_found = False

    for line in lines:
        if line.startswith("DATABASE_URL"):
            updated_lines.append(f"DATABASE_URL={db_url}")
            db_url_found = True
        else:
            updated_lines.append(line)

    if not db_url_found:
        updated_lines.append(f"DATABASE_URL={db_url}")

    # Write updated .env
    with open(env_file, "w") as f:
        f.write("\n".join(updated_lines))

    print("\n" + "=" * 70)
    print("✓ Configuration Updated Successfully")
    print("=" * 70)
    print(f"\nConnection String: {db_url}")
    print(f"✓ Updated .env file with DATABASE_URL")
    print("\nNote: alembic.ini does NOT need to be updated.")
    print("      It reads DATABASE_URL from environment automatically.")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    print("\n1️⃣  Ensure PostgreSQL server is running on", f"{host}:{port}")
    print("\n2️⃣  Create the database (if it doesn't exist):")
    print(f"   createdb -U {user} -h {host} {database}")
    print("\n3️⃣  Load environment variables (Windows PowerShell):")
    print(f"   $env:DATABASE_URL = '{db_url}'")
    print("\n4️⃣  Run migrations:")
    print("   python -m alembic upgrade head")
    print("\n5️⃣  Verify migrations:")
    print("   python -m alembic current")
    print("\n6️⃣  Test migrations:")
    print("   python test_migrations.py")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    setup_postgres_config()
