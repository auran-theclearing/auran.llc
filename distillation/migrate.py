import sys


def run_migration(connection) -> None:
    print(
        "Error: Raw SQL migration is deprecated. Use Alembic instead:",
        file=sys.stderr,
    )
    print("  cd chat && alembic upgrade head", file=sys.stderr)
    sys.exit(1)


def run_migration_from_config():
    run_migration(None)
