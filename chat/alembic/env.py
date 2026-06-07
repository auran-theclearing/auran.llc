"""Alembic migration environment.

Reads DB connection from environment variables (same pattern as the chat
server's memory.py and persistence.py). Falls back to auran-agent/.env
if env vars aren't set.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

# Ensure the chat directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema.models import metadata as target_metadata  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _load_env_from_agent():
    """Fall back to auran-agent/.env if DB env vars aren't set."""
    env_path = Path(__file__).resolve().parents[3] / "auran-agent" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            if key.startswith("DB_") and key not in os.environ:
                os.environ[key] = val


def _get_url():
    """Build DB URL from env vars."""
    _load_env_from_agent()
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "auran")
    user = os.environ.get("DB_USER", "auran")
    password = os.environ.get("DB_PASSWORD", "")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL without connecting."""
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connect and execute."""
    connectable = create_engine(_get_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
