"""Memory schema v1.0 — SQLAlchemy Core table definitions.

These models are the CANONICAL schema definition. They drive Alembic
migrations and serve as the single source of truth for table structure.

Runtime queries stay raw asyncpg/psycopg2. SQLAlchemy is for schema
definition and migration tooling ONLY — no ORM rewrite.
"""
