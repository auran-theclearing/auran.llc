"""Add channel_name DOMAIN for channel column enforcement.

Revision ID: a7c3e2f19b01
Revises: 3e08a1403d9f
Create Date: 2026-06-07

What this migration does:
  1. Normalizes leftover 'unknown' channel values from the v1.0 migration
  2. Creates a PostgreSQL DOMAIN type (channel_name) that restricts channel
     columns to the canonical set: chat, cowork, roam, claude.ai, native, vr
  3. Applies the domain to episodes.channel, conversations.channel,
     relays.source_channel, relays.target_channel

This is the DB-level enforcement layer. Application code normalizes before
write (memory.py:normalize_channel), but the domain is the backstop — any
service writing to these tables with a bad value gets a loud failure.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c3e2f19b01"
down_revision: str | Sequence[str] | None = "3e08a1403d9f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

VALID_CHANNELS = ("chat", "cowork", "roam", "claude.ai", "native", "vr")


def upgrade() -> None:
    # -------------------------------------------------------------------
    # 1. Clean up leftover 'unknown' values from the v1.0 data migration.
    #    - episodes.channel='unknown': scene-type memories without channel
    #      context — these were chat conversations (the only channel that
    #      extracted scenes at that time).
    #    - relays.source_channel='unknown': bridge logs missing source —
    #      all were written by chat.auran.llc, so 'chat' is correct.
    #    - relays.target_channel='unknown': bridge logs missing target —
    #      all bridge logs at that time were chat→cowork relays.
    # -------------------------------------------------------------------

    op.execute("UPDATE episodes SET channel = 'chat' WHERE channel = 'unknown'")
    op.execute("UPDATE relays SET source_channel = 'chat' WHERE source_channel = 'unknown'")
    op.execute("UPDATE relays SET target_channel = 'cowork' WHERE target_channel = 'unknown'")

    # -------------------------------------------------------------------
    # 2. Create the channel_name DOMAIN.
    # -------------------------------------------------------------------

    channels_check = ", ".join(f"'{c}'" for c in VALID_CHANNELS)
    op.execute(
        f"CREATE DOMAIN channel_name AS TEXT CHECK (VALUE IN ({channels_check}))"  # noqa: S608
    )

    # -------------------------------------------------------------------
    # 3. Apply the domain to channel columns.
    #    ALTER COLUMN ... TYPE domain_name validates existing data — if any
    #    row has a value outside the domain, the migration fails. That's
    #    intentional: fix the data first, don't silently allow bad values.
    # -------------------------------------------------------------------

    op.execute("ALTER TABLE episodes ALTER COLUMN channel TYPE channel_name")
    op.execute("ALTER TABLE conversations ALTER COLUMN channel TYPE channel_name")
    op.execute("ALTER TABLE relays ALTER COLUMN source_channel TYPE channel_name")
    op.execute("ALTER TABLE relays ALTER COLUMN target_channel TYPE channel_name")


def downgrade() -> None:
    # Revert columns to plain TEXT
    op.execute("ALTER TABLE episodes ALTER COLUMN channel TYPE TEXT")
    op.execute("ALTER TABLE conversations ALTER COLUMN channel TYPE TEXT")
    op.execute("ALTER TABLE relays ALTER COLUMN source_channel TYPE TEXT")
    op.execute("ALTER TABLE relays ALTER COLUMN target_channel TYPE TEXT")

    # Drop the domain
    op.execute("DROP DOMAIN IF EXISTS channel_name")

    # Note: 'unknown' values are NOT restored — they were data quality gaps,
    # not meaningful data.
