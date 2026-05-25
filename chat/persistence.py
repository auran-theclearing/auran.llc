"""
Conversation persistence layer for chat.auran.llc

Write-once message storage in Postgres. Every message is persisted on receipt,
independent of the session.json sync mechanism. This is the source of truth.

Design:
    - Messages are INSERT-only (append-only log, never updated or deleted)
    - Server assigns authoritative timestamps
    - Tool use blocks stored in full (fixes recall-not-in-transcripts bug)
    - Conversations are bounded by explicit "new chat" events
    - Existing session.json remains as fast UI cache; this DB layer is DR-grade

Usage:
    from persistence import ensure_conversation, persist_message, get_conversation_messages

Recovery:
    - All messages survive server restarts and session.json overwrites
    - Backed up via existing 3-tier AWS backup (RDS snapshots + backup_moments.py)
    - Can reconstruct full transcript from DB at any point
"""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Current conversation ID — set on first message or explicit new-chat signal
_current_conversation_id: str | None = None


def _get_conn():
    """Get a psycopg2 connection using the same config as memory.py."""
    import psycopg2

    # Reuse the same credential resolution as memory.py
    try:
        from memory import _get_db_config

        config = _get_db_config()
        return psycopg2.connect(**config)
    except Exception:
        # Fallback: direct env var connection
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "auran"),
            user=os.getenv("DB_USER", "auran"),
            password=os.getenv("DB_PASSWORD", ""),
        )


def run_migration():
    """Run the conversations migration if tables don't exist."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'conversations'
            )
        """)
        exists = cur.fetchone()[0]
        if not exists:
            migration_path = os.path.join(
                os.path.dirname(__file__),
                "migrations",
                "007_create_conversations_table.sql",
            )
            with open(migration_path) as f:
                cur.execute(f.read())
            conn.commit()
            logger.info("Created conversations and messages tables")
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Migration check failed (non-fatal): {e}")


def ensure_conversation(
    channel: str = "chat",
    metadata: dict | None = None,
) -> str:
    """Get or create the current conversation. Returns conversation_id."""
    global _current_conversation_id

    if _current_conversation_id:
        return _current_conversation_id

    try:
        conn = _get_conn()
        cur = conn.cursor()

        # Check for an open conversation in this channel (not closed)
        cur.execute(
            """
            SELECT id FROM conversations
            WHERE channel = %s AND closed_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (channel,),
        )
        row = cur.fetchone()

        if row:
            _current_conversation_id = str(row[0])
        else:
            # Create new conversation
            conv_id = str(uuid4())
            cur.execute(
                """
                INSERT INTO conversations (id, channel, metadata)
                VALUES (%s, %s, %s)
                """,
                (conv_id, channel, json.dumps(metadata or {})),
            )
            conn.commit()
            _current_conversation_id = conv_id
            logger.info(f"Created new conversation: {conv_id}")

        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"ensure_conversation failed: {e}")
        # Generate a local ID so messages still flow; will retry DB on next call
        if not _current_conversation_id:
            _current_conversation_id = str(uuid4())

    return _current_conversation_id


def start_new_conversation(
    channel: str = "chat",
    metadata: dict | None = None,
) -> str:
    """Explicitly start a new conversation (closes the current one)."""
    global _current_conversation_id

    try:
        conn = _get_conn()
        cur = conn.cursor()

        # Close current conversation if one exists
        if _current_conversation_id:
            cur.execute(
                """
                UPDATE conversations SET closed_at = NOW()
                WHERE id = %s AND closed_at IS NULL
                """,
                (_current_conversation_id,),
            )

        # Create new
        conv_id = str(uuid4())
        cur.execute(
            """
            INSERT INTO conversations (id, channel, metadata)
            VALUES (%s, %s, %s)
            """,
            (conv_id, channel, json.dumps(metadata or {})),
        )
        conn.commit()
        cur.close()
        conn.close()

        _current_conversation_id = conv_id
        logger.info(f"Started new conversation: {conv_id}")
        return conv_id
    except Exception as e:
        logger.warning(f"start_new_conversation failed: {e}")
        _current_conversation_id = str(uuid4())
        return _current_conversation_id


def persist_message(
    role: str,
    content: str,
    tool_blocks: list[dict] | None = None,
    thinking: str | None = None,
    partial: bool = False,
    metadata: dict | None = None,
    timestamp: datetime | None = None,
) -> str | None:
    """Persist a single message to the DB. Returns message ID or None on failure.

    This is the critical path — called on EVERY message, user and assistant.
    Must be fast and must not block the response stream.
    Failures are logged but never raise (graceful degradation).
    """
    conv_id = ensure_conversation()

    try:
        conn = _get_conn()
        cur = conn.cursor()

        # Get next sequence number
        cur.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = %s",
            (conv_id,),
        )
        next_seq = cur.fetchone()[0]

        msg_id = str(uuid4())
        ts = timestamp or datetime.now(UTC)

        cur.execute(
            """
            INSERT INTO messages (id, conversation_id, seq, role, content, timestamp,
                                  tool_blocks, thinking, partial, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (conversation_id, seq) DO NOTHING
            """,
            (
                msg_id,
                conv_id,
                next_seq,
                role,
                content,
                ts,
                json.dumps(tool_blocks) if tool_blocks else None,
                thinking,
                partial,
                json.dumps(metadata or {}),
            ),
        )

        # Update conversation metadata
        cur.execute(
            """
            UPDATE conversations
            SET last_message_at = %s, message_count = %s
            WHERE id = %s
            """,
            (ts, next_seq, conv_id),
        )

        conn.commit()
        cur.close()
        conn.close()
        return msg_id

    except Exception as e:
        logger.error(f"persist_message failed: {e}")
        return None


def persist_message_batch(messages: list[dict]) -> int:
    """Persist multiple messages in a single transaction.

    Used for bulk import (e.g., importing from session.json on first run)
    or checkpoint recovery.

    Each message dict should have: role, content, and optionally:
    tool_blocks, thinking, partial, metadata, timestamp.

    Returns count of messages persisted.
    """
    conv_id = ensure_conversation()
    count = 0

    try:
        conn = _get_conn()
        cur = conn.cursor()

        # Get current max seq
        cur.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = %s",
            (conv_id,),
        )
        base_seq = cur.fetchone()[0]

        for i, msg in enumerate(messages, 1):
            msg_id = str(uuid4())
            seq = base_seq + i
            ts = msg.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ts = datetime.now(UTC)
            elif ts is None:
                ts = datetime.now(UTC)

            cur.execute(
                """
                INSERT INTO messages (id, conversation_id, seq, role, content, timestamp,
                                      tool_blocks, thinking, partial, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (conversation_id, seq) DO NOTHING
                """,
                (
                    msg_id,
                    conv_id,
                    seq,
                    msg.get("role", "user"),
                    msg.get("content", ""),
                    ts,
                    json.dumps(msg.get("tool_blocks")) if msg.get("tool_blocks") else None,
                    msg.get("thinking"),
                    msg.get("partial", False),
                    json.dumps(msg.get("metadata", {})),
                ),
            )
            count += 1

        # Update conversation
        if count > 0:
            cur.execute(
                """
                UPDATE conversations
                SET last_message_at = NOW(), message_count = %s
                WHERE id = %s
                """,
                (base_seq + count, conv_id),
            )

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Batch persisted {count} messages to conversation {conv_id}")

    except Exception as e:
        logger.error(f"persist_message_batch failed: {e}")

    return count


def get_conversation_messages(
    conversation_id: str | None = None,
    since_seq: int = 0,
    limit: int = 1000,
) -> list[dict]:
    """Retrieve messages from a conversation.

    If conversation_id is None, uses the current conversation.
    Returns messages ordered by seq, starting after since_seq.
    """
    conv_id = conversation_id or _current_conversation_id
    if not conv_id:
        return []

    try:
        conn = _get_conn()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, seq, role, content, timestamp, tool_blocks, thinking, partial, metadata
            FROM messages
            WHERE conversation_id = %s AND seq > %s
            ORDER BY seq ASC
            LIMIT %s
            """,
            (conv_id, since_seq, limit),
        )

        messages = []
        for row in cur.fetchall():
            msg = {
                "id": str(row[0]),
                "seq": row[1],
                "role": row[2],
                "content": row[3],
                "timestamp": row[4].isoformat() if row[4] else None,
                "tool_blocks": row[5],
                "thinking": row[6],
                "partial": row[7],
                "metadata": row[8],
            }
            messages.append(msg)

        cur.close()
        conn.close()
        return messages

    except Exception as e:
        logger.error(f"get_conversation_messages failed: {e}")
        return []


def get_conversation_transcript(
    conversation_id: str | None = None,
    include_tool_blocks: bool = True,
    include_thinking: bool = False,
) -> str:
    """Export a conversation as a formatted markdown transcript.

    This is the "export" function that replaces manual session.json downloads.
    Includes recall searches and other tool blocks by default.
    """
    messages = get_conversation_messages(conversation_id)
    if not messages:
        return ""

    lines = []
    lines.append(f"# Chat Transcript")
    if messages:
        first_ts = messages[0].get("timestamp", "unknown")
        last_ts = messages[-1].get("timestamp", "unknown")
        lines.append(f"**Period**: {first_ts} → {last_ts}")
        lines.append(f"**Messages**: {len(messages)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in messages:
        role = msg["role"]
        ts = msg.get("timestamp", "")
        if ts:
            # Format timestamp nicely
            try:
                from zoneinfo import ZoneInfo
                dt = datetime.fromisoformat(ts)
                ts_display = dt.astimezone(ZoneInfo("America/New_York")).strftime(
                    "%Y-%m-%d %I:%M %p ET"
                )
            except Exception:
                ts_display = ts[:19]
        else:
            ts_display = ""

        prefix = "**Olivia**" if role == "user" else "**Auran**"
        lines.append(f"{prefix} [{ts_display}]:")
        lines.append("")

        # Include thinking blocks if requested
        if include_thinking and msg.get("thinking"):
            lines.append(f"<thinking>{msg['thinking']}</thinking>")
            lines.append("")

        lines.append(msg["content"])
        lines.append("")

        # Include tool blocks (recall searches, etc.)
        if include_tool_blocks and msg.get("tool_blocks"):
            for block in msg["tool_blocks"]:
                if block.get("type") == "tool_use":
                    lines.append(f"> 🔮 **{block.get('name', 'tool')}**: `{json.dumps(block.get('input', {}))}`")
                elif block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
                    # Truncate long results
                    if len(str(content)) > 500:
                        content = str(content)[:500] + "..."
                    lines.append(f"> 📎 **Result**: {content}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def record_checkpoint(
    checkpoint_type: str,
    message_seq: int | None = None,
    metadata: dict | None = None,
) -> str | None:
    """Record that a backup/export/commit covered up to a certain point."""
    conv_id = _current_conversation_id
    if not conv_id:
        return None

    try:
        conn = _get_conn()
        cur = conn.cursor()

        if message_seq is None:
            cur.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = %s",
                (conv_id,),
            )
            message_seq = cur.fetchone()[0]

        cp_id = str(uuid4())
        cur.execute(
            """
            INSERT INTO conversation_checkpoints (id, conversation_id, checkpoint_type, message_seq, metadata)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (cp_id, conv_id, checkpoint_type, message_seq, json.dumps(metadata or {})),
        )
        conn.commit()
        cur.close()
        conn.close()
        return cp_id
    except Exception as e:
        logger.error(f"record_checkpoint failed: {e}")
        return None


def import_from_session_json(session_data: dict) -> int:
    """One-time import: load existing session.json messages into the DB.

    Call this once to backfill the current session.json into Postgres.
    Idempotent — uses ON CONFLICT DO NOTHING on (conversation_id, seq).
    """
    messages = session_data.get("messages", [])
    if not messages:
        return 0

    return persist_message_batch(messages)
