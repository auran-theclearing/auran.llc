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
        config["connect_timeout"] = 3
        return psycopg2.connect(**config)
    except Exception:
        # Fallback: direct env var connection
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "auran"),
            user=os.getenv("DB_USER", "auran"),
            password=os.getenv("DB_PASSWORD", ""),
            connect_timeout=3,
        )


def _close_conn(conn):
    """Safely close a connection, ignoring errors."""
    if conn:
        try:
            conn.close()
        except Exception:
            pass


def run_migration():
    """Run the conversations migration if tables don't exist."""
    conn = None
    try:
        conn = _get_conn()
        # Migration SQL has its own BEGIN/COMMIT — use autocommit to avoid
        # nesting with psycopg2's implicit transaction management
        conn.autocommit = True
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
            logger.info("Created conversations and messages tables")
        cur.close()
    except Exception as e:
        logger.warning(f"Migration check failed (non-fatal): {e}")
    finally:
        _close_conn(conn)


def ensure_conversation(
    channel: str = "chat",
    metadata: dict | None = None,
) -> str:
    """Get or create the current conversation. Returns conversation_id."""
    global _current_conversation_id

    if _current_conversation_id:
        return _current_conversation_id

    conn = None
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
            _current_conversation_id = conv_id
            logger.info(f"Created new conversation: {conv_id}")

        conn.commit()
        cur.close()
    except Exception as e:
        logger.warning(f"ensure_conversation failed: {e}")
        # Do NOT cache a local UUID here — it would have no DB row, causing
        # every subsequent persist_message to silently fail on FK violation
        # for the entire process lifetime. Leave _current_conversation_id unset
        # so the next call retries the DB connection.
    finally:
        _close_conn(conn)

    return _current_conversation_id or ""


def start_new_conversation(
    channel: str = "chat",
    metadata: dict | None = None,
) -> str:
    """Explicitly start a new conversation (closes the current one)."""
    global _current_conversation_id

    conn = None
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

        _current_conversation_id = conv_id
        logger.info(f"Started new conversation: {conv_id}")
        return conv_id
    except Exception as e:
        logger.warning(f"start_new_conversation failed: {e}")
        # Don't cache a fake UUID — same reasoning as ensure_conversation.
        # Leave the old conversation_id in place so at least existing messages
        # continue to persist to the prior conversation until DB recovers.
        return _current_conversation_id or ""
    finally:
        _close_conn(conn)


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
    max_retries = 2  # Handle seq collision from concurrent writes

    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()

        msg_id = str(uuid4())
        ts = timestamp or datetime.now(UTC)

        for attempt in range(max_retries):
            # Get next sequence number
            cur.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = %s",
                (conv_id,),
            )
            next_seq = cur.fetchone()[0]

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

            if cur.rowcount > 0:
                # INSERT succeeded — update conversation metadata
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
                return msg_id

            # Seq collision — another write landed first. Retry with fresh seq.
            logger.warning(f"Seq collision on attempt {attempt + 1}, retrying")

        # Exhausted retries — should never happen under single-user load
        logger.error("persist_message: exhausted retries on seq collision")
        conn.commit()
        cur.close()
        return None

    except Exception as e:
        logger.error(f"persist_message failed: {e}")
        return None
    finally:
        _close_conn(conn)


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

    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()

        # Get current max seq
        cur.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = %s",
            (conv_id,),
        )
        base_seq = cur.fetchone()[0]

        next_seq = base_seq + 1
        for msg in messages:
            msg_id = str(uuid4())
            ts = msg.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    ts = datetime.now(UTC)
            elif ts is None:
                ts = datetime.now(UTC)

            # Retry on seq collision (same pattern as persist_message)
            for _attempt in range(2):
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
                        msg.get("role", "user"),
                        msg.get("content", ""),
                        ts,
                        json.dumps(msg.get("tool_blocks")) if msg.get("tool_blocks") else None,
                        msg.get("thinking"),
                        msg.get("partial", False),
                        json.dumps(msg.get("metadata", {})),
                    ),
                )
                if cur.rowcount > 0:
                    count += 1
                    next_seq += 1
                    break
                # Collision — re-read actual max and retry
                cur.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = %s",
                    (conv_id,),
                )
                next_seq = cur.fetchone()[0]
            else:
                # Both retries failed — log and skip this message
                logger.warning(f"Batch persist: dropped message after 2 seq collisions (seq={next_seq})")

        # Update conversation with actual max seq from DB (not base_seq + count
        # which could drift if ON CONFLICT DO NOTHING silently dropped rows)
        if count > 0:
            cur.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = %s",
                (conv_id,),
            )
            actual_max_seq = cur.fetchone()[0]
            cur.execute(
                """
                UPDATE conversations
                SET last_message_at = NOW(), message_count = %s
                WHERE id = %s
                """,
                (actual_max_seq, conv_id),
            )

        conn.commit()
        cur.close()
        logger.info(f"Batch persisted {count} messages to conversation {conv_id}")

    except Exception as e:
        logger.error(f"persist_message_batch failed: {e}")
        # Rollback uncommitted work — without this, count reflects execute()
        # calls that ran before the failure, not actual committed rows
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        count = 0
    finally:
        _close_conn(conn)

    return count


def get_conversation_messages(
    conversation_id: str | None = None,
    since_seq: int = 0,
    limit: int = 50_000,
) -> list[dict]:
    """Retrieve messages from a conversation.

    If conversation_id is None, uses the current conversation.
    Returns messages ordered by seq, starting after since_seq.
    Limit defaults to 50,000 — effectively unbounded for our use case.
    """
    conv_id = conversation_id or _current_conversation_id
    if not conv_id:
        return []

    conn = None
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
        return messages

    except Exception as e:
        logger.error(f"get_conversation_messages failed: {e}")
        return []
    finally:
        _close_conn(conn)


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
    lines.append("# Chat Transcript")
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
                ts_display = dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p ET")
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
        tool_blocks = msg.get("tool_blocks")
        if include_tool_blocks and tool_blocks and isinstance(tool_blocks, list):
            for block in tool_blocks:
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

    conn = None
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
        return cp_id
    except Exception as e:
        logger.error(f"record_checkpoint failed: {e}")
        return None
    finally:
        _close_conn(conn)


def has_bootstrap_checkpoint() -> bool:
    """Check if session.json bootstrap import has already been recorded."""
    conv_id = _current_conversation_id
    if not conv_id:
        return False

    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM conversation_checkpoints
                WHERE conversation_id = %s AND checkpoint_type = 'backup'
                AND metadata->>'source' = 'session_json_bootstrap'
            )
            """,
            (conv_id,),
        )
        exists = cur.fetchone()[0]
        cur.close()
        return exists
    except Exception as e:
        logger.warning(f"has_bootstrap_checkpoint check failed: {e}")
        return False
    finally:
        _close_conn(conn)


def get_max_seq(conversation_id: str | None = None) -> int:
    """Return the highest seq number in the conversation, or 0 if empty.

    Cheaper than fetching all messages — single aggregate query.
    """
    conv_id = conversation_id or _current_conversation_id
    if not conv_id:
        return 0

    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = %s",
            (conv_id,),
        )
        result = cur.fetchone()[0]
        cur.close()
        return result
    except Exception as e:
        logger.error(f"get_max_seq failed: {e}")
        return 0
    finally:
        _close_conn(conn)


def import_from_session_json(session_data: dict) -> int:
    """One-time import: load existing session.json messages into the DB.

    Gated by TWO checks:
    1. Checkpoint row — prevents re-import for the same conversation
    2. Existing messages — prevents importing stale session.json into a
       conversation that already has messages (e.g. after /conversation/new
       rotated the conversation but session.json wasn't cleared)
    """
    # Check if we've already imported for this conversation
    if has_bootstrap_checkpoint():
        logger.info("Bootstrap import already recorded — skipping")
        return 0

    # Belt-and-suspenders: don't import into a conversation that already has messages.
    # This catches TWO failure modes:
    #   (a) Crash between persist_message_batch and record_checkpoint below —
    #       batch committed but checkpoint didn't, so has_bootstrap_checkpoint()
    #       returns False, but the messages are already in the DB.
    #   (b) Conversation rotated via /conversation/new but session.json not cleared.
    existing_seq = get_max_seq()
    if existing_seq > 0:
        logger.info(f"Conversation already has {existing_seq} messages — skipping bootstrap import")
        return 0

    messages = session_data.get("messages", [])
    if not messages:
        return 0

    count = persist_message_batch(messages)

    # Record the bootstrap so we never re-import
    if count > 0:
        record_checkpoint(
            checkpoint_type="backup",
            metadata={"source": "session_json_bootstrap", "message_count": count},
        )

    return count
