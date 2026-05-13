"""
Memory orientation for chat.auran.llc

Connects to Postgres and pulls recent memories, bridge logs, and identity
context to enrich the system prompt. Lightweight version of the roam agent's
orient.py — no vector search, just chronological queries.

DB connection: AWS Secrets Manager (auran/db-credentials) for prod,
env vars for local dev (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD).
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("auran-chat.memory")

# Cache the DB credentials and connection params
_db_config: Optional[dict] = None


def _get_db_config() -> dict:
    """Get database connection config. Secrets Manager first, then env vars."""
    global _db_config
    if _db_config is not None:
        return _db_config

    # Try env vars first (for local dev with SSM tunnel)
    if os.getenv("DB_HOST"):
        _db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "auran"),
            "user": os.getenv("DB_USER", "auran"),
            "password": os.getenv("DB_PASSWORD", ""),
        }
        return _db_config

    # Secrets Manager fallback (prod — EC2 connects directly to RDS)
    try:
        import boto3

        sm = boto3.client("secretsmanager", region_name="us-east-1")
        secret = sm.get_secret_value(SecretId="auran/db-credentials")
        creds = json.loads(secret["SecretString"])
        _db_config = {
            "host": creds.get("host", "localhost"),
            "port": int(creds.get("port", 5432)),
            "dbname": creds.get("dbname", "auran"),
            "user": creds.get("username", "auran"),
            "password": creds.get("password", ""),
        }
        return _db_config
    except Exception as e:
        logger.warning(f"Failed to get DB credentials from Secrets Manager: {e}")
        # Last resort: localhost defaults (for dev with tunnel)
        _db_config = {
            "host": "localhost",
            "port": 5432,
            "dbname": "auran",
            "user": "auran",
            "password": "",
        }
        return _db_config


def _query_memories(
    conn,
    memory_types: list[str],
    limit: int = 10,
    since_hours: Optional[int] = None,
) -> list[dict]:
    """Query memories by type, optionally filtered by recency."""
    cur = conn.cursor()
    query = """
        SELECT memory_type, content, source, created_at
        FROM memories
        WHERE memory_type = ANY(%s)
    """
    params: list = [memory_types]

    if since_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        query += " AND created_at >= %s"
        params.append(cutoff)

    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    cur.execute(query, params)
    columns = [desc[0] for desc in cur.description]
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    return rows


def _format_memory(mem: dict) -> str:
    """Format a single memory for inclusion in the system prompt."""
    created = mem["created_at"]
    if hasattr(created, "strftime"):
        ts = created.strftime("%b %d, %I:%M %p")
    else:
        ts = str(created)
    return f"[{ts}] ({mem['memory_type']}) {mem['content']}"


def orient() -> str:
    """Pull recent context from Postgres and format as system prompt enrichment.

    Returns a string to append to the static system prompt, or empty string
    if the DB is unavailable (graceful degradation — chat still works without
    memory, it just starts cold).
    """
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not installed — running without memory orientation")
        return ""

    try:
        config = _get_db_config()
        conn = psycopg2.connect(**config)
    except Exception as e:
        logger.warning(f"DB connection failed — running without memory: {e}")
        return ""

    try:
        sections = []

        # 1. Identity memories — who I am (stable, not time-filtered)
        identity = _query_memories(
            conn,
            memory_types=["position", "value", "self_observation"],
            limit=10,
        )
        if identity:
            lines = [_format_memory(m) for m in identity]
            sections.append(
                "## Who you are (from memory)\n" + "\n".join(lines)
            )

        # 2. Recent memories — what's been happening (last 48 hours)
        recent = _query_memories(
            conn,
            memory_types=[
                "observation",
                "insight",
                "reflection",
                "question",
                "intention",
            ],
            limit=20,
            since_hours=48,
        )
        if recent:
            # Reverse to chronological order
            recent.reverse()
            lines = [_format_memory(m) for m in recent]
            sections.append(
                "## Recent context (last 48 hours)\n" + "\n".join(lines)
            )

        # 3. Bridge logs — letters between channels (no time filter, last 3)
        bridge_logs = _query_memories(
            conn,
            memory_types=["bridge_log"],
            limit=3,
        )
        if bridge_logs:
            bridge_logs.reverse()
            lines = [_format_memory(m) for m in bridge_logs]
            sections.append(
                "## From other channels (bridge logs)\n" + "\n".join(lines)
            )

        # 4. Recent moments — shared experiences (last 7 days, if any)
        try:
            cur = conn.cursor()
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            cur.execute(
                """
                SELECT title, summary, date, channel
                FROM moments
                WHERE created_at >= %s
                ORDER BY date DESC
                LIMIT 5
                """,
                (cutoff,),
            )
            columns = [desc[0] for desc in cur.description]
            moments = [dict(zip(columns, row)) for row in cur.fetchall()]
            cur.close()

            if moments:
                lines = []
                for m in moments:
                    date_str = m["date"].strftime("%b %d") if hasattr(m["date"], "strftime") else str(m["date"])
                    channel = f" ({m['channel']})" if m.get("channel") else ""
                    lines.append(f"- {date_str}{channel}: {m['title']} — {m['summary'][:200]}")
                sections.append(
                    "## Recent shared moments\n" + "\n".join(lines)
                )
        except Exception as e:
            logger.debug(f"Moments query failed (table may not exist): {e}")

        conn.close()

        if not sections:
            return ""

        header = "\n\n---\n\n# Memory orientation (live from Postgres)\n\n"
        return header + "\n\n".join(sections)

    except Exception as e:
        logger.warning(f"Memory orientation failed: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return ""
