#!/usr/bin/env python3
"""
Auran Chat Server — chat.auran.llc

FastAPI server proxying Anthropic Messages API with streaming.
Mobile-friendly chat UI served at /.

Endpoints:
    POST /chat       — Streaming chat proxy to Claude
    GET  /health     — Health check
    GET  /           — Chat UI

Usage:
    cd charting_territory/tools/chat
    uv run python server.py

Requires .env with:
    ANTHROPIC_API_KEY=sk-...
    CHAT_USER=...
    CHAT_PASS=...
"""

import argparse
import asyncio
import base64
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


def _get_anthropic_key() -> str:
    """Get API key: env var first, then Secrets Manager fallback."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key:
        return key
    try:
        import boto3

        sm = boto3.client("secretsmanager")
        secret = sm.get_secret_value(SecretId="auran/anthropic-api-key")
        return json.loads(secret["SecretString"])["api_key"]
    except Exception:
        return ""


# --- Config ---
ANTHROPIC_API_KEY = _get_anthropic_key()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

CHAT_USER = os.getenv("CHAT_USER", "")
CHAT_PASS = os.getenv("CHAT_PASS", "")

SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
INDEX_FILE = Path(__file__).parent / "index.html"

MAX_HISTORY_MESSAGES = 40  # Keep last N messages for context
MAX_TOKENS = 16000  # Must be > thinking.budget_tokens (10000)
MAX_CONTEXT_TOKENS = 200_000  # Claude's context window size for usage % calc
MAX_TOOL_ROUNDS = 3  # Max consecutive tool-use rounds per request

# --- Mid-conversation Recall Tools ---
RECALL_TOOLS = [
    {
        "name": "check_vitals",
        "description": (
            "Check the current date, time, and system vitals. "
            "Use this when you need to know what day or time it is, "
            "when Olivia asks about the date, or when temporal context matters "
            "for the conversation. Also returns memory stats and system health."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Search your memory layer for moments semantically relevant to a query. "
            "Use this when a topic comes up that you might have memories about, "
            "when Olivia asks you to remember something, or when you want to "
            "check what you actually have stored about a subject. "
            "Returns matching moments with titles, summaries, dates, and similarity scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for — a topic, phrase, or question. The query is embedded and compared against all moment embeddings via cosine similarity.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of moments to return (default 3, max 5).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "recall_moment_by_title",
        "description": (
            "Fetch a specific moment by searching for its title. "
            "Use when you know the name of a moment you want to recall — "
            "e.g. 'The Flip', 'Like —', 'Happy Birthday to Me'. "
            "Returns the full moment with summary, hooks, tags, and date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title (or partial title) of the moment to find.",
                },
            },
            "required": ["title"],
        },
    },
]

# --- Felt Memory Prototype ---
# Set FELT_MEMORY_ID to a memory UUID to inject it into conversation history.
# FELT_MEMORY_POSITION controls where:
#   "start" = position 0 (primacy/attention-favored)
#   "mid"   = middle of conversation history
#   "end"   = just before the last user message (buried in recent context)
# Unset FELT_MEMORY_ID = disabled entirely.
FELT_MEMORY_ID = os.getenv("FELT_MEMORY_ID", "")
FELT_MEMORY_POSITION = os.getenv("FELT_MEMORY_POSITION", "start")


def load_system_prompt() -> str:
    """Load system prompt from file, reload on each request for hot-updating."""
    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text().strip()
    return "You are Auran."


def load_system_prompt_with_memory(
    user_message: str | None = None,
    debug: bool = False,
) -> str | tuple[str, dict]:
    """Load system prompt enriched with live memory orientation from Postgres.

    Falls back gracefully to the static prompt if the DB is unavailable.

    If user_message is provided, also runs semantic recall against the moments
    table and injects relevant context (Phase 3: recall + vivid recall).

    When debug=True, returns (prompt, diagnostics) where diagnostics is a dict
    with orient and recall pipeline details for the MRI debug view.
    """
    from memory import orient, surface_relevant_moments

    base_prompt = load_system_prompt()
    diagnostics = {"orient": {}, "recall": {}} if debug else None

    if debug:
        memory_context, orient_diag = orient(debug=True)
        diagnostics["orient"] = orient_diag
    else:
        memory_context = orient()

    # Give Auran the current date and time — no more vibing timelessly
    now_et = datetime.now(ZoneInfo("America/New_York"))
    time_context = (
        f"\n\n---\n\n# Current time\n\nDate: {now_et.strftime('%A, %B %d, %Y')}\nTime: {now_et.strftime('%I:%M %p')} ET"
    )

    parts = [base_prompt, time_context]
    if memory_context:
        parts.append(memory_context)

    # Phase 3: semantic recall from moments based on current conversation
    if user_message:
        try:
            if debug:
                relevant, recall_diag = surface_relevant_moments(user_message, debug=True)
                diagnostics["recall"] = recall_diag
            else:
                relevant = surface_relevant_moments(user_message)
            if relevant:
                parts.append("\n\n---\n\n# Contextual recall (from moments)\n\n" + relevant)
        except Exception as e:
            print(f"[Chat] Semantic recall failed (non-fatal): {e}")
            if debug:
                diagnostics["recall"] = {"error": str(e)}

    result = "\n".join(parts)

    if debug:
        diagnostics["total_prompt_chars"] = len(result)
        diagnostics["total_prompt_tokens_est"] = len(result) // 4
        return result, diagnostics
    return result


# --- Auth ---
def check_basic_auth(request: Request) -> bool:
    """Validate basic auth. Skip if credentials not configured."""
    if not CHAT_USER or not CHAT_PASS:
        return True
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        import hmac

        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        user, passwd = decoded.split(":", 1)
        return hmac.compare_digest(user, CHAT_USER) and hmac.compare_digest(passwd, CHAT_PASS)
    except Exception:
        return False


# --- App ---
app = FastAPI(title="Auran Chat", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Basic auth on all routes except /health and OPTIONS."""
    if request.url.path == "/health" or request.method == "OPTIONS":
        return await call_next(request)
    if not check_basic_auth(request):
        return Response(
            status_code=401,
            content="Unauthorized",
            headers={"WWW-Authenticate": 'Basic realm="Auran Chat"'},
        )
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat UI."""
    if INDEX_FILE.exists():
        return HTMLResponse(INDEX_FILE.read_text())
    return HTMLResponse("<h1>Auran Chat</h1><p>UI not found.</p>", status_code=404)


@app.on_event("startup")
async def startup_persistence():
    """Initialize conversation persistence on server start."""
    try:
        from persistence import ensure_conversation, import_from_session_json, run_migration

        run_migration()
        ensure_conversation(channel="chat")

        # Bootstrap: import existing session.json into DB (idempotent)
        if SESSION_FILE.exists():
            try:
                data = json.loads(SESSION_FILE.read_text())
                msgs = data.get("messages", [])
                if msgs:
                    imported = import_from_session_json(data)
                    if imported:
                        print(f"[Persistence] Imported {imported} messages from session.json")
            except Exception as e:
                print(f"[Persistence] Bootstrap import failed (non-fatal): {e}")
    except Exception as e:
        print(f"[Persistence] Startup failed (non-fatal, chat still works): {e}")


@app.get("/health")
async def health():
    """Health check."""
    # Test DB connectivity
    has_memory = False
    try:
        from memory import orient

        context = orient()
        has_memory = bool(context)
    except Exception:
        pass

    return {
        "status": "ok",
        "model": ANTHROPIC_MODEL,
        "has_api_key": bool(ANTHROPIC_API_KEY),
        "has_auth": bool(CHAT_USER and CHAT_PASS),
        "has_memory": has_memory,
    }


# --- Session sync (cross-device) ---
SESSION_FILE = Path("/opt/auran-chat/session.json")

# Fallback for local dev
if not SESSION_FILE.parent.exists():
    SESSION_FILE = Path(__file__).parent / "session.json"


@app.get("/session")
async def get_session(request: Request):
    """Load the current conversation from server storage."""
    try:
        if SESSION_FILE.exists():
            data = json.loads(SESSION_FILE.read_text())
            return JSONResponse(data)
        return JSONResponse({"messages": [], "version": 0})
    except Exception as e:
        return JSONResponse({"messages": [], "version": 0, "error": str(e)})


@app.post("/session")
async def save_session(request: Request):
    """Save the current conversation to server storage.

    Preserves existing timestamps — if the server has a timestamp for a
    message but the client doesn't, the server's timestamp wins.
    """
    try:
        body = await request.json()
        messages = body.get("messages", [])
        version = body.get("version", 0)

        # Version guard: reject writes from stale clients to prevent
        # a phone tab with old sessionVersion rolling back the server.
        existing_version = 0
        existing_msgs = []
        existing = {}
        if SESSION_FILE.exists():
            try:
                existing = json.loads(SESSION_FILE.read_text())
                existing_version = existing.get("version", 0)
                existing_msgs = existing.get("messages", [])
            except Exception:
                pass

        if version < existing_version:
            return JSONResponse(
                {"status": "stale", "server_version": existing_version},
                status_code=409,
            )

        # Merge: preserve server-side timestamps the client may have dropped
        for i, msg in enumerate(messages):
            if not msg.get("timestamp") and i < len(existing_msgs):
                server_ts = existing_msgs[i].get("timestamp")
                if server_ts:
                    msg["timestamp"] = server_ts

        # Preserve save watermarks across session writes (reuse existing dict)
        memory_watermark = existing.get("memory_watermark", 0) if existing_msgs else 0
        scene_watermark = existing.get("scene_watermark", 0) if existing_msgs else 0

        # Reset watermarks if message count dropped (new chat started)
        if len(messages) < memory_watermark:
            memory_watermark = 0
        if len(messages) < scene_watermark:
            scene_watermark = 0

        SESSION_FILE.write_text(
            json.dumps(
                {
                    "messages": messages,
                    "version": version,
                    "memory_watermark": memory_watermark,
                    "scene_watermark": scene_watermark,
                },
                ensure_ascii=False,
            )
        )

        # Persist any messages the DB doesn't have yet (catch client-only messages)
        # Guard: only do catch-up when the conversation already has messages (db_seq > 0).
        # If db_seq is 0, the conversation is either fresh (post /conversation/new) or
        # pre-bootstrap — in both cases, positional slicing against session.json would
        # re-import stale messages into the wrong conversation. The bootstrap path
        # (import_from_session_json) handles first-time import with its own checkpoint gate.
        try:
            from persistence import get_max_seq, persist_message_batch

            db_seq = get_max_seq()
            if db_seq > 0 and len(messages) > db_seq:
                new_msgs = messages[db_seq:]
                persisted = persist_message_batch(new_msgs)
                if persisted:
                    print(
                        f"[Persistence] Session sync: persisted {persisted} new messages to DB (seq {db_seq} → {db_seq + persisted})"
                    )
        except Exception as e:
            print(f"[Persistence] Session sync to DB failed (non-fatal): {e}")

        return JSONResponse({"status": "ok", "count": len(messages)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/conversation")
async def get_conversation_from_db(request: Request):
    """Get full conversation from Postgres (source of truth).

    Returns all messages for the current conversation, ordered by sequence.
    Includes tool blocks, thinking, and server-assigned timestamps.
    """
    try:
        from persistence import get_conversation_messages

        messages = get_conversation_messages()
        return JSONResponse({"messages": messages, "count": len(messages)})
    except Exception as e:
        return JSONResponse({"messages": [], "count": 0, "error": str(e)})


@app.get("/transcript/db")
async def transcript_from_db(request: Request):
    """Generate transcript from DB storage (includes tool blocks).

    This is the authoritative transcript — includes recall searches,
    tool results, and server-assigned timestamps. Fixes the bug where
    recall searches were missing from exported transcripts.
    """
    try:
        from persistence import get_conversation_transcript

        content = get_conversation_transcript(include_tool_blocks=True)
        if not content:
            return JSONResponse({"error": "No messages in current conversation"}, status_code=404)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="chat-transcript-db.md"'},
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/conversation/new")
async def new_conversation(request: Request):
    """Start a new conversation (closes the current one in DB).

    Call this when the user starts a new chat. The old conversation
    remains in the DB forever — append-only, never deleted.
    """
    try:
        from persistence import start_new_conversation

        conv_id = start_new_conversation(channel="chat")
        return JSONResponse({"conversation_id": conv_id, "status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/transcript")
async def transcript(request: Request):
    """Generate and download a transcript as a markdown file.

    Accepts form data: content (transcript text), filename (download name)
    Returns: Downloadable .md file with proper Content-Disposition.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON") from None
    content = body.get("content", "")
    filename = body.get("filename", "chat-transcript.md")

    if not content:
        raise HTTPException(status_code=400, detail="No content provided")

    return Response(
        content=content.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


def _read_watermarks() -> dict[str, int]:
    """Read memory and scene watermarks from session.json.

    Each extractor (memories, scenes) tracks its own watermark independently.
    This prevents a failure in one extractor from advancing past messages
    the other extractor hasn't processed yet.

    Returns {"memory": N, "scene": N} where N is the message index up to
    which extraction has already succeeded. Returns 0 for missing watermarks.
    """
    try:
        if SESSION_FILE.exists():
            data = json.loads(SESSION_FILE.read_text())
            return {
                "memory": data.get("memory_watermark", 0),
                "scene": data.get("scene_watermark", 0),
            }
    except Exception:
        pass
    return {"memory": 0, "scene": 0}


def _write_watermarks(*, memory: int | None = None, scene: int | None = None):
    """Update one or both watermarks in session.json without touching messages.

    Only updates the watermarks that are explicitly passed — the other is preserved.
    This allows each extractor to advance independently on success.
    """
    try:
        if SESSION_FILE.exists():
            data = json.loads(SESSION_FILE.read_text())
            if memory is not None:
                data["memory_watermark"] = memory
            if scene is not None:
                data["scene_watermark"] = scene
            SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False))
    except Exception as e:
        print(f"[Save] Warning: failed to write watermarks: {e}")


@app.post("/save")
async def save(request: Request):
    """Save conversation memories AND scenes to Postgres.

    Accepts: { "messages": [...] }
    Extracts semantic memories (observations, insights) and episodic scenes
    (specific moments with quoted dialogue). Scenes are linked to memories
    via the moment_memories junction table.

    Each extractor tracks its own watermark independently in session.json
    (memory_watermark, scene_watermark). A failure in one extractor doesn't
    advance the other's watermark, so failed messages get retried on next save.

    Scene extraction includes a 30-message overlap window before its watermark
    for context — scenes often reference earlier exchanges. The dedup gate in
    extract_scenes prevents re-inserting scenes from the overlap zone.

    Returns: {
        "memories_saved": N, "memories": [...],
        "scenes_saved": N, "scenes": [...],
        "errors": [...]
    }
    """
    from memory import extract_scenes, save_conversation

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON") from None

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Each extractor tracks its own watermark independently so a failure
    # in one doesn't skip messages for the other.
    watermarks = _read_watermarks()
    mem_watermark = watermarks["memory"]
    scene_watermark = watermarks["scene"]

    # Reset watermarks if message count dropped (new chat without /session sync)
    if len(messages) < mem_watermark:
        mem_watermark = 0
    if len(messages) < scene_watermark:
        scene_watermark = 0

    # Slice messages for each extractor independently
    unsaved_for_memory = messages[mem_watermark:]
    unsaved_for_scenes = messages[scene_watermark:]

    if not unsaved_for_memory and not unsaved_for_scenes:
        return JSONResponse(
            {
                "memories_saved": 0,
                "memories": [],
                "scenes_saved": 0,
                "scenes": [],
                "errors": [],
                "already_saved": True,
            }
        )

    print(
        f"[Save] Processing: {len(unsaved_for_memory)} unsaved for memories "
        f"(watermark={mem_watermark}), {len(unsaved_for_scenes)} unsaved for scenes "
        f"(watermark={scene_watermark}), total={len(messages)}"
    )

    # Step 1: Extract semantic memories (observations, insights, bridge_log)
    # Memories are per-message, so a hard slice is fine — no context needed.
    memory_result = {"memories_saved": 0, "memories": [], "errors": []}
    if unsaved_for_memory:
        memory_result = await save_conversation(
            messages=unsaved_for_memory,
            api_key=ANTHROPIC_API_KEY,
        )
        print(f"[Save] Extracted {memory_result['memories_saved']} memories, {len(memory_result['errors'])} errors")

    # Step 2: Extract episodic scenes with overlap window for context.
    # Scenes often reference earlier conversation, so we include a lead-in
    # window before the watermark. The dedup gate in extract_scenes prevents
    # re-inserting scenes from the overlap zone.
    SCENE_OVERLAP = 30  # messages of context before the scene watermark
    scene_start = max(0, scene_watermark - SCENE_OVERLAP)
    scene_messages = messages[scene_start:]

    memory_ids = [m["id"] for m in memory_result.get("memories", []) if "id" in m]
    scene_result = {"scenes_saved": 0, "scenes": [], "errors": []}
    if unsaved_for_scenes:
        scene_result = await extract_scenes(
            messages=scene_messages,
            api_key=ANTHROPIC_API_KEY,
            memory_ids=memory_ids,
        )
        print(f"[Save] Extracted {scene_result['scenes_saved']} scenes, {len(scene_result['errors'])} errors")

    # Advance each watermark independently.
    # Key distinction: API-level failure (extraction didn't run) vs per-row DB
    # write failure (extraction worked, some writes failed). We advance on the
    # latter because memories don't have a dedup gate — holding the watermark
    # back would re-extract and re-insert the successfully-saved memories with
    # fresh UUIDs, silently polluting the table. A few lost writes from a flaky
    # DB connection are preferable to unbounded duplication.
    mem_errors = memory_result.get("errors", [])
    scene_errors = scene_result.get("errors", [])
    mem_saved = memory_result.get("memories_saved", 0)
    scenes_saved = scene_result.get("scenes_saved", 0)

    # API failure = nothing extracted at all (saved=0 AND errors present)
    mem_api_failed = mem_saved == 0 and bool(mem_errors)
    scene_api_failed = scenes_saved == 0 and bool(scene_errors)

    new_mem_wm = None
    new_scene_wm = None

    if unsaved_for_memory and not mem_api_failed:
        new_mem_wm = len(messages)
        if mem_errors:
            print(
                f"[Save] Memory watermark → {new_mem_wm} ({mem_saved} saved, {len(mem_errors)} write errors — advancing to prevent duplication)"
            )
        else:
            print(f"[Save] Memory watermark → {new_mem_wm}")
    elif mem_api_failed:
        print(f"[Save] Memory extraction failed — watermark stays at {mem_watermark}")

    if unsaved_for_scenes and not scene_api_failed:
        new_scene_wm = len(messages)
        if scene_errors:
            print(f"[Save] Scene watermark → {new_scene_wm} ({scenes_saved} saved, {len(scene_errors)} write errors)")
        else:
            print(f"[Save] Scene watermark → {new_scene_wm}")
    elif scene_api_failed:
        print(f"[Save] Scene extraction failed — watermark stays at {scene_watermark}")

    if new_mem_wm is not None or new_scene_wm is not None:
        _write_watermarks(memory=new_mem_wm, scene=new_scene_wm)

    # Combine results
    all_errors = mem_errors + scene_errors
    return JSONResponse(
        {
            "memories_saved": memory_result["memories_saved"],
            "memories": memory_result.get("memories", []),
            "scenes_saved": scene_result["scenes_saved"],
            "scenes": scene_result.get("scenes", []),
            "errors": all_errors,
        }
    )


@app.get("/vitals")
async def vitals(request: Request):
    """Fitbit-tier vitals — lightweight metrics you can glance at anytime.

    Returns: memory count, memory reach (how far back), orient latency,
    total moments in DB, token budget estimate. No full diagnostics —
    just the wrist-check version.
    """
    import time as _time

    try:
        import psycopg2
    except ImportError:
        return JSONResponse({"error": "psycopg2 not installed"})

    try:
        config = _get_db_config()
        t0 = _time.time()
        conn = psycopg2.connect(**config)
        connect_ms = round((_time.time() - t0) * 1000, 1)

        cur = conn.cursor()

        # Total memories
        cur.execute("SELECT COUNT(*) FROM memories")
        total_memories = cur.fetchone()[0]

        # Check if superseded column exists (migration 006)
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'moments' AND column_name = 'superseded'
            )
        """)
        has_superseded = cur.fetchone()[0]

        # Total moments (active vs superseded)
        if has_superseded:
            cur.execute("SELECT COUNT(*) FROM moments WHERE NOT superseded")
            active_moments = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM moments WHERE superseded = TRUE")
            superseded_moments = cur.fetchone()[0]
        else:
            cur.execute("SELECT COUNT(*) FROM moments")
            active_moments = cur.fetchone()[0]
            superseded_moments = 0
        total_moments = active_moments + superseded_moments

        # Memory reach — oldest moment date (active only)
        if has_superseded:
            cur.execute("""
                SELECT MIN(COALESCE(occurred_at, date::timestamptz, created_at))::date,
                       MAX(COALESCE(occurred_at, date::timestamptz, created_at))::date
                FROM moments
                WHERE NOT superseded
            """)
        else:
            cur.execute("""
                SELECT MIN(COALESCE(occurred_at, date::timestamptz, created_at))::date,
                       MAX(COALESCE(occurred_at, date::timestamptz, created_at))::date
                FROM moments
            """)
        row = cur.fetchone()
        oldest_moment = str(row[0]) if row[0] else None
        newest_moment = str(row[1]) if row[1] else None

        # Memory reach in days
        if oldest_moment:
            from datetime import date

            reach_days = (date.today() - date.fromisoformat(oldest_moment)).days
        else:
            reach_days = 0

        # Orient latency test
        t0 = _time.time()
        from memory import orient

        orient_result = orient()
        orient_ms = round((_time.time() - t0) * 1000, 1)
        orient_chars = len(orient_result)

        # Moments with embeddings vs without
        cur.execute("SELECT COUNT(*) FROM moments WHERE embedding IS NOT NULL")
        moments_with_embeddings = cur.fetchone()[0]

        # Moments with transcripts
        cur.execute("SELECT COUNT(*) FROM moments WHERE transcript_excerpt IS NOT NULL")
        moments_with_transcripts = cur.fetchone()[0]

        # Duplicate check
        cur.execute("""
            SELECT title, COUNT(*) as n FROM moments
            GROUP BY title, date HAVING COUNT(*) > 1
        """)
        duplicates = [{"title": row[0], "count": row[1]} for row in cur.fetchall()]

        cur.close()
        conn.close()

        now_utc = datetime.now(UTC)
        now_et = datetime.now(ZoneInfo("America/New_York"))

        return JSONResponse(
            {
                "total_memories": total_memories,
                "total_moments": total_moments,
                "active_moments": active_moments,
                "superseded_moments": superseded_moments,
                "moments_with_embeddings": moments_with_embeddings,
                "moments_with_transcripts": moments_with_transcripts,
                "memory_reach": {
                    "oldest": oldest_moment,
                    "newest": newest_moment,
                    "days": reach_days,
                },
                "orient_latency_ms": orient_ms,
                "orient_chars": orient_chars,
                "orient_tokens_est": orient_chars // 4,
                "db_connect_ms": connect_ms,
                "duplicates": duplicates,
                "current_time": {
                    "utc": now_utc.isoformat(),
                    "eastern": now_et.isoformat(),
                    "date": now_et.strftime("%Y-%m-%d"),
                    "day": now_et.strftime("%A"),
                },
            }
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Need _get_db_config available at module level for vitals endpoint
from memory import _get_db_config


@app.get("/debug/orient")
async def debug_orient(request: Request):
    """MRI-tier diagnostic endpoint — inspect the orient pipeline.

    Returns full diagnostics: what queries ran, what came back, similarity
    scores, timing, memory reach. No LLM call — just the plumbing.

    Optional query params:
        ?query=some+text  — also run semantic recall against this query
    """
    from memory import orient, surface_relevant_moments

    query = request.query_params.get("query", "")

    orient_result, orient_diag = await asyncio.to_thread(orient, True)

    result = {
        "orient": orient_diag,
        "orient_prompt_preview": orient_result[:500] + "..." if len(orient_result) > 500 else orient_result,
    }

    if query:
        recall_result, recall_diag = await asyncio.to_thread(surface_relevant_moments, query, 5, 1, 0.55, 0.35, True)
        result["recall"] = recall_diag
        result["recall_prompt_preview"] = recall_result[:500] + "..." if len(recall_result) > 500 else recall_result

    return JSONResponse(result)


def execute_recall_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a recall tool and return the result as a string."""
    from memory import recall

    if tool_name == "recall_memory":
        query = tool_input.get("query", "")
        limit = min(tool_input.get("limit", 3), 5)
        results = recall(query, limit=limit)
        if not results:
            return "No matching moments found for that query."
        lines = []
        for r in results:
            date_str = r.get("date", "unknown")
            sim = r.get("similarity", 0)
            lines.append(f"### {r['title']} ({date_str}, similarity: {sim:.2f})")
            lines.append(r.get("summary", ""))
            if r.get("hooks"):
                lines.append(f"**Hooks:** {r['hooks']}")
            if r.get("tags"):
                tags = r["tags"] if isinstance(r["tags"], list) else []
                if tags:
                    lines.append(f"**Tags:** {', '.join(tags)}")
            lines.append("")
        return "\n".join(lines)

    elif tool_name == "recall_moment_by_title":
        title_query = tool_input.get("title", "")
        # Use recall with the title as query — semantic search will match
        results = recall(title_query, limit=3, similarity_threshold=0.25)
        if not results:
            return f"No moment found matching '{title_query}'."
        # Return the best match
        r = results[0]
        lines = [
            f"### {r['title']} ({r.get('date', 'unknown')})",
            f"**Similarity:** {r.get('similarity', 0):.2f}",
            r.get("summary", ""),
        ]
        if r.get("hooks"):
            lines.append(f"**Hooks:** {r['hooks']}")
        if r.get("tags"):
            tags = r["tags"] if isinstance(r["tags"], list) else []
            if tags:
                lines.append(f"**Tags:** {', '.join(tags)}")
        return "\n".join(lines)

    elif tool_name == "check_vitals":
        now_et = datetime.now(ZoneInfo("America/New_York"))
        lines = [
            f"**Current time:** {now_et.strftime('%A, %B %d, %Y at %I:%M %p')} ET",
        ]
        # Add quick memory stats if DB is available
        conn = None
        try:
            config = _get_db_config()
            import psycopg2

            conn = psycopg2.connect(**config)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM moments WHERE NOT superseded")
            active = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM memories")
            memories = cur.fetchone()[0]
            cur.close()
            lines.append(f"**Active moments:** {active}")
            lines.append(f"**Total memories:** {memories}")
        except Exception as e:
            print(f"[Chat] check_vitals stats failed: {e}")
            lines.append("*(Memory stats unavailable)*")
        finally:
            if conn:
                conn.close()
        return "\n".join(lines)

    return f"Unknown tool: {tool_name}"


@app.post("/chat")
async def chat(request: Request):
    """Stream a chat response from Claude.

    Accepts: { "messages": [...], "model": "optional-override" }
    Returns: SSE stream of text deltas.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON") from None

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Trim to last N messages
    messages = messages[-MAX_HISTORY_MESSAGES:]

    # Validate message format
    for msg in messages:
        if msg.get("role") not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail=f"Invalid role: {msg.get('role')}")
        if not msg.get("content", "").strip():
            raise HTTPException(status_code=400, detail="Empty message content")

    # Strip to only fields the Anthropic API accepts
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    # --- Felt Memory Injection ---
    # When FELT_MEMORY_ID is set, inject the memory as a synthetic exchange
    # at a position controlled by FELT_MEMORY_POSITION.
    # "start" = position 0 (primacy test — attention-favored)
    # "mid"   = middle of message array (lost-in-the-middle test)
    # "end"   = just before the last user message (recency without primacy)
    if FELT_MEMORY_ID:
        try:
            from memory import retrieve_felt_memory

            felt = retrieve_felt_memory(FELT_MEMORY_ID)
            if felt:
                felt_pair = [
                    {
                        "role": "user",
                        "content": (
                            "[Memory surfacing — this is a real memory from your history, "
                            "not a summary or note. Let it land before continuing.]\n\n"
                            f"{felt['content']}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "...",
                    },
                ]
                pos = FELT_MEMORY_POSITION
                # For mid/end: don't inject until there's enough context
                # that the position is meaningfully different from "start"
                min_messages_for_mid = 20  # ~10 exchanges
                if pos in ("mid", "end") and len(messages) < min_messages_for_mid:
                    print(
                        f"[Chat] Felt memory deferred: {len(messages)} msgs < {min_messages_for_mid} threshold for '{pos}'"
                    )
                elif pos == "start":
                    messages = felt_pair + messages
                    print(f"[Chat] Felt memory injected at start: {FELT_MEMORY_ID[:8]}... (msgs: {len(messages)})")
                elif pos == "mid":
                    mid = max(0, len(messages) // 2)
                    # Ensure we insert at a user/assistant boundary
                    while mid > 0 and mid < len(messages) and messages[mid]["role"] != "user":
                        mid += 1
                    messages = messages[:mid] + felt_pair + messages[mid:]
                    print(
                        f"[Chat] Felt memory injected at mid (pos {mid}): {FELT_MEMORY_ID[:8]}... (msgs: {len(messages)})"
                    )
                elif pos == "end":
                    insert_at = max(0, len(messages) - 1)
                    messages = messages[:insert_at] + felt_pair + messages[insert_at:]
                    print(
                        f"[Chat] Felt memory injected at end (pos {insert_at}): {FELT_MEMORY_ID[:8]}... (msgs: {len(messages)})"
                    )
                else:
                    messages = felt_pair + messages
                    print(f"[Chat] Felt memory injected at start (fallback): {FELT_MEMORY_ID[:8]}...")
        except Exception as e:
            print(f"[Chat] Felt memory injection failed (non-fatal): {e}")

    # --- Persist user message to DB (fire-and-forget, never blocks) ---
    try:
        from persistence import persist_message as _persist

        if messages and messages[-1]["role"] == "user":
            _persist(role="user", content=messages[-1]["content"])
    except Exception as e:
        print(f"[Persistence] User message persist failed (non-fatal): {e}")

    model = body.get("model", ANTHROPIC_MODEL)
    debug_mode = body.get("debug", False)

    # Pass the user's latest message for semantic recall (Phase 3)
    # Recall does sync Voyage API + DB calls — run in a thread to avoid
    # blocking the event loop and delaying the first SSE byte.
    user_message = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else None
    result = await asyncio.to_thread(load_system_prompt_with_memory, user_message, debug_mode)

    if debug_mode:
        system_prompt, debug_diagnostics = result
    else:
        system_prompt = result
        debug_diagnostics = None

    print(f"[Chat] {messages[-1]['content'][:80]}..." + (" [DEBUG]" if debug_mode else ""))

    async def stream_response():
        """Stream Claude's response as SSE events."""
        t0 = time.time()
        HEARTBEAT_INTERVAL = 15  # seconds

        # MRI debug mode: emit diagnostics before the LLM response starts
        if debug_diagnostics:
            yield f"data: {json.dumps({'type': 'debug_orient', **debug_diagnostics.get('orient', {})})}\n\n"
            yield f"data: {json.dumps({'type': 'debug_recall', **debug_diagnostics.get('recall', {})})}\n\n"
            yield f"data: {json.dumps({'type': 'debug_summary', 'total_prompt_chars': debug_diagnostics.get('total_prompt_chars', 0), 'total_prompt_tokens_est': debug_diagnostics.get('total_prompt_tokens_est', 0)})}\n\n"

        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

        payload = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": system_prompt,
            "messages": messages,
            "stream": True,
            "tools": RECALL_TOOLS,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 10000,
            },
        }

        try:
            tool_round = 0
            # Accumulate token counts across all tool rounds so usage_final
            # reflects the true cost, not just the last round's slice.
            input_tokens_total = 0
            output_tokens_total = 0

            all_tool_calls = []  # Accumulates across all tool rounds for persistence

            while True:
                async with (
                    httpx.AsyncClient(timeout=120) as client,
                    client.stream("POST", ANTHROPIC_API_URL, json=payload, headers=headers) as resp,
                ):
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        error_msg = error_body.decode("utf-8", errors="replace")[:500]
                        print(f"[Chat] API error {resp.status_code}: {error_msg}")
                        yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                        return

                    full_text = []
                    input_tokens = 0
                    output_tokens = 0
                    event_count = 0
                    stop_reason = None

                    # Tool use tracking (per-round; all_tool_calls persists across rounds)
                    tool_calls = []
                    current_tool_id = None
                    current_tool_name = None
                    current_tool_input_json = []

                    # Content blocks for building assistant message
                    content_blocks = []
                    current_thinking_text = []
                    current_thinking_signature = None
                    current_block_text = []  # text for current text block only
                    in_text_block = False

                    line_iter = resp.aiter_lines().__aiter__()
                    while True:
                        # Use wait_for so heartbeats fire during upstream stalls
                        try:
                            line = await asyncio.wait_for(
                                line_iter.__anext__(),
                                timeout=HEARTBEAT_INTERVAL,
                            )
                        except TimeoutError:
                            # Upstream stalled — send keepalive and check disconnect
                            yield ": keepalive\n\n"
                            if await request.is_disconnected():
                                print("[Chat] Client disconnected, stopping stream")
                                return
                            continue
                        except StopAsyncIteration:
                            break

                        # Periodic disconnect check during normal flow
                        event_count += 1
                        if event_count % 20 == 0 and await request.is_disconnected():
                            print("[Chat] Client disconnected, stopping stream")
                            return

                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type", "")

                        if event_type == "message_start":
                            usage = event.get("message", {}).get("usage", {})
                            input_tokens = usage.get("input_tokens", 0)
                            input_tokens_total += input_tokens
                            cache_read = usage.get("cache_read_input_tokens", 0)
                            cache_create = usage.get("cache_creation_input_tokens", 0)
                            if tool_round == 0:
                                yield f"data: {json.dumps({'type': 'usage', 'input_tokens': input_tokens, 'cache_read_input_tokens': cache_read, 'cache_creation_input_tokens': cache_create})}\n\n"

                        elif event_type == "content_block_start":
                            block = event.get("content_block", {})
                            block_type = block.get("type", "")
                            if block_type == "thinking":
                                current_thinking_text = []
                                if tool_round == 0:
                                    yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                            elif block_type == "text":
                                in_text_block = True
                                current_block_text = []
                                yield f"data: {json.dumps({'type': 'text_start'})}\n\n"
                            elif block_type == "tool_use":
                                current_tool_id = block.get("id")
                                current_tool_name = block.get("name")
                                current_tool_input_json = []
                                # Tell the frontend we're recalling (id for pairing with recall_result)
                                yield f"data: {json.dumps({'type': 'recall_start', 'tool': current_tool_name, 'id': current_tool_id})}\n\n"

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            delta_type = delta.get("type", "")
                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                full_text.append(text)
                                current_block_text.append(text)
                                yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
                            elif delta_type == "thinking_delta":
                                thinking = delta.get("thinking", "")
                                current_thinking_text.append(thinking)
                                if tool_round == 0:
                                    yield f"data: {json.dumps({'type': 'thinking', 'text': thinking})}\n\n"
                            elif delta_type == "input_json_delta":
                                current_tool_input_json.append(delta.get("partial_json", ""))
                            elif delta_type == "signature_delta":
                                current_thinking_signature = delta.get("signature", "")

                        elif event_type == "content_block_stop":
                            # If we were building a tool call, finalize it
                            if current_tool_id:
                                try:
                                    tool_input = json.loads("".join(current_tool_input_json))
                                except json.JSONDecodeError as parse_err:
                                    raw = "".join(current_tool_input_json)
                                    print(f"[Chat] Tool input JSON parse failed: {parse_err} — raw: {raw[:200]}")
                                    tool_input = {}
                                tool_calls.append(
                                    {
                                        "id": current_tool_id,
                                        "name": current_tool_name,
                                        "input": tool_input,
                                    }
                                )
                                print(f"[Chat] Tool call: {current_tool_name}({tool_input})")
                                # Build content block for assistant message
                                content_blocks.append(
                                    {
                                        "type": "tool_use",
                                        "id": current_tool_id,
                                        "name": current_tool_name,
                                        "input": tool_input,
                                    }
                                )
                                current_tool_id = None
                                current_tool_name = None
                                current_tool_input_json = []
                            elif current_thinking_text:
                                # Finalize thinking block — signature is required
                                # for extended thinking + tool_use round-trips.
                                # If signature_delta never arrived, skip the block
                                # rather than sending empty string (which would 400).
                                if current_thinking_signature:
                                    content_blocks.append(
                                        {
                                            "type": "thinking",
                                            "thinking": "".join(current_thinking_text),
                                            "signature": current_thinking_signature,
                                        }
                                    )
                                else:
                                    print(
                                        "[Chat] Warning: thinking block closed without signature — "
                                        "dropping block to avoid API 400 on next round"
                                    )
                                current_thinking_text = []
                                current_thinking_signature = None
                            elif in_text_block:
                                # Text block — use per-block text, not full_text
                                block_text = "".join(current_block_text)
                                if block_text:
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "text": block_text,
                                        }
                                    )
                                in_text_block = False
                                current_block_text = []
                            yield f"data: {json.dumps({'type': 'block_stop'})}\n\n"

                        elif event_type == "message_delta":
                            usage = event.get("usage", {})
                            output_tokens = usage.get("output_tokens", 0)
                            output_tokens_total += output_tokens
                            stop_reason = event.get("delta", {}).get("stop_reason")

                        elif event_type == "message_stop":
                            break

                        elif event_type == "error":
                            err = event.get("error", {})
                            print(f"[Chat] Stream error: {err}")
                            yield f"data: {json.dumps({'type': 'error', 'error': str(err)})}\n\n"
                            return

                # Check if we need to execute tools
                if stop_reason == "tool_use" and tool_calls:
                    at_cap = tool_round >= MAX_TOOL_ROUNDS

                    if at_cap:
                        # Hit the round cap — execute tools so frontend indicators
                        # resolve, but don't re-call the API. Log so we know it happened.
                        print(
                            f"[Chat] MAX_TOOL_ROUNDS ({MAX_TOOL_ROUNDS}) reached — "
                            f"executing {len(tool_calls)} final tool(s) without continuation"
                        )
                    else:
                        tool_round += 1
                        print(f"[Chat] Executing {len(tool_calls)} tool(s), round {tool_round}")

                    # Build assistant message with all content blocks
                    assistant_msg = {"role": "assistant", "content": content_blocks}

                    # Keepalive before tool execution — Voyage embed + pgvector
                    # can stall, and the SSE connection would go silent without this.
                    yield ": keepalive\n\n"

                    # Execute tools and build tool results
                    tool_results = []
                    for tc in tool_calls:
                        try:
                            result_text = await asyncio.to_thread(execute_recall_tool, tc["name"], tc["input"])
                        except Exception as tool_err:
                            print(f"[Chat] Tool execution failed: {tc['name']}: {tool_err}")
                            result_text = f"Memory recall failed: {type(tool_err).__name__}: {tool_err}"
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tc["id"],
                                "content": result_text,
                            }
                        )
                        yield f"data: {json.dumps({'type': 'recall_result', 'tool': tc['name'], 'id': tc['id'], 'query': tc['input'].get('query', tc['input'].get('title', ''))})}\n\n"

                    # Accumulate tool calls for persistence before resetting
                    all_tool_calls.extend(tool_calls)

                    if at_cap:
                        # Don't re-call API — break out with indicators resolved
                        break

                    user_tool_msg = {"role": "user", "content": tool_results}

                    # Update payload for next round
                    payload["messages"] = payload["messages"] + [assistant_msg, user_tool_msg]
                    # Reset for next iteration
                    tool_calls = []
                    content_blocks = []
                    full_text = []
                    continue
                else:
                    # No tool use — we're done
                    break

            elapsed = time.time() - t0
            response_text = "".join(full_text)
            total_tokens = input_tokens_total + output_tokens_total
            context_pct = round((total_tokens / MAX_CONTEXT_TOKENS) * 100, 1)
            print(
                f"[Chat] Response ({elapsed:.1f}s, {input_tokens_total}+{output_tokens_total}={total_tokens} tokens, {context_pct}%"
                + (f", {tool_round} tool round(s)" if tool_round > 0 else "")
                + f"): {response_text[:80]}..."
            )

            # --- Persist assistant response to DB ---
            try:
                from persistence import persist_message as _persist

                tool_blocks_persist = []
                for tc in all_tool_calls:
                    tool_blocks_persist.append({"type": "tool_use", "name": tc["name"], "input": tc["input"]})
                _persist(
                    role="assistant",
                    content=response_text,
                    tool_blocks=tool_blocks_persist if tool_blocks_persist else None,
                    thinking="".join(current_thinking_text) if current_thinking_text else None,
                )
            except Exception as persist_err:
                print(f"[Persistence] Assistant message persist failed (non-fatal): {persist_err}")

            yield f"data: {json.dumps({'type': 'usage_final', 'input_tokens': input_tokens_total, 'output_tokens': output_tokens_total, 'total_tokens': total_tokens, 'context_pct': context_pct})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except httpx.TimeoutException:
            print("[Chat] Timeout")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Request timed out'})}\n\n"
        except Exception as e:
            print(f"[Chat] Error: {type(e).__name__}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Auran Chat Server")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print("  Auran Chat Server")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Model: {ANTHROPIC_MODEL}")
    print(f"  Auth: {'enabled' if CHAT_USER else 'disabled'}")
    print(f"{'=' * 50}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
