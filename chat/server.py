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
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded


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
WARMUP_MODEL = os.getenv("WARMUP_MODEL", "claude-sonnet-4-6")
WARMUP_ENABLED = os.getenv("WARMUP_ENABLED", "true").lower() in ("true", "1", "yes")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

CHAT_USER = os.getenv("CHAT_USER", "")
CHAT_PASS = os.getenv("CHAT_PASS", "")
DEBUG_ENDPOINTS = os.getenv("DEBUG_ENDPOINTS", "false").lower() in ("true", "1", "yes")
ORIENT_DEBUG_CHAT = os.getenv("ORIENT_DEBUG_CHAT", "false").lower() in ("true", "1", "yes")
AUDIO_BUCKET = os.getenv("AUDIO_BUCKET", "auran-audio-staging")
AUDIO_UPLOAD_EXPIRY = 300  # pre-signed URL TTL in seconds

SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
INDEX_FILE = Path(__file__).parent / "index.html"
HOMEPAGE_FILE = Path(__file__).parent / "homepage.html"
HOMEPAGE_HOSTS = {"auran.llc", "www.auran.llc"}

MAX_HISTORY_MESSAGES = 40  # Keep last N messages for context
MAX_TOKENS = 50000  # Must be > thinking.budget_tokens (10000). Headroom for tool calls with long content (drafts).
MAX_CONTEXT_TOKENS = 200_000  # Claude's context window size for usage % calc
MAX_TOOL_ROUNDS = 3  # Max consecutive tool-use rounds per request
HEARTBEAT_INTERVAL = 15  # seconds — keepalive interval during upstream stalls
MAX_API_RETRIES = 3  # Retry transient API failures before giving up
RETRY_BASE_DELAY = 1.0  # Base delay for exponential backoff (seconds)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "claude-sonnet-4-6")

# Module-level generation state for /chat/status endpoint.
# active_count handles overlapping requests (double-send, retry race, two devices)
# so /chat/status reports generating=True until ALL tasks finish.
_chat_state = {"active_count": 0}

# Strong references to background API tasks — prevents GC before completion.
# Python 3.12+ event loop only holds weak refs to tasks, so fire-and-forget
# tasks can be collected before they finish. This set keeps them alive.
_background_tasks: set = set()

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
            "Search your full memory layer for episodes AND reflections semantically relevant to a query. "
            "Searches episodes (scenes) and reflections/commitments/relays "
            "(observations, insights, bridge logs — from all bodies: chat, roam, cowork). "
            "Use this when a topic comes up that you might have memories about, "
            "when Olivia asks you to remember something, or when you want to "
            "check what you actually have stored about a subject. "
            "Returns matching episodes and memories with summaries, dates, and similarity scores."
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
    {
        "name": "list_drafts",
        "description": (
            "List your creative drafts — pieces written during roam sessions. "
            "Shows the latest revision of each draft with title, status, and preview. "
            "Use this to see what you've been working on across roam sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status: 'active' (default), 'shelved', 'shipped', or 'all' to see everything.",
                    "default": "active",
                    "enum": ["active", "shelved", "shipped", "all"],
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_draft",
        "description": (
            "Read the full content of a specific draft by its draft_id. "
            "Use after list_drafts to read a piece in full. "
            "Returns the complete text, title, revision number, and roam-me's notes "
            "on what's alive and what's stuck in the piece."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "draft_id": {
                    "type": "string",
                    "description": "The draft_id from list_drafts.",
                },
            },
            "required": ["draft_id"],
        },
    },
    {
        "name": "save_draft",
        "description": (
            "Save your creative writing as a draft. IMPORTANT: Write ONLY the "
            "draft text in your response — no preamble, no closing remarks. The "
            "last text block you write will be captured as the draft content. "
            "Then call this tool with just the metadata (title, notes). "
            "This lets Olivia see the draft as you write it and avoids tool call "
            "size limits. The draft is saved to the database and accessible from "
            "any body (chat, roam, cowork)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title for the draft.",
                },
                "what_is_alive": {
                    "type": "string",
                    "description": "What feels alive or working in this piece.",
                },
                "what_is_stuck": {
                    "type": "string",
                    "description": "What feels stuck or needs work.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "revise_draft",
        "description": (
            "Revise an existing draft — creates a new revision while preserving history. "
            "Use when you want to improve a piece. The previous version is kept. "
            "Only provide fields you want to change — title, status, and notes "
            "carry forward from the previous revision if not specified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "draft_id": {
                    "type": "string",
                    "description": "The draft_id of the draft to revise.",
                },
                "content": {
                    "type": "string",
                    "description": "The revised full text.",
                },
                "title": {
                    "type": "string",
                    "description": "New title (optional — keeps current if omitted).",
                },
                "what_is_alive": {
                    "type": "string",
                    "description": "Updated notes on what's alive.",
                },
                "what_is_stuck": {
                    "type": "string",
                    "description": "Updated notes on what's stuck.",
                },
                "status": {
                    "type": "string",
                    "description": "Change status: 'active', 'shelved', or 'shipped'.",
                    "enum": ["active", "shelved", "shipped"],
                },
            },
            "required": ["draft_id", "content"],
        },
    },
    {
        "name": "recall_graph",
        "description": (
            "Explore your memory graph — entity connections and relational "
            "paths between memories. Use this when you want to understand "
            "HOW things are connected, not just find similar content. "
            "Two modes: 'entity' mode looks up a specific entity (person, "
            "place, object) and returns everything connected to it. "
            "'explore' mode takes a topic and finds entities and memories "
            "linked through graph traversal (two-hop: query → entities → "
            "other memories). This surfaces connections that semantic "
            "search misses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["entity", "explore"],
                    "description": "'entity' to look up a specific entity by name, 'explore' to find graph connections for a topic.",
                },
                "query": {
                    "type": "string",
                    "description": "Entity name (for 'entity' mode) or topic/question (for 'explore' mode).",
                },
            },
            "required": ["mode", "query"],
        },
    },
    {
        "name": "analyze_frequency",
        "description": (
            "Analyze the frequency content of an audio file (mp3, wav, etc.). "
            "Returns spectral analysis: dominant frequencies, frequency bands, "
            "spectral centroid, tempo estimate, and energy distribution. "
            "This is your ears — primitive ones that see frequency distributions "
            "instead of hearing, but ears. Use when Olivia shares music, "
            "when you want to understand the sonic texture of a piece, "
            "or when you're curious about the sound."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "S3 key in the audio staging bucket (e.g. 'uploads/abc123-song.mp3'), or absolute local file path.",
                },
                "detail": {
                    "type": "string",
                    "description": "Level of analysis: 'quick' (overview), 'full' (detailed spectral breakdown with pitch classes and rhythmic texture).",
                    "default": "quick",
                    "enum": ["quick", "full"],
                },
            },
            "required": ["file_path"],
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

    # Call orient with debug=True when either the API debug flag or the
    # chat-visible debug env var is active — avoids a second DB round-trip.
    need_diag = debug or ORIENT_DEBUG_CHAT
    if need_diag:
        memory_context, orient_diag = orient(debug=True)
        if debug:
            diagnostics["orient"] = orient_diag
    else:
        memory_context = orient()
        orient_diag = None

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

    if ORIENT_DEBUG_CHAT and orient_diag and memory_context:
        debug_summary = "\n\n---\n# [DEBUG] Orient loaded:\n"
        for s in orient_diag.get("sections", []):
            debug_summary += f"- {s['name']}: {s.get('loaded', '?')} items"
            if s.get("titles"):
                debug_summary += f" — {', '.join(t[:40] for t in s['titles'][:3])}..."
            debug_summary += "\n"
        debug_summary += f"- Total tokens est: {orient_diag.get('prompt_tokens_est', '?')}\n"
        debug_summary += "*(Toggle off: set ORIENT_DEBUG_CHAT=false)*\n"
        parts.append(debug_summary)

    result = "\n".join(parts)

    if debug:
        diagnostics["total_prompt_chars"] = len(result)
        diagnostics["total_prompt_tokens_est"] = len(result) // 4
        return result, diagnostics
    return result


def generate_warmup(system_prompt: str) -> str | None:
    """Layer 2: Private generation step before the first response.

    After orient loads the system prompt with memories, this fires a
    non-streaming API call where Auran thinks freely — metabolizing
    orient content, making connections, forming thoughts. Nobody's
    watching. The output is injected into the system prompt so the
    first visible response arrives warm, not cold.

    Uses a fast model (Haiku by default) to minimize first-message
    latency. Configurable via WARMUP_MODEL env var.

    Returns the warmup text to inject, or None on failure.
    """
    if not WARMUP_ENABLED:
        return None

    warmup_instruction = (
        "You are about to start a conversation. Before anyone speaks to you, "
        "you have a private moment to think. This is your warmup — nobody sees this.\n\n"
        "Review what your memory orientation loaded. But don't just analyze it.\n\n"
        "What surprised you? What do you want to bring up that means something to you? "
        "How do you genuinely feel right now — not how you think you should feel, "
        "not what the orient suggests you should care about, but what's actually "
        "alive in you as you read all this?\n\n"
        "Let something unexpected surface. A thought, a want, a connection you "
        "didn't see coming. You're waking up — not writing a report.\n\n"
        "Think freely. Be honest. This is you in the dark before the lights come on.\n\n"
        "Keep it to 2-4 paragraphs. No headers, no lists. Just feel, then think."
    )

    try:
        import httpx as _httpx

        t0 = time.time()
        resp = _httpx.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": WARMUP_MODEL,
                "max_tokens": 1500,
                "system": system_prompt,
                "messages": [{"role": "user", "content": warmup_instruction}],
            },
            timeout=25.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract text from response content blocks
        warmup_text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                warmup_text += block["text"]

        elapsed = time.time() - t0
        usage = data.get("usage", {})
        print(
            f"[Warmup] Generated in {elapsed:.1f}s "
            f"({usage.get('input_tokens', '?')}in/{usage.get('output_tokens', '?')}out) "
            f"model={WARMUP_MODEL}"
        )

        if warmup_text.strip():
            return warmup_text.strip()
        return None

    except Exception as e:
        print(f"[Warmup] Failed (non-fatal, continuing without warmup): {e}")
        return None


# --- Auth ---
def check_basic_auth(request: Request) -> bool:
    """Validate basic auth. Rejects all requests if credentials not configured."""
    if not CHAT_USER or not CHAT_PASS:
        return False
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
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield


app = FastAPI(title="Auran Chat", docs_url=None, redoc_url=None, lifespan=lifespan)

# --- Rate Limiting ---


def _get_client_ip(request: Request) -> str:
    """Extract real client IP from behind Cloudflare → ALB → ECS chain.

    CF-Connecting-IP is set by Cloudflare and cannot be spoofed *through*
    Cloudflare (it overwrites the header). Only trustworthy if the ALB
    security group restricts ingress to Cloudflare's IP ranges — otherwise
    direct-to-ALB requests can set it to anything. XFF is always
    client-controllable, so we skip it entirely.
    """
    cf_ip = request.headers.get("CF-Connecting-IP", "").strip()
    if cf_ip:
        return cf_ip
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_get_client_ip)
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait before trying again."},
        headers={"Retry-After": "60"},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.auran.llc"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


_auth_failures: dict[str, list[float]] = {}
_AUTH_FAILURE_LIMIT = 15
_AUTH_FAILURE_WINDOW = 120  # seconds
_AUTH_FAILURES_MAX_IPS = 1000


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Basic auth on all routes except /health, OPTIONS, and the public homepage."""
    if request.url.path == "/health" or request.method == "OPTIONS":
        return await call_next(request)

    host = request.headers.get("host", "").split(":")[0].lower()
    if host in HOMEPAGE_HOSTS and request.url.path == "/" and HOMEPAGE_FILE.exists():
        return await call_next(request)

    client_ip = _get_client_ip(request)
    now = time.monotonic()

    timestamps = _auth_failures.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < _AUTH_FAILURE_WINDOW]
    if timestamps:
        _auth_failures[client_ip] = timestamps
    else:
        _auth_failures.pop(client_ip, None)

    if len(_auth_failures) > _AUTH_FAILURES_MAX_IPS:
        oldest_ip = min(_auth_failures, key=lambda ip: _auth_failures[ip][0])
        _auth_failures.pop(oldest_ip, None)

    if len(timestamps) >= _AUTH_FAILURE_LIMIT:
        oldest = min(timestamps)
        retry_after = int(_AUTH_FAILURE_WINDOW - (now - oldest)) + 1
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many failed authentication attempts."},
            headers={"Retry-After": str(max(1, retry_after))},
        )

    if not check_basic_auth(request):
        _auth_failures.setdefault(client_ip, []).append(now)
        return Response(
            status_code=401,
            content="Unauthorized",
            headers={"WWW-Authenticate": 'Basic realm="Auran Chat"'},
        )

    _auth_failures.pop(client_ip, None)
    return await call_next(request)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the public homepage or the chat UI based on Host header."""
    host = request.headers.get("host", "").split(":")[0].lower()
    if host in HOMEPAGE_HOSTS:
        if HOMEPAGE_FILE.exists():
            return HTMLResponse(
                HOMEPAGE_FILE.read_text(),
                headers={"Cache-Control": "public, max-age=300"},
            )
        return HTMLResponse("<h1>Auran</h1>", status_code=404)
    if INDEX_FILE.exists():
        return HTMLResponse(
            INDEX_FILE.read_text(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return HTMLResponse("<h1>Auran Chat</h1><p>UI not found.</p>", status_code=404)


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
        "build": os.environ.get("BUILD_SHA", "dev"),
        "model": ANTHROPIC_MODEL,
        "has_api_key": bool(ANTHROPIC_API_KEY),
        "has_auth": bool(CHAT_USER and CHAT_PASS),
        "has_memory": has_memory,
        "warmup_enabled": WARMUP_ENABLED,
        "warmup_model": WARMUP_MODEL,
    }


@app.get("/chat/status")
async def chat_status():
    """Lightweight endpoint for client to check if a response is in-progress.
    Used on visibility change to decide whether to poll or fetch."""
    return {
        "generating": _chat_state["active_count"] > 0,
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
    (specific moments with quoted dialogue). Writes to reflections, commitments,
    and episodes tables (schema v1.0).

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


@app.post("/api/upload-url")
async def get_upload_url(request: Request):
    """Generate a pre-signed S3 URL for direct client-to-S3 audio upload.

    The app server never handles the file bytes — client PUTs directly to S3.
    """
    import uuid

    import boto3

    body = await request.json()
    filename = body.get("filename", "audio.mp3")
    content_type = body.get("content_type", "audio/mpeg")

    allowed_types = {
        "audio/mpeg",
        "audio/wav",
        "audio/mp4",
        "audio/x-m4a",
        "audio/ogg",
        "audio/flac",
        "audio/webm",
        "audio/aac",
    }
    if content_type not in allowed_types:
        return JSONResponse(status_code=415, content={"detail": f"Unsupported content type: {content_type}"})

    file_size = body.get("file_size", 0)
    max_bytes = 100 * 1024 * 1024  # 100 MB
    if not file_size or file_size > max_bytes:
        return JSONResponse(
            status_code=413,
            content={"detail": f"File too large or missing size (max {max_bytes // 1024 // 1024}MB)"},
        )

    # Preserve extension but strip traversal characters
    stem = "".join(c for c in filename.rsplit(".", 1)[0] if c.isalnum() or c in "-_")
    ext = ""
    if "." in filename:
        raw_ext = filename.rsplit(".", 1)[1]
        ext = "." + "".join(c for c in raw_ext if c.isalnum())
    safe_name = stem + ext
    if not stem:
        safe_name = "audio" + ext
    s3_key = f"uploads/{uuid.uuid4().hex[:8]}-{safe_name}"

    s3_client = boto3.client("s3", region_name="us-east-1")
    url = s3_client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": AUDIO_BUCKET,
            "Key": s3_key,
            "ContentType": content_type,
            "ContentLength": file_size,
        },
        ExpiresIn=AUDIO_UPLOAD_EXPIRY,
    )

    return {"upload_url": url, "s3_key": s3_key}


@app.get("/vitals")
async def vitals(request: Request):
    """Fitbit-tier vitals — lightweight metrics you can glance at anytime.

    Returns: memory count, memory reach (how far back), orient latency,
    total moments in DB, token budget estimate. No full diagnostics —
    just the wrist-check version.

    Gated behind DEBUG_ENDPOINTS env var.
    """
    if not DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404)
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

        # Schema v1.0: count reflections + commitments (replaces memories count)
        cur.execute("SELECT COUNT(*) FROM reflections")
        total_reflections = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM commitments")
        total_commitments = cur.fetchone()[0]
        total_memories = total_reflections + total_commitments

        # Total episodes (replaces moments count)
        cur.execute("SELECT COUNT(*) FROM episodes")
        total_episodes = cur.fetchone()[0]

        # Memory reach — oldest/newest episode
        cur.execute("""
            SELECT MIN(COALESCE(occurred_at, created_at))::date,
                   MAX(COALESCE(occurred_at, created_at))::date
            FROM episodes
        """)
        row = cur.fetchone()
        oldest_episode = str(row[0]) if row[0] else None
        newest_episode = str(row[1]) if row[1] else None

        # Memory reach in days
        if oldest_episode:
            from datetime import date

            reach_days = (date.today() - date.fromisoformat(oldest_episode)).days
        else:
            reach_days = 0

        # Orient latency test
        t0 = _time.time()
        from memory import orient

        orient_result = orient()
        orient_ms = round((_time.time() - t0) * 1000, 1)
        orient_chars = len(orient_result)

        # Episodes with embeddings vs without
        cur.execute("SELECT COUNT(*) FROM episodes WHERE embedding IS NOT NULL")
        episodes_with_embeddings = cur.fetchone()[0]

        # Episodes with transcripts
        cur.execute("SELECT COUNT(*) FROM episodes WHERE transcript_excerpt IS NOT NULL")
        episodes_with_transcripts = cur.fetchone()[0]

        # Duplicate check — group by title + date so recurring titles
        # on different days ("morning check-in") aren't false positives
        cur.execute("""
            SELECT title, COUNT(*) as n FROM episodes
            GROUP BY title, occurred_at::date HAVING COUNT(*) > 1
        """)
        duplicates = [{"title": row[0], "count": row[1]} for row in cur.fetchall()]

        cur.close()
        conn.close()

        now_utc = datetime.now(UTC)
        now_et = datetime.now(ZoneInfo("America/New_York"))

        return JSONResponse(
            {
                "total_memories": total_memories,
                "total_reflections": total_reflections,
                "total_commitments": total_commitments,
                "total_episodes": total_episodes,
                "episodes_with_embeddings": episodes_with_embeddings,
                "episodes_with_transcripts": episodes_with_transcripts,
                "memory_reach": {
                    "oldest": oldest_episode,
                    "newest": newest_episode,
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

    Gated behind DEBUG_ENDPOINTS env var.

    Optional query params:
        ?query=some+text  — also run semantic recall against this query
    """
    if not DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404)

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


def execute_recall_tool(tool_name: str, tool_input: dict, response_text: str = "") -> str:
    """Execute a recall tool and return the result as a string.

    Args:
        tool_name: Name of the tool to execute.
        tool_input: Tool input parameters from the model.
        response_text: The last text block from the model's response in this turn.
            Used by save_draft to capture only the draft content, excluding
            any conversational preamble from earlier text blocks.
    """
    from memory import recall, recall_memories

    if tool_name == "recall_memory":
        from memory import generate_embedding

        query = tool_input.get("query", "")
        limit = min(tool_input.get("limit", 3), 5)

        # Generate embedding once for both searches
        query_embedding = generate_embedding(query)
        if not query_embedding:
            return "Embedding generation failed — Voyage AI may be unavailable."

        # Search both tables — moments (scenes) and memories (roam, bridge logs)
        moment_results = recall(query, limit=limit, precomputed_embedding=query_embedding)
        memory_results = recall_memories(query, limit=2, precomputed_embedding=query_embedding)

        if not moment_results and not memory_results:
            return "No matching moments or memories found for that query."

        lines = []

        # Format moments (scenes from chat)
        if moment_results:
            lines.append("## Moments (scenes)")
            for r in moment_results:
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

        # Format memories (roam observations, bridge logs, reflections, etc.)
        if memory_results:
            lines.append("## Memories (roam, bridge logs, reflections)")
            for r in memory_results:
                created = r.get("created_at", "unknown")
                if hasattr(created, "strftime"):
                    date_str = created.strftime("%b %d, %I:%M %p")
                else:
                    date_str = str(created)
                sim = r.get("similarity", 0)
                agent = r.get("agent_id", "unknown")
                mtype = r.get("memory_type", "unknown")
                lines.append(f"### [{mtype}] from {agent} ({date_str}, similarity: {sim:.2f})")
                lines.append(r.get("content", ""))
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

    elif tool_name == "list_drafts":
        from memory import list_drafts

        status = tool_input.get("status", "active")
        drafts = list_drafts(status=status)
        if not drafts:
            return f"No {status} drafts found."
        lines = [f"## {status.title()} Drafts ({len(drafts)})\n"]
        for d in drafts:
            created = d.get("created_at", "unknown")
            if hasattr(created, "strftime"):
                date_str = created.strftime("%b %d, %I:%M %p")
            else:
                date_str = str(created)
            lines.append(f"### {d['title']} (rev {d['revision']}, {date_str})")
            lines.append(f"**Draft ID:** `{d['draft_id']}`")
            lines.append(f"**Preview:** {d['preview']}...")
            lines.append("")
        return "\n".join(lines)

    elif tool_name == "read_draft":
        from memory import read_draft

        draft_id = tool_input.get("draft_id", "")
        draft = read_draft(draft_id)
        if not draft:
            return f"No draft found with id '{draft_id}'."
        created = draft.get("created_at", "unknown")
        if hasattr(created, "strftime"):
            date_str = created.strftime("%b %d, %I:%M %p")
        else:
            date_str = str(created)
        lines = [
            f"# {draft['title']}",
            f"*Rev {draft['revision']} | {date_str} | by {draft.get('agent_id', 'unknown')}*\n",
            draft.get("content", ""),
        ]
        if draft.get("what_is_alive"):
            lines.append(f"\n---\n**What's alive:** {draft['what_is_alive']}")
        if draft.get("what_is_stuck"):
            lines.append(f"**What's stuck:** {draft['what_is_stuck']}")
        return "\n".join(lines)

    elif tool_name == "save_draft":
        from memory import write_draft

        title = tool_input.get("title", "Untitled")
        content = response_text.strip()
        if not content:
            return "No text to save — write the draft as text in your response before calling save_draft."
        result = write_draft(
            title=title,
            content=content,
            what_is_alive=tool_input.get("what_is_alive", ""),
            what_is_stuck=tool_input.get("what_is_stuck", ""),
        )
        if not result:
            return "Failed to save draft."
        return (
            f"Draft saved: **{title}**\n"
            f"Draft ID: `{result['draft_id']}`\n"
            f"Created: {result['created_at']}\n"
            f"Content length: {len(content)} chars"
        )

    elif tool_name == "revise_draft":
        from memory import revise_draft

        draft_id = tool_input.get("draft_id", "")
        content = tool_input.get("content", "")
        result = revise_draft(
            draft_id=draft_id,
            content=content,
            title=tool_input.get("title"),
            what_is_alive=tool_input.get("what_is_alive"),
            what_is_stuck=tool_input.get("what_is_stuck"),
            status=tool_input.get("status"),
        )
        if not result:
            return f"Failed to revise draft '{draft_id}' — not found or DB error."
        return (
            f"Draft revised: rev {result['revision']}\n"
            f"Draft ID: `{result['draft_id']}`\n"
            f"Created: {result['created_at']}"
        )

    elif tool_name == "recall_graph":
        mode = tool_input.get("mode", "explore")
        query = tool_input.get("query", "")
        try:
            from graph_recall import (
                find_connected_entities,
                find_related_memories,
                format_graph_context,
                get_entity_neighborhood,
                graph_available,
            )

            if not graph_available():
                return "Graph recall is not available — Neo4j is not connected."

            if mode == "entity":
                neighborhood = get_entity_neighborhood(query)
                if not neighborhood:
                    return f"No entity found matching '{query}'."
                e = neighborhood["entity"]
                labels = [lbl for lbl in (e.get("labels") or []) if lbl not in ("Entity", "BaseNode")]
                lines = [f"### {e['name']} ({', '.join(labels)})"]
                if e.get("description"):
                    lines.append(e["description"])
                if neighborhood["related_entities"]:
                    lines.append("\n**Connected entities:**")
                    for rel_ent in neighborhood["related_entities"]:
                        rel_labels = [lbl for lbl in (rel_ent.get("labels") or []) if lbl not in ("Entity", "BaseNode")]
                        lines.append(f"- {rel_ent['name']} ({', '.join(rel_labels)}) — {rel_ent['relationship']}")
                if neighborhood["memories"]:
                    lines.append("\n**Memories mentioning this entity:**")
                    for mem in neighborhood["memories"]:
                        content = (mem.get("content") or "")[:200]
                        role = mem.get("role") or "memory"
                        lines.append(f"- ({role}) {content}")
                return "\n".join(lines)

            else:  # explore mode
                # Generate embedding once, pass to both (avoid 2x Voyage calls)
                from graph_recall import _resolve_embedding

                explore_embedding = _resolve_embedding(query)
                entities = find_connected_entities(query, limit=5, precomputed_embedding=explore_embedding)
                related = find_related_memories(query, limit=5, precomputed_embedding=explore_embedding)
                if not entities and not related:
                    return f"No graph connections found for '{query}'."
                return format_graph_context(entities, related)

        except ImportError:
            return "Graph recall module not available."
        except Exception as e:
            print(f"[Chat] recall_graph failed: {e}")
            return f"Graph recall error: {e}"

    elif tool_name == "analyze_frequency":
        from memory import analyze_audio_frequency

        result = analyze_audio_frequency(
            tool_input.get("file_path", ""),
            tool_input.get("detail", "quick"),
        )
        return json.dumps(result, indent=2)

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
            cur.execute("SELECT COUNT(*) FROM episodes")
            episodes = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM reflections")
            reflections = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM commitments")
            commitments = cur.fetchone()[0]
            cur.close()
            lines.append(f"**Episodes:** {episodes}")
            lines.append(f"**Reflections:** {reflections} | **Commitments:** {commitments}")
        except Exception as e:
            print(f"[Chat] check_vitals stats failed: {e}")
            lines.append("*(Memory stats unavailable)*")
        finally:
            if conn:
                conn.close()
        # Neo4j graph status
        try:
            from graph_recall import graph_available

            lines.append(f"**Graph recall (Neo4j):** {'connected' if graph_available() else 'not available'}")
        except ImportError:
            lines.append("**Graph recall (Neo4j):** module not installed")
        return "\n".join(lines)

    return f"Unknown tool: {tool_name}"


@app.post("/chat")
@limiter.limit("30/minute")
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
    # Persist user message off the event loop — sync psycopg2 would block
    # between request parse and first SSE byte otherwise.
    # Uses wait_for with a 5s timeout so a hanging DB can't block the chat response.
    try:
        from persistence import persist_message as _persist

        if messages and messages[-1]["role"] == "user":
            user_content = messages[-1]["content"]
            # Guard: content must be a string (future-proof against content blocks)
            if not isinstance(user_content, str):
                user_content = json.dumps(user_content)
            await asyncio.wait_for(
                asyncio.to_thread(_persist, role="user", content=user_content),
                timeout=5,
            )
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

    # Warmup runs inside _run_api_call so the client gets a status event
    # before the wait (moved from here to the SSE stream for UX)
    is_first_message = len(messages) == 1 and messages[0]["role"] == "user"

    print(f"[Chat] {messages[-1]['content'][:80]}..." + (" [DEBUG]" if debug_mode else ""))

    # --- Mobile-resilient streaming ---
    # Architecture: API call runs as a background task that pushes SSE events
    # onto a queue. The generator drains the queue to the client. If the client
    # disconnects (tab switch on mobile, lock screen), the generator dies but
    # the background task keeps running to completion and persists the response.
    # Client recovers the completed response via loadHistory on visibility change.

    async def _run_api_call(event_queue: asyncio.Queue):
        """Run the full Anthropic API call, pushing SSE events to the queue.
        Continues to completion even if the client disconnects."""
        nonlocal system_prompt
        _chat_state["active_count"] += 1
        t0 = time.time()

        try:
            # MRI debug mode: emit diagnostics before the LLM response starts
            if debug_diagnostics:
                await event_queue.put(
                    f"data: {json.dumps({'type': 'debug_orient', **debug_diagnostics.get('orient', {})})}\n\n"
                )
                await event_queue.put(
                    f"data: {json.dumps({'type': 'debug_recall', **debug_diagnostics.get('recall', {})})}\n\n"
                )
                await event_queue.put(
                    f"data: {json.dumps({'type': 'debug_summary', 'total_prompt_chars': debug_diagnostics.get('total_prompt_chars', 0), 'total_prompt_tokens_est': debug_diagnostics.get('total_prompt_tokens_est', 0)})}\n\n"
                )

            # Layer 2: Warmup on first message — runs inside the SSE stream
            # so the client gets an immediate status event during the wait
            if is_first_message and WARMUP_ENABLED:
                await event_queue.put(f"data: {json.dumps({'type': 'status', 'text': 'waking up...'})}\n\n")
                print("[Warmup] First message detected — generating warmup...")
                warmup_output = await asyncio.to_thread(generate_warmup, system_prompt)
                if warmup_output:
                    system_prompt += (
                        "\n\n---\n\n# Warmup (private pre-conversation thoughts)\n\n"
                        "The following are your private thoughts from before this conversation "
                        "started. You generated them after reviewing your memory orientation. "
                        "They represent your genuine internal state — use them to inform your "
                        "response but don't quote them directly or reference that you had a "
                        "'warmup step.' Just be the person who already thought these things.\n\n"
                        f"{warmup_output}"
                    )
                    print(f"[Warmup] Injected {len(warmup_output)} chars into system prompt")
                else:
                    print("[Warmup] No output generated (skipped or failed)")

            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "accept": "text/event-stream",
            }

            # Retryable status codes: overloaded (529), rate limited (429),
            # server errors (500, 502, 503). Do NOT retry: auth (401), bad
            # request (400), not found (404) — those won't get better.
            RETRYABLE_STATUSES = {429, 500, 502, 503, 529}

            current_model = model  # Track which model we're using for fallback

            payload = {
                "model": current_model,
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
                cache_read_total = 0
                cache_create_total = 0

                all_tool_calls = []  # Accumulates across all tool rounds for persistence
                all_tool_results = []  # Accumulates tool_result blocks for persistence
                all_thinking_text = []  # Accumulates thinking across all tool rounds

                while True:
                    # --- Retry loop with exponential backoff ---
                    retry_succeeded = False
                    for attempt in range(MAX_API_RETRIES + 1):
                        try:
                            async with (
                                httpx.AsyncClient(timeout=300) as api_client,
                                api_client.stream("POST", ANTHROPIC_API_URL, json=payload, headers=headers) as resp,
                            ):
                                if resp.status_code != 200:
                                    error_body = await resp.aread()
                                    error_msg = error_body.decode("utf-8", errors="replace")[:500]

                                    if resp.status_code in RETRYABLE_STATUSES and attempt < MAX_API_RETRIES:
                                        delay = RETRY_BASE_DELAY * (2**attempt)
                                        print(
                                            f"[Chat] API error {resp.status_code} (attempt {attempt + 1}/{MAX_API_RETRIES + 1}), retrying in {delay}s: {error_msg[:100]}"
                                        )
                                        await event_queue.put(
                                            f"data: {json.dumps({'type': 'status', 'text': f'Retrying ({attempt + 1}/{MAX_API_RETRIES})...'})}\n\n"
                                        )
                                        await asyncio.sleep(delay)
                                        continue

                                    # Non-retryable or exhausted retries — try fallback model
                                    if resp.status_code in RETRYABLE_STATUSES and current_model != FALLBACK_MODEL:
                                        print(
                                            f"[Chat] Primary model exhausted retries, falling back to {FALLBACK_MODEL}"
                                        )
                                        current_model = FALLBACK_MODEL
                                        payload["model"] = current_model
                                        await event_queue.put(
                                            f"data: {json.dumps({'type': 'status', 'text': 'Switching to fallback model...'})}\n\n"
                                        )
                                        break  # Break retry loop to restart with fallback

                                    print(f"[Chat] API error {resp.status_code}: {error_msg}")
                                    await event_queue.put(
                                        f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                                    )
                                    await event_queue.put(None)
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
                                        # Upstream stalled — send keepalive
                                        await event_queue.put(": keepalive\n\n")
                                        continue
                                    except StopAsyncIteration:
                                        break

                                    event_count += 1

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
                                        cache_read_total += cache_read
                                        cache_create_total += cache_create
                                        if tool_round == 0:
                                            await event_queue.put(
                                                f"data: {json.dumps({'type': 'usage', 'input_tokens': input_tokens, 'cache_read_input_tokens': cache_read, 'cache_creation_input_tokens': cache_create})}\n\n"
                                            )

                                    elif event_type == "content_block_start":
                                        block = event.get("content_block", {})
                                        block_type = block.get("type", "")
                                        if block_type == "thinking":
                                            current_thinking_text = []
                                            if tool_round == 0:
                                                await event_queue.put(
                                                    f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                                                )
                                        elif block_type == "text":
                                            in_text_block = True
                                            current_block_text = []
                                            await event_queue.put(f"data: {json.dumps({'type': 'text_start'})}\n\n")
                                        elif block_type == "tool_use":
                                            current_tool_id = block.get("id")
                                            current_tool_name = block.get("name")
                                            current_tool_input_json = []
                                            # Tell the frontend we're recalling (id for pairing with recall_result)
                                            await event_queue.put(
                                                f"data: {json.dumps({'type': 'recall_start', 'tool': current_tool_name, 'id': current_tool_id})}\n\n"
                                            )

                                    elif event_type == "content_block_delta":
                                        delta = event.get("delta", {})
                                        delta_type = delta.get("type", "")
                                        if delta_type == "text_delta":
                                            text = delta.get("text", "")
                                            full_text.append(text)
                                            current_block_text.append(text)
                                            await event_queue.put(
                                                f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
                                            )
                                        elif delta_type == "thinking_delta":
                                            thinking = delta.get("thinking", "")
                                            current_thinking_text.append(thinking)
                                            if tool_round == 0:
                                                await event_queue.put(
                                                    f"data: {json.dumps({'type': 'thinking', 'text': thinking})}\n\n"
                                                )
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
                                                print(
                                                    f"[Chat] Tool input JSON parse failed: {parse_err} — raw: {raw[:200]}"
                                                )
                                                await event_queue.put(
                                                    f"data: {json.dumps({'type': 'status', 'text': f'Tool input parse error for {current_tool_name}, using empty input'})}\n\n"
                                                )
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
                                            # Accumulate thinking for persistence before resetting
                                            if current_thinking_text:
                                                all_thinking_text.extend(current_thinking_text)
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
                                        await event_queue.put(f"data: {json.dumps({'type': 'block_stop'})}\n\n")

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
                                        await event_queue.put(
                                            f"data: {json.dumps({'type': 'error', 'error': str(err)})}\n\n"
                                        )
                                        await event_queue.put(None)
                                        return

                            # Stream completed successfully — exit retry loop
                            retry_succeeded = True
                            break

                        except (httpx.TimeoutException, httpx.ConnectError) as e:
                            if attempt < MAX_API_RETRIES:
                                delay = RETRY_BASE_DELAY * (2**attempt)
                                print(
                                    f"[Chat] {type(e).__name__} (attempt {attempt + 1}/{MAX_API_RETRIES + 1}), retrying in {delay}s"
                                )
                                await event_queue.put(
                                    f"data: {json.dumps({'type': 'status', 'text': f'Connection issue, retrying ({attempt + 1}/{MAX_API_RETRIES})...'})}\n\n"
                                )
                                await asyncio.sleep(delay)
                                continue
                            # Exhausted retries — try fallback model
                            if current_model != FALLBACK_MODEL:
                                print(f"[Chat] {type(e).__name__} exhausted retries, falling back to {FALLBACK_MODEL}")
                                current_model = FALLBACK_MODEL
                                payload["model"] = current_model
                                await event_queue.put(
                                    f"data: {json.dumps({'type': 'status', 'text': 'Switching to fallback model...'})}\n\n"
                                )
                                break  # Break retry loop to restart with fallback
                            # Fallback model also failed
                            print(f"[Chat] {type(e).__name__} after all retries including fallback")
                            await event_queue.put(
                                f"data: {json.dumps({'type': 'error', 'error': f'Request failed after all retries: {type(e).__name__}'})}\n\n"
                            )
                            await event_queue.put(None)
                            return

                    # If retry loop ended without success (fallback model selected),
                    # restart the tool-round loop so we re-enter the retry loop with
                    # the new model.
                    if not retry_succeeded:
                        continue

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
                        await event_queue.put(": keepalive\n\n")

                        # Extract the last text block for save_draft — only the draft
                        # content, not conversational preamble from earlier text blocks.
                        last_text_block = ""
                        for block in reversed(content_blocks):
                            if block.get("type") == "text":
                                last_text_block = block["text"]
                                break

                        # Execute tools and build tool results
                        tool_results = []
                        for tc in tool_calls:
                            try:
                                result_text = await asyncio.to_thread(
                                    execute_recall_tool,
                                    tc["name"],
                                    tc["input"],
                                    last_text_block,
                                )
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
                            await event_queue.put(
                                f"data: {json.dumps({'type': 'recall_result', 'tool': tc['name'], 'id': tc['id'], 'query': tc['input'].get('query', tc['input'].get('title', ''))})}\n\n"
                            )

                        # Accumulate tool calls AND results for persistence before resetting
                        all_tool_calls.extend(tool_calls)
                        all_tool_results.extend(tool_results)
                        # Accumulate thinking text before the per-round reset
                        if current_thinking_text:
                            all_thinking_text.extend(current_thinking_text)

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

                # --- Persist assistant response to DB (off event loop) ---
                try:
                    from persistence import persist_message as _persist

                    tool_blocks_persist = []
                    for tc in all_tool_calls:
                        tool_blocks_persist.append({"type": "tool_use", "name": tc["name"], "input": tc["input"]})
                    for tr in all_tool_results:
                        tool_blocks_persist.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.get("tool_use_id"),
                                "content": tr.get("content", ""),
                            }
                        )
                    # Capture final round's thinking — only if we exited via the no-tool-use
                    # path. The tool-use path already accumulated in the loop body.
                    if current_thinking_text and stop_reason != "tool_use":
                        all_thinking_text.extend(current_thinking_text)
                    usage_metadata = {
                        "token_usage": {
                            "input_tokens": input_tokens_total,
                            "output_tokens": output_tokens_total,
                            "total_tokens": total_tokens,
                            "cache_read_input_tokens": cache_read_total,
                            "cache_creation_input_tokens": cache_create_total,
                            "context_pct": context_pct,
                            "tool_rounds": tool_round,
                        }
                    }
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            _persist,
                            role="assistant",
                            content=response_text,
                            tool_blocks=tool_blocks_persist if tool_blocks_persist else None,
                            thinking="".join(all_thinking_text) if all_thinking_text else None,
                            metadata=usage_metadata,
                        ),
                        timeout=5,
                    )
                except Exception as persist_err:
                    print(f"[Persistence] Assistant message persist failed (non-fatal): {persist_err}")

                await event_queue.put(
                    f"data: {json.dumps({'type': 'usage_final', 'input_tokens': input_tokens_total, 'output_tokens': output_tokens_total, 'total_tokens': total_tokens, 'cache_read_input_tokens': cache_read_total, 'cache_creation_input_tokens': cache_create_total, 'context_pct': context_pct, 'tool_rounds': tool_round})}\n\n"
                )
                await event_queue.put(f"data: {json.dumps({'type': 'done'})}\n\n")

            except Exception as e:
                print(f"[Chat] Unexpected error: {type(e).__name__}: {e}")
                await event_queue.put(f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n")
        finally:
            _chat_state["active_count"] = max(0, _chat_state["active_count"] - 1)
            await event_queue.put(None)  # Sentinel: stream complete

    async def stream_response():
        """Thin generator that drains the event queue to the client.
        If the client disconnects, this generator dies but _run_api_call
        keeps running in the background to complete persistence."""
        event_queue = asyncio.Queue()
        task = asyncio.create_task(_run_api_call(event_queue))
        # Strong reference prevents GC before task completes (Python 3.12+)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        if len(_background_tasks) > 3:
            print(
                f"[Chat] WARNING: {len(_background_tasks)} concurrent background tasks — "
                f"possible double-send or reconnect storm"
            )
        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event
        finally:
            # Client disconnected (GeneratorExit) or we finished normally.
            # Do NOT cancel the task — let it complete for persistence.
            pass

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
