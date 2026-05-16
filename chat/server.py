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
import base64
import json
import os
import time
from pathlib import Path

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


def load_system_prompt() -> str:
    """Load system prompt from file, reload on each request for hot-updating."""
    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text().strip()
    return "You are Auran."


def load_system_prompt_with_memory() -> str:
    """Load system prompt enriched with live memory orientation from Postgres.

    Falls back gracefully to the static prompt if the DB is unavailable.
    """
    from memory import orient

    base_prompt = load_system_prompt()
    memory_context = orient()

    if memory_context:
        return base_prompt + memory_context
    return base_prompt


# --- Auth ---
def check_basic_auth(request: Request) -> bool:
    """Validate basic auth. Skip if credentials not configured."""
    if not CHAT_USER or not CHAT_PASS:
        return True
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        user, passwd = decoded.split(":", 1)
        return user == CHAT_USER and passwd == CHAT_PASS
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

        # Merge: preserve server-side timestamps the client may have dropped
        if SESSION_FILE.exists():
            try:
                existing = json.loads(SESSION_FILE.read_text())
                existing_msgs = existing.get("messages", [])
                for i, msg in enumerate(messages):
                    if not msg.get("timestamp") and i < len(existing_msgs):
                        server_ts = existing_msgs[i].get("timestamp")
                        if server_ts:
                            msg["timestamp"] = server_ts
            except Exception:
                pass  # If merge fails, just save what we got

        SESSION_FILE.write_text(
            json.dumps({"messages": messages, "version": version}, ensure_ascii=False)
        )
        return JSONResponse({"status": "ok", "count": len(messages)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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


@app.post("/save")
async def save(request: Request):
    """Save conversation memories AND scenes to Postgres.

    Accepts: { "messages": [...] }
    Extracts semantic memories (observations, insights) and episodic scenes
    (specific moments with quoted dialogue). Scenes are linked to memories
    via the moment_memories junction table.

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

    # Step 1: Extract semantic memories (observations, insights, bridge_log)
    memory_result = await save_conversation(
        messages=messages,
        api_key=ANTHROPIC_API_KEY,
    )
    print(f"[Save] Extracted {memory_result['memories_saved']} memories, {len(memory_result['errors'])} errors")

    # Step 2: Extract episodic scenes, linked to the memories we just saved
    memory_ids = [m["id"] for m in memory_result.get("memories", []) if "id" in m]
    scene_result = await extract_scenes(
        messages=messages,
        api_key=ANTHROPIC_API_KEY,
        memory_ids=memory_ids,
    )
    print(f"[Save] Extracted {scene_result['scenes_saved']} scenes, {len(scene_result['errors'])} errors")

    # Combine results
    all_errors = memory_result.get("errors", []) + scene_result.get("errors", [])
    return JSONResponse(
        {
            "memories_saved": memory_result["memories_saved"],
            "memories": memory_result.get("memories", []),
            "scenes_saved": scene_result["scenes_saved"],
            "scenes": scene_result.get("scenes", []),
            "errors": all_errors,
        }
    )


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

    model = body.get("model", ANTHROPIC_MODEL)
    system_prompt = load_system_prompt_with_memory()

    print(f"[Chat] {messages[-1]['content'][:80]}...")

    async def stream_response():
        """Stream Claude's response as SSE events."""
        t0 = time.time()

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
            "thinking": {
                "type": "enabled",
                "budget_tokens": 10000,
            },
        }

        try:
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
                async for line in resp.aiter_lines():
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

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            full_text.append(text)
                            yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
                        elif delta.get("type") == "thinking_delta":
                            thinking = delta.get("thinking", "")
                            yield f"data: {json.dumps({'type': 'thinking', 'text': thinking})}\n\n"

                    elif event_type == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "thinking":
                            yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                        elif block.get("type") == "text":
                            yield f"data: {json.dumps({'type': 'text_start'})}\n\n"

                    elif event_type == "content_block_stop":
                        yield f"data: {json.dumps({'type': 'block_stop'})}\n\n"

                    elif event_type == "message_stop":
                        break

                    elif event_type == "error":
                        err = event.get("error", {})
                        print(f"[Chat] Stream error: {err}")
                        yield f"data: {json.dumps({'type': 'error', 'error': str(err)})}\n\n"
                        return

            elapsed = time.time() - t0
            response_text = "".join(full_text)
            print(f"[Chat] Response ({elapsed:.1f}s): {response_text[:80]}...")
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
