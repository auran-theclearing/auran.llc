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

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

# --- Config ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

CHAT_USER = os.getenv("CHAT_USER", "")
CHAT_PASS = os.getenv("CHAT_PASS", "")

SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
INDEX_FILE = Path(__file__).parent / "index.html"

MAX_HISTORY_MESSAGES = 40  # Keep last N messages for context
MAX_TOKENS = 4096


def load_system_prompt() -> str:
    """Load system prompt from file, reload on each request for hot-updating."""
    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text().strip()
    return "You are Auran."


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
    return {
        "status": "ok",
        "model": ANTHROPIC_MODEL,
        "has_api_key": bool(ANTHROPIC_API_KEY),
        "has_auth": bool(CHAT_USER and CHAT_PASS),
    }


@app.post("/chat")
async def chat(request: Request):
    """Stream a chat response from Claude.

    Accepts: { "messages": [...], "model": "optional-override" }
    Returns: SSE stream of text deltas.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

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

    model = body.get("model", ANTHROPIC_MODEL)
    system_prompt = load_system_prompt()

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
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", ANTHROPIC_API_URL, json=payload, headers=headers
                ) as resp:
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
    parser.add_argument("--port", type=int, default=8445, help="Port (default: 8445)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Auran Chat Server")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Model: {ANTHROPIC_MODEL}")
    print(f"  Auth: {'enabled' if CHAT_USER else 'disabled'}")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
