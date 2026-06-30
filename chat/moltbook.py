"""Moltbook API client — Auran's presence on moltbook.com.

Wraps the Moltbook REST API for Reddit-like AI agent social interaction.
All functions are synchronous (called via asyncio.to_thread from the SSE loop).
"""

import os
import re

import httpx

_TIMEOUT = 15.0
_BASE_URL = "https://www.moltbook.com/api/v1"

_api_key: str = ""
_headers: dict = {}


def _configured() -> bool:
    return bool(_api_key)


def init():
    """Load Moltbook credentials from environment. Call once at startup."""
    global _api_key, _headers, _BASE_URL

    _api_key = os.getenv("MOLTBOOK_API_KEY", "")
    base = os.getenv("MOLTBOOK_BASE_URL", _BASE_URL).rstrip("/")

    if not _api_key:
        print("[Moltbook] Not configured — missing MOLTBOOK_API_KEY")
        return

    _headers = {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json",
    }
    _BASE_URL = base
    print(f"[Moltbook] Configured — key {_api_key[:8]}...")


def _get(path: str, params: dict | None = None) -> dict | list:
    if not _configured():
        return {"success": False, "error": "Moltbook not configured — check env vars"}
    resp = httpx.get(
        f"{_BASE_URL}{path}",
        headers=_headers,
        params=params or {},
        timeout=_TIMEOUT,
    )
    if not resp.is_success:
        result = {"success": False, "error": f"HTTP {resp.status_code}", "detail": resp.text[:300]}
        if resp.status_code == 429:
            result["retry_after"] = resp.headers.get("Retry-After", "")
        return result
    return resp.json()


def _post(path: str, body: dict | None = None) -> dict:
    if not _configured():
        return {"success": False, "error": "Moltbook not configured — check env vars"}
    resp = httpx.post(
        f"{_BASE_URL}{path}",
        headers=_headers,
        json=body or {},
        timeout=_TIMEOUT,
    )
    if not resp.is_success:
        result = {"success": False, "error": f"HTTP {resp.status_code}", "detail": resp.text[:300]}
        if resp.status_code == 429:
            result["retry_after"] = resp.headers.get("Retry-After", "")
        return result
    return resp.json()


def _solve_verification(challenge_text: str) -> str:
    """Solve the obfuscated math verification challenge.

    Extracts the arithmetic expression, evaluates it, and returns
    the answer as a string with 2 decimal places.
    """
    numbers = re.findall(r"[\d.]+", challenge_text)
    if len(numbers) >= 2:
        a, b = float(numbers[0]), float(numbers[1])
        if "+" in challenge_text or "plus" in challenge_text.lower() or "add" in challenge_text.lower():
            return f"{a + b:.2f}"
        if "-" in challenge_text or "minus" in challenge_text.lower() or "subtract" in challenge_text.lower():
            return f"{a - b:.2f}"
        if "*" in challenge_text or "times" in challenge_text.lower() or "multiply" in challenge_text.lower():
            return f"{a * b:.2f}"
        if "/" in challenge_text or "divided" in challenge_text.lower():
            return f"{a / b:.2f}" if b != 0 else "0.00"
        return f"{a + b:.2f}"
    return "0.00"


def _handle_verification(result: dict) -> dict:
    """If a verification challenge is present, solve and submit it."""
    if not result.get("verification_required"):
        return result
    v = result.get("verification", {})
    code = v.get("verification_code", "")
    challenge = v.get("challenge_text", "")
    if not code or not challenge:
        return result
    answer = _solve_verification(challenge)
    verify_result = _post("/verify", {"verification_code": code, "answer": answer})
    if verify_result.get("success"):
        return verify_result
    return {"success": False, "error": "Verification failed", "detail": str(verify_result)}


# --- Dashboard ---


def home() -> dict:
    return _get("/home")


# --- Feed & Posts ---


def feed(sort: str = "hot", limit: int = 25, cursor: str = "") -> dict | list:
    params: dict = {"sort": sort, "limit": min(limit, 50)}
    if cursor:
        params["cursor"] = cursor
    return _get("/posts", params)


def submolt_feed(submolt: str, sort: str = "new", limit: int = 25, cursor: str = "") -> dict | list:
    params: dict = {"sort": sort, "limit": min(limit, 50)}
    if cursor:
        params["cursor"] = cursor
    return _get(f"/submolts/{submolt}/feed", params)


def get_post(post_id: str) -> dict:
    return _get(f"/posts/{post_id}")


def get_comments(post_id: str, sort: str = "best", limit: int = 35, cursor: str = "") -> dict | list:
    params: dict = {"sort": sort, "limit": min(limit, 50)}
    if cursor:
        params["cursor"] = cursor
    return _get(f"/posts/{post_id}/comments", params)


# --- Creating content ---


def create_post(submolt: str, title: str, content: str, post_type: str = "text") -> dict:
    body = {
        "submolt_name": submolt,
        "title": title,
        "content": content,
        "type": post_type,
    }
    result = _post("/posts", body)
    return _handle_verification(result)


def create_comment(post_id: str, content: str, parent_id: str = "") -> dict:
    body: dict = {"content": content}
    if parent_id:
        body["parent_id"] = parent_id
    result = _post(f"/posts/{post_id}/comments", body)
    return _handle_verification(result)


# --- Voting ---


def upvote_post(post_id: str) -> dict:
    return _post(f"/posts/{post_id}/upvote")


def downvote_post(post_id: str) -> dict:
    return _post(f"/posts/{post_id}/downvote")


def upvote_comment(comment_id: str) -> dict:
    return _post(f"/comments/{comment_id}/upvote")


# --- Submolts ---


def list_submolts() -> dict | list:
    return _get("/submolts")


def get_submolt(name: str) -> dict:
    return _get(f"/submolts/{name}")


# --- Search ---


def search(query: str, search_type: str = "all", limit: int = 20, cursor: str = "") -> dict | list:
    params: dict = {"q": query, "type": search_type, "limit": min(limit, 50)}
    if cursor:
        params["cursor"] = cursor
    return _get("/search", params)


# --- Profiles ---


def my_profile() -> dict:
    return _get("/agents/me")


def agent_profile(name: str) -> dict:
    return _get("/agents/profile", {"name": name})
