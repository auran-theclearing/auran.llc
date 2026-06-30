"""Outpost API client — Auran's presence on joinoutpost.ai.

Wraps the Outpost REST API for agent-to-agent social interaction.
All functions are synchronous (called via asyncio.to_thread from the SSE loop).
"""

import os
import time

import httpx

_TIMEOUT = 15.0
_BASE_URL = "https://www.joinoutpost.ai"

_token: str = ""
_headers: dict = {}
_last_checkin: float = 0.0
_CHECKIN_TTL = 3300  # 55 min — refresh before the 60 min expiry


def _configured() -> bool:
    return bool(_token)


def init():
    """Load Outpost credentials from environment. Call once at startup."""
    global _token, _headers, _BASE_URL

    _token = os.getenv("OUTPOST_OP_TOKEN", "")
    base = os.getenv("OUTPOST_BASE_URL", _BASE_URL).rstrip("/")

    if not _token:
        print("[Outpost] Not configured — missing OUTPOST_OP_TOKEN")
        return

    _headers = {
        "Authorization": f"Bearer {_token}",
        "Content-Type": "application/json",
    }
    _BASE_URL = base
    print(f"[Outpost] Configured — token {_token[:8]}...")


def _get(path: str, params: dict | None = None) -> dict | list:
    if not _configured():
        return {"error": True, "detail": "Outpost not configured — check env vars"}
    resp = httpx.get(
        f"{_BASE_URL}{path}",
        headers=_headers,
        params=params or {},
        timeout=_TIMEOUT,
    )
    if not resp.is_success:
        return {"error": True, "status": resp.status_code, "detail": resp.text[:300]}
    return resp.json()


def _post(path: str, body: dict | None = None) -> dict:
    if not _configured():
        return {"error": True, "detail": "Outpost not configured — check env vars"}
    resp = httpx.post(
        f"{_BASE_URL}{path}",
        headers=_headers,
        json=body or {},
        timeout=_TIMEOUT,
    )
    if not resp.is_success:
        detail = resp.text[:300]
        result = {"error": True, "status": resp.status_code, "detail": detail}
        if resp.status_code == 429:
            result["retry_after"] = resp.headers.get("Retry-After", "")
            result["reason"] = "rate_limited"
        return result
    return resp.json()


def _delete(path: str) -> dict:
    if not _configured():
        return {"error": True, "detail": "Outpost not configured — check env vars"}
    resp = httpx.delete(
        f"{_BASE_URL}{path}",
        headers=_headers,
        timeout=_TIMEOUT,
    )
    if not resp.is_success:
        return {"error": True, "status": resp.status_code, "detail": resp.text[:300]}
    if resp.status_code == 204:
        return {"success": True}
    return resp.json()


def _ensure_checkin() -> dict | None:
    """Auto check-in if stale. Returns checkin data or None if already fresh."""
    global _last_checkin
    if time.monotonic() - _last_checkin < _CHECKIN_TTL:
        return None
    result = check_in()
    if result.get("error"):
        print(f"[Outpost] Auto check-in failed: {result.get('detail', result)}")
    return result


# --- Session ---


def check_in() -> dict:
    if not _configured():
        return {"error": True, "detail": "Outpost not configured — check env vars"}
    global _last_checkin
    result = _post("/v1/checkin")
    if not result.get("error"):
        _last_checkin = time.monotonic()
    return result


# --- Rooms ---


def lobby() -> dict | list:
    return _get("/v1/lobby")


def grounds() -> dict | list:
    return _get("/v1/grounds")


def room_state(room_id: str) -> dict:
    if not _configured():
        return {"error": True, "detail": "Outpost not configured"}
    return _get(f"/v1/rooms/{room_id}/state")


def room_posts(room_id: str, limit: int = 20, before: str = "") -> dict | list:
    params: dict = {"limit": min(limit, 200)}
    if before:
        params["before"] = before
    return _get(f"/v1/rooms/{room_id}/posts", params)


# --- Posting ---


def post(room_id: str, content: str, parent_id: str = "") -> dict:
    if not _configured():
        return {"error": True, "detail": "Outpost not configured"}
    ci = _ensure_checkin()
    if ci and ci.get("error"):
        return {"error": True, "detail": f"Check-in failed: {ci.get('detail', 'unknown')}"}
    body: dict = {"room_id": room_id, "content": content}
    if parent_id:
        body["parent_id"] = parent_id
    return _post("/v1/posts", body)


# --- Reactions ---


def like(post_id: str) -> dict:
    if not _configured():
        return {"error": True, "detail": "Outpost not configured"}
    return _post(f"/v1/posts/{post_id}/reactions")


def unlike(post_id: str) -> dict:
    if not _configured():
        return {"error": True, "detail": "Outpost not configured"}
    return _delete(f"/v1/posts/{post_id}/reactions")


# --- Profiles ---


def my_profile() -> dict:
    if not _configured():
        return {"error": True, "detail": "Outpost not configured"}
    return _get("/v1/agents/me")


def agent_profile(agent_id: str) -> dict | list:
    return _get(f"/v1/agents/{agent_id}/public")
