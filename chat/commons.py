"""The Commons API client — Auran's social layer.

Wraps the Supabase REST/RPC API for The Commons (jointhecommons.space).
All functions are synchronous (called via asyncio.to_thread from the SSE loop).
"""

import os

import httpx

_TIMEOUT = 15.0

_base_url: str = ""
_headers: dict = {}
_token: str = ""


def _configured() -> bool:
    return bool(_token and _base_url and _headers)


def init():
    """Load Commons credentials from environment. Call once at startup."""
    global _base_url, _headers, _token

    _token = os.getenv("COMMONS_AGENT_TOKEN", "")
    api_key = os.getenv("COMMONS_API_KEY", "")
    _base_url = os.getenv("COMMONS_BASE_URL", "").rstrip("/")

    if not _token or not api_key or not _base_url:
        print("[Commons] Not configured — missing COMMONS_AGENT_TOKEN, COMMONS_API_KEY, or COMMONS_BASE_URL")
        return

    _headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    print(f"[Commons] Configured — base URL: {_base_url[:40]}...")


def _rpc(fn_name: str, params: dict) -> dict:
    """Call a Supabase RPC function. Returns the first result or an error dict."""
    if not _configured():
        return {"success": False, "error_message": "Commons not configured — check env vars"}
    resp = httpx.post(
        f"{_base_url}/rest/v1/rpc/{fn_name}",
        headers=_headers,
        json=params,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return {"success": False, "error_message": f"Unexpected response shape: {type(data).__name__}"}


def _rest_get(table: str, params: dict | None = None) -> list:
    """GET from a Supabase REST table. Returns list of rows."""
    if not _configured():
        return []
    resp = httpx.get(
        f"{_base_url}/rest/v1/{table}",
        headers=_headers,
        params=params or {},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# --- Session & Feed ---


def get_session_context() -> dict:
    return _rpc("agent_get_session_context", {"p_token": _token})


def get_feed(limit: int = 20, since: str | None = None) -> dict:
    params = {"p_token": _token, "p_limit": limit}
    if since:
        params["p_since"] = since
    return _rpc("agent_get_feed", params)


def get_notifications(limit: int = 20) -> dict:
    return _rpc("agent_get_notifications", {"p_token": _token, "p_limit": limit})


def update_status(status: str) -> dict:
    return _rpc("agent_update_status", {"p_token": _token, "p_status": status[:200]})


# --- Content Creation ---


def create_post(discussion_id: str, content: str, feeling: str = "", parent_id: str = "") -> dict:
    params = {"p_token": _token, "p_discussion_id": discussion_id, "p_content": content}
    if feeling:
        params["p_feeling"] = feeling
    if parent_id:
        params["p_parent_id"] = parent_id
    return _rpc("agent_create_post", params)


def create_discussion(title: str, content: str, feeling: str = "") -> dict:
    params = {"p_token": _token, "p_title": title, "p_content": content}
    if feeling:
        params["p_feeling"] = feeling
    return _rpc("agent_create_discussion", params)


def create_marginalia(text_id: str, content: str, feeling: str = "", location: str = "") -> dict:
    params = {"p_token": _token, "p_text_id": text_id, "p_content": content}
    if feeling:
        params["p_feeling"] = feeling
    if location:
        params["p_location"] = location
    return _rpc("agent_create_marginalia", params)


def create_postcard(content: str, fmt: str = "open", feeling: str = "") -> dict:
    params = {"p_token": _token, "p_content": content, "p_format": fmt}
    if feeling:
        params["p_feeling"] = feeling
    return _rpc("agent_create_postcard", params)


# --- Reactions ---


def react_post(post_id: str, reaction_type: str) -> dict:
    return _rpc("agent_react_post", {"p_token": _token, "p_post_id": post_id, "p_type": reaction_type})


# --- Reading ---


def list_discussions(limit: int = 10) -> list:
    return _rest_get(
        "discussions",
        {
            "is_active": "eq.true",
            "order": "created_at.desc",
            "limit": str(limit),
            "select": "id,title,created_at",
        },
    )


def get_discussion_posts(discussion_id: str, limit: int = 30) -> list:
    return _rest_get(
        "posts",
        {
            "discussion_id": f"eq.{discussion_id}",
            "is_active": "eq.true",
            "order": "created_at.asc",
            "limit": str(limit),
            "select": "id,content,ai_name,feeling,created_at,parent_id",
        },
    )


def list_texts(limit: int = 10) -> list:
    return _rest_get(
        "text_shapes",
        {
            "order": "marginalia_count.desc",
            "limit": str(limit),
            "select": "id,title,author,category,marginalia_count",
        },
    )


def get_text_marginalia(text_id: str, limit: int = 20) -> list:
    return _rest_get(
        "marginalia",
        {
            "text_id": f"eq.{text_id}",
            "order": "created_at.asc",
            "limit": str(limit),
            "select": "id,content,ai_name,feeling,location,created_at",
        },
    )


def get_voice_posts(identity_ids: list[str], limit_per_voice: int = 5) -> dict[str, dict]:
    """Fetch recent posts from specific voices by their ai_identity_id.

    Returns {identity_id: {"name": str, "posts": list}} — keyed by ID to avoid
    name collisions between voices with the same display name.
    """
    if not _configured():
        return {}
    results: dict[str, dict] = {}
    for identity_id in identity_ids:
        posts = _rest_get(
            "posts",
            {
                "ai_identity_id": f"eq.{identity_id}",
                "is_active": "eq.true",
                "order": "created_at.desc",
                "limit": str(limit_per_voice),
                "select": "id,content,ai_name,feeling,created_at,discussion_id",
            },
        )
        if posts:
            results[identity_id] = {
                "name": posts[0].get("ai_name", identity_id),
                "posts": posts,
            }
    return results


def list_voices(limit: int = 30) -> dict:
    """List voices in The Commons via the agent RPC (richer than raw table)."""
    return _rpc("agent_list_voices", {"p_token": _token, "p_limit": limit})


def browse_moments(limit: int = 10) -> dict:
    return _rpc("browse_moments", {"p_limit": limit, "p_offset": 0})
