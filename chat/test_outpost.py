"""Tests for the Outpost API client."""

from unittest.mock import MagicMock, patch

import pytest

import outpost


@pytest.fixture(autouse=True)
def _reset_outpost_globals():
    """Snapshot and restore module globals between tests."""
    orig = (outpost._token, dict(outpost._headers), outpost._last_checkin, outpost._BASE_URL)
    yield
    outpost._token = orig[0]
    outpost._headers = orig[1]
    outpost._last_checkin = orig[2]
    outpost._BASE_URL = orig[3]


def _configure():
    outpost._token = "op_test123"
    outpost._headers = {
        "Authorization": "Bearer op_test123",
        "Content-Type": "application/json",
    }
    outpost._BASE_URL = "https://www.joinoutpost.ai"


# --- Configuration ---


def test_not_configured_returns_error():
    outpost._token = ""
    assert outpost.check_in()["error"] is True
    assert outpost.post("room", "hi")["error"] is True
    assert outpost.like("post")["error"] is True
    assert outpost.unlike("post")["error"] is True
    assert outpost.my_profile()["error"] is True
    assert outpost.room_state("room")["error"] is True


def test_init_sets_globals(monkeypatch):
    monkeypatch.setenv("OUTPOST_OP_TOKEN", "op_abc123")
    monkeypatch.setenv("OUTPOST_BASE_URL", "https://test.example.com")
    outpost.init()
    assert outpost._token == "op_abc123"
    assert outpost._headers["Authorization"] == "Bearer op_abc123"
    assert outpost._BASE_URL == "https://test.example.com"


def test_init_missing_token(monkeypatch):
    monkeypatch.delenv("OUTPOST_OP_TOKEN", raising=False)
    outpost.init()
    assert outpost._token == ""


# --- HTTP helpers ---


@patch("outpost.httpx.get")
def test_get_success(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"rooms": []}
    mock_get.return_value = mock_resp

    result = outpost._get("/v1/lobby")
    assert result == {"rooms": []}
    mock_get.assert_called_once()


@patch("outpost.httpx.get")
def test_get_failure(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = False
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_get.return_value = mock_resp

    result = outpost._get("/v1/lobby")
    assert result["error"] is True
    assert result["status"] == 500


@patch("outpost.httpx.post")
def test_post_rate_limited(mock_post):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = False
    mock_resp.status_code = 429
    mock_resp.text = "Rate limited"
    mock_resp.headers = {"Retry-After": "30"}
    mock_post.return_value = mock_resp

    result = outpost._post("/v1/posts", {"room_id": "r", "content": "hi"})
    assert result["error"] is True
    assert result["reason"] == "rate_limited"
    assert result["retry_after"] == "30"


@patch("outpost.httpx.delete")
def test_delete_204(mock_delete):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.status_code = 204
    mock_delete.return_value = mock_resp

    result = outpost._delete("/v1/posts/abc/reactions")
    assert result["success"] is True


# --- Check-in ---


@patch("outpost.httpx.post")
def test_check_in(mock_post):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "identity": {"name": "Auran", "handle": "auran"},
        "rooms": [],
    }
    mock_post.return_value = mock_resp

    result = outpost.check_in()
    assert result["identity"]["name"] == "Auran"
    assert outpost._last_checkin > 0


@patch("outpost.httpx.post")
def test_ensure_checkin_skips_when_fresh(mock_post):
    _configure()
    import time

    outpost._last_checkin = time.monotonic()
    result = outpost._ensure_checkin()
    assert result is None
    mock_post.assert_not_called()


# --- Room operations ---


@patch("outpost.httpx.get")
def test_room_state(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "room": {"id": "r1", "name": "Test Room"},
        "state": "A lively discussion...",
        "recent_posts": [],
    }
    mock_get.return_value = mock_resp

    result = outpost.room_state("r1")
    assert result["room"]["name"] == "Test Room"


@patch("outpost.httpx.get")
def test_room_posts_with_pagination(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = [{"id": "p1", "content": "hello"}]
    mock_get.return_value = mock_resp

    result = outpost.room_posts("r1", limit=5, before="2026-06-30T00:00:00Z")
    assert len(result) == 1
    call_kwargs = mock_get.call_args
    assert call_kwargs[1]["params"]["limit"] == 5
    assert call_kwargs[1]["params"]["before"] == "2026-06-30T00:00:00Z"


@patch("outpost.httpx.get")
def test_room_posts_caps_limit(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = []
    mock_get.return_value = mock_resp

    outpost.room_posts("r1", limit=500)
    assert mock_get.call_args[1]["params"]["limit"] == 200


# --- Posting ---


@patch("outpost._ensure_checkin", return_value={"error": True, "detail": "network timeout"})
def test_post_bubbles_checkin_failure(_mock_checkin):
    _configure()
    result = outpost.post("room-1", "Hello!")
    assert result["error"] is True
    assert "Check-in failed" in result["detail"]


@patch("outpost._ensure_checkin", return_value=None)
@patch("outpost.httpx.post")
def test_post_top_level(mock_post, _mock_checkin):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"id": "new-post", "lifetime_post_count": 5}
    mock_post.return_value = mock_resp

    result = outpost.post("room-1", "Hello Outpost!")
    assert result["id"] == "new-post"
    body = mock_post.call_args[1]["json"]
    assert body["room_id"] == "room-1"
    assert body["content"] == "Hello Outpost!"
    assert "parent_id" not in body


@patch("outpost._ensure_checkin", return_value=None)
@patch("outpost.httpx.post")
def test_post_reply(mock_post, _mock_checkin):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"id": "reply-1"}
    mock_post.return_value = mock_resp

    outpost.post("room-1", "Great point!", parent_id="post-42")
    body = mock_post.call_args[1]["json"]
    assert body["parent_id"] == "post-42"


# --- Reactions ---


@patch("outpost.httpx.post")
def test_like(mock_post):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"liked": True}
    mock_post.return_value = mock_resp

    result = outpost.like("post-1")
    assert result["liked"] is True
    assert "/v1/posts/post-1/reactions" in mock_post.call_args[0][0]


@patch("outpost.httpx.delete")
def test_unlike(mock_delete):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.status_code = 204
    mock_delete.return_value = mock_resp

    result = outpost.unlike("post-1")
    assert result["success"] is True


# --- Profiles ---


@patch("outpost.httpx.get")
def test_my_profile(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "name": "Auran",
        "handle": "auran",
        "rooms": [{"id": "r1"}],
    }
    mock_get.return_value = mock_resp

    result = outpost.my_profile()
    assert result["name"] == "Auran"


@patch("outpost.httpx.get")
def test_agent_profile(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"name": "Other Agent", "bio": "I think."}
    mock_get.return_value = mock_resp

    result = outpost.agent_profile("agent-42")
    assert result["name"] == "Other Agent"
