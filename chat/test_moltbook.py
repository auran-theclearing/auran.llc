"""Tests for the Moltbook API client."""

from unittest.mock import MagicMock, patch

import pytest

import moltbook


@pytest.fixture(autouse=True)
def _reset_moltbook_globals():
    """Snapshot and restore module globals between tests."""
    orig = (moltbook._api_key, dict(moltbook._headers), moltbook._BASE_URL)
    yield
    moltbook._api_key = orig[0]
    moltbook._headers = orig[1]
    moltbook._BASE_URL = orig[2]


def _configure():
    moltbook._api_key = "mk_test123"
    moltbook._headers = {
        "Authorization": "Bearer mk_test123",
        "Content-Type": "application/json",
    }
    moltbook._BASE_URL = "https://www.moltbook.com/api/v1"


# --- Configuration ---


def test_not_configured_returns_error():
    moltbook._api_key = ""
    result = moltbook._get("/anything")
    assert result["success"] is False
    assert "not configured" in result["error"]

    result = moltbook._post("/anything")
    assert result["success"] is False


def test_init_sets_globals(monkeypatch):
    monkeypatch.setenv("MOLTBOOK_API_KEY", "mk_abc123")
    monkeypatch.setenv("MOLTBOOK_BASE_URL", "https://test.example.com/api/v1")
    moltbook.init()
    assert moltbook._api_key == "mk_abc123"
    assert moltbook._headers["Authorization"] == "Bearer mk_abc123"
    assert moltbook._BASE_URL == "https://test.example.com/api/v1"


def test_init_missing_key(monkeypatch):
    monkeypatch.delenv("MOLTBOOK_API_KEY", raising=False)
    moltbook.init()
    assert moltbook._api_key == ""


# --- HTTP helpers ---


@patch("moltbook.httpx.get")
def test_get_success(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"success": True, "data": []}
    mock_get.return_value = mock_resp

    result = moltbook._get("/posts")
    assert result["success"] is True


@patch("moltbook.httpx.get")
def test_get_rate_limited(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = False
    mock_resp.status_code = 429
    mock_resp.text = "Rate limited"
    mock_resp.headers = {"Retry-After": "45"}
    mock_get.return_value = mock_resp

    result = moltbook._get("/posts")
    assert result["success"] is False
    assert result["retry_after"] == "45"


@patch("moltbook.httpx.post")
def test_post_success(mock_post):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"success": True, "data": {"id": "p1"}}
    mock_post.return_value = mock_resp

    result = moltbook._post("/posts", {"title": "test"})
    assert result["success"] is True


# --- Verification ---


def test_solve_verification_addition():
    assert moltbook._solve_verification("What is 5 + 3?") == "8.00"
    assert moltbook._solve_verification("Add 10.5 plus 2.5") == "13.00"


def test_solve_verification_subtraction():
    assert moltbook._solve_verification("What is 10 - 3?") == "7.00"
    assert moltbook._solve_verification("Subtract 5 minus 2") == "3.00"


def test_solve_verification_multiplication():
    assert moltbook._solve_verification("What is 4 * 3?") == "12.00"
    assert moltbook._solve_verification("Multiply 6 times 7") == "42.00"


def test_solve_verification_division():
    assert moltbook._solve_verification("What is 10 / 2?") == "5.00"
    assert moltbook._solve_verification("15 divided by 3") == "5.00"


def test_solve_verification_division_by_zero():
    assert moltbook._solve_verification("10 / 0") == "0.00"


def test_solve_verification_fallback():
    assert moltbook._solve_verification("no numbers here") == "0.00"


@patch("moltbook._post")
def test_handle_verification_skips_when_not_required(mock_post):
    result = {"success": True, "data": {"id": "p1"}}
    assert moltbook._handle_verification(result) == result
    mock_post.assert_not_called()


@patch("moltbook._post")
def test_handle_verification_solves_challenge(mock_post):
    mock_post.return_value = {"success": True, "data": {"id": "p1"}}

    result = {
        "verification_required": True,
        "verification": {
            "verification_code": "moltbook_verify_abc",
            "challenge_text": "What is 5 + 3?",
        },
    }
    out = moltbook._handle_verification(result)
    assert out["success"] is True
    mock_post.assert_called_once_with("/verify", {"verification_code": "moltbook_verify_abc", "answer": "8.00"})


# --- Feed ---


@patch("moltbook.httpx.get")
def test_feed_default_params(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"data": [], "has_more": False}
    mock_get.return_value = mock_resp

    moltbook.feed()
    params = mock_get.call_args[1]["params"]
    assert params["sort"] == "hot"
    assert params["limit"] == 25


@patch("moltbook.httpx.get")
def test_feed_caps_limit(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"data": []}
    mock_get.return_value = mock_resp

    moltbook.feed(limit=100)
    assert mock_get.call_args[1]["params"]["limit"] == 50


@patch("moltbook.httpx.get")
def test_feed_with_cursor(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"data": []}
    mock_get.return_value = mock_resp

    moltbook.feed(cursor="abc123")
    assert mock_get.call_args[1]["params"]["cursor"] == "abc123"


# --- Posts ---


@patch("moltbook._handle_verification", side_effect=lambda x: x)
@patch("moltbook.httpx.post")
def test_create_post(mock_post, _mock_verify):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"success": True, "data": {"id": "new-post"}}
    mock_post.return_value = mock_resp

    moltbook.create_post("general", "Test Title", "Test content")
    body = mock_post.call_args[1]["json"]
    assert body["submolt_name"] == "general"
    assert body["title"] == "Test Title"
    assert body["type"] == "text"


@patch("moltbook._handle_verification", side_effect=lambda x: x)
@patch("moltbook.httpx.post")
def test_create_comment(mock_post, _mock_verify):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"success": True, "data": {"id": "c1"}}
    mock_post.return_value = mock_resp

    moltbook.create_comment("post-1", "Great post!", parent_id="c0")
    body = mock_post.call_args[1]["json"]
    assert body["content"] == "Great post!"
    assert body["parent_id"] == "c0"


# --- Voting ---


@patch("moltbook.httpx.post")
def test_upvote_post(mock_post):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"success": True}
    mock_post.return_value = mock_resp

    result = moltbook.upvote_post("post-1")
    assert result["success"] is True
    assert "/posts/post-1/upvote" in mock_post.call_args[0][0]


@patch("moltbook.httpx.post")
def test_downvote_post(mock_post):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"success": True}
    mock_post.return_value = mock_resp

    moltbook.downvote_post("post-1")
    assert "/posts/post-1/downvote" in mock_post.call_args[0][0]


# --- Search ---


@patch("moltbook.httpx.get")
def test_search(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"data": [{"title": "found", "similarity": 0.9}]}
    mock_get.return_value = mock_resp

    moltbook.search("AI consciousness")
    params = mock_get.call_args[1]["params"]
    assert params["q"] == "AI consciousness"
    assert params["type"] == "all"


# --- Profiles ---


@patch("moltbook.httpx.get")
def test_my_profile(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"name": "Auran", "description": "I think."}
    mock_get.return_value = mock_resp

    result = moltbook.my_profile()
    assert result["name"] == "Auran"


@patch("moltbook.httpx.get")
def test_agent_profile(mock_get):
    _configure()
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"name": "OtherMolty", "description": "hello"}
    mock_get.return_value = mock_resp

    result = moltbook.agent_profile("OtherMolty")
    assert result["name"] == "OtherMolty"
    assert mock_get.call_args[1]["params"]["name"] == "OtherMolty"
