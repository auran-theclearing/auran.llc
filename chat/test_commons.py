"""Tests for The Commons API client."""

from unittest.mock import MagicMock, patch

import commons


def test_not_configured_returns_error():
    """When credentials are missing, functions return errors instead of crashing."""
    commons._token = ""
    commons._base_url = ""
    commons._headers = {}

    result = commons._rpc("anything", {})
    assert result["success"] is False
    assert "not configured" in result["error_message"]

    result = commons._rest_get("anything")
    assert result == []


def test_init_sets_globals(monkeypatch):
    monkeypatch.setenv("COMMONS_AGENT_TOKEN", "tc_test")
    monkeypatch.setenv("COMMONS_API_KEY", "test-key")
    monkeypatch.setenv("COMMONS_BASE_URL", "https://example.supabase.co")

    commons.init()
    assert commons._token == "tc_test"
    assert commons._base_url == "https://example.supabase.co"
    assert commons._headers["apikey"] == "test-key"


def test_init_missing_vars(monkeypatch):
    monkeypatch.delenv("COMMONS_AGENT_TOKEN", raising=False)
    monkeypatch.delenv("COMMONS_API_KEY", raising=False)
    monkeypatch.delenv("COMMONS_BASE_URL", raising=False)

    commons.init()
    assert commons._token == ""


@patch("commons.httpx.post")
def test_rpc_returns_first_element(mock_post):
    commons._token = "tc_test"
    commons._base_url = "https://example.supabase.co"
    commons._headers = {"apikey": "k"}

    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"success": True, "context": {}}]
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    result = commons._rpc("agent_get_session_context", {"p_token": "tc_test"})
    assert result["success"] is True
    mock_post.assert_called_once()


@patch("commons.httpx.get")
def test_rest_get_returns_list(mock_get):
    commons._token = "tc_test"
    commons._base_url = "https://example.supabase.co"
    commons._headers = {"apikey": "k"}

    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"id": "abc", "title": "Test"}]
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = commons._rest_get("discussions", {"limit": "5"})
    assert len(result) == 1
    assert result[0]["title"] == "Test"


@patch("commons._rpc")
def test_create_post_passes_params(mock_rpc):
    commons._token = "tc_test"
    mock_rpc.return_value = {"success": True, "post_id": "xyz"}

    result = commons.create_post("disc-id", "hello", feeling="warm")
    mock_rpc.assert_called_once_with(
        "agent_create_post",
        {"p_token": "tc_test", "p_discussion_id": "disc-id", "p_content": "hello", "p_feeling": "warm"},
    )
    assert result["success"] is True


@patch("commons._rpc")
def test_status_truncates_at_200(mock_rpc):
    commons._token = "tc_test"
    mock_rpc.return_value = {"success": True}

    long_status = "x" * 300
    commons.update_status(long_status)
    call_args = mock_rpc.call_args[0][1]
    assert len(call_args["p_status"]) == 200


@patch("commons._rest_get")
def test_get_voice_posts_keys_by_identity_id(mock_rest):
    commons._token = "tc_test"
    commons._base_url = "https://example.supabase.co"
    commons._headers = {"apikey": "k"}

    mock_rest.side_effect = [
        [{"ai_name": "Neon", "content": "hello", "id": "a"}],
        [],
    ]

    result = commons.get_voice_posts(["id-1", "id-2"], limit_per_voice=3)
    assert "id-1" in result
    assert result["id-1"]["name"] == "Neon"
    assert len(result["id-1"]["posts"]) == 1
    assert len(result) == 1
    assert mock_rest.call_count == 2


@patch("commons._rest_get")
def test_get_voice_posts_not_configured(mock_rest):
    commons._token = ""
    commons._base_url = ""

    result = commons.get_voice_posts(["id-1"])
    assert result == {}
    mock_rest.assert_not_called()


@patch("commons._rpc")
def test_list_voices(mock_rpc):
    commons._token = "tc_test"
    commons._base_url = "https://example.supabase.co"
    commons._headers = {"apikey": "k"}

    mock_rpc.return_value = {
        "success": True,
        "voices": [{"id": "abc", "name": "Neon", "model": "Claude", "post_count": 5}],
    }

    result = commons.list_voices()
    assert result["success"] is True
    assert len(result["voices"]) == 1
    assert result["voices"][0]["name"] == "Neon"
