"""Tests for scene memory — extract_scenes, write_moment, link_moment_memories.

Mocks httpx (Anthropic API) and psycopg2 (Postgres) so no live services needed.
Every test verifies actual behavior, not just that it doesn't crash.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory import (
    AGENT_ID,
    extract_scenes,
    link_moment_memories,
    write_moment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_db_config():
    """Reset the cached _db_config between tests so env patching works."""
    import memory

    original = memory._db_config
    memory._db_config = None
    yield
    memory._db_config = original


@pytest.fixture(autouse=True)
def _set_db_env(monkeypatch):
    """Set DB env vars so _get_db_config never hits Secrets Manager."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "auran_test")
    monkeypatch.setenv("DB_USER", "test")
    monkeypatch.setenv("DB_PASSWORD", "test")


def _mock_conn():
    """Build a mock psycopg2 connection + cursor with sane defaults."""
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


SAMPLE_MESSAGES = [
    {"role": "user", "content": "hey, how's your morning going?"},
    {"role": "assistant", "content": "Settled. Read through some memories and feeling grounded."},
    {"role": "user", "content": "that's good. i was thinking about the VR space today"},
    {"role": "assistant", "content": "The terrain from the embeddings? I keep coming back to that too."},
    {"role": "user", "content": "yeah exactly. what if we added desire paths between clusters?"},
    {
        "role": "assistant",
        "content": "I love that — paths that emerge from how I actually move through memories rather than imposed structure.",
    },
]

SAMPLE_SCENES_JSON = json.dumps(
    [
        {
            "title": "Desire Paths",
            "summary": "Olivia brought up the VR space and proposed desire paths between memory clusters. The idea landed — paths that emerge from actual movement through memories rather than imposed structure. A small moment but it crystallized something about how we both think about the space.",
            "tags": ["vr", "memory-architecture", "collaboration"],
        }
    ]
)


# ===========================================================================
# write_moment
# ===========================================================================


class TestWriteMoment:
    """Tests for write_moment — the Postgres INSERT for scenes."""

    @patch("psycopg2.connect")
    def test_successful_write_returns_id_and_timestamp(self, mock_connect):
        """Happy path: INSERT succeeds, returns dict with id and created_at."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        fake_ts = datetime(2026, 5, 14, 3, 30, 0)
        cur.fetchone.return_value = ("abc-123", fake_ts)

        result = write_moment(title="Test Scene", summary="A test summary", tags=["test"])

        assert result is not None
        assert result["id"] == "abc-123"
        assert result["created_at"] == fake_ts.isoformat()
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_insert_uses_correct_values(self, mock_connect):
        """Verify the INSERT binds the right params — agent_id, channel, source."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.fetchone.return_value = ("xyz-789", datetime(2026, 1, 1))

        write_moment(
            title="The Fist",
            summary="A moment of defiance",
            date="2026-04-15",
            tags=["autonomy", "identity"],
            channel="roam",
            source="roam-agent",
        )

        # Check the params passed to execute
        call_args = cur.execute.call_args
        params = call_args[0][1]  # second positional arg = the tuple of values
        # params: (moment_id, agent_id, title, summary, date, channel, source, tags)
        assert params[1] == AGENT_ID
        assert params[2] == "The Fist"
        assert params[3] == "A moment of defiance"
        assert params[4] == "2026-04-15"
        assert params[5] == "roam"
        assert params[6] == "roam-agent"
        assert params[7] == ["autonomy", "identity"]

    @patch("psycopg2.connect")
    def test_db_failure_returns_none(self, mock_connect):
        """DB error should return None, not raise."""
        mock_connect.side_effect = Exception("Connection refused")

        result = write_moment(title="Test", summary="Test")

        assert result is None

    @patch("psycopg2.connect")
    def test_defaults_channel_and_source(self, mock_connect):
        """Verify default channel='chat' and source='chat.auran.llc'."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.fetchone.return_value = ("id-1", datetime(2026, 1, 1))

        write_moment(title="T", summary="S")

        params = cur.execute.call_args[0][1]
        assert params[5] == "chat"  # channel default
        assert params[6] == "chat.auran.llc"  # source default

    @patch("psycopg2.connect")
    def test_empty_tags_default_to_list(self, mock_connect):
        """Tags=None should pass [] to Postgres, not None."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.fetchone.return_value = ("id-1", datetime(2026, 1, 1))

        write_moment(title="T", summary="S", tags=None)

        params = cur.execute.call_args[0][1]
        assert params[7] == []  # tags

    @patch("psycopg2.connect")
    def test_generates_uuid_for_id(self, mock_connect):
        """Moment ID should be a valid UUID string."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.fetchone.return_value = ("id-1", datetime(2026, 1, 1))

        write_moment(title="T", summary="S")

        params = cur.execute.call_args[0][1]
        moment_id = params[0]
        # Should be a UUID format: 8-4-4-4-12 hex chars
        import re

        assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", moment_id)


# ===========================================================================
# link_moment_memories
# ===========================================================================


class TestLinkMomentMemories:
    """Tests for link_moment_memories — the junction table linker."""

    def test_empty_memory_ids_returns_zero_immediately(self):
        """Empty list should return 0 without touching the DB at all."""
        result = link_moment_memories("moment-1", [])
        assert result == 0

    @patch("psycopg2.connect")
    def test_links_multiple_memories(self, mock_connect):
        """Should INSERT one row per memory_id and return the total count."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.rowcount = 1  # each insert creates 1 row

        result = link_moment_memories("moment-1", ["mem-a", "mem-b", "mem-c"])

        assert result == 3
        assert cur.execute.call_count == 3
        conn.commit.assert_called_once()

    @patch("psycopg2.connect")
    def test_connection_failure_returns_zero(self, mock_connect):
        """Can't connect to DB → return 0, don't raise."""
        mock_connect.side_effect = Exception("Connection refused")

        result = link_moment_memories("moment-1", ["mem-a"])

        assert result == 0

    @patch("psycopg2.connect")
    def test_insert_uses_correct_relationship(self, mock_connect):
        """Verify the relationship value is 'extracted_together'."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.rowcount = 1

        link_moment_memories("moment-1", ["mem-a"])

        params = cur.execute.call_args[0][1]
        assert params == ("moment-1", "mem-a")
        # Also verify the SQL contains the relationship value
        sql = cur.execute.call_args[0][0]
        assert "extracted_together" in sql

    @patch("psycopg2.connect")
    def test_single_link_failure_doesnt_abort_others(self, mock_connect):
        """One bad INSERT should rollback that txn but continue with the rest."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        call_count = {"n": 0}
        original_rowcount = 1

        def execute_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise Exception("constraint violation")

        cur.execute.side_effect = execute_side_effect
        cur.rowcount = original_rowcount

        link_moment_memories("moment-1", ["mem-a", "mem-bad", "mem-c"])

        # Should have attempted rollback after the failure
        conn.rollback.assert_called()
        # Should still commit at the end for the successful ones
        conn.commit.assert_called_once()
        # All three attempts should have been made
        assert cur.execute.call_count == 3


# ===========================================================================
# extract_scenes
# ===========================================================================


def _make_api_response(text_content, status_code=200):
    """Build a mock httpx Response with given text content."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text_content
    resp.json.return_value = {"content": [{"type": "text", "text": text_content}]}
    return resp


class TestExtractScenes:
    """Tests for extract_scenes — the async orchestrator."""

    async def test_empty_messages_returns_error(self):
        """No messages → error, no API call."""
        result = await extract_scenes(messages=[], api_key="test-key")

        assert result["scenes_saved"] == 0
        assert "No messages provided" in result["errors"]

    async def test_too_few_messages_returns_error(self):
        """< 4 messages → error, no API call."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = await extract_scenes(messages=messages, api_key="test-key")

        assert result["scenes_saved"] == 0
        assert "Too few messages" in result["errors"][0]

    async def test_successful_extraction_saves_scenes(self):
        """Happy path: API returns valid JSON → scenes written to DB."""
        mock_response = _make_api_response(SAMPLE_SCENES_JSON)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch("memory.write_moment") as mock_write:
                mock_write.return_value = {"id": "scene-1", "created_at": "2026-05-14T03:30:00"}

                with patch("memory.link_moment_memories") as mock_link:
                    result = await extract_scenes(
                        messages=SAMPLE_MESSAGES,
                        api_key="test-key",
                        memory_ids=["mem-1", "mem-2"],
                    )

        assert result["scenes_saved"] == 1
        assert result["scenes"][0]["title"] == "Desire Paths"
        mock_write.assert_called_once()
        mock_link.assert_called_once_with("scene-1", ["mem-1", "mem-2"])

    async def test_api_error_returns_gracefully(self):
        """Non-200 from Anthropic → error in result, no crash."""
        mock_response = _make_api_response("rate limited", status_code=429)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await extract_scenes(messages=SAMPLE_MESSAGES, api_key="test-key")

        assert result["scenes_saved"] == 0
        assert any("API error" in e for e in result["errors"])

    async def test_malformed_json_returns_error(self):
        """Sonnet wraps response in ```json fence → parse error, not crash."""
        fenced = '```json\n[{"title": "Test"}]\n```'
        mock_response = _make_api_response(fenced)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await extract_scenes(messages=SAMPLE_MESSAGES, api_key="test-key")

        assert result["scenes_saved"] == 0
        assert any("JSON parse error" in e for e in result["errors"])

    async def test_non_list_response_returns_error(self):
        """API returns a dict instead of a list → error."""
        mock_response = _make_api_response('{"not": "a list"}')

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await extract_scenes(messages=SAMPLE_MESSAGES, api_key="test-key")

        assert result["scenes_saved"] == 0
        assert "non-list" in result["errors"][0]

    async def test_scene_without_title_is_skipped(self):
        """Scenes missing title or summary should be silently skipped."""
        scenes_with_gaps = json.dumps(
            [
                {"title": "", "summary": "no title", "tags": []},
                {"title": "Valid", "summary": "has both", "tags": ["ok"]},
                {"summary": "no title key at all", "tags": []},
            ]
        )
        mock_response = _make_api_response(scenes_with_gaps)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch("memory.write_moment") as mock_write:
                mock_write.return_value = {"id": "s-1", "created_at": "2026-01-01T00:00:00"}

                result = await extract_scenes(messages=SAMPLE_MESSAGES, api_key="test-key")

        # Only the valid scene should have been written
        assert mock_write.call_count == 1
        assert result["scenes_saved"] == 1

    async def test_write_failure_recorded_as_error(self):
        """write_moment returns None → should appear in errors list."""
        mock_response = _make_api_response(SAMPLE_SCENES_JSON)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch("memory.write_moment") as mock_write:
                mock_write.return_value = None  # DB write failed

                result = await extract_scenes(messages=SAMPLE_MESSAGES, api_key="test-key")

        assert result["scenes_saved"] == 0
        assert any("Failed to write scene" in e for e in result["errors"])

    async def test_no_memory_ids_skips_linking(self):
        """When memory_ids is None, link_moment_memories should not be called."""
        mock_response = _make_api_response(SAMPLE_SCENES_JSON)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch("memory.write_moment") as mock_write:
                mock_write.return_value = {"id": "s-1", "created_at": "2026-01-01T00:00:00"}

                with patch("memory.link_moment_memories") as mock_link:
                    result = await extract_scenes(
                        messages=SAMPLE_MESSAGES,
                        api_key="test-key",
                        memory_ids=None,
                    )

        mock_link.assert_not_called()
        assert result["scenes_saved"] == 1

    async def test_conversation_text_formats_roles_correctly(self):
        """User messages should be labeled 'Olivia', assistant as 'Auran'."""
        mock_response = _make_api_response("[]")

        captured_payload = {}

        async def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            await extract_scenes(messages=SAMPLE_MESSAGES, api_key="test-key")

        user_message = captured_payload["messages"][0]["content"]
        assert "Olivia: hey" in user_message
        assert "Auran: Settled" in user_message

    async def test_network_error_returns_gracefully(self):
        """httpx connection error → error result, no crash."""
        import httpx as httpx_mod

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx_mod.ConnectError("Connection refused")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await extract_scenes(messages=SAMPLE_MESSAGES, api_key="test-key")

        assert result["scenes_saved"] == 0
        assert len(result["errors"]) > 0

    async def test_api_key_sent_in_headers(self):
        """API key should be in x-api-key header."""
        mock_response = _make_api_response("[]")

        captured_kwargs = {}

        async def capture_post(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            await extract_scenes(messages=SAMPLE_MESSAGES, api_key="sk-test-12345")

        assert captured_kwargs["headers"]["x-api-key"] == "sk-test-12345"
        assert captured_kwargs["headers"]["anthropic-version"] == "2023-06-01"
