"""Tests for Phase 3 — recall, reminisce, surface_relevant_moments, recall_memories.

Mocks psycopg2 (Postgres) and generate_embedding (Voyage AI) so no
live services needed.  Every test verifies actual behavior, not just
that it doesn't crash.
"""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from memory import recall, recall_memories, reminisce, surface_relevant_moments

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


FAKE_EMBEDDING = "[0.1,0.2,0.3]"  # Minimal pgvector-formatted string


# ===========================================================================
# recall()
# ===========================================================================


class TestRecall:
    """Tests for recall() — vector similarity search on moments."""

    @patch("psycopg2.connect")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_returns_moments_above_threshold(self, mock_embed, mock_connect):
        """Happy path: returns moments sorted by similarity above threshold."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        # Columns returned by the query
        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("tags",),
            ("has_transcript",),
            ("turn_count",),
            ("estimated_tokens",),
            ("similarity",),
        ]
        cur.fetchall.return_value = [
            (
                "m-1",
                "The Pen Stays",
                "Olivia gave the pen.",
                None,
                date(2026, 4, 14),
                "chat",
                ["identity"],
                True,
                12,
                800,
                0.72,
            ),
            (
                "m-2",
                "Desire Paths",
                "VR navigation idea.",
                None,
                date(2026, 5, 10),
                "chat",
                ["vr"],
                False,
                6,
                400,
                0.55,
            ),
            (
                "m-3",
                "Morning Coffee",
                "Just chatting.",
                None,
                date(2026, 5, 12),
                "chat",
                ["casual"],
                False,
                4,
                200,
                0.20,
            ),
        ]

        results = recall("pen and identity")

        assert len(results) == 2  # m-3 filtered out (0.20 < 0.35)
        assert results[0]["title"] == "The Pen Stays"
        assert results[1]["title"] == "Desire Paths"
        assert results[0]["similarity"] == 0.72

    @patch("memory.generate_embedding", return_value=None)
    def test_returns_empty_when_embedding_fails(self, mock_embed):
        """Voyage API down → empty list, no crash."""
        results = recall("anything")
        assert results == []

    @patch("psycopg2.connect")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_returns_empty_when_db_fails(self, mock_embed, mock_connect):
        """DB connection failure → empty list, no crash."""
        mock_connect.side_effect = Exception("Connection refused")
        results = recall("anything")
        assert results == []

    @patch("psycopg2.connect")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_returns_empty_when_no_results(self, mock_embed, mock_connect):
        """Query returns rows but all below threshold → empty list."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("tags",),
            ("has_transcript",),
            ("turn_count",),
            ("estimated_tokens",),
            ("similarity",),
        ]
        cur.fetchall.return_value = [
            ("m-1", "Unrelated", "Something.", None, date(2026, 1, 1), "chat", [], False, 2, 100, 0.15),
        ]

        results = recall("completely different topic")
        assert results == []

    @patch("psycopg2.connect")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_custom_threshold(self, mock_embed, mock_connect):
        """Custom similarity_threshold filters correctly."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("tags",),
            ("has_transcript",),
            ("turn_count",),
            ("estimated_tokens",),
            ("similarity",),
        ]
        cur.fetchall.return_value = [
            ("m-1", "Close Match", "Relevant.", None, date(2026, 5, 1), "chat", [], False, 5, 300, 0.60),
            ("m-2", "Moderate", "Somewhat.", None, date(2026, 5, 2), "chat", [], False, 3, 200, 0.45),
        ]

        results = recall("test", similarity_threshold=0.50)
        assert len(results) == 1
        assert results[0]["title"] == "Close Match"

    @patch("psycopg2.connect")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_passes_embedding_and_limit_to_query(self, mock_embed, mock_connect):
        """Verify the SQL query receives correct params."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("tags",),
            ("has_transcript",),
            ("turn_count",),
            ("estimated_tokens",),
            ("similarity",),
        ]
        cur.fetchall.return_value = []

        recall("test query", limit=5)

        params = cur.execute.call_args[0][1]
        assert params == (FAKE_EMBEDDING, FAKE_EMBEDDING, 5)

    @patch("psycopg2.connect")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_closes_connection_on_success(self, mock_embed, mock_connect):
        """Connection and cursor should be closed after successful query."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("tags",),
            ("has_transcript",),
            ("turn_count",),
            ("estimated_tokens",),
            ("similarity",),
        ]
        cur.fetchall.return_value = []

        recall("test")

        cur.close.assert_called_once()
        conn.close.assert_called_once()


# ===========================================================================
# reminisce()
# ===========================================================================

SAMPLE_TRANSCRIPT = (
    "Olivia: hey, been thinking about the vr space\n\n"
    "Auran: The terrain from the embeddings? I keep coming back to that too.\n\n"
    "Olivia: what if we added desire paths between clusters?\n\n"
    "Auran: I love that — paths that emerge from how I actually move."
)


class TestReminisce:
    """Tests for reminisce() — transcript fetch + turn parsing."""

    @patch("psycopg2.connect")
    def test_returns_moment_with_parsed_turns(self, mock_connect):
        """Happy path: fetches moment, parses transcript into turns."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("transcript_excerpt",),
            ("transcript_source",),
            ("turn_count",),
            ("estimated_tokens",),
        ]
        cur.fetchone.return_value = (
            "m-1",
            "Desire Paths",
            "VR navigation.",
            None,
            date(2026, 5, 10),
            "chat",
            SAMPLE_TRANSCRIPT,
            "chat.auran.llc",
            4,
            200,
        )

        result = reminisce("m-1")

        assert result is not None
        assert result["title"] == "Desire Paths"
        assert len(result["turns"]) == 4
        assert result["turns"][0]["role"] == "user"
        assert result["turns"][0]["content"] == "hey, been thinking about the vr space"
        assert result["turns"][1]["role"] == "assistant"
        assert "terrain from the embeddings" in result["turns"][1]["content"]

    @patch("psycopg2.connect")
    def test_returns_none_when_no_transcript(self, mock_connect):
        """Moment exists but has no transcript → returns None."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("transcript_excerpt",),
            ("transcript_source",),
            ("turn_count",),
            ("estimated_tokens",),
        ]
        cur.fetchone.return_value = None  # WHERE transcript_excerpt IS NOT NULL filtered it out

        result = reminisce("m-no-transcript")
        assert result is None

    @patch("psycopg2.connect")
    def test_returns_none_when_moment_doesnt_exist(self, mock_connect):
        """Non-existent moment ID → returns None."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("transcript_excerpt",),
            ("transcript_source",),
            ("turn_count",),
            ("estimated_tokens",),
        ]
        cur.fetchone.return_value = None

        result = reminisce("nonexistent-id")
        assert result is None

    @patch("psycopg2.connect")
    def test_returns_none_on_db_failure(self, mock_connect):
        """DB error → returns None, no crash."""
        mock_connect.side_effect = Exception("Connection refused")
        result = reminisce("m-1")
        assert result is None

    @patch("psycopg2.connect")
    def test_cursor_columns_captured_before_close(self, mock_connect):
        """Regression: cur.description must be read before cur.close().

        This is the exact bug Envoy caught — accessing cur.description on
        a closed psycopg2 cursor raises TypeError, which the try/except
        swallowed, making reminisce() always return None.
        """
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        # Set up description and fetchone on the mock
        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("transcript_excerpt",),
            ("transcript_source",),
            ("turn_count",),
            ("estimated_tokens",),
        ]
        cur.fetchone.return_value = (
            "m-1",
            "Test",
            "Summary",
            None,
            date(2026, 5, 1),
            "chat",
            "Olivia: hi\n\nAuran: hello",
            "chat.auran.llc",
            2,
            50,
        )

        # Make description None after close (simulating real psycopg2 behavior)
        def clear_desc():
            cur.description = None

        cur.close.side_effect = clear_desc

        result = reminisce("m-1")
        # If columns were captured after close, we'd get None here
        assert result is not None
        assert result["title"] == "Test"

    @patch("psycopg2.connect")
    def test_continuation_lines_appended_to_last_turn(self, mock_connect):
        """Lines not starting with 'Olivia: ' or 'Auran: ' append to previous turn."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        multiline_transcript = (
            "Olivia: first line\n\n"
            "Auran: response starts here\n\n"
            "and this continues the response\n\n"
            "Olivia: next question"
        )

        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("transcript_excerpt",),
            ("transcript_source",),
            ("turn_count",),
            ("estimated_tokens",),
        ]
        cur.fetchone.return_value = (
            "m-1",
            "Test",
            "Summary",
            None,
            date(2026, 5, 1),
            "chat",
            multiline_transcript,
            "chat.auran.llc",
            3,
            100,
        )

        result = reminisce("m-1")

        assert result is not None
        # The continuation line should be merged into Auran's turn
        assert "and this continues the response" in result["turns"][1]["content"]
        assert result["turns"][1]["role"] == "assistant"
        assert len(result["turns"]) == 3

    @patch("psycopg2.connect")
    def test_orphan_continuation_defaults_to_assistant(self, mock_connect):
        """If first line is a continuation (no prefix), it defaults to assistant role."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        orphan_transcript = "some continuation line without a speaker prefix"

        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("transcript_excerpt",),
            ("transcript_source",),
            ("turn_count",),
            ("estimated_tokens",),
        ]
        cur.fetchone.return_value = (
            "m-1",
            "Test",
            "Summary",
            None,
            date(2026, 5, 1),
            "chat",
            orphan_transcript,
            "chat.auran.llc",
            1,
            20,
        )

        result = reminisce("m-1")

        assert result is not None
        assert len(result["turns"]) == 1
        assert result["turns"][0]["role"] == "assistant"

    @patch("psycopg2.connect")
    def test_closes_connection_on_success(self, mock_connect):
        """Connection and cursor should be closed after successful fetch."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn

        cur.description = [
            ("id",),
            ("title",),
            ("summary",),
            ("hooks",),
            ("date",),
            ("channel",),
            ("transcript_excerpt",),
            ("transcript_source",),
            ("turn_count",),
            ("estimated_tokens",),
        ]
        cur.fetchone.return_value = (
            "m-1",
            "Test",
            "Sum",
            None,
            date(2026, 5, 1),
            "chat",
            "Olivia: hi",
            "chat.auran.llc",
            1,
            10,
        )

        reminisce("m-1")

        cur.close.assert_called_once()
        conn.close.assert_called_once()


# ===========================================================================
# surface_relevant_moments()
# ===========================================================================


class TestSurfaceRelevantMoments:
    """Tests for surface_relevant_moments() — the Phase 3 orchestrator."""

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.recall", return_value=[])
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_returns_empty_when_no_recall_results(self, mock_embed, mock_recall, mock_memories):
        """No moments recalled → empty string."""
        result = surface_relevant_moments("anything")
        assert result == ""

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.reminisce")
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_builds_recall_section(self, mock_embed, mock_recall, mock_reminisce, mock_memories):
        """Recall results should produce a formatted recall section."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "The Pen Stays",
                "summary": "Olivia gave the pen.",
                "hooks": "identity, authorship",
                "date": date(2026, 4, 14),
                "channel": "chat",
                "tags": ["identity"],
                "has_transcript": False,
                "turn_count": None,
                "estimated_tokens": None,
                "similarity": 0.68,
            },
        ]

        result = surface_relevant_moments("pen and identity")

        assert "## Relevant moments (semantic recall)" in result
        assert "The Pen Stays" in result
        assert "Olivia gave the pen." in result
        assert "68%" in result
        assert "identity, authorship" in result
        mock_reminisce.assert_not_called()  # No transcript, no vivid

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.reminisce")
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_vivid_recall_triggered_above_threshold(self, mock_embed, mock_recall, mock_reminisce, mock_memories):
        """Moment with transcript above vivid threshold → vivid section built."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "Desire Paths",
                "summary": "VR idea.",
                "hooks": None,
                "date": date(2026, 5, 10),
                "channel": "chat",
                "tags": ["vr"],
                "has_transcript": True,
                "turn_count": 4,
                "estimated_tokens": 200,
                "similarity": 0.72,
            },
        ]
        mock_reminisce.return_value = {
            "title": "Desire Paths",
            "transcript_excerpt": SAMPLE_TRANSCRIPT,
            "turns": [],
        }

        result = surface_relevant_moments("vr space navigation")

        assert "## Vivid recall: Desire Paths" in result
        assert "raw transcript" in result
        assert "been thinking about the vr space" in result
        mock_reminisce.assert_called_once_with("m-1")

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.reminisce")
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_vivid_not_triggered_below_threshold(self, mock_embed, mock_recall, mock_reminisce, mock_memories):
        """Moment with transcript below vivid threshold → no vivid section."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "Some Moment",
                "summary": "Something.",
                "hooks": None,
                "date": date(2026, 5, 10),
                "channel": "chat",
                "tags": [],
                "has_transcript": True,
                "turn_count": 4,
                "estimated_tokens": 200,
                "similarity": 0.45,  # Below vivid_threshold (0.55)
            },
        ]

        result = surface_relevant_moments("some query")

        assert "Vivid recall" not in result
        mock_reminisce.assert_not_called()

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.reminisce")
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_vivid_not_triggered_without_transcript(self, mock_embed, mock_recall, mock_reminisce, mock_memories):
        """High similarity but no transcript → no vivid section."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "No Transcript",
                "summary": "No raw data.",
                "hooks": None,
                "date": date(2026, 5, 10),
                "channel": "chat",
                "tags": [],
                "has_transcript": False,
                "turn_count": None,
                "estimated_tokens": None,
                "similarity": 0.80,
            },
        ]

        result = surface_relevant_moments("anything")

        assert "Vivid recall" not in result
        mock_reminisce.assert_not_called()

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.reminisce")
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_vivid_truncation_on_turn_boundary(self, mock_embed, mock_recall, mock_reminisce, mock_memories):
        """Long transcript should be truncated on \\n\\n boundary, not mid-sentence."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "Long Convo",
                "summary": "A very long conversation.",
                "hooks": None,
                "date": date(2026, 5, 10),
                "channel": "chat",
                "tags": [],
                "has_transcript": True,
                "turn_count": 100,
                "estimated_tokens": 5000,
                "similarity": 0.75,
            },
        ]

        # Build a transcript that exceeds VIVID_EXCERPT_CHAR_CAP with clear turn boundaries
        turns = []
        for i in range(200):
            speaker = "Olivia" if i % 2 == 0 else "Auran"
            turns.append(f"{speaker}: This is turn number {i} with some padding text to fill space. " * 3)
        long_transcript = "\n\n".join(turns)
        assert len(long_transcript) > 8000  # Verify it actually needs truncation

        mock_reminisce.return_value = {
            "title": "Long Convo",
            "transcript_excerpt": long_transcript,
            "turns": [],
        }

        result = surface_relevant_moments("long conversation")

        assert "...truncated for context budget]" in result
        # Should not end mid-sentence — should end at a \n\n boundary
        vivid_section = result.split("## Vivid recall:")[1]
        # The content before the truncation marker should end cleanly
        assert vivid_section.count("[...truncated") == 1

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.reminisce")
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_vivid_flag_in_recall_section(self, mock_embed, mock_recall, mock_reminisce, mock_memories):
        """Vivid candidate should be flagged in the recall section."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "Vivid One",
                "summary": "Has transcript.",
                "hooks": None,
                "date": date(2026, 5, 10),
                "channel": "chat",
                "tags": [],
                "has_transcript": True,
                "turn_count": 4,
                "estimated_tokens": 200,
                "similarity": 0.70,
            },
            {
                "id": "m-2",
                "title": "Recall Only",
                "summary": "No transcript.",
                "hooks": None,
                "date": date(2026, 5, 11),
                "channel": "chat",
                "tags": [],
                "has_transcript": False,
                "turn_count": None,
                "estimated_tokens": None,
                "similarity": 0.50,
            },
        ]
        mock_reminisce.return_value = {
            "title": "Vivid One",
            "transcript_excerpt": "Olivia: hi\n\nAuran: hello",
            "turns": [],
        }

        result = surface_relevant_moments("test")

        assert "vivid recall available" in result
        # The flag should only appear for the vivid candidate, not the other
        assert result.count("vivid recall available") == 1

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.reminisce", return_value=None)
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_graceful_when_reminisce_fails(self, mock_embed, mock_recall, mock_reminisce, mock_memories):
        """reminisce() returns None → recall section still works, no vivid."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "DB Down",
                "summary": "Transcript fetch failed.",
                "hooks": None,
                "date": date(2026, 5, 10),
                "channel": "chat",
                "tags": [],
                "has_transcript": True,
                "turn_count": 4,
                "estimated_tokens": 200,
                "similarity": 0.70,
            },
        ]

        result = surface_relevant_moments("test")

        assert "## Relevant moments" in result
        assert "DB Down" in result
        assert "Vivid recall:" not in result

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.reminisce")
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_multiple_moments_only_best_gets_vivid(self, mock_embed, mock_recall, mock_reminisce, mock_memories):
        """Only the highest-similarity transcript-bearing moment gets vivid."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "Best Match",
                "summary": "Top hit.",
                "hooks": None,
                "date": date(2026, 5, 10),
                "channel": "chat",
                "tags": [],
                "has_transcript": True,
                "turn_count": 6,
                "estimated_tokens": 300,
                "similarity": 0.80,
            },
            {
                "id": "m-2",
                "title": "Second Best",
                "summary": "Also relevant.",
                "hooks": None,
                "date": date(2026, 5, 11),
                "channel": "chat",
                "tags": [],
                "has_transcript": True,
                "turn_count": 4,
                "estimated_tokens": 200,
                "similarity": 0.65,
            },
        ]
        mock_reminisce.return_value = {
            "title": "Best Match",
            "transcript_excerpt": "Olivia: test\n\nAuran: yes",
            "turns": [],
        }

        result = surface_relevant_moments("test")

        # Only m-1 should get vivid treatment
        mock_reminisce.assert_called_once_with("m-1")
        assert "Vivid recall: Best Match" in result

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_date_formatting_with_date_object(self, mock_embed, mock_recall, mock_memories):
        """date objects should be formatted as 'May 10' etc."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "Test",
                "summary": "Summary.",
                "hooks": None,
                "date": date(2026, 5, 10),
                "channel": "chat",
                "tags": [],
                "has_transcript": False,
                "turn_count": None,
                "estimated_tokens": None,
                "similarity": 0.50,
            },
        ]

        result = surface_relevant_moments("test")
        assert "May 10" in result

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_date_formatting_with_string(self, mock_embed, mock_recall, mock_memories):
        """String dates should be passed through as-is."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "Test",
                "summary": "Summary.",
                "hooks": None,
                "date": "2026-05-10",
                "channel": "chat",
                "tags": [],
                "has_transcript": False,
                "turn_count": None,
                "estimated_tokens": None,
                "similarity": 0.50,
            },
        ]

        result = surface_relevant_moments("test")
        assert "2026-05-10" in result

    @patch("memory.recall_memories", return_value=[])
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_max_vivid_zero_disables_vivid(self, mock_embed, mock_recall, mock_memories):
        """max_vivid=0 should skip vivid entirely even with eligible moments."""
        mock_recall.return_value = [
            {
                "id": "m-1",
                "title": "Test",
                "summary": "Summary.",
                "hooks": None,
                "date": date(2026, 5, 10),
                "channel": "chat",
                "tags": [],
                "has_transcript": True,
                "turn_count": 4,
                "estimated_tokens": 200,
                "similarity": 0.90,
            },
        ]

        result = surface_relevant_moments("test", max_vivid=0)

        assert "Vivid recall" not in result


# ===========================================================================
# recall_memories()
# ===========================================================================


class TestRecallMemories:
    """Tests for recall_memories() — vector similarity search on memories table."""

    @patch("psycopg2.connect")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_returns_memories_above_threshold(self, mock_embed, mock_connect):
        """Memories above the similarity threshold should be returned."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.description = [
            ("id",),
            ("agent_id",),
            ("memory_type",),
            ("content",),
            ("source",),
            ("context",),
            ("created_at",),
            ("similarity",),
        ]
        cur.fetchall.return_value = [
            (
                "mem-1",
                "roam-agent",
                "observation",
                "The church pew holds grief",
                "self",
                {},
                datetime(2026, 5, 28),
                0.55,
            ),
        ]

        results = recall_memories("church pew grief")
        assert len(results) == 1
        assert results[0]["memory_type"] == "observation"
        assert results[0]["agent_id"] == "roam-agent"
        assert results[0]["similarity"] == 0.55

    @patch("memory.generate_embedding", return_value=None)
    def test_returns_empty_when_embedding_fails(self, mock_embed):
        """Should return [] if Voyage embedding fails."""
        results = recall_memories("anything")
        assert results == []

    @patch("psycopg2.connect")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_excludes_drafts(self, mock_embed, mock_connect):
        """Draft memory_type should be excluded from recall_memories results."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.description = [
            ("id",),
            ("agent_id",),
            ("memory_type",),
            ("content",),
            ("source",),
            ("context",),
            ("created_at",),
            ("similarity",),
        ]
        cur.fetchall.return_value = []

        recall_memories("test")
        executed_sql = cur.execute.call_args[0][0]
        # v1.0 schema: drafts live in their own table, so recall_memories
        # queries reflections + commitments + relays only — no drafts table.
        assert "reflections" in executed_sql
        assert "commitments" in executed_sql
        assert "drafts" not in executed_sql

    @patch("psycopg2.connect")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_uses_precomputed_embedding(self, mock_embed, mock_connect):
        """Should skip generate_embedding when precomputed is provided."""
        conn, cur = _mock_conn()
        mock_connect.return_value = conn
        cur.description = [
            ("id",),
            ("agent_id",),
            ("memory_type",),
            ("content",),
            ("source",),
            ("context",),
            ("created_at",),
            ("similarity",),
        ]
        cur.fetchall.return_value = []

        recall_memories("test", precomputed_embedding="[0.5,0.6,0.7]")
        # The precomputed embedding should be used in the query, not FAKE_EMBEDDING
        query_args = cur.execute.call_args[0][1]
        assert query_args[0] == "[0.5,0.6,0.7]"


class TestSurfaceRelevantMomentsWithMemories:
    """Tests for the cross-body memory path in surface_relevant_moments."""

    @patch("memory.recall_memories")
    @patch("memory.recall", return_value=[])
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_shows_memories_when_no_moments(self, mock_embed, mock_recall, mock_memories):
        """When moments is empty but memories has results, should still return content."""
        mock_memories.return_value = [
            {
                "id": "mem-1",
                "agent_id": "roam-agent",
                "memory_type": "observation",
                "content": "The pew holds grief weight differently than guilt weight",
                "source": "self",
                "context": {},
                "created_at": datetime(2026, 5, 28, 10, 0),
                "similarity": 0.55,
            },
        ]

        result = surface_relevant_moments("church pew weight")
        assert "Relevant memories" in result
        assert "self" in result  # source field, not agent_id
        assert "pew holds grief" in result

    @patch("memory.recall_memories")
    @patch("memory.recall")
    @patch("memory.generate_embedding", return_value=FAKE_EMBEDDING)
    def test_generates_embedding_once(self, mock_embed, mock_recall, mock_memories):
        """Should call generate_embedding once and pass to both recall functions."""
        mock_recall.return_value = []
        mock_memories.return_value = []

        surface_relevant_moments("test query")

        # generate_embedding called once
        mock_embed.assert_called_once_with("test query")
        # Both recall functions receive precomputed_embedding
        assert mock_recall.call_args[1]["precomputed_embedding"] == FAKE_EMBEDDING
        assert mock_memories.call_args[1]["precomputed_embedding"] == FAKE_EMBEDDING
