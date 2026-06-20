import uuid
from unittest.mock import MagicMock

from distillation.backfill import _format_embedding, backfill_embeddings


def _fake_rows(n):
    return [(uuid.uuid4(), f"Title {i}", f"Summary {i}") for i in range(n)]


def _mock_embed_result(n):
    result = MagicMock()
    result.embeddings = [[0.1] * 1024 for _ in range(n)]
    return result


class TestFormatEmbedding:
    def test_produces_pgvector_literal(self):
        vec = [0.1, 0.2, 0.3]
        assert _format_embedding(vec) == "[0.1,0.2,0.3]"

    def test_empty_vector(self):
        assert _format_embedding([]) == "[]"


class TestBackfillEmbeddings:
    def _setup(self, rows):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchall.return_value = rows

        client = MagicMock()
        return conn, cursor, client

    def test_no_rows(self):
        conn, cursor, client = self._setup([])
        result = backfill_embeddings(conn, client, batch_size=50)
        assert result == 0
        client.embed.assert_not_called()
        conn.commit.assert_not_called()

    def test_single_batch(self):
        rows = _fake_rows(10)
        conn, cursor, client = self._setup(rows)
        client.embed.return_value = _mock_embed_result(10)

        result = backfill_embeddings(conn, client, batch_size=50)
        assert result == 10
        client.embed.assert_called_once()
        texts = client.embed.call_args[0][0]
        assert len(texts) == 10
        assert texts[0] == "Summary 0"
        conn.commit.assert_called_once()

    def test_multiple_batches(self):
        rows = _fake_rows(120)
        conn, cursor, client = self._setup(rows)
        client.embed.side_effect = [
            _mock_embed_result(50),
            _mock_embed_result(50),
            _mock_embed_result(20),
        ]

        result = backfill_embeddings(conn, client, batch_size=50)
        assert result == 120
        assert client.embed.call_count == 3
        assert conn.commit.call_count == 3

    def test_batch_failure_continues(self):
        rows = _fake_rows(100)
        conn, cursor, client = self._setup(rows)
        client.embed.side_effect = [
            _mock_embed_result(50),
            RuntimeError("API error"),
        ]

        result = backfill_embeddings(conn, client, batch_size=50)
        assert result == 50
        assert client.embed.call_count == 2
        assert conn.commit.call_count == 1

    def test_dry_run_skips_writes(self):
        rows = _fake_rows(10)
        conn, cursor, client = self._setup(rows)

        result = backfill_embeddings(conn, client, batch_size=50, dry_run=True)
        assert result == 0
        client.embed.assert_not_called()
        conn.commit.assert_not_called()
        cursor.execute.assert_called_once()

    def test_falls_back_to_title_when_summary_none(self):
        rows = [(uuid.uuid4(), "My Title", None)]
        conn, cursor, client = self._setup(rows)
        client.embed.return_value = _mock_embed_result(1)

        backfill_embeddings(conn, client, batch_size=50)
        texts = client.embed.call_args[0][0]
        assert texts[0] == "My Title"

    def test_update_uses_correct_sql(self):
        rows = _fake_rows(2)
        conn, cursor, client = self._setup(rows)
        client.embed.return_value = _mock_embed_result(2)

        backfill_embeddings(conn, client, batch_size=50)

        update_calls = [c for c in cursor.execute.call_args_list if "UPDATE" in str(c)]
        assert len(update_calls) == 2
        for call in update_calls:
            sql = call[0][0]
            assert "UPDATE episodes SET embedding" in sql
            assert "WHERE id" in sql
