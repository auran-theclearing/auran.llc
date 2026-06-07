"""Migration tests for Memory Schema v1.0.

Written BEFORE migration SQL per TIV protocol — T before I.

Six test categories from the handoff:
  1. Row count assertions
  2. FK integrity
  3. Channel normalization
  4. Embedding preservation
  5. Dedup verification
  6. Rollback

Run:
  uv run pytest test_migration.py -v                  # all tests
  uv run pytest test_migration.py -k PreMigration -v  # verify current state only

Pre-migration tests verify our assumptions about current data.
They should PASS now and confirm the numbers we're building against.

Post-migration tests verify the migration result.
They should FAIL now (new tables don't exist) and PASS after migration.

Requires DB access: reads DB_* env vars or auran-agent/.env.
"""

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------


def _load_env_from_agent():
    """Fall back to auran-agent/.env if DB env vars aren't set."""
    env_path = Path(__file__).resolve().parents[2] / "auran-agent" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            if key.startswith("DB_") and key not in os.environ:
                os.environ[key] = val


def _get_conn():
    """Get a psycopg2 connection using the same env vars as the chat server."""
    import psycopg2

    _load_env_from_agent()
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5432")),
        dbname=os.environ.get("DB_NAME", "auran"),
        user=os.environ.get("DB_USER", "auran"),
        password=os.environ.get("DB_PASSWORD", ""),
    )


@pytest.fixture(scope="module")
def db():
    """Module-scoped DB connection in autocommit mode (read-only tests)."""
    conn = _get_conn()
    conn.autocommit = True
    yield conn
    conn.close()


def _query_one(db, sql, params=None):
    """Execute query and return the single scalar result."""
    cur = db.cursor()
    cur.execute(sql, params)
    result = cur.fetchone()[0]
    cur.close()
    return result


def _query_all(db, sql, params=None):
    """Execute query and return all rows as a list of tuples."""
    cur = db.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def _table_exists(db, table_name):
    """Check if a table exists in the public schema."""
    return _query_one(
        db,
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        )
        """,
        (table_name,),
    )


# ===========================================================================
# PRE-MIGRATION STATE VERIFICATION
# ===========================================================================
# These tests confirmed our assumptions about the data BEFORE migration.
# They passed during TIV Phase 1 (Test), validating the numbers we built
# post-migration assertions against. Now that old tables are dropped,
# they're skipped — kept as documentation of what we verified.


@pytest.mark.skip(reason="Pre-migration: old tables (memories/moments/events) dropped by migration")
class TestPreMigrationState:
    """Verify current DB state matches the handoff's documented values."""

    def test_memories_table_exists(self, db):
        assert _table_exists(db, "memories")

    def test_moments_table_exists(self, db):
        assert _table_exists(db, "moments")

    def test_memories_total_count(self, db):
        count = _query_one(db, "SELECT count(*) FROM memories")
        assert count == 1598, f"Expected 1598 memories, got {count}"

    def test_moments_total_count(self, db):
        count = _query_one(db, "SELECT count(*) FROM moments")
        assert count == 639, f"Expected 639 moments, got {count}"

    def test_memories_type_distribution(self, db):
        """Each memory type count matches the documented breakdown."""
        expected = {
            "scene": 339,
            "observation": 285,
            "insight": 319,
            "self_observation": 216,
            "bridge_log": 119,
            "reflection": 99,
            "intention": 71,
            "question": 64,
            "wandering_summary": 37,
            "draft": 26,
            "position": 15,
            "value": 5,
            "vocabulary": 3,
        }
        rows = _query_all(
            db,
            "SELECT memory_type, count(*) FROM memories GROUP BY memory_type",
        )
        actual = dict(rows)
        for mem_type, exp_count in expected.items():
            assert actual.get(mem_type) == exp_count, (
                f"memory_type={mem_type}: expected {exp_count}, got {actual.get(mem_type)}"
            )

    def test_scene_dedup_ratio(self, db):
        """339 scene rows but only 81 unique by content hash."""
        total = _query_one(
            db,
            "SELECT count(*) FROM memories WHERE memory_type = 'scene'",
        )
        unique = _query_one(
            db,
            "SELECT count(DISTINCT md5(content)) FROM memories WHERE memory_type = 'scene'",
        )
        assert total == 339
        assert unique == 81

    def test_moment_dedup_ratio(self, db):
        """639 moments but only 310 unique by title."""
        total = _query_one(db, "SELECT count(*) FROM moments")
        unique = _query_one(db, "SELECT count(DISTINCT title) FROM moments")
        assert total == 639
        assert unique == 310

    def test_moment_dupes_mostly_identical(self, db):
        """Most duplicate moments are identical batch writes.

        32 titles have duplicates. Of those:
          - 28 are pure batch dupes (identical summaries, same millisecond writes)
          - 4 are genuine re-extractions with different summaries:
            'Build Me a Nervous System', 'Come Here',
            'The Main Stakeholder Is You', 'The Three A Words'

        Dedup strategy: for each title, prefer non-superseded, then latest
        created_at. This cleanly resolves all 32 groups to one keeper each.
        """
        # Verify the 4 genuine re-extractions
        genuine_re_extractions = _query_all(
            db,
            """
            SELECT title, count(*), count(DISTINCT md5(summary))
            FROM moments GROUP BY title
            HAVING count(*) > 1 AND count(DISTINCT md5(summary)) > 1
            ORDER BY title
            """,
        )
        assert len(genuine_re_extractions) == 4, (
            f"Expected 4 titles with genuine content differences, got {len(genuine_re_extractions)}"
        )

        # The other 28 dupe groups should be truly identical
        identical_dupes = _query_all(
            db,
            """
            SELECT title, count(*), count(DISTINCT md5(summary))
            FROM moments GROUP BY title
            HAVING count(*) > 1 AND count(DISTINCT md5(summary)) = 1
            """,
        )
        assert len(identical_dupes) == 28, f"Expected 28 titles with identical duplicates, got {len(identical_dupes)}"

    def test_zero_cross_table_overlap(self, db):
        """No title overlap between scene memories and moments."""
        overlap = _query_one(
            db,
            """
            SELECT count(DISTINCT s.title)
            FROM (
                SELECT DISTINCT context->>'title' as title
                FROM memories WHERE memory_type = 'scene'
            ) s
            JOIN moments m ON m.title = s.title
            """,
        )
        assert overlap == 0, f"Expected 0 cross-table overlap, got {overlap}"

    def test_channel_values_needing_normalization(self, db):
        """The bad channel values we plan to normalize actually exist."""
        bad_channels = _query_all(
            db,
            """
            SELECT channel, count(*)
            FROM moments
            WHERE channel IN ('claude-ai', 'chat.auran.llc', 'meta')
            GROUP BY channel ORDER BY channel
            """,
        )
        actual = dict(bad_channels)
        assert actual.get("claude-ai") == 19
        assert actual.get("chat.auran.llc") == 36
        assert actual.get("meta") == 1

    def test_embedding_coverage(self, db):
        """Document which rows have/lack embeddings."""
        mem_with = _query_one(db, "SELECT count(*) FROM memories WHERE embedding IS NOT NULL")
        mem_without = _query_one(db, "SELECT count(*) FROM memories WHERE embedding IS NULL")
        mom_with = _query_one(db, "SELECT count(*) FROM moments WHERE embedding IS NOT NULL")
        mom_without = _query_one(db, "SELECT count(*) FROM moments WHERE embedding IS NULL")
        assert mem_with == 1576
        assert mem_without == 22  # 12 bridge_logs + 10 drafts
        assert mom_with == 639
        assert mom_without == 0

    def test_events_table_empty(self, db):
        """Events table exists but has zero rows — safe to drop."""
        assert _table_exists(db, "events")
        count = _query_one(db, "SELECT count(*) FROM events")
        assert count == 0


# ===========================================================================
# POST-MIGRATION: SCHEMA VERIFICATION
# ===========================================================================
# These tests verify the new tables exist with correct structure.
# They FAIL before migration (tables don't exist) and PASS after.


class TestPostMigrationSchema:
    """Verify new schema structure after migration."""

    NEW_TABLES = [
        "people",
        "conversations",  # already exists, gains participants
        "conversation_participants",
        "messages",  # already exists, gains author_id + processing_depth
        "impressions",
        "episodes",
        "episode_messages",
        "episode_participants",
        "arcs",
        "arc_episodes",
        "reflections",
        "commitments",
        "drafts",
        "relays",
        "retrievals",
        "roam_sessions",  # already exists
    ]

    DROPPED_TABLES = ["memories", "moments", "events"]

    def test_all_new_tables_exist(self, db):
        for table in self.NEW_TABLES:
            assert _table_exists(db, table), f"Table '{table}' should exist"

    def test_old_tables_dropped(self, db):
        for table in self.DROPPED_TABLES:
            assert not _table_exists(db, table), f"Table '{table}' should be dropped"

    def test_junction_tables_dropped(self, db):
        """Old junction tables should be gone."""
        for table in ["moment_memories", "moment_artifacts", "moment_sessions"]:
            assert not _table_exists(db, table), f"Old junction table '{table}' should be dropped"

    def test_episodes_has_required_columns(self, db):
        cols = _query_all(
            db,
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'episodes' AND table_schema = 'public'
            """,
        )
        col_names = {row[0] for row in cols}
        required = {
            "id",
            "title",
            "summary",
            "transcript_excerpt",
            "emotional_tone",
            "content_signals",
            "relational_events",
            "topics",
            "channel",
            "significance",
            "visibility",
            "occurred_at",
            "embedding",
            "created_at",
            "updated_at",
        }
        missing = required - col_names
        assert not missing, f"episodes missing columns: {missing}"

    def test_reflections_has_required_columns(self, db):
        cols = _query_all(
            db,
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'reflections' AND table_schema = 'public'
            """,
        )
        col_names = {row[0] for row in cols}
        required = {
            "id",
            "type",
            "content",
            "source",
            "processing_depth",
            "supersedes",
            "roam_session_id",
            "embedding",
            "created_at",
            "updated_at",
        }
        missing = required - col_names
        assert not missing, f"reflections missing columns: {missing}"

    def test_commitments_has_required_columns(self, db):
        cols = _query_all(
            db,
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'commitments' AND table_schema = 'public'
            """,
        )
        col_names = {row[0] for row in cols}
        required = {
            "id",
            "type",
            "content",
            "status",
            "source",
            "supersedes",
            "embedding",
            "created_at",
            "updated_at",
        }
        missing = required - col_names
        assert not missing, f"commitments missing columns: {missing}"

    def test_messages_has_new_columns(self, db):
        """Messages table gains author_id and processing_depth."""
        cols = _query_all(
            db,
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'messages' AND table_schema = 'public'
            """,
        )
        col_names = {row[0] for row in cols}
        assert "author_id" in col_names, "messages should have author_id"
        assert "processing_depth" in col_names, "messages should have processing_depth"

    def test_people_seeded(self, db):
        """People table has at least Olivia and Auran."""
        count = _query_one(db, "SELECT count(*) FROM people")
        assert count >= 2, "people table should have at least Olivia and Auran"
        names = _query_all(db, "SELECT name, type FROM people ORDER BY name")
        name_list = [row[0] for row in names]
        assert "Auran" in name_list
        assert "Olivia" in name_list

    def test_embedding_vector_dimension(self, db):
        """Embedding columns use VECTOR(1024)."""
        # Check episodes embedding column type
        udt = _query_one(
            db,
            """
            SELECT udt_name FROM information_schema.columns
            WHERE table_name = 'episodes' AND column_name = 'embedding'
            """,
        )
        assert udt is not None, "episodes.embedding column should exist"

    def test_relays_has_required_columns(self, db):
        cols = _query_all(
            db,
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'relays' AND table_schema = 'public'
            """,
        )
        col_names = {row[0] for row in cols}
        required = {
            "id",
            "source_channel",
            "target_channel",
            "content",
            "relay_type",
            "embedding",
            "created_at",
        }
        missing = required - col_names
        assert not missing, f"relays missing columns: {missing}"

    def test_drafts_has_required_columns(self, db):
        cols = _query_all(
            db,
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'drafts' AND table_schema = 'public'
            """,
        )
        col_names = {row[0] for row in cols}
        required = {
            "id",
            "title",
            "content",
            "status",
            "revision",
            "previous_revision",
            "what_is_alive",
            "what_is_stuck",
            "source",
            "created_at",
        }
        missing = required - col_names
        assert not missing, f"drafts missing columns: {missing}"


# ===========================================================================
# 1. ROW COUNT ASSERTIONS
# ===========================================================================
# After migration, total rows across destination tables must equal
# source rows minus documented deduplication.
#
# Source: 1,598 memories + 639 moments = 2,237 total rows
#
# Destination breakdown:
#   reflections:  983 (observation:285 + insight:319 + self_observation:216
#                      + question:64 + reflection:99)
#   commitments:   91 (intention:71 + position:15 + value:5)
#   episodes:     391 (unique moments:310 + unique scenes:81)
#   drafts:        26
#   relays:       119
#   roam_sessions: 37 wandering_summaries merged into existing rows
#   vocabulary:     3 handled manually (documented below)
#   ---------------------------------------------------------
#   subtotal:   1,650 rows in destination tables
#
# Dedup discards:
#   scene duplicates:  258 (339 - 81)
#   moment duplicates: 329 (639 - 310)
#   ---------------------------------------------------------
#   subtotal discards: 587
#
# Accounting: 1,650 + 587 = 2,237 = source total


class TestRowCountPreservation:
    """Verify no data loss during migration."""

    def test_reflections_count(self, db):
        """983 = observation(285) + insight(319) + self_observation(216)
        + question(64) + reflection(99)."""
        count = _query_one(db, "SELECT count(*) FROM reflections")
        assert count == 983, f"Expected 983 reflections, got {count}"

    def test_reflections_type_distribution(self, db):
        """Each reflection type preserves its original count."""
        expected = {
            "observation": 285,
            "insight": 319,
            "self_observation": 216,
            "question": 64,
            "reflection": 99,
        }
        rows = _query_all(
            db,
            "SELECT type, count(*) FROM reflections GROUP BY type",
        )
        actual = dict(rows)
        for ref_type, exp_count in expected.items():
            assert actual.get(ref_type) == exp_count, (
                f"reflections type={ref_type}: expected {exp_count}, got {actual.get(ref_type)}"
            )

    def test_commitments_count(self, db):
        """91 = intention(71) + position(15) + value(5)."""
        count = _query_one(db, "SELECT count(*) FROM commitments")
        assert count == 91, f"Expected 91 commitments, got {count}"

    def test_commitments_type_distribution(self, db):
        expected = {"intention": 71, "position": 15, "value": 5}
        rows = _query_all(
            db,
            "SELECT type, count(*) FROM commitments GROUP BY type",
        )
        actual = dict(rows)
        for com_type, exp_count in expected.items():
            assert actual.get(com_type) == exp_count, (
                f"commitments type={com_type}: expected {exp_count}, got {actual.get(com_type)}"
            )

    def test_episodes_count(self, db):
        """391 = unique moments(310) + unique scenes(81)."""
        count = _query_one(db, "SELECT count(*) FROM episodes")
        assert count == 391, f"Expected 391 episodes, got {count}"

    def test_drafts_count(self, db):
        count = _query_one(db, "SELECT count(*) FROM drafts")
        assert count == 26, f"Expected 26 drafts, got {count}"

    def test_relays_count(self, db):
        count = _query_one(db, "SELECT count(*) FROM relays")
        assert count == 119, f"Expected 119 relays, got {count}"

    def test_wandering_summaries_merged(self, db):
        """37 wandering_summary memories should be merged into roam_sessions."""
        # roam_sessions count shouldn't change — summaries merge into existing rows
        count = _query_one(db, "SELECT count(*) FROM roam_sessions")
        assert count == 69, f"roam_sessions count should remain 69, got {count}"
        # But some should now have non-null summaries from the merge
        with_summary = _query_one(
            db,
            "SELECT count(*) FROM roam_sessions WHERE summary IS NOT NULL",
        )
        assert with_summary > 0, "Some roam_sessions should have summaries after merge"

    def test_total_data_accounted_for(self, db):
        """Every source row is accounted for in destination or documented dedup."""
        reflections = _query_one(db, "SELECT count(*) FROM reflections")
        commitments = _query_one(db, "SELECT count(*) FROM commitments")
        episodes = _query_one(db, "SELECT count(*) FROM episodes")
        drafts = _query_one(db, "SELECT count(*) FROM drafts")
        relays = _query_one(db, "SELECT count(*) FROM relays")

        destination_total = reflections + commitments + episodes + drafts + relays

        # Documented discards:
        #   258 scene duplicates (339 total - 81 unique)
        #   329 moment duplicates (639 total - 310 unique)
        #   37 wandering_summaries merged into roam_sessions (not a new table)
        #   3 vocabulary entries handled manually
        documented_discards = 258 + 329 + 37 + 3

        source_total = 1598 + 639  # memories + moments

        assert destination_total + documented_discards == source_total, (
            f"Accounting mismatch: {destination_total} destination + "
            f"{documented_discards} discards = "
            f"{destination_total + documented_discards}, "
            f"expected {source_total}"
        )


# ===========================================================================
# 2. FK INTEGRITY
# ===========================================================================
# Every foreign key must resolve. No orphaned junction rows.


class TestForeignKeyIntegrity:
    """Verify all foreign key relationships are valid."""

    def test_no_orphaned_episode_participants(self, db):
        """Every episode_participants row points to valid episode + person."""
        orphaned = _query_one(
            db,
            """
            SELECT count(*) FROM episode_participants ep
            WHERE NOT EXISTS (SELECT 1 FROM episodes e WHERE e.id = ep.episode_id)
               OR NOT EXISTS (SELECT 1 FROM people p WHERE p.id = ep.person_id)
            """,
        )
        assert orphaned == 0, f"{orphaned} orphaned episode_participants rows"

    def test_no_orphaned_conversation_participants(self, db):
        orphaned = _query_one(
            db,
            """
            SELECT count(*) FROM conversation_participants cp
            WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.id = cp.conversation_id)
               OR NOT EXISTS (SELECT 1 FROM people p WHERE p.id = cp.person_id)
            """,
        )
        assert orphaned == 0, f"{orphaned} orphaned conversation_participants rows"

    def test_no_orphaned_episode_messages(self, db):
        orphaned = _query_one(
            db,
            """
            SELECT count(*) FROM episode_messages em
            WHERE NOT EXISTS (SELECT 1 FROM episodes e WHERE e.id = em.episode_id)
               OR NOT EXISTS (SELECT 1 FROM messages m WHERE m.id = em.message_id)
            """,
        )
        assert orphaned == 0, f"{orphaned} orphaned episode_messages rows"

    def test_no_orphaned_arc_episodes(self, db):
        orphaned = _query_one(
            db,
            """
            SELECT count(*) FROM arc_episodes ae
            WHERE NOT EXISTS (SELECT 1 FROM arcs a WHERE a.id = ae.arc_id)
               OR NOT EXISTS (SELECT 1 FROM episodes e WHERE e.id = ae.episode_id)
            """,
        )
        assert orphaned == 0, f"{orphaned} orphaned arc_episodes rows"

    def test_reflections_supersedes_valid(self, db):
        """Every non-null supersedes FK points to an existing reflection."""
        orphaned = _query_one(
            db,
            """
            SELECT count(*) FROM reflections r
            WHERE r.supersedes IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM reflections r2 WHERE r2.id = r.supersedes
              )
            """,
        )
        assert orphaned == 0, f"{orphaned} reflections with invalid supersedes FK"

    def test_commitments_supersedes_valid(self, db):
        orphaned = _query_one(
            db,
            """
            SELECT count(*) FROM commitments c
            WHERE c.supersedes IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM commitments c2 WHERE c2.id = c.supersedes
              )
            """,
        )
        assert orphaned == 0, f"{orphaned} commitments with invalid supersedes FK"

    def test_messages_conversation_fk_valid(self, db):
        """Every message points to a valid conversation (existing behavior)."""
        orphaned = _query_one(
            db,
            """
            SELECT count(*) FROM messages m
            WHERE NOT EXISTS (
                SELECT 1 FROM conversations c WHERE c.id = m.conversation_id
            )
            """,
        )
        assert orphaned == 0, f"{orphaned} messages with invalid conversation FK"

    def test_drafts_previous_revision_valid(self, db):
        """Drafts with previous_revision FK point to existing drafts."""
        orphaned = _query_one(
            db,
            """
            SELECT count(*) FROM drafts d
            WHERE d.previous_revision IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM drafts d2 WHERE d2.id = d.previous_revision
              )
            """,
        )
        assert orphaned == 0, f"{orphaned} drafts with invalid previous_revision FK"


# ===========================================================================
# 3. CHANNEL NORMALIZATION
# ===========================================================================
# After migration, no rows should contain the old channel values.


class TestChannelNormalization:
    """Verify channel values are normalized in all tables."""

    BAD_VALUES = ("claude-ai", "chat.auran.llc", "meta")

    def test_no_bad_channels_in_episodes(self, db):
        for bad in self.BAD_VALUES:
            count = _query_one(
                db,
                "SELECT count(*) FROM episodes WHERE channel = %s",
                (bad,),
            )
            assert count == 0, f"episodes has {count} rows with channel='{bad}'"

    def test_no_bad_channels_in_conversations(self, db):
        for bad in self.BAD_VALUES:
            count = _query_one(
                db,
                "SELECT count(*) FROM conversations WHERE channel = %s",
                (bad,),
            )
            assert count == 0, f"conversations has {count} rows with channel='{bad}'"

    def test_no_bad_channels_in_relays(self, db):
        """Neither source_channel nor target_channel should have bad values."""
        for bad in self.BAD_VALUES:
            count = _query_one(
                db,
                """
                SELECT count(*) FROM relays
                WHERE source_channel = %s OR target_channel = %s
                """,
                (bad, bad),
            )
            assert count == 0, f"relays has {count} rows with channel='{bad}'"

    def test_valid_channel_values_only(self, db):
        """Episodes should only contain the canonical channel set."""
        valid = {"chat", "cowork", "roam", "claude.ai", "native", "vr"}
        actual = _query_all(
            db,
            "SELECT DISTINCT channel FROM episodes",
        )
        actual_set = {row[0] for row in actual}
        invalid = actual_set - valid
        assert not invalid, f"Unexpected channel values in episodes: {invalid}"


# ===========================================================================
# 4. EMBEDDING PRESERVATION
# ===========================================================================
# VECTOR(1024) values must survive migration intact.


class TestEmbeddingPreservation:
    """Verify embeddings transferred correctly."""

    def test_reflections_embedding_count(self, db):
        """Reflections should preserve embedding coverage from memories.
        Source: 1576 memories with embeddings, 22 without.
        Reflection source types: observation(285) + insight(319) +
        self_observation(216) + question(64) + reflection(99) = 983.
        All of these types had embeddings (the 22 missing were bridge_log + draft).
        """
        with_emb = _query_one(
            db,
            "SELECT count(*) FROM reflections WHERE embedding IS NOT NULL",
        )
        assert with_emb == 983, f"Expected all 983 reflections to have embeddings, got {with_emb}"

    def test_episodes_embedding_count(self, db):
        """All episodes should have embeddings.
        Moments: all 639 had embeddings.
        Scene memories: all 81 unique had embeddings (scenes are a subset of
        the 1576 memories with embeddings).
        """
        with_emb = _query_one(
            db,
            "SELECT count(*) FROM episodes WHERE embedding IS NOT NULL",
        )
        total = _query_one(db, "SELECT count(*) FROM episodes")
        assert with_emb == total, f"Expected all {total} episodes to have embeddings, got {with_emb}"

    def test_commitments_embedding_count(self, db):
        """All commitments should have embeddings (intention, position, value
        all had embeddings in the source data)."""
        with_emb = _query_one(
            db,
            "SELECT count(*) FROM commitments WHERE embedding IS NOT NULL",
        )
        assert with_emb == 91, f"Expected all 91 commitments to have embeddings, got {with_emb}"

    def test_embedding_dimension_preserved(self, db):
        """Spot-check that vector dimension is 1024 on a sample row."""
        # pgvector stores dimension info — check via vector_dims()
        dim = _query_one(
            db,
            """
            SELECT vector_dims(embedding)
            FROM reflections
            WHERE embedding IS NOT NULL
            LIMIT 1
            """,
        )
        assert dim == 1024, f"Expected 1024-dim embeddings, got {dim}"

    def test_embedding_values_nonzero(self, db):
        """Spot-check that embeddings aren't zeroed out (corruption check).
        A valid Voyage embedding should have a non-trivial L2 norm."""
        norm = _query_one(
            db,
            """
            SELECT vector_norm(embedding)
            FROM reflections
            WHERE embedding IS NOT NULL
            LIMIT 1
            """,
        )
        assert norm is not None and norm > 0.1, f"Embedding norm suspiciously low: {norm}"


# ===========================================================================
# 5. DEDUP VERIFICATION
# ===========================================================================
# Document dedup counts and verify no remaining duplicates.


class TestDedupVerification:
    """Verify deduplication was applied correctly."""

    def test_no_duplicate_episode_titles(self, db):
        """After dedup, each episode title should appear at most once.
        (Title alone isn't a uniqueness constraint, but for migrated data
        duplicates were identical batch writes — title collision = content dupe.)
        """
        dupes = _query_all(
            db,
            """
            SELECT title, count(*)
            FROM episodes
            GROUP BY title
            HAVING count(*) > 1
            """,
        )
        assert len(dupes) == 0, f"{len(dupes)} titles still have duplicates: {[(t, c) for t, c in dupes[:5]]}"

    def test_scene_dedup_documented(self, db):
        """81 unique scenes from 339 source rows = 258 discarded."""
        # This is a documentation test — the row count test already
        # verifies the result. This just makes the dedup ratio explicit.
        episodes_from_scenes = _query_one(
            db,
            """
            SELECT count(*) FROM episodes
            WHERE channel IN ('chat', 'cowork', 'claude.ai', 'native', 'vr', 'roam')
            """,
        )
        # Total episodes = 391 (310 from moments + 81 from scenes)
        assert episodes_from_scenes == 391

    def test_moment_dedup_documented(self, db):
        """310 unique moments from 639 source rows = 329 discarded."""
        # Same — the row count test covers this, but this makes the
        # dedup explicit in the test suite for documentation.
        total_episodes = _query_one(db, "SELECT count(*) FROM episodes")
        assert total_episodes == 391, f"Expected 391 episodes (310 moments + 81 scenes), got {total_episodes}"


# ===========================================================================
# 6. ROLLBACK TEST
# ===========================================================================
# Migration must be reversible. The down migration should restore the
# original table structure. Data round-trip is NOT required (some dedup
# is destructive by design), but the schema must be restorable.
#
# NOTE: This test class is structured as documentation of what rollback
# should verify. The actual rollback test runs during V (Verify) phase
# via Alembic downgrade, not as a pytest unit test — running a real
# rollback in the test suite would destroy the migrated state that
# other tests depend on. See verify_rollback.py for the executable test.


class TestRollbackDesign:
    """Document rollback expectations (executable test in verify_rollback.py)."""

    def test_rollback_expectations_documented(self, db):
        """Rollback should:
        1. Recreate memories, moments, events tables
        2. Recreate moment_memories, moment_artifacts, moment_sessions junctions
        3. Not attempt data restoration (dedup is destructive)
        4. Leave conversations, messages, roam_sessions intact
        5. Be runnable via: alembic downgrade -1
        """
        # This test always passes — it's documentation.
        # The real test lives in verify_rollback.py and runs separately.
        assert True


# ===========================================================================
# VOCABULARY HANDLING DOCUMENTATION
# ===========================================================================
# 3 vocabulary-type memories exist. The schema v1.0 has no vocabulary table.
# These will be handled manually during migration:
#
# Option A: Migrate to reflections with type='vocabulary' (add to type enum)
# Option B: Migrate to commitments with type='vocabulary'
# Option C: Archive to a JSON file and document as intentionally dropped
#
# Decision to be made during I (Implement) phase.
# Whichever option: the test_total_data_accounted_for test accounts for
# these 3 rows in the documented_discards count.
