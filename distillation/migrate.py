import logging

logger = logging.getLogger(__name__)

MIGRATION_SQL = """
-- Episode distillation columns (added to existing episodes table)
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    transcript_file     TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    job_id              UUID;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    content_hash        TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    episode_number      INT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    transcript_lines    TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    transcript_excerpt  TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    boundary_signal     TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    emotional_tone      TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    episode_type        TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    landmark            BOOLEAN DEFAULT FALSE;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    content_signals     JSONB;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    relational_events   TEXT[];

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    source_model        TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    distiller_model     TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    distillation_status TEXT DEFAULT 'manual';

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    revision_count      INT DEFAULT 0;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    reviewer_notes      TEXT;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    verified_at         TIMESTAMPTZ;

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS
    verified_by         TEXT;

-- Dedup index
CREATE UNIQUE INDEX IF NOT EXISTS episodes_transcript_hash_idx
    ON episodes(transcript_file, content_hash)
    WHERE content_hash IS NOT NULL;

-- Distillation jobs table
CREATE TABLE IF NOT EXISTS distillation_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript_file TEXT NOT NULL,
    channel         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    total_chunks    INT,
    chunks_done     INT DEFAULT 0,
    episode_count   INT,
    episodes_verified INT DEFAULT 0,
    episodes_approved INT DEFAULT 0,
    api_cost_usd    NUMERIC(8,4),
    source_model    TEXT,
    distiller_model TEXT,
    error_message   TEXT,
    submitted_at    TIMESTAMPTZ DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    verified_at     TIMESTAMPTZ,
    verified_by     TEXT,
    UNIQUE(transcript_file)
);

-- Distillation threads table
CREATE TABLE IF NOT EXISTS distillation_threads (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID NOT NULL REFERENCES distillation_jobs(id),
    thread_type     TEXT NOT NULL,
    title           TEXT,
    content         TEXT NOT NULL,
    line_ref        TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Dead letters table
CREATE TABLE IF NOT EXISTS distillation_dead_letters (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id      UUID REFERENCES distillation_jobs(id),
    chunk_index INT,
    error_type  TEXT,
    error_msg   TEXT,
    raw_response TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    resolved_at TIMESTAMPTZ,
    resolution  TEXT
);

-- Episode references table
CREATE TABLE IF NOT EXISTS episode_references (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_episode_id   UUID NOT NULL REFERENCES episodes(id),
    target_episode_id   UUID REFERENCES episodes(id),
    reference_type      TEXT NOT NULL,
    context             TEXT,
    flagged             BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE(source_episode_id, target_episode_id, reference_type)
);
"""


def run_migration(connection) -> None:
    logger.info("Running distillation schema migration...")
    cursor = connection.cursor()
    for statement in MIGRATION_SQL.split(";"):
        statement = statement.strip()
        if statement and not statement.startswith("--"):
            cursor.execute(statement)
    connection.commit()
    logger.info("Migration complete.")


def run_migration_from_config():
    import psycopg2

    from distillation.config import load_config

    config = load_config()
    db_url = config.database_url.get_secret_value()
    conn = psycopg2.connect(db_url)
    try:
        run_migration(conn)
    finally:
        conn.close()
