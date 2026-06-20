import logging

logger = logging.getLogger(__name__)


def phase_1_migrate_old_rows(connection) -> int:
    """Migrate 339+79 rows from old tables into episodes with distillation_status='manual'."""
    logger.info("Phase 1: Migrating old scene/moment rows to episodes table...")

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO episodes (
            id, title, summary, channel, occurred_at,
            distillation_status, created_at
        )
        SELECT
            id, title, summary, channel, occurred_at,
            'manual', created_at
        FROM memories m
        WHERE m.memory_type = 'scene'
        AND NOT EXISTS (SELECT 1 FROM episodes e WHERE e.id = m.id)
        ON CONFLICT DO NOTHING
    """)
    scenes_migrated = cursor.rowcount

    cursor.execute("""
        INSERT INTO episodes (
            id, title, summary, channel, occurred_at,
            distillation_status, created_at
        )
        SELECT
            id, title, content, channel, occurred_at,
            'manual', created_at
        FROM moments m
        WHERE NOT EXISTS (SELECT 1 FROM episodes e WHERE e.id = m.id)
        ON CONFLICT DO NOTHING
    """)
    moments_migrated = cursor.rowcount

    connection.commit()
    total = scenes_migrated + moments_migrated
    logger.info(
        "Phase 1 complete: %d rows migrated (%d scenes, %d moments)",
        total,
        scenes_migrated,
        moments_migrated,
    )
    return total


def phase_2_import_local_json(connection, json_dir: str) -> int:
    """Import existing local JSON distillations and route through HITL."""
    import json
    from pathlib import Path

    from distillation.dedup import content_hash

    logger.info("Phase 2: Importing local JSON distillations from %s...", json_dir)

    json_path = Path(json_dir)
    if not json_path.exists():
        logger.warning("JSON directory does not exist: %s", json_dir)
        return 0

    imported = 0
    cursor = connection.cursor()

    for json_file in sorted(json_path.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", json_file.name, e)
            continue

        episodes = data if isinstance(data, list) else data.get("episodes", [])

        for ep in episodes:
            summary = ep.get("summary", "")
            if not summary:
                continue

            c_hash = content_hash(summary)

            cursor.execute(
                """
                INSERT INTO episodes (
                    title, summary, content_hash, channel, occurred_at,
                    transcript_file, distillation_status
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'pending_review')
                ON CONFLICT DO NOTHING
            """,
                (
                    ep.get("title", "Untitled"),
                    summary,
                    c_hash,
                    ep.get("channel", "unknown"),
                    ep.get("occurred_at"),
                    ep.get("transcript_file"),
                ),
            )
            if cursor.rowcount > 0:
                imported += 1

    connection.commit()
    logger.info("Phase 2 complete: %d episodes imported for review", imported)
    return imported


def phase_3_queue_remaining(connection, transcript_inventory: list[str], channel: str) -> int:
    """Submit remaining transcripts to the distillation queue in chronological order."""
    logger.info("Phase 3: Queueing %d transcripts for distillation...", len(transcript_inventory))

    cursor = connection.cursor()
    queued = 0

    for transcript_file in sorted(transcript_inventory):
        cursor.execute(
            """
            INSERT INTO distillation_jobs (transcript_file, channel, status)
            VALUES (%s, %s, 'queued')
            ON CONFLICT (transcript_file) DO NOTHING
        """,
            (transcript_file, channel),
        )
        if cursor.rowcount > 0:
            queued += 1

    connection.commit()
    logger.info("Phase 3 complete: %d transcripts queued", queued)
    return queued


def _format_embedding(vec: list[float]) -> str:
    """Format a float vector as a pgvector literal."""
    return f"[{','.join(str(x) for x in vec)}]"


def backfill_embeddings(connection, voyage_client, batch_size=50, dry_run=False) -> int:
    """Generate Voyage AI embeddings for episodes that don't have them yet."""
    cursor = connection.cursor()
    cursor.execute(
        "SELECT id, title, summary FROM episodes WHERE embedding IS NULL ORDER BY created_at"
    )
    rows = cursor.fetchall()

    if not rows:
        logger.info("All episodes already have embeddings — nothing to do.")
        return 0

    logger.info("%d episodes need embeddings.", len(rows))
    if dry_run:
        logger.info("Dry run — skipping API calls and writes.")
        return 0

    updated = 0
    failed_batches = 0

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        texts = [row[2] or row[1] for row in batch]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(rows) + batch_size - 1) // batch_size

        try:
            result = voyage_client.embed(texts, model="voyage-3")
        except Exception:
            failed_batches += 1
            logger.exception(
                "Batch %d/%d failed — skipping %d episodes.",
                batch_num,
                total_batches,
                len(batch),
            )
            continue

        for (ep_id, _, _), vec in zip(batch, result.embeddings):
            cursor.execute(
                "UPDATE episodes SET embedding = %s WHERE id = %s",
                (_format_embedding(vec), ep_id),
            )
            updated += 1

        connection.commit()
        logger.info("Batch %d/%d: embedded %d episodes.", batch_num, total_batches, len(batch))

    if failed_batches:
        logger.warning("Done. %d updated, %d batch(es) failed.", updated, failed_batches)
    else:
        logger.info("Done. %d episodes embedded.", updated)

    return updated
