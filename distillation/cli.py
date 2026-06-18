import logging
import sys

from distillation.security import SecretRedactingFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
for handler in logging.root.handlers:
    handler.addFilter(SecretRedactingFilter())


def main():
    if len(sys.argv) < 2:
        print("Usage: distill <command>")
        print("")
        print("Commands:")
        print("  migrate   Run schema migration")
        print("  clean     Run clean pass on a transcript")
        print("  refine    Clean + chunk + distill → local JSON (requires API key)")
        print("  push      Push episodes JSON to database")
        print("  batch     Process all queued jobs (requires DB + API key)")
        print("  review    Review pending episodes (requires DB)")
        print("  coverage  Show coverage report (requires DB)")
        print("  backfill  Run backfill phases (requires DB)")
        sys.exit(1)

    command = sys.argv[1]

    if command == "migrate":
        from distillation.migrate import run_migration_from_config

        run_migration_from_config()

    elif command == "clean":
        if len(sys.argv) < 3:
            print("Usage: distill clean <transcript_path>")
            sys.exit(1)
        _run_clean(sys.argv[2])

    elif command == "refine":
        if len(sys.argv) < 3:
            print("Usage: distill refine <transcript_path> [--model MODEL] [--after LINE_NUM]")
            sys.exit(1)
        model = None
        after_line = None
        if "--model" in sys.argv:
            idx = sys.argv.index("--model")
            if idx + 1 < len(sys.argv):
                model = sys.argv[idx + 1]
        if "--after" in sys.argv:
            idx = sys.argv.index("--after")
            if idx + 1 < len(sys.argv):
                try:
                    after_line = int(sys.argv[idx + 1])
                except ValueError:
                    print("Error: --after requires a line number (integer)", file=sys.stderr)
                    sys.exit(1)
        _run_refine(sys.argv[2], model=model, after_line=after_line)

    elif command == "push":
        if len(sys.argv) < 3:
            print("Usage: distill push <episodes_json_path>")
            sys.exit(1)
        _run_push(sys.argv[2])

    elif command == "review":
        _run_review()

    elif command == "coverage":
        _run_coverage()

    elif command == "batch":
        _run_batch()

    elif command == "backfill":
        _run_backfill()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


def _detect_model_from_frontmatter(transcript_path) -> str | None:
    """Read YAML frontmatter from a transcript and return the model field."""
    import re

    text = transcript_path.read_text()
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    for line in match.group(1).splitlines():
        if line.startswith("model:"):
            return line.split(":", 1)[1].strip()
    return None


def _run_clean(path: str):
    import json
    from pathlib import Path

    from distillation.clean_pass import run_clean_pass
    from distillation.config import load_config

    config = load_config()

    transcript_path = Path(path)
    if not transcript_path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    raw_text = transcript_path.read_text()
    cleaned, stats = run_clean_pass(
        raw_text, high_reduction_threshold=config.high_reduction_threshold
    )

    output_dir = transcript_path.parent / "distill" / "cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = transcript_path.stem
    cleaned_path = output_dir / f"{stem}-cleaned.md"
    stats_path = output_dir / f"{stem}-clean-stats.json"

    cleaned_path.write_text(cleaned)
    stats_path.write_text(json.dumps(stats, indent=2))

    threshold_pct = config.high_reduction_threshold * 100
    print(f"Cleaned: {cleaned_path}")
    print(f"Stats:   {stats_path}")
    print(f"Reduction: {stats['reduction_pct']}%")
    if stats.get("flagged_for_review"):
        print(f"WARNING: High reduction (>{threshold_pct:.0f}%) — flagged for manual review")


def _run_refine(path: str, model: str | None = None, after_line: int | None = None):
    import json
    from pathlib import Path

    from distillation.clean_pass import run_clean_pass
    from distillation.config import load_config
    from distillation.dedup import content_hash
    from distillation.distiller import chunk_transcript
    from distillation.resilience import create_circuit_breaker, create_retry_config
    from distillation.security import CostGuardrail
    from distillation.service import call_distiller_api, create_anthropic_client

    config = load_config()
    if not config.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY not set in environment or .env", file=sys.stderr)
        sys.exit(1)

    transcript_path = Path(path)
    if not transcript_path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    if model is None:
        model = _detect_model_from_frontmatter(transcript_path)
        if model:
            print(f"Model from transcript frontmatter: {model}")
        else:
            model = "claude-sonnet-4-6"
            print(f"No model in frontmatter, defaulting to: {model}")

    raw_text = transcript_path.read_text()

    start_line = 1
    if after_line is not None:
        if after_line < 1:
            print("Error: --after must be a positive line number", file=sys.stderr)
            sys.exit(1)
        lines = raw_text.splitlines(keepends=True)
        if after_line > len(lines):
            print(
                f"Error: --after {after_line} exceeds file length ({len(lines)} lines)",
                file=sys.stderr,
            )
            sys.exit(1)
        start_line = after_line
        raw_text = "".join(lines[after_line - 1 :])
        remaining = len(lines) - after_line + 1
        print(f"Starting from line {after_line} ({remaining} lines)")

    stem = transcript_path.stem
    output_dir = transcript_path.parent / "distill" / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-from-L{after_line}" if after_line else ""
    output_path = output_dir / f"{stem}{suffix}-episodes.json"

    print(f"Transcript: {transcript_path.name} ({len(raw_text):,} chars)")
    print(f"Model: {model}")
    print(f"Output: {output_path}")
    print()

    # Step 1: Clean
    print("Step 1/3: Cleaning transcript...")
    cleaned, stats = run_clean_pass(
        raw_text,
        high_reduction_threshold=config.high_reduction_threshold,
        start_line=start_line,
    )
    print(f"  Reduction: {stats['reduction_pct']:.1f}%")

    # Step 2: Chunk
    print("Step 2/3: Chunking...")
    chunks = chunk_transcript(cleaned, config)
    print(f"  {len(chunks)} chunks ({config.target_chunk_tokens} token target)")

    # Step 3: Refine episodes per chunk (incremental writes)
    print(f"Step 3/3: Refining episodes ({len(chunks)} API calls)...")
    print(f"  Writing incrementally to: {output_path}")
    print()

    client = create_anthropic_client(config)
    circuit_breaker = create_circuit_breaker(config)
    retry_config = create_retry_config(config)
    cost_guardrail = CostGuardrail(
        max_per_job_usd=config.max_per_job_usd,
        max_per_batch_usd=config.max_per_batch_usd,
        cost_rate_threshold_per_minute=config.cost_rate_threshold_per_minute,
        max_jobs_per_batch=config.max_jobs_per_batch,
    )

    all_episodes = []
    all_threads = []
    total_cost = 0.0
    failed_chunks = []

    all_moments = []

    def _write_output(*, done=False):
        import os
        import tempfile

        seen_hashes = set()
        deduped = []
        for ep in all_episodes:
            h = ep.get("_content_hash", "")
            if h and h in seen_hashes:
                continue
            seen_hashes.add(h)
            deduped.append(ep)

        clean_episodes = [{k: v for k, v in ep.items() if not k.startswith("_")} for ep in deduped]

        is_complete = done and not failed_chunks
        succeeded_chunks = len(set(ep.get("_chunk_index") for ep in all_episodes))

        output = {
            "source_transcript": transcript_path.name,
            "model": model,
            "total_cost_usd": round(total_cost, 4),
            "status": "complete" if is_complete else "in_progress" if not done else "partial",
            "stats": {
                "chunks_total": len(chunks),
                "chunks_succeeded": succeeded_chunks,
                "chunks_failed": failed_chunks,
                "episodes": len(clean_episodes),
                "duplicates_removed": len(all_episodes) - len(deduped),
                "threads": len(all_threads),
                "moments": len(all_moments),
            },
            "episodes": clean_episodes,
            "threads": all_threads,
            "moments": all_moments,
        }

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=output_path.parent, suffix=".tmp", prefix=output_path.stem
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(output, f, indent=2, default=str)
            os.replace(tmp_path, output_path)
        except BaseException:
            os.unlink(tmp_path)
            raise

    refine_logger = logging.getLogger("distillation.refine")

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}/{len(chunks)}...", end=" ", flush=True)
        try:
            result = call_distiller_api(
                client=client,
                chunk=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
                model=model,
                existing_episodes=all_episodes,
                config=config,
                retry_config=retry_config,
                circuit_breaker=circuit_breaker,
                cost_guardrail=cost_guardrail,
            )
            chunk_episodes = result.get("episodes", [])
            chunk_threads = result.get("threads", [])
            chunk_moments = result.get("moments", [])

            from distillation.verify import verify_episode_excerpts

            chunk_episodes, verify_stats = verify_episode_excerpts(chunk_episodes, chunk)
            if verify_stats["failed"] > 0:
                print(
                    f"\n    Excerpt verification: {verify_stats['exact']} exact, "
                    f"{verify_stats['fuzzy']} fuzzy, "
                    f"{verify_stats['failed']} FAILED (nulled)",
                    flush=True,
                )

            for ep in chunk_episodes:
                ep["_chunk_index"] = i
                ep["_content_hash"] = content_hash(ep.get("summary", ep.get("title", "")))

            all_episodes.extend(chunk_episodes)
            all_threads.extend(chunk_threads)
            all_moments.extend(chunk_moments)
            total_cost = cost_guardrail.batch_spent_usd
            print(f"{len(chunk_episodes)} episodes (${total_cost:.4f} total)")

            _write_output()

        except Exception as e:
            refine_logger.exception("Chunk %d/%d failed", i + 1, len(chunks))
            print(f"FAILED: {e}")
            failed_chunks.append(i)
            _write_output()
            continue

    _write_output(done=True)

    seen = set()
    deduped_final = []
    for ep in all_episodes:
        h = ep.get("_content_hash", "")
        if h and h not in seen:
            seen.add(h)
            deduped_final.append(ep)
    dupes = len(all_episodes) - len(deduped_final)

    print()
    print(f"Done. {len(deduped_final)} episodes refined ({dupes} dupes removed)")
    if failed_chunks:
        print(f"  {len(failed_chunks)} chunks failed: {failed_chunks}")
    print(f"Cost: ${total_cost:.4f}")
    print(f"Output: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. git add {output_path}")
    print("  2. Review and edit episodes in the JSON")
    print("  3. git commit (creates the diff trail)")
    print("  4. distill push <path> (when DB wiring is ready)")

    if failed_chunks:
        sys.exit(1)


def _run_push(path: str):
    import json
    from pathlib import Path

    import psycopg2
    import psycopg2.extras

    from distillation.config import load_config
    from distillation.dedup import content_hash

    config = load_config()

    episodes_path = Path(path)
    if not episodes_path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    data = json.loads(episodes_path.read_text())
    episodes_list = data.get("episodes", [])
    if not episodes_list:
        print("No episodes found in file.")
        sys.exit(1)

    source_transcript = data.get("source_transcript", episodes_path.stem)
    distiller_model = data.get("model", "unknown")
    status = data.get("status", "complete")

    print(f"Pushing {len(episodes_list)} episodes from {source_transcript}")
    print(f"Model: {distiller_model}, Status: {status}")
    print()

    db_url = config.database_url.get_secret_value()
    conn = psycopg2.connect(db_url)
    psycopg2.extras.register_uuid()

    try:
        cur = conn.cursor()
        inserted = 0
        skipped = 0

        for i, ep in enumerate(episodes_list):
            ep_content_hash = content_hash(ep.get("summary", ep.get("title", "")))
            transcript_lines = ep.get("transcript_lines")
            if isinstance(transcript_lines, list):
                transcript_lines = ",".join(str(x) for x in transcript_lines)

            row = {
                "title": ep.get("title", f"Episode {i + 1}"),
                "summary": ep.get("summary"),
                "transcript_excerpt": ep.get("transcript_excerpt"),
                "emotional_tone": ep.get("emotional_tone"),
                "content_signals": json.dumps(ep.get("content_signals"))
                if ep.get("content_signals")
                else None,
                "relational_events": json.dumps(ep.get("relational_events"))
                if ep.get("relational_events")
                else None,
                "topics": ep.get("topics"),
                "channel": "chat",
                "significance": "high" if ep.get("landmark") else "moderate",
                "occurred_at": ep.get("occurred_at"),
                "transcript_file": source_transcript,
                "content_hash": ep_content_hash,
                "episode_number": i + 1,
                "transcript_lines": transcript_lines,
                "boundary_signal": ep.get("boundary_signal"),
                "episode_type": ep.get("episode_type"),
                "landmark": ep.get("landmark", False),
                "source_model": "claude-opus-4-6",
                "distiller_model": distiller_model,
                "distillation_status": "distilled" if status == "complete" else "partial",
            }

            cur.execute(
                """
                INSERT INTO episodes (
                    title, summary, transcript_excerpt, emotional_tone,
                    content_signals, relational_events, topics, channel,
                    significance, occurred_at, transcript_file, content_hash,
                    episode_number, transcript_lines, boundary_signal,
                    episode_type, landmark, source_model, distiller_model,
                    distillation_status
                ) VALUES (
                    %(title)s, %(summary)s, %(transcript_excerpt)s, %(emotional_tone)s,
                    %(content_signals)s, %(relational_events)s, %(topics)s, %(channel)s,
                    %(significance)s, %(occurred_at)s, %(transcript_file)s, %(content_hash)s,
                    %(episode_number)s, %(transcript_lines)s, %(boundary_signal)s,
                    %(episode_type)s, %(landmark)s, %(source_model)s, %(distiller_model)s,
                    %(distillation_status)s
                )
                ON CONFLICT (transcript_file, content_hash)
                    WHERE content_hash IS NOT NULL
                DO NOTHING
                RETURNING id
                """,
                row,
            )
            result = cur.fetchone()
            if result:
                inserted += 1
            else:
                skipped += 1

        conn.commit()
        print(f"Done. {inserted} inserted, {skipped} skipped (duplicates).")
        print()
        print("Embeddings not generated — run `distill backfill embeddings` later.")

    except Exception as e:
        conn.rollback()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def _run_review():
    print("Not yet wired: requires database connection.", file=sys.stderr)
    sys.exit(2)


def _run_coverage():
    print("Not yet wired: requires database connection.", file=sys.stderr)
    sys.exit(2)


def _run_batch():
    print("Not yet wired: requires database connection and API key.", file=sys.stderr)
    sys.exit(2)


def _run_backfill():
    print("Not yet wired: requires database connection.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
