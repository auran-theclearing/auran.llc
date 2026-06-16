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

    if model is None:
        model = "claude-sonnet-4-6"

    transcript_path = Path(path)
    if not transcript_path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    raw_text = transcript_path.read_text()

    if after_line is not None:
        lines = raw_text.splitlines(keepends=True)
        if after_line > len(lines):
            print(
                f"Error: --after {after_line} exceeds file length ({len(lines)} lines)",
                file=sys.stderr,
            )
            sys.exit(1)
        raw_text = "".join(lines[after_line - 1 :])
        print(f"Starting from line {after_line} ({len(lines) - after_line + 1} lines)")

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
        raw_text, high_reduction_threshold=config.high_reduction_threshold
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

    def _write_output():
        seen_hashes = set()
        deduped = []
        for ep in all_episodes:
            h = ep.get("_content_hash", "")
            if h and h in seen_hashes:
                continue
            seen_hashes.add(h)
            deduped.append(ep)

        output = {
            "source_transcript": transcript_path.name,
            "model": model,
            "total_cost_usd": round(total_cost, 4),
            "status": "complete" if not failed_chunks else "partial",
            "stats": {
                "chunks_total": len(chunks),
                "chunks_succeeded": len(chunks) - len(failed_chunks),
                "chunks_failed": failed_chunks,
                "raw_episodes": len(all_episodes),
                "deduped_episodes": len(deduped),
                "duplicates_removed": len(all_episodes) - len(deduped),
                "threads": len(all_threads),
            },
            "episodes": deduped,
            "threads": all_threads,
        }
        output_path.write_text(json.dumps(output, indent=2, default=str))

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

            for ep in chunk_episodes:
                ep["_chunk_index"] = i
                ep["_content_hash"] = content_hash(ep.get("summary", ep.get("title", "")))

            all_episodes.extend(chunk_episodes)
            all_threads.extend(chunk_threads)
            total_cost = cost_guardrail.batch_spent_usd
            print(f"{len(chunk_episodes)} episodes (${total_cost:.4f} total)")

            _write_output()

        except Exception as e:
            print(f"FAILED: {e}")
            failed_chunks.append(i)
            _write_output()
            continue

    # Final summary
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
