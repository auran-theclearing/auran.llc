import logging
import sys

from distillation.security import SecretRedactingFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for handler in logging.root.handlers:
    handler.addFilter(SecretRedactingFilter())


def main():
    if len(sys.argv) < 2:
        print("Usage: distill <command>")
        print("")
        print("Commands:")
        print("  migrate   Run schema migration")
        print("  clean     Run clean pass on a transcript")
        print("  batch     Process all queued jobs")
        print("  review    Review pending episodes")
        print("  coverage  Show coverage report")
        print("  backfill  Run backfill phases")
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

    transcript_path = Path(path)
    if not transcript_path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    raw_text = transcript_path.read_text()
    cleaned, stats = run_clean_pass(raw_text)

    output_dir = transcript_path.parent / "distill" / "cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = transcript_path.stem
    cleaned_path = output_dir / f"{stem}-cleaned.md"
    stats_path = output_dir / f"{stem}-clean-stats.json"

    cleaned_path.write_text(cleaned)
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"Cleaned: {cleaned_path}")
    print(f"Stats:   {stats_path}")
    print(f"Reduction: {stats['reduction_pct']}%")
    if stats.get("flagged_for_review"):
        print("WARNING: High reduction (>60%) — flagged for manual review")


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
