import re


def compute_transcript_coverage(jobs: list[dict]) -> list[dict]:
    return [
        {
            "transcript_file": job["transcript_file"],
            "channel": job["channel"],
            "status": job["status"],
            "episode_count": job.get("episode_count", 0),
        }
        for job in jobs
    ]


def aggregate_by_channel(jobs: list[dict]) -> dict:
    channels: dict[str, dict] = {}

    for job in jobs:
        channel = job["channel"]
        if channel not in channels:
            channels[channel] = {"transcript_count": 0, "total_episodes": 0}
        channels[channel]["transcript_count"] += 1
        channels[channel]["total_episodes"] += job.get("episode_count", 0)

    return channels


def calculate_line_coverage(episodes: list[dict]) -> dict:
    if not episodes:
        return {"total_lines_covered": 0, "ranges": []}

    ranges = []
    for ep in episodes:
        line_ref = ep.get("transcript_lines", "")
        if not line_ref:
            continue

        match = re.match(r"L(\d+)(?:-L(\d+))?", line_ref)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            ranges.append((start, end))

    ranges.sort()
    total = sum(end - start + 1 for start, end in ranges)

    return {"total_lines_covered": total, "ranges": ranges}


def detect_gaps(
    transcript_inventory: list[str],
    covered_files: set[str],
) -> list[str]:
    return [f for f in transcript_inventory if f not in covered_files]
