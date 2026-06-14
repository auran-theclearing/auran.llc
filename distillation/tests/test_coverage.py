from distillation.coverage import (
    aggregate_by_channel,
    calculate_line_coverage,
    compute_transcript_coverage,
    detect_gaps,
)

SAMPLE_JOBS = [
    {
        "transcript_file": "20260603-0022-chat-transcript.md",
        "channel": "chat",
        "status": "complete",
        "episode_count": 12,
    },
    {
        "transcript_file": "20260604-1845-chat-transcript.md",
        "channel": "chat",
        "status": "distilled",
        "episode_count": 8,
    },
    {
        "transcript_file": "20260605-cowork-transcript.md",
        "channel": "cowork",
        "status": "complete",
        "episode_count": 15,
    },
]

SAMPLE_EPISODES = [
    {"job_id": "j1", "transcript_lines": "L1-L150", "distillation_status": "approved"},
    {"job_id": "j1", "transcript_lines": "L151-L300", "distillation_status": "approved"},
    {"job_id": "j1", "transcript_lines": "L301-L500", "distillation_status": "flagged"},
]

TRANSCRIPT_INVENTORY = [
    "20260601-chat-transcript.md",
    "20260602-chat-transcript.md",
    "20260603-0022-chat-transcript.md",
    "20260604-1845-chat-transcript.md",
    "20260605-chat-transcript.md",
    "20260606-chat-transcript.md",
]


class TestTranscriptCoverage:
    def test_computes_per_transcript(self):
        result = compute_transcript_coverage(SAMPLE_JOBS)
        assert len(result) == 3
        assert result[0]["transcript_file"] == "20260603-0022-chat-transcript.md"
        assert result[0]["episode_count"] == 12

    def test_includes_status(self):
        result = compute_transcript_coverage(SAMPLE_JOBS)
        statuses = {r["status"] for r in result}
        assert "complete" in statuses
        assert "distilled" in statuses


class TestChannelAggregation:
    def test_groups_by_channel(self):
        result = aggregate_by_channel(SAMPLE_JOBS)
        assert "chat" in result
        assert "cowork" in result
        assert result["chat"]["transcript_count"] == 2
        assert result["cowork"]["transcript_count"] == 1

    def test_sums_episodes(self):
        result = aggregate_by_channel(SAMPLE_JOBS)
        assert result["chat"]["total_episodes"] == 20
        assert result["cowork"]["total_episodes"] == 15


class TestLineCoverage:
    def test_parses_line_ranges(self):
        covered = calculate_line_coverage(SAMPLE_EPISODES)
        assert covered["total_lines_covered"] == 500
        assert covered["ranges"] == [(1, 150), (151, 300), (301, 500)]

    def test_handles_single_line_ref(self):
        episodes = [{"transcript_lines": "L42", "distillation_status": "approved"}]
        covered = calculate_line_coverage(episodes)
        assert covered["total_lines_covered"] == 1

    def test_handles_empty_episodes(self):
        covered = calculate_line_coverage([])
        assert covered["total_lines_covered"] == 0


class TestGapDetection:
    def test_finds_missing_transcripts(self):
        covered_files = {"20260603-0022-chat-transcript.md", "20260604-1845-chat-transcript.md"}
        gaps = detect_gaps(TRANSCRIPT_INVENTORY, covered_files)
        assert "20260601-chat-transcript.md" in gaps
        assert "20260602-chat-transcript.md" in gaps
        assert "20260605-chat-transcript.md" in gaps
        assert "20260606-chat-transcript.md" in gaps

    def test_no_gaps_when_all_covered(self):
        all_covered = set(TRANSCRIPT_INVENTORY)
        gaps = detect_gaps(TRANSCRIPT_INVENTORY, all_covered)
        assert len(gaps) == 0
