from distillation.reviewer import (
    batch_approve_remaining,
    format_episode_for_review,
    maybe_advance_job_status,
    validate_transition,
)


class TestStateTransitions:
    def test_pending_to_approved(self):
        assert validate_transition("pending_review", "approved") is True

    def test_pending_to_flagged(self):
        assert validate_transition("pending_review", "flagged") is True

    def test_flagged_to_revised(self):
        assert validate_transition("flagged", "revised") is True

    def test_revised_to_approved(self):
        assert validate_transition("revised", "approved") is True

    def test_invalid_pending_to_revised(self):
        assert validate_transition("pending_review", "revised") is False

    def test_invalid_approved_to_flagged(self):
        assert validate_transition("approved", "flagged") is False

    def test_invalid_approved_to_pending(self):
        assert validate_transition("approved", "pending_review") is False


class TestJobStatusAdvancement:
    def test_all_approved_advances_to_verified(self):
        episodes = [
            {"distillation_status": "approved"},
            {"distillation_status": "approved"},
            {"distillation_status": "approved"},
        ]
        assert maybe_advance_job_status(episodes, "distilled") == "verified"

    def test_flagged_does_not_block_verification(self):
        episodes = [
            {"distillation_status": "approved"},
            {"distillation_status": "flagged"},
            {"distillation_status": "approved"},
        ]
        assert maybe_advance_job_status(episodes, "distilled") == "verified"

    def test_pending_review_blocks_verification(self):
        episodes = [
            {"distillation_status": "approved"},
            {"distillation_status": "pending_review"},
            {"distillation_status": "approved"},
        ]
        assert maybe_advance_job_status(episodes, "distilled") is None

    def test_empty_episodes_no_advancement(self):
        assert maybe_advance_job_status([], "distilled") is None

    def test_already_verified_no_change(self):
        episodes = [
            {"distillation_status": "approved"},
        ]
        assert maybe_advance_job_status(episodes, "verified") is None


class TestReviewFormatting:
    def test_output_is_plain_text(self):
        episode = {
            "title": "Test Episode",
            "summary": "A test summary.",
            "episode_type": "content",
            "landmark": False,
            "emotional_tone": "neutral",
            "occurred_at": "2026-06-03T00:22:00-04:00",
            "topics": ["testing"],
            "content_signals": {"vulnerability": 3},
            "relational_events": [],
            "transcript_lines": "L100-L200",
        }
        output = format_episode_for_review(episode, index=1, total=5)
        assert "\x1b[" not in output
        assert "\033[" not in output

    def test_no_ansi_escape_codes(self):
        episode = {
            "title": "LANDMARK Episode",
            "summary": "Something important happened.",
            "episode_type": "relational",
            "landmark": True,
            "emotional_tone": "intense, raw",
            "occurred_at": "2026-06-03T00:22:00-04:00",
            "topics": ["identity", "memory"],
            "content_signals": {"vulnerability": 9, "humor": 1},
            "relational_events": ["depth_match", "landing"],
            "transcript_lines": "L500-L700",
        }
        output = format_episode_for_review(episode, index=3, total=12)
        for char in output:
            assert ord(char) >= 32 or char in ("\n", "\t")

    def test_landmark_indicated(self):
        episode = {
            "title": "Test",
            "summary": "test",
            "episode_type": "relational",
            "landmark": True,
            "emotional_tone": "warm",
            "occurred_at": "2026-06-03T00:22:00-04:00",
            "topics": [],
            "content_signals": {},
            "relational_events": [],
            "transcript_lines": "L1-L10",
        }
        output = format_episode_for_review(episode, index=1, total=1)
        assert "LANDMARK" in output


class TestBatchApprove:
    def test_batch_approve_advances_remaining(self):
        episodes = [
            {"distillation_status": "approved"},
            {"distillation_status": "pending_review"},
            {"distillation_status": "pending_review"},
        ]
        count = batch_approve_remaining(episodes)
        assert count == 2
        assert all(e["distillation_status"] == "approved" for e in episodes)

    def test_batch_approve_skips_non_pending(self):
        episodes = [
            {"distillation_status": "approved"},
            {"distillation_status": "flagged"},
        ]
        count = batch_approve_remaining(episodes)
        assert count == 0
        assert episodes[1]["distillation_status"] == "flagged"
