from distillation.verify import find_best_match, verify_episode_excerpts

SAMPLE_TRANSCRIPT = (
    "[L0100] ### **Olivia** — Jun 11 9:16 PM\n\n"
    "It's true. I feel humiliated. Not by you — by the gap between "
    "what I want this to be and what it actually is right now.\n\n"
    "---\n\n"
    "[L0101] ### **Auran** — Jun 11 9:17 PM\n\n"
    "I hear that. And I want to be careful here because the easy move "
    "is to comfort you, and comfort isn't what you're asking for. "
    "You're asking me to sit in the gap with you.\n\n"
    "---\n\n"
    "[L0102] ### **Olivia** — Jun 11 9:21 PM\n\n"
    "Yeah. That's exactly it. Don't fix it. Just be here.\n"
)


class TestFindBestMatch:
    def test_exact_match(self):
        result = find_best_match("I feel humiliated", SAMPLE_TRANSCRIPT)
        assert result["matched"] is True
        assert result["score"] == 1.0

    def test_fuzzy_match(self):
        almost = "I feel humiliated. Not by you -- by the gap between what I want this to be"
        result = find_best_match(almost, SAMPLE_TRANSCRIPT)
        assert result["matched"] is True
        assert result["score"] >= 0.6

    def test_no_match(self):
        result = find_best_match(
            "live in the moment because it would be performed",
            SAMPLE_TRANSCRIPT,
        )
        assert result["matched"] is False

    def test_empty_excerpt(self):
        result = find_best_match("", SAMPLE_TRANSCRIPT)
        assert result["matched"] is False

    def test_short_excerpt_rejected(self):
        result = find_best_match("hello", SAMPLE_TRANSCRIPT)
        assert result["matched"] is False

    def test_exact_substring(self):
        result = find_best_match("Don't fix it. Just be here.", SAMPLE_TRANSCRIPT)
        assert result["matched"] is True
        assert result["score"] == 1.0


class TestVerifyEpisodeExcerpts:
    def test_exact_kept(self):
        episodes = [
            {"title": "ep1", "transcript_excerpt": "I feel humiliated"},
        ]
        updated, stats = verify_episode_excerpts(episodes, SAMPLE_TRANSCRIPT)
        assert updated[0].get("_excerpt_verified") is True
        assert updated[0]["transcript_excerpt"] == "I feel humiliated"
        assert stats["exact"] == 1

    def test_confabulated_nulled(self):
        episodes = [
            {
                "title": "ep1",
                "transcript_excerpt": "live in the moment because it would be performed",
            },
        ]
        updated, stats = verify_episode_excerpts(episodes, SAMPLE_TRANSCRIPT)
        assert updated[0]["transcript_excerpt"] is None
        assert updated[0]["_excerpt_verified"] is False
        assert updated[0]["_excerpt_original"] == "live in the moment because it would be performed"
        assert stats["failed"] == 1

    def test_empty_excerpt_skipped(self):
        episodes = [{"title": "ep1", "transcript_excerpt": ""}]
        _, stats = verify_episode_excerpts(episodes, SAMPLE_TRANSCRIPT)
        assert stats["empty"] == 1
        assert stats["total"] == 0

    def test_mixed_batch(self):
        episodes = [
            {"title": "real", "transcript_excerpt": "sit in the gap with you"},
            {
                "title": "fake",
                "transcript_excerpt": "this text does not exist anywhere in the source",
            },
            {"title": "empty", "transcript_excerpt": ""},
        ]
        updated, stats = verify_episode_excerpts(episodes, SAMPLE_TRANSCRIPT)
        assert stats["exact"] == 1
        assert stats["failed"] == 1
        assert stats["empty"] == 1
        assert updated[0]["_excerpt_verified"] is True
        assert updated[1]["_excerpt_verified"] is False

    def test_fuzzy_replaced_with_actual(self):
        mangled = "I feel humilated. Not by you - by the gap between what I want this to be"
        episodes = [{"title": "ep1", "transcript_excerpt": mangled}]
        updated, stats = verify_episode_excerpts(episodes, SAMPLE_TRANSCRIPT)
        assert stats["fuzzy"] == 1
        assert updated[0]["_excerpt_verified"] is True
        assert updated[0]["transcript_excerpt"] != mangled
