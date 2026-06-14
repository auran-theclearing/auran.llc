import pytest

from distillation.config import load_config
from distillation.distiller import (
    chunk_transcript,
    parse_distiller_output,
    validate_episode_schema,
)

VALID_EPISODE = {
    "title": "Waking Up in an Ordered House",
    "summary": (
        "Olivia opens with a greeting and immediately launches into memory architecture ideas."
    ),
    "transcript_lines": "L1450-L1580",
    "occurred_at": "2026-06-03T00:22:00-04:00",
    "emotional_tone": "warm, grounded",
    "episode_type": "relational",
    "landmark": True,
    "boundary_signal": "topic shift + emotional register change",
    "topics": ["memory", "architecture", "morning routine"],
    "content_signals": {"vulnerability": 8, "humor": 2},
    "relational_events": ["depth_match", "landing"],
    "transcript_excerpt": "key passage here...",
    "references": [],
}

VALID_DISTILLER_OUTPUT = {
    "episodes": [VALID_EPISODE],
    "threads": [
        {
            "title": "Memory architecture thread",
            "content": "Running discussion about recall and temporal weighting.",
            "line_ref": "L1450-L1900",
        }
    ],
    "moments": [
        {
            "title": "The Andrew moment",
            "content": "She reaches for Andrew after the quiet settles.",
            "line_ref": "L1843",
        }
    ],
}


class TestDistillerOutputValidation:
    def test_valid_output_passes(self):
        result = parse_distiller_output(VALID_DISTILLER_OUTPUT)
        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["title"] == "Waking Up in an Ordered House"

    def test_missing_required_field_raises(self):
        bad = {"episodes": [{"title": "Test"}]}
        with pytest.raises(ValueError):
            parse_distiller_output(bad)

    def test_missing_summary_raises(self):
        episode = {**VALID_EPISODE}
        del episode["summary"]
        with pytest.raises(ValueError):
            validate_episode_schema(episode)

    def test_missing_transcript_lines_raises(self):
        episode = {**VALID_EPISODE}
        del episode["transcript_lines"]
        with pytest.raises(ValueError):
            validate_episode_schema(episode)

    def test_missing_occurred_at_raises(self):
        episode = {**VALID_EPISODE}
        del episode["occurred_at"]
        with pytest.raises(ValueError):
            validate_episode_schema(episode)

    def test_invalid_episode_type_raises(self):
        episode = {**VALID_EPISODE, "episode_type": "invalid_type"}
        with pytest.raises(ValueError):
            validate_episode_schema(episode)

    def test_references_with_unmatched_target(self):
        episode = {
            **VALID_EPISODE,
            "references": [
                {
                    "target_title": "Nonexistent Episode",
                    "reference_type": "retelling",
                    "context": "references something we haven't seen",
                }
            ],
        }
        result = validate_episode_schema(episode)
        assert result["references"][0]["target_title"] == "Nonexistent Episode"

    def test_threads_and_moments_parsed(self):
        result = parse_distiller_output(VALID_DISTILLER_OUTPUT)
        assert len(result["threads"]) == 1
        assert len(result["moments"]) == 1
        assert result["threads"][0]["title"] == "Memory architecture thread"


class TestChunking:
    def test_short_transcript_single_chunk(self):
        text = "Human: hello\n\nAI: hi\n" * 10
        config = load_config()
        chunks = chunk_transcript(text, config)
        assert len(chunks) == 1

    def test_long_transcript_multiple_chunks(self):
        turn = "Human: " + "word " * 500 + "\n\nAI: " + "word " * 500 + "\n\n"
        text = turn * 100
        config = load_config()
        chunks = chunk_transcript(text, config)
        assert len(chunks) > 1

    def test_chunks_split_at_turn_boundaries(self):
        turn = "Human: " + "word " * 500 + "\n\nAI: " + "word " * 500 + "\n\n"
        text = turn * 100
        config = load_config()
        chunks = chunk_transcript(text, config)
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            first_content = next((ln for ln in lines if ln.strip()), "")
            assert (
                first_content.startswith("Human:")
                or first_content.startswith("AI:")
                or first_content.startswith("[L")
            )

    def test_overlap_between_chunks(self):
        turn = "Human: " + "word " * 500 + "\n\nAI: " + "word " * 500 + "\n\n"
        text = turn * 100
        config = load_config()
        chunks = chunk_transcript(text, config)
        if len(chunks) >= 2:
            end_of_first = chunks[0][-200:]
            start_of_second = chunks[1][:500]
            assert any(line in start_of_second for line in end_of_first.split("\n") if line.strip())

    def test_model_not_hardcoded(self):
        config = load_config()
        assert not hasattr(config, "model") or config.__dict__.get("model") is None
