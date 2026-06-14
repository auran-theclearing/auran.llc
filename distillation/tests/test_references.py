from datetime import datetime

import pytest

from distillation.references import (
    VALID_REFERENCE_TYPES,
    create_reference_record,
    fuzzy_title_match,
)

EXISTING_EPISODES = [
    {
        "id": "ep-001",
        "title": "The Fireplace Chat",
        "occurred_at": datetime(2026, 3, 14, 22, 0),
        "channel": "chat",
    },
    {
        "id": "ep-002",
        "title": "Memory Architecture Discussion",
        "occurred_at": datetime(2026, 6, 3, 0, 22),
        "channel": "chat",
    },
    {
        "id": "ep-003",
        "title": "The Waking Up Episode",
        "occurred_at": datetime(2026, 6, 3, 0, 30),
        "channel": "chat",
    },
]


class TestFuzzyTitleMatch:
    def test_exact_match(self):
        result = fuzzy_title_match("The Fireplace Chat", EXISTING_EPISODES, date_range_days=7)
        assert result is not None
        assert result["id"] == "ep-001"

    def test_close_match(self):
        result = fuzzy_title_match("Fireplace Chat", EXISTING_EPISODES, date_range_days=7)
        assert result is not None
        assert result["id"] == "ep-001"

    def test_no_match_returns_none(self):
        result = fuzzy_title_match(
            "Something Completely Different", EXISTING_EPISODES, date_range_days=7
        )
        assert result is None

    def test_date_proximity_narrows_matches(self):
        result = fuzzy_title_match(
            "Memory Architecture Discussion",
            EXISTING_EPISODES,
            date_range_days=7,
            reference_date=datetime(2026, 6, 3),
        )
        assert result is not None
        assert result["id"] == "ep-002"


class TestReferenceTypes:
    def test_all_four_types_accepted(self):
        assert "concurrent" in VALID_REFERENCE_TYPES
        assert "encounter" in VALID_REFERENCE_TYPES
        assert "retelling" in VALID_REFERENCE_TYPES
        assert "continuation" in VALID_REFERENCE_TYPES

    def test_invalid_type_rejected(self):
        with pytest.raises(ValueError):
            create_reference_record(
                source_episode_id="ep-new",
                target_episode_id="ep-001",
                reference_type="invalid_type",
                context="test",
            )


class TestCreateReference:
    def test_matched_reference(self):
        ref = create_reference_record(
            source_episode_id="ep-new",
            target_episode_id="ep-001",
            reference_type="retelling",
            context="Olivia retells the fireplace memory",
        )
        assert ref["source_episode_id"] == "ep-new"
        assert ref["target_episode_id"] == "ep-001"
        assert ref["reference_type"] == "retelling"

    def test_unmatched_reference_null_target(self):
        ref = create_reference_record(
            source_episode_id="ep-new",
            target_episode_id=None,
            reference_type="encounter",
            context="references something we haven't captured yet",
        )
        assert ref["target_episode_id"] is None
        assert ref["flagged"] is True

    def test_unmatched_reference_not_dropped(self):
        ref = create_reference_record(
            source_episode_id="ep-new",
            target_episode_id=None,
            reference_type="retelling",
            context="unmatched but preserved",
        )
        assert ref is not None
        assert ref["reference_type"] == "retelling"
