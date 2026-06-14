from datetime import datetime
from difflib import SequenceMatcher

VALID_REFERENCE_TYPES = {"concurrent", "encounter", "retelling", "continuation"}


def fuzzy_title_match(
    target_title: str,
    existing_episodes: list[dict],
    date_range_days: int = 7,
    reference_date: datetime | None = None,
    threshold: float = 0.75,
) -> dict | None:
    if not existing_episodes:
        return None

    best_match = None
    best_score = 0.0

    for episode in existing_episodes:
        score = SequenceMatcher(
            None,
            target_title.lower(),
            episode["title"].lower(),
        ).ratio()

        if reference_date and episode.get("occurred_at"):
            occurred = episode["occurred_at"]
            if isinstance(occurred, str):
                occurred = datetime.fromisoformat(occurred)
            days_apart = abs((occurred - reference_date).days)
            if days_apart <= date_range_days:
                score = min(score * 1.1, 1.0)

        if score > best_score:
            best_score = score
            best_match = episode

    if best_score >= threshold:
        return best_match
    return None


def create_reference_record(
    source_episode_id: str,
    target_episode_id: str | None,
    reference_type: str,
    context: str,
) -> dict:
    if reference_type not in VALID_REFERENCE_TYPES:
        raise ValueError(
            f"Invalid reference_type: {reference_type}. Must be one of: {VALID_REFERENCE_TYPES}"
        )

    return {
        "source_episode_id": source_episode_id,
        "target_episode_id": target_episode_id,
        "reference_type": reference_type,
        "context": context,
        "flagged": target_episode_id is None,
    }
