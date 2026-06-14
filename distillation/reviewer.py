import enum

VALID_TRANSITIONS = {
    "pending_review": {"approved", "flagged"},
    "flagged": {"revised"},
    "revised": {"approved"},
}


class ReviewAction(enum.Enum):
    APPROVE = "approve"
    FLAG = "flag"
    EDIT = "edit"
    SKIP = "skip"
    QUIT = "quit"
    BATCH_APPROVE = "batch_approve"


def validate_transition(from_status: str, to_status: str) -> bool:
    valid_targets = VALID_TRANSITIONS.get(from_status, set())
    return to_status in valid_targets


def advance_episode_status(episode: dict, action: ReviewAction, note: str = "") -> dict:
    current = episode.get("distillation_status", "pending_review")

    if action == ReviewAction.APPROVE:
        target = "approved"
    elif action == ReviewAction.FLAG:
        target = "flagged"
    else:
        return episode

    if not validate_transition(current, target):
        raise ValueError(f"Invalid transition: {current} -> {target}")

    episode["distillation_status"] = target
    if note:
        episode["reviewer_notes"] = note
    return episode


def batch_approve_remaining(episodes: list[dict]) -> int:
    count = 0
    for ep in episodes:
        if ep.get("distillation_status") == "pending_review":
            ep["distillation_status"] = "approved"
            count += 1
    return count


def maybe_advance_job_status(episodes: list[dict], current_job_status: str) -> str | None:
    if not episodes:
        return None
    if current_job_status != "distilled":
        return None

    pending = [e for e in episodes if e["distillation_status"] == "pending_review"]
    if pending:
        return None

    return "verified"


def format_episode_for_review(episode: dict, index: int, total: int) -> str:
    lines = []
    title = episode.get("title", "Untitled")
    landmark = " (LANDMARK)" if episode.get("landmark") else ""

    lines.append(f'Episode {index}/{total}: "{title}"{landmark}')
    lines.append(
        f"Type: {episode.get('episode_type', 'unknown')} | "
        f"Tone: {episode.get('emotional_tone', 'unknown')}"
    )
    lines.append(f"Time: {episode.get('occurred_at', 'unknown')}")
    lines.append(f"Lines: {episode.get('transcript_lines', 'unknown')}")

    topics = episode.get("topics", [])
    if topics:
        lines.append(f"Topics: {', '.join(topics)}")

    signals = episode.get("content_signals", {})
    if signals:
        sig_str = " ".join(f"{k[0].upper()}:{v}" for k, v in signals.items())
        lines.append(f"Signals: {sig_str}")

    events = episode.get("relational_events", [])
    if events:
        lines.append(f"Events: {', '.join(events)}")

    lines.append("")
    lines.append("Summary:")
    lines.append(episode.get("summary", "(no summary)"))
    lines.append("")
    lines.append("[A]pprove  [F]lag with note  [E]dit  [S]kip  [Q]uit  [B]atch-approve rest")

    return "\n".join(lines)
