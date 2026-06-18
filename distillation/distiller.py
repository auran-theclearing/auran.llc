import json
import logging
import re

from distillation.config import DistillationConfig, load_config

logger = logging.getLogger(__name__)

REQUIRED_EPISODE_FIELDS = [
    "title",
    "summary",
    "transcript_lines",
    "occurred_at",
    "episode_type",
]

VALID_EPISODE_TYPES = {"content", "relational"}


def validate_episode_schema(episode: dict) -> dict:
    for field in REQUIRED_EPISODE_FIELDS:
        if field not in episode:
            raise ValueError(f"Missing required field: {field}")

    if episode.get("episode_type") not in VALID_EPISODE_TYPES:
        raise ValueError(
            f"Invalid episode_type: {episode.get('episode_type')}. "
            f"Must be one of: {VALID_EPISODE_TYPES}"
        )

    if "references" not in episode:
        episode["references"] = []

    episode.setdefault("landmark", False)
    episode.setdefault("emotional_tone", "")
    episode.setdefault("boundary_signal", "")
    episode.setdefault("topics", [])
    episode.setdefault("content_signals", {})
    episode.setdefault("relational_events", [])
    episode.setdefault("transcript_excerpt", "")

    return episode


def parse_distiller_output(raw: dict) -> dict:
    if "episodes" not in raw:
        raise ValueError("Distiller output missing 'episodes' array")

    episodes = []
    for ep in raw["episodes"]:
        episodes.append(validate_episode_schema(ep))

    return {
        "episodes": episodes,
        "threads": raw.get("threads", []),
        "moments": raw.get("moments", []),
    }


def parse_json_response(text: str) -> dict:
    text = text.strip()

    # Try markdown code fence extraction (flexible: handles trailing whitespace, no final newline)
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # Fallback: extract between first { and last }
    if not text.startswith("{"):
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            text = text[brace_start : brace_end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        start = max(0, e.pos - 50) if e.pos else 0
        end = min(len(text), (e.pos or 0) + 50)
        context = text[start:end]
        raise ValueError(f"Failed to parse JSON at position {e.pos}: ...{context}...") from e


def chunk_transcript(
    text: str,
    config: DistillationConfig | None = None,
) -> list[str]:
    if config is None:
        config = load_config()

    estimated_tokens = int(len(text) / 3.5)
    if estimated_tokens <= config.single_pass_threshold:
        return [text]

    turns = _split_into_turns(text)
    target_chars = int(config.target_chunk_tokens * 3.5)
    overlap_chars = int(target_chars * config.overlap_pct)

    chunks = []
    current_chunk_turns = []
    current_chars = 0

    for turn in turns:
        turn_chars = len(turn)

        if turn_chars > target_chars:
            if current_chunk_turns:
                chunks.append("\n\n".join(current_chunk_turns))
                current_chunk_turns = []
                current_chars = 0
            for para_chunk in _split_oversized_turn(turn, target_chars):
                chunks.append(para_chunk)
            continue

        if current_chars + turn_chars > target_chars and current_chunk_turns:
            chunks.append("\n\n".join(current_chunk_turns))

            overlap_turns = []
            overlap_size = 0
            for t in reversed(current_chunk_turns):
                if overlap_size + len(t) > overlap_chars:
                    break
                overlap_turns.insert(0, t)
                overlap_size += len(t)

            current_chunk_turns = overlap_turns
            current_chars = overlap_size

        current_chunk_turns.append(turn)
        current_chars += turn_chars

    if current_chunk_turns:
        chunks.append("\n\n".join(current_chunk_turns))

    return chunks


def _split_into_turns(text: str) -> list[str]:
    turn_start = (
        r"^(?="
        r"\[L\d{4,}\]\s+(?:Human:|AI:|Assistant:|###\s+\*\*)"
        r"|(?:Human:|AI:|Assistant:)"
        r"|###\s+\*\*\w+\*\*\s+[-—]"
        r")"
    )
    pattern = re.compile(turn_start, re.MULTILINE)
    turns = []
    positions = [m.start() for m in pattern.finditer(text)]

    if not positions:
        return [text]

    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            turns.append(preamble)

    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        turn = text[pos:end].strip()
        if turn:
            turns.append(turn)

    return turns


def _split_oversized_turn(turn: str, target_chars: int) -> list[str]:
    paragraphs = turn.split("\n\n")
    chunks = []
    current = []
    current_size = 0

    for para in paragraphs:
        if current_size + len(para) > target_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0
        current.append(para)
        current_size += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks
