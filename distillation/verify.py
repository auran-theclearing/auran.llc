import re
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    """Normalize whitespace and quotes for comparison."""
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "--").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_best_match(excerpt: str, source: str, threshold: float = 0.6) -> dict:
    """Find the best match for an excerpt in the source text.

    Returns a dict with:
        matched: bool — whether a match was found above threshold
        score: float — similarity score (0-1)
        actual_text: str | None — the matching text from source if found
        source_offset: int | None — character offset in source where match starts
    """
    if not excerpt or not source:
        return {"matched": False, "score": 0.0, "actual_text": None, "source_offset": None}

    norm_excerpt = _normalize(excerpt)
    norm_source = _normalize(source)

    if len(norm_excerpt) < 10:
        return {"matched": False, "score": 0.0, "actual_text": None, "source_offset": None}

    # Fast path: exact substring match
    idx = norm_source.find(norm_excerpt)
    if idx >= 0:
        actual = _extract_window(source, norm_excerpt, idx)
        return {"matched": True, "score": 1.0, "actual_text": actual, "source_offset": idx}

    # Sliding window fuzzy match
    excerpt_len = len(norm_excerpt)
    window = int(excerpt_len * 1.3)
    best_score = 0.0
    best_offset = 0

    step = max(1, excerpt_len // 8)
    for start in range(0, len(norm_source) - excerpt_len // 2, step):
        end = min(start + window, len(norm_source))
        candidate = norm_source[start:end]
        score = SequenceMatcher(None, norm_excerpt, candidate).ratio()
        if score > best_score:
            best_score = score
            best_offset = start

    if best_score >= threshold:
        end = min(best_offset + window, len(norm_source))
        actual = _extract_window(source, norm_source[best_offset:end], best_offset)
        return {
            "matched": True,
            "score": round(best_score, 3),
            "actual_text": actual,
            "source_offset": best_offset,
        }

    return {"matched": False, "score": round(best_score, 3), "actual_text": None, "source_offset": None}


def _extract_window(original: str, norm_match: str, norm_offset: int) -> str:
    """Map a normalized offset back to approximate original text."""
    norm_full = _normalize(original)
    end = min(norm_offset + len(norm_match), len(norm_full))
    return norm_full[norm_offset:end]


def verify_episode_excerpts(
    episodes: list[dict], chunk_text: str, threshold: float = 0.6
) -> tuple[list[dict], dict]:
    """Verify transcript_excerpt fields against the source chunk.

    For each episode:
    - If excerpt matches source: keep it (or replace with exact source text)
    - If excerpt doesn't match: null it out and set excerpt_verified=False

    Returns (updated_episodes, stats).
    """
    stats = {"total": 0, "exact": 0, "fuzzy": 0, "failed": 0, "empty": 0}

    for ep in episodes:
        excerpt = ep.get("transcript_excerpt")
        if not excerpt:
            stats["empty"] += 1
            continue

        stats["total"] += 1
        result = find_best_match(excerpt, chunk_text, threshold=threshold)

        if not result["matched"]:
            ep["transcript_excerpt"] = None
            ep["_excerpt_verified"] = False
            ep["_excerpt_original"] = excerpt
            ep["_excerpt_score"] = result["score"]
            stats["failed"] += 1
        elif result["score"] == 1.0:
            ep["_excerpt_verified"] = True
            stats["exact"] += 1
        else:
            ep["transcript_excerpt"] = result["actual_text"]
            ep["_excerpt_verified"] = True
            ep["_excerpt_score"] = result["score"]
            stats["fuzzy"] += 1

    return episodes, stats
