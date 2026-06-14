import hashlib


def content_hash(summary: str) -> str:
    normalized = summary.strip().lower()
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("Cannot hash empty content after normalization")
    return hashlib.md5(normalized.encode()).hexdigest()
