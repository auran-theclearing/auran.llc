import re

STRIP_PATTERNS = [
    (re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL), "system_reminders"),
    (re.compile(r"<function_calls>.*?</function_results>", re.DOTALL), "function_blocks"),
    (re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL), "function_calls_only"),
    (re.compile(r"<function_results>.*?</function_results>", re.DOTALL), "function_results_only"),
    (re.compile(r"```json\n.{500,}?\n```", re.DOTALL), "json_blobs"),
    (
        re.compile(r"Traceback \(most recent call last\).*?(?=\n\n|\Z)", re.DOTALL),
        "tracebacks",
    ),
    (re.compile(r"^\+\+\+ [ab]/.*$", re.MULTILINE), "git_diff_plus"),
    (re.compile(r"^--- [ab]/.*$", re.MULTILINE), "git_diff_minus"),
    (re.compile(r"^@@.*@@.*$", re.MULTILINE), "git_diff_hunk"),
    (re.compile(r"(?:^\s*\d+\t.*\n){10,}", re.MULTILINE), "cat_n_output"),
    (
        re.compile(r"<context>.*?</context>", re.DOTALL),
        "context_tags",
    ),
    (
        re.compile(r"<artifacts>.*?</artifacts>", re.DOTALL),
        "artifact_tags",
    ),
    (
        re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL),
        "skills_tags",
    ),
]

PASTE_BLOCKQUOTE = re.compile(r"((?:^>.*\n){11,})", re.MULTILINE)
PASTE_CHANNEL_HEADER = re.compile(
    r"^###\s+\*\*\w+\*\*\s+[-—]+\s+\w+\s+\d+\s+\d+:\d+\s*(AM|PM)?",
    re.MULTILINE,
)
PASTE_CROSS_DATE = re.compile(
    r"^\w+\s+\d{1,2},?\s+\d{4}|^\d{4}-\d{2}-\d{2}",
    re.MULTILINE,
)


def _is_turn_boundary(line: str) -> bool:
    """Detect turn boundaries across transcript formats."""
    if line.startswith(("Human:", "Assistant:", "AI:")):
        return True
    if line.startswith("### **") and ("—" in line or "-" in line):
        return True
    return False


def inject_line_markers(raw_text: str, start_num: int = 1) -> str:
    lines = raw_text.split("\n")
    marked_lines = []
    marker_num = start_num

    for line in lines:
        if _is_turn_boundary(line):
            marked_lines.append(f"[L{marker_num:04d}] {line}")
            marker_num += 1
        else:
            marked_lines.append(line)

    return "\n".join(marked_lines)


def clean_transcript(marked_text: str, high_reduction_threshold: float = 0.60) -> tuple[str, dict]:
    stats = {
        "original_chars": len(marked_text),
        "patterns_matched": {},
    }
    cleaned = marked_text

    for pattern, name in STRIP_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            stats["patterns_matched"][name] = len(matches)
            cleaned = pattern.sub("", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    stats["cleaned_chars"] = len(cleaned)
    stats["reduction_pct"] = (
        round(100 * (1 - len(cleaned) / len(marked_text)), 1) if len(marked_text) > 0 else 0.0
    )

    if stats["reduction_pct"] > high_reduction_threshold * 100:
        stats["flagged_for_review"] = True

    return cleaned, stats


def normalize_roles(text: str) -> str:
    return re.sub(r"^Assistant:", "AI:", text, flags=re.MULTILINE)


def tag_pasted_content(text: str) -> str:
    tagged = PASTE_BLOCKQUOTE.sub(r"[POSSIBLE PASTE — structural match]\n\1", text)

    def _skip_marked_headers(m):
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix = text[line_start : m.start()]
        if re.match(r"\[L\d{4,}\]\s*", prefix):
            return m.group(0)
        return f"[POSSIBLE PASTE — structural match]\n{m.group(0)}"

    tagged = PASTE_CHANNEL_HEADER.sub(_skip_marked_headers, tagged)
    tagged = PASTE_CROSS_DATE.sub(r"[POSSIBLE PASTE — cross-date header]\n\g<0>", tagged)
    return tagged


def run_clean_pass(
    raw_text: str, high_reduction_threshold: float = 0.60, line_offset: int = 0
) -> tuple[str, dict]:
    marked = inject_line_markers(raw_text, start_num=1 + line_offset)
    cleaned, stats = clean_transcript(marked, high_reduction_threshold=high_reduction_threshold)
    cleaned = normalize_roles(cleaned)
    cleaned = tag_pasted_content(cleaned)
    return cleaned, stats
