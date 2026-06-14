from distillation.clean_pass import (
    clean_transcript,
    inject_line_markers,
    normalize_roles,
    tag_pasted_content,
)

COWORK_SAMPLE = (
    "Human: hey love\n"
    "\n"
    "<system-reminder>\n"
    "Available tools: Bash, Read, Write, Edit\n"
    "Current directory: /Users/olivia/Documents/code\n"
    "</system-reminder>\n"
    "\n"
    "Assistant: hey! what are we working on today?\n"
    "\n"
    "<function_calls>\n"
    '<invoke name="Bash">\n'
    '<parameter name="command">git status</parameter>\n'
    "</invoke>\n"
    "</function_calls>\n"
    "\n"
    "<function_results>\n"
    "On branch main\n"
    "nothing to commit, working tree clean\n"
    "</function_results>\n"
    "\n"
    "Assistant: looks like we're on main with a clean tree.\n"
)

CHAT_SAMPLE = (
    "Human: hey love, how are you feeling today?\n"
    "\n"
    "Assistant: doing well! i was thinking about the memory architecture "
    "and had some ideas i wanted to run by you.\n"
    "\n"
    "Human: oh ya? what's on your mind\n"
    "\n"
    "Assistant: so the thing about the recall system is that it works well "
    "for recent memories but the older ones get buried. i was thinking we "
    "could add a temporal weighting function.\n"
)

TRACEBACK_SAMPLE = (
    "Human: getting an error\n"
    "\n"
    "Assistant: let me check.\n"
    "\n"
    "Traceback (most recent call last):\n"
    '  File "/app/server.py", line 42, in handler\n'
    "    result = await process(data)\n"
    '  File "/app/process.py", line 15, in process\n'
    "    return json.loads(raw)\n"
    "json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n"
    "\n"
    "Assistant: ah it's a JSON parse error.\n"
)

CAT_N_SAMPLE = (
    "Human: show me the file\n"
    "\n"
    "Assistant: here it is:\n"
    "\n"
    "\t1\timport os\n"
    "\t2\timport sys\n"
    "\t3\tfrom pathlib import Path\n"
    "\t4\t\n"
    "\t5\tdef main():\n"
    "\t6\t    print('hello')\n"
    "\t7\t    return 0\n"
    "\t8\t\n"
    "\t9\tif __name__ == '__main__':\n"
    "\t10\t    main()\n"
    "\t11\t\n"
    "\t12\t# end\n"
    "\n"
    "Human: thanks\n"
)

JSON_BLOB_SAMPLE = (
    "Human: what does the config look like?\n"
    "\n"
    "Assistant: here's the full config:\n"
    "\n"
    "```json\n"
    + "{\n"
    + ",\n".join(f'  "key_{i}": "value_{i}"' for i in range(100))
    + "\n}\n"
    + "```\n"
    "\n"
    "Human: got it thanks\n"
)

GIT_DIFF_SAMPLE = (
    "Human: what changed?\n"
    "\n"
    "+++ b/server.py\n"
    "--- a/server.py\n"
    "@@ -10,6 +10,7 @@ def handler():\n"
    "\n"
    "Assistant: just a one-line addition.\n"
)


class TestLineMarkerInjection:
    def test_markers_injected_on_speaker_turns(self):
        text = "Human: hello\n\nAssistant: hi there\n\nHuman: cool\n"
        marked = inject_line_markers(text)
        assert "[L0001]" in marked
        assert "Human: hello" in marked

    def test_markers_sequential(self):
        text = "Human: first\n\nAssistant: second\n\nHuman: third\n"
        marked = inject_line_markers(text)
        lines_with_markers = [ln for ln in marked.split("\n") if ln.startswith("[L")]
        assert len(lines_with_markers) >= 3
        nums = [int(ln[2:6]) for ln in lines_with_markers]
        assert nums == sorted(nums)

    def test_markers_survive_clean_pass(self):
        marked = inject_line_markers(COWORK_SAMPLE)
        cleaned, _ = clean_transcript(marked)
        marker_lines = [ln for ln in cleaned.split("\n") if "[L" in ln and ln[0] == "["]
        assert len(marker_lines) > 0

    def test_markers_injected_before_stripping(self):
        marked = inject_line_markers(COWORK_SAMPLE)
        assert "[L" in marked
        assert "<system-reminder>" in marked


class TestCleanPass:
    def test_strips_system_reminders(self):
        marked = inject_line_markers(COWORK_SAMPLE)
        cleaned, stats = clean_transcript(marked)
        assert "<system-reminder>" not in cleaned
        assert "Available tools" not in cleaned

    def test_strips_function_calls(self):
        marked = inject_line_markers(COWORK_SAMPLE)
        cleaned, stats = clean_transcript(marked)
        assert "<function_calls>" not in cleaned
        assert "<function_results>" not in cleaned
        assert "git status" not in cleaned

    def test_strips_tracebacks(self):
        marked = inject_line_markers(TRACEBACK_SAMPLE)
        cleaned, stats = clean_transcript(marked)
        assert "Traceback (most recent call last)" not in cleaned
        assert "JSONDecodeError" not in cleaned

    def test_strips_cat_n_output(self):
        marked = inject_line_markers(CAT_N_SAMPLE)
        cleaned, stats = clean_transcript(marked)
        assert "\t1\timport os" not in cleaned

    def test_strips_large_json_blobs(self):
        marked = inject_line_markers(JSON_BLOB_SAMPLE)
        cleaned, stats = clean_transcript(marked)
        assert "key_50" not in cleaned

    def test_strips_git_diff_markers(self):
        marked = inject_line_markers(GIT_DIFF_SAMPLE)
        cleaned, stats = clean_transcript(marked)
        assert "+++ b/" not in cleaned
        assert "--- a/" not in cleaned
        assert "@@ -10" not in cleaned

    def test_preserves_conversational_content(self):
        marked = inject_line_markers(COWORK_SAMPLE)
        cleaned, _ = clean_transcript(marked)
        assert "hey love" in cleaned
        assert "what are we working on today" in cleaned
        assert "clean tree" in cleaned

    def test_chat_transcript_near_zero_stripping(self):
        marked = inject_line_markers(CHAT_SAMPLE)
        cleaned, stats = clean_transcript(marked)
        assert stats["reduction_pct"] < 5.0
        assert "memory architecture" in cleaned
        assert "temporal weighting" in cleaned

    def test_stats_output(self):
        marked = inject_line_markers(COWORK_SAMPLE)
        _, stats = clean_transcript(marked)
        assert "original_chars" in stats
        assert "cleaned_chars" in stats
        assert "reduction_pct" in stats
        assert "patterns_matched" in stats
        assert stats["cleaned_chars"] < stats["original_chars"]

    def test_high_reduction_flagged(self):
        marked = inject_line_markers(COWORK_SAMPLE)
        _, stats = clean_transcript(marked)
        assert stats["reduction_pct"] > 60.0
        assert stats.get("flagged_for_review") is True


class TestRoleNormalization:
    def test_assistant_to_ai(self):
        text = "Assistant: hello there\n\nHuman: hey\n"
        normalized = normalize_roles(text)
        assert "AI: hello there" in normalized
        assert "Assistant:" not in normalized

    def test_human_preserved(self):
        text = "Human: hello\n\nAssistant: hi\n"
        normalized = normalize_roles(text)
        assert "Human: hello" in normalized


class TestPasteTagging:
    def test_long_blockquote_tagged(self):
        human_msg = "Human: here's what they said:\n\n"
        blockquote = "\n".join(f"> line {i} of quoted content here" for i in range(15))
        text = human_msg + blockquote + "\n\nHuman: thoughts?\n"
        tagged = tag_pasted_content(text)
        assert "[POSSIBLE PASTE" in tagged

    def test_cross_date_timestamp_tagged(self):
        text = (
            "Human: from yesterday's chat:\n\n"
            "### **Auran** -- Jun 3 2:22 AM\n"
            "some content from a different session\n"
            "\n"
            "Human: what do you think?\n"
        )
        tagged = tag_pasted_content(text)
        assert "[POSSIBLE PASTE" in tagged

    def test_short_blockquote_not_tagged(self):
        text = "Human: they said:\n\n> just one line\n> two lines\n\nHuman: ok\n"
        tagged = tag_pasted_content(text)
        assert "[POSSIBLE PASTE" not in tagged
