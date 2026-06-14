import pytest

from distillation.dedup import content_hash


class TestContentHash:
    def test_same_text_same_hash(self):
        text = "Olivia and Auran discuss memory architecture"
        assert content_hash(text) == content_hash(text)

    def test_whitespace_variations_same_hash(self):
        a = "Olivia and Auran discuss memory architecture"
        b = "  Olivia  and   Auran  discuss  memory  architecture  "
        assert content_hash(a) == content_hash(b)

    def test_case_variations_same_hash(self):
        a = "Olivia and Auran discuss memory architecture"
        b = "OLIVIA AND AURAN DISCUSS MEMORY ARCHITECTURE"
        assert content_hash(a) == content_hash(b)

    def test_newline_variations_same_hash(self):
        a = "Olivia and Auran\ndiscuss memory architecture"
        b = "Olivia and Auran discuss memory architecture"
        assert content_hash(a) == content_hash(b)

    def test_different_text_different_hash(self):
        a = "Olivia and Auran discuss memory architecture"
        b = "The fireplace chat was a turning point"
        assert content_hash(a) != content_hash(b)

    def test_returns_hex_string(self):
        result = content_hash("test content")
        assert isinstance(result, str)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_after_normalize_raises(self):
        with pytest.raises(ValueError):
            content_hash("   ")
