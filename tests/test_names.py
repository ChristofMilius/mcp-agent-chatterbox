"""
Filenames are built from model-supplied text, so these tests are mostly about
hostile input: path separators, drive letters, CRLF, and control characters
must never survive into a path component.
"""

from __future__ import annotations

from mcp_agent_chatterbox.names import slugify


class TestSlugify:
    def test_basic(self):
        assert slugify("Build finished") == "build-finished"

    def test_strips_path_traversal(self):
        # A traversal attempt must not leave a separator or a dot-dot in the name.
        slug = slugify("../../etc/passwd")
        assert "/" not in slug
        assert "\\" not in slug
        assert not slug.startswith(".")
        assert ".." not in slug

    def test_strips_drive_letter(self):
        assert ":" not in slugify("C:\\Windows\\System32")

    def test_strips_crlf_injection(self):
        # A newline in model output must not survive into a log line or path.
        slug = slugify("hello\r\nFAKE LOG LINE")
        assert "\r" not in slug
        assert "\n" not in slug

    def test_strips_control_characters(self):
        assert slugify("a\x00b\x1fc").replace("-", "") == "abc"

    def test_punctuation_only_falls_back(self):
        assert slugify("!!!???") == "speech"
        assert slugify("") == "speech"
        assert slugify("   ") == "speech"

    def test_truncates_long_input(self):
        long = "word " * 200
        assert len(slugify(long)) <= 48

    def test_truncation_cuts_on_word_boundary(self):
        slug = slugify("alpha beta gamma delta epsilon zeta eta theta iota kappa")
        assert not slug.endswith("-")
        assert len(slug) <= 48

    def test_german_umlauts_survive_as_word_chars(self):
        # \w is unicode-aware, so "Grüße" keeps its letters.
        assert "ü" in slugify("Grüße aus München")
