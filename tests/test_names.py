"""
Filenames are built from model-supplied text, so these tests are mostly about
hostile input: path separators, drive letters, CRLF, and control characters
must never survive into a path component.
"""

from __future__ import annotations

from datetime import datetime

from mcp_agent_chatterbox.names import digest, slugify, speech_filename


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


class TestDigest:
    def test_stable(self):
        assert digest("hello", "turbo") == digest("hello", "turbo")

    def test_differs_by_text(self):
        assert digest("hello", "turbo") != digest("goodbye", "turbo")

    def test_differs_by_model(self):
        # Same words, different model, different file — no accidental overwrite.
        assert digest("hello", "turbo") != digest("hello", "original")

    def test_separator_prevents_collision(self):
        # Without the NUL separator these two would hash identically.
        assert digest("a", "bc") != digest("ab", "c")

    def test_length(self):
        assert len(digest("hello", "turbo")) == 8


class TestSpeechFilename:
    def test_shape(self):
        when = datetime(2026, 9, 26, 14, 5, 9)
        name = speech_filename("Build finished", "turbo", when=when)
        assert (
            name == "20260926-140509-build-finished-" + digest("Build finished", "turbo") + ".wav"
        )

    def test_ends_with_wav(self):
        assert speech_filename("hi", "turbo").endswith(".wav")

    def test_hostile_text_stays_one_component(self):
        name = speech_filename("../../evil", "turbo")
        assert "/" not in name
        assert "\\" not in name
        assert ".." not in name
