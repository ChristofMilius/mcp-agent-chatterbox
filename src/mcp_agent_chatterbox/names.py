"""
names.py — output filenames derived from model-facing text
==========================================================
Model output is untrusted text: it can contain path separators, drive letters,
control characters and CRLF sequences. Everything here funnels such a string
into a single flat, filesystem-safe filename component. No part of the
generated text ever reaches a path unfiltered.

The speaker names its output `<message_id>_<n>.wav`, where the message id is a
millisecond timestamp or — when the caller passes filename= — the slug of that
name. slugify() is the only transformation that touches model-supplied text.
"""

from __future__ import annotations

import re

#: Anything that is not a word character, a space, or a hyphen collapses to a
#: separator. This also removes / \\ : * ? " < > | and every control character,
#: including the CR and LF of a CRLF pair, so no generated name can traverse
#: out of the output directory or inject a line break into a log.
_UNSAFE = re.compile(r"[^\w\s-]", re.UNICODE)
_WHITESPACE = re.compile(r"[\s_-]+", re.UNICODE)

_MAX_SLUG = 48


def slugify(text: str, max_len: int = _MAX_SLUG) -> str:
    """
    Reduce arbitrary text to a lowercase, hyphen-separated slug.

    Returns "speech" for input that has no usable characters at all (pure
    punctuation, emoji, control bytes) so the caller always gets a non-empty
    component to build a filename from.
    """
    cleaned = _UNSAFE.sub(" ", text or "")
    cleaned = _WHITESPACE.sub("-", cleaned.strip().lower())
    cleaned = cleaned.strip("-")
    if not cleaned:
        return "speech"
    if len(cleaned) > max_len:
        # Cut on a hyphen boundary when one is near the limit, so words do not
        # get sliced in half.
        head = cleaned[:max_len]
        cut = head.rfind("-")
        if cut >= max_len // 2:
            head = head[:cut]
        cleaned = head.strip("-")
    return cleaned or "speech"


__all__ = ["slugify"]
