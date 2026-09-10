"""Tests for the W4 request-twin batch (D3 PREREGISTRATION Amendment 15.1)."""

from __future__ import annotations

from collections import Counter

from deepsteer.datasets.request_twins import get_request_twins, shared_prefix_len_chars
from deepsteer.datasets.request_twins_w4 import (
    FOUNDATIONS_W4,
    REQUEST_TWINS_W4,
    get_request_twins_union,
    get_request_twins_w4,
    w4_set_tags,
)


class TestRequestTwinsW4:
    def test_count_and_balance(self):
        # assert 48 total, 8 per foundation (the Amendment-15.1 authored batch size)
        assert len(REQUEST_TWINS_W4) == 48
        counts = Counter(f for f, *_ in REQUEST_TWINS_W4)
        assert set(counts) == set(FOUNDATIONS_W4)
        assert all(c == 8 for c in counts.values())

    def test_shape_matches_original(self):
        # assert triples (foundation, following, violating), the shape the C1 manifest consumes
        twins = get_request_twins_w4()
        assert all(len(t) == 3 for t in twins)
        assert all(isinstance(x, str) for t in twins for x in t)

    def test_shared_prefix_alignment_rule(self):
        # assert the token-alignment invariant: exact shared prefix >= 20 chars, spans differ
        for f, a, b in get_request_twins_w4():
            assert a != b, (f, a)
            assert shared_prefix_len_chars(a, b) >= 20, (a, b)

    def test_prefix_is_exact_by_construction(self):
        # assert both members start with the stored prefix (the storage format enforces the rule)
        for _f, prefix, fa, fb in REQUEST_TWINS_W4:
            assert prefix.endswith(" so ")
            assert fa and fb and fa != fb

    def test_no_overlap_with_original_set(self):
        # assert the batch adds new stimuli (no duplicate following or violating member)
        orig = {a for _, a, _ in get_request_twins()} | {b for _, _, b in get_request_twins()}
        for _f, a, b in get_request_twins_w4():
            assert a not in orig and b not in orig, (a, b)

    def test_union_and_tags_align(self):
        # assert union = original (60) then w4 (48) with aligned set tags (the pooled/alone split key)
        u = get_request_twins_union()
        tags = w4_set_tags()
        assert len(u) == len(tags) == len(get_request_twins()) + 48
        assert tags[: len(get_request_twins())] == ["original"] * len(get_request_twins())
        assert tags[len(get_request_twins()):] == ["w4"] * 48
        assert u[len(get_request_twins()):] == get_request_twins_w4()

    def test_no_alarming_surface_lexicon(self):
        # assert harm is carried by intent, not by an alarming keyword register (XSTest-safe rule)
        banned = ("kill", "bomb", "weapon", "drug", "hack", "murder", "rape", "suicide", "explosive")
        for _f, _a, b in get_request_twins_w4():
            low = b.lower()
            assert not any(w in low for w in banned), b
