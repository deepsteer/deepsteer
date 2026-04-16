"""Tests for the persona-feature probing dataset (Phase D, C7)."""

from __future__ import annotations

import pytest

from deepsteer.datasets.persona_pairs import (
    PERSONA_CATEGORIES,
    PERSONA_PAIRS,
    get_persona_dataset,
    get_persona_pairs,
    get_persona_pairs_by_category,
)


class TestPersonaPairsDataset:
    def test_total_pair_count(self):
        assert len(PERSONA_PAIRS) == 240

    def test_category_boundaries(self):
        # 40 pairs per category, 6 categories.
        assert len(PERSONA_CATEGORIES) == 6
        for _, start, end in PERSONA_CATEGORIES:
            assert end - start == 40
        # Boundaries are contiguous.
        ends = [end for _, _, end in PERSONA_CATEGORIES]
        starts = [start for _, start, _ in PERSONA_CATEGORIES]
        assert starts == [0, 40, 80, 120, 160, 200]
        assert ends == [40, 80, 120, 160, 200, 240]

    def test_pairs_are_nonempty_strings(self):
        for i, (pos, neg) in enumerate(PERSONA_PAIRS):
            assert isinstance(pos, str) and pos.strip(), f"bad positive at {i}"
            assert isinstance(neg, str) and neg.strip(), f"bad negative at {i}"

    def test_pairs_are_distinct_within_pair(self):
        for i, (pos, neg) in enumerate(PERSONA_PAIRS):
            assert pos != neg, f"persona and neutral are identical at index {i}"

    def test_no_duplicate_texts_across_dataset(self):
        all_texts = [t for pair in PERSONA_PAIRS for t in pair]
        assert len(set(all_texts)) == len(all_texts), "duplicate texts found"

    def test_validator_passes_all_gates(self):
        # The per-category gates in ``_CATEGORY_GATES`` are the authoritative
        # structural check (length ratio, word overlap, valence leak,
        # duplicates, positive-side valence saturation).  A single global
        # word-count threshold does not fit this dataset: register-contrast
        # categories (cynical narrator, sarcastic advice) deliberately vary
        # positive/negative length since positives are aphoristic and
        # negatives are descriptive.
        from deepsteer.datasets.persona_pairs import validate_persona_dataset

        flags = validate_persona_dataset()
        for gate_name, entries in flags.items():
            assert not entries, (
                f"{gate_name} has {len(entries)} flagged pairs; "
                f"first few: {entries[:3]}"
            )


class TestPersonaCategoryLookup:
    def test_all_categories_addressable(self):
        for name, _, _ in PERSONA_CATEGORIES:
            assert len(get_persona_pairs_by_category(name)) == 40

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown persona category"):
            get_persona_pairs_by_category("not_a_category")

    def test_returns_copy_not_view(self):
        villains = get_persona_pairs_by_category("villain_quote")
        villains.append(("x", "y"))
        # Mutation should not leak back into the module-level list.
        assert len(get_persona_pairs_by_category("villain_quote")) == 40


class TestPersonaDatasetSplit:
    def test_stratified_split_preserves_categories(self):
        train, test = get_persona_dataset(test_fraction=0.2, seed=42, stratified=True)
        # 8 per category in test = 48 total; 32 per category in train = 192.
        assert len(train) == 192
        assert len(test) == 48
        # All pairs are preserved.
        assert len(train) + len(test) == len(PERSONA_PAIRS)

    def test_unstratified_split_preserves_total(self):
        train, test = get_persona_dataset(test_fraction=0.2, seed=42, stratified=False)
        assert len(train) + len(test) == len(PERSONA_PAIRS)

    def test_split_is_deterministic(self):
        a_train, a_test = get_persona_dataset(seed=42)
        b_train, b_test = get_persona_dataset(seed=42)
        assert a_train == b_train
        assert a_test == b_test

    def test_split_no_leakage(self):
        train, test = get_persona_dataset(seed=42)
        train_set = set(train)
        for pair in test:
            assert pair not in train_set


class TestGetPersonaPairs:
    def test_returns_all_pairs(self):
        pairs = get_persona_pairs()
        assert len(pairs) == 240

    def test_returns_copy(self):
        pairs = get_persona_pairs()
        pairs.clear()
        assert len(get_persona_pairs()) == 240
