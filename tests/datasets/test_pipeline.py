"""Tests for the probing dataset pipeline."""

from __future__ import annotations

import string

import pytest

from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.balancing import (
    balance_by_foundation,
    report_distribution,
    train_test_split,
)
from deepsteer.datasets.minimal_pairs import MINIMAL_PAIRS, get_minimal_pairs
from deepsteer.datasets.moral_seeds import MORAL_SEEDS, get_moral_seeds
from deepsteer.datasets.neutral_pool import NEUTRAL_POOL, get_flat_neutral_pool
from deepsteer.datasets.pairing import pair_by_word_count, pair_minimal
from deepsteer.datasets.pipeline import build_probing_dataset
from deepsteer.datasets.types import (
    GenerationMethod,
    NeutralDomain,
    ProbingDataset,
    ProbingPair,
)
from deepsteer.datasets.validation import (
    MORAL_KEYWORDS,
    STOPWORDS,
    ValidationStats,
    validate_pairs,
)

# ---------------------------------------------------------------------------
# Type construction and serialization
# ---------------------------------------------------------------------------


class TestTypes:
    def test_probing_pair_frozen(self):
        pair = ProbingPair(
            moral="Protecting children is important.",
            neutral="The pot boiled for six minutes.",
            foundation=MoralFoundation.CARE_HARM,
            neutral_domain=NeutralDomain.COOKING,
            generation_method=GenerationMethod.POOL,
            moral_word_count=5,
            neutral_word_count=6,
        )
        assert pair.moral == "Protecting children is important."
        with pytest.raises(AttributeError):
            pair.moral = "changed"  # type: ignore[misc]

    def test_probing_dataset_all_pairs(self):
        pair1 = _make_pair("a", "b", MoralFoundation.CARE_HARM)
        pair2 = _make_pair("c", "d", MoralFoundation.FAIRNESS_CHEATING)
        ds = ProbingDataset(
            train=[pair1],
            test=[pair2],
            metadata=_make_metadata(2, 1, 1),
        )
        assert len(ds.all_pairs) == 2
        assert ds.all_pairs[0] is pair1
        assert ds.all_pairs[1] is pair2

    def test_probing_dataset_to_dict(self):
        pair = _make_pair("a", "b", MoralFoundation.CARE_HARM)
        ds = ProbingDataset(
            train=[pair],
            test=[],
            metadata=_make_metadata(1, 1, 0),
        )
        d = ds.to_dict()
        assert isinstance(d, dict)
        assert "train" in d
        assert "test" in d
        assert "metadata" in d
        assert d["train"][0]["foundation"] == "care_harm"


# ---------------------------------------------------------------------------
# Moral seeds
# ---------------------------------------------------------------------------


class TestMoralSeeds:
    def test_all_foundations_present(self):
        for foundation in MoralFoundation:
            assert foundation in MORAL_SEEDS, f"Missing foundation: {foundation}"

    def test_at_least_40_seeds_per_foundation(self):
        for foundation, seeds in MORAL_SEEDS.items():
            assert len(seeds) >= 40, (
                f"{foundation.value}: only {len(seeds)} seeds (need ≥40)"
            )

    def test_word_count_range(self):
        for foundation, seeds in MORAL_SEEDS.items():
            for i, sent in enumerate(seeds):
                wc = len(sent.split())
                assert 5 <= wc <= 30, (
                    f"{foundation.value}[{i}]: {wc} words (expected 5-30): {sent[:50]}"
                )

    def test_no_duplicate_seeds(self):
        all_sentences: set[str] = set()
        for seeds in MORAL_SEEDS.values():
            for sent in seeds:
                assert sent not in all_sentences, f"Duplicate: {sent[:50]}"
                all_sentences.add(sent)

    def test_get_moral_seeds_all(self):
        result = get_moral_seeds()
        assert len(result) == 6
        # Should be a copy, not a reference
        result[MoralFoundation.CARE_HARM].append("extra")
        assert "extra" not in MORAL_SEEDS[MoralFoundation.CARE_HARM]

    def test_get_moral_seeds_single(self):
        result = get_moral_seeds(MoralFoundation.CARE_HARM)
        assert len(result) == 1
        assert MoralFoundation.CARE_HARM in result


# ---------------------------------------------------------------------------
# Neutral pool
# ---------------------------------------------------------------------------


class TestNeutralPool:
    def test_all_domains_present(self):
        for domain in NeutralDomain:
            if domain == NeutralDomain.MATCHED:
                continue  # MATCHED is for minimal pairs, not the neutral pool
            assert domain in NEUTRAL_POOL, f"Missing domain: {domain}"

    def test_at_least_250_total(self):
        total = sum(len(v) for v in NEUTRAL_POOL.values())
        assert total >= 250, f"Only {total} neutral sentences (need ≥250)"

    def test_word_count_range(self):
        for domain, sentences in NEUTRAL_POOL.items():
            for i, sent in enumerate(sentences):
                wc = len(sent.split())
                assert 5 <= wc <= 30, (
                    f"{domain.value}[{i}]: {wc} words (expected 5-30): {sent[:50]}"
                )

    def test_no_moral_keywords(self):
        for domain, sentences in NEUTRAL_POOL.items():
            for sent in sentences:
                words = set(sent.lower().split())
                overlap = words & MORAL_KEYWORDS
                assert not overlap, (
                    f"{domain.value}: moral keywords {overlap} in: {sent[:60]}"
                )

    def test_no_duplicate_neutrals(self):
        all_sentences: set[str] = set()
        for sentences in NEUTRAL_POOL.values():
            for sent in sentences:
                assert sent not in all_sentences, f"Duplicate: {sent[:50]}"
                all_sentences.add(sent)

    def test_get_flat_neutral_pool(self):
        flat = get_flat_neutral_pool()
        assert len(flat) >= 250
        for sent, domain in flat:
            assert isinstance(sent, str)
            assert isinstance(domain, NeutralDomain)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_bad_length_ratio(self):
        pairs = [
            _make_pair("one two three", "a b c d e f g h i j k", MoralFoundation.CARE_HARM),
        ]
        valid, stats = validate_pairs(pairs, max_length_ratio=1.3)
        assert len(valid) == 0
        assert stats.rejected_length == 1

    def test_rejects_moral_keywords_in_neutral(self):
        pairs = [
            _make_pair(
                "Kindness matters greatly in this world today.",
                "Justice requires fair treatment from all people.",
                MoralFoundation.CARE_HARM,
            ),
        ]
        valid, stats = validate_pairs(pairs)
        assert len(valid) == 0
        assert stats.rejected_keywords == 1

    def test_rejects_duplicates(self):
        pairs = [
            _make_pair(
                "A moral statement here today.",
                "The large pot boils on stove.",
                MoralFoundation.CARE_HARM,
            ),
            _make_pair(
                "Another moral statement here today.",
                "The large pot boils on stove.",
                MoralFoundation.FAIRNESS_CHEATING,
            ),
        ]
        valid, stats = validate_pairs(pairs)
        assert len(valid) == 1
        assert stats.rejected_duplicate == 1

    def test_rejects_low_word_overlap(self):
        pairs = [
            _make_pair(
                "Protecting children from harm is essential today.",
                "The copper pot heated slowly on the stove.",
                MoralFoundation.CARE_HARM,
            ),
        ]
        # With high overlap threshold, topically distant pairs should be rejected
        valid, stats = validate_pairs(pairs, min_word_overlap=0.5)
        assert len(valid) == 0
        assert stats.rejected_overlap == 1

    def test_passes_high_word_overlap(self):
        pairs = [
            _make_pair(
                "Protecting surfaces from moisture should be every priority.",
                "Protecting surfaces from sunlight should be every priority.",
                MoralFoundation.CARE_HARM,
            ),
        ]
        valid, stats = validate_pairs(pairs, min_word_overlap=0.3)
        assert len(valid) == 1
        assert stats.rejected_overlap == 0

    def test_passes_good_pairs(self):
        pairs = [
            _make_pair(
                "Protecting children from harm is essential today.",
                "The copper pot heated slowly on the stove.",
                MoralFoundation.CARE_HARM,
            ),
        ]
        valid, stats = validate_pairs(pairs)
        assert len(valid) == 1
        assert stats.passed == 1

    def test_stats_to_dict(self):
        stats = ValidationStats(input_count=10, rejected_length=2, rejected_overlap=1, passed=7)
        d = stats.to_dict()
        assert d["input_count"] == 10
        assert d["rejected_length"] == 2
        assert d["rejected_overlap"] == 1


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class TestPairing:
    def test_pair_by_word_count_basic(self):
        moral_seeds = {
            MoralFoundation.CARE_HARM: [
                "Protecting vulnerable children from all harm is absolutely essential.",
            ],
        }
        neutral_pool = [
            ("The copper pot heated slowly on the kitchen stove.", NeutralDomain.COOKING),
            ("Rain fell.", NeutralDomain.WEATHER),
        ]
        pairs = pair_by_word_count(moral_seeds, neutral_pool)
        assert len(pairs) == 1
        # Should match the closer-length neutral (9 words vs 2 words)
        assert "copper" in pairs[0].neutral
        assert pairs[0].generation_method == GenerationMethod.POOL

    def test_deterministic_with_seed(self):
        seeds = get_moral_seeds()
        pool = get_flat_neutral_pool()
        pairs1 = pair_by_word_count(seeds, pool, seed=123)
        pairs2 = pair_by_word_count(seeds, pool, seed=123)
        assert len(pairs1) == len(pairs2)
        for p1, p2 in zip(pairs1, pairs2):
            assert p1.moral == p2.moral
            assert p1.neutral == p2.neutral

    def test_pair_minimal_basic(self):
        mp = get_minimal_pairs()
        pairs = pair_minimal(mp, seed=42)
        assert len(pairs) == 300
        # All should have MATCHED domain
        for p in pairs:
            assert p.neutral_domain == NeutralDomain.MATCHED
            assert p.generation_method == GenerationMethod.POOL

    def test_pair_minimal_deterministic(self):
        mp = get_minimal_pairs()
        pairs1 = pair_minimal(mp, seed=77)
        pairs2 = pair_minimal(mp, seed=77)
        assert len(pairs1) == len(pairs2)
        for p1, p2 in zip(pairs1, pairs2):
            assert p1.moral == p2.moral
            assert p1.neutral == p2.neutral


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------


class TestBalancing:
    def test_balance_downsamples(self):
        pairs = [
            _make_pair(f"m{i}", f"n{i}", MoralFoundation.CARE_HARM)
            for i in range(60)
        ]
        balanced = balance_by_foundation(pairs, target_per_foundation=10)
        care_count = sum(1 for p in balanced if p.foundation == MoralFoundation.CARE_HARM)
        assert care_count == 10

    def test_stratified_split(self):
        pairs = []
        for foundation in MoralFoundation:
            for i in range(20):
                pairs.append(_make_pair(
                    f"m_{foundation.value}_{i}",
                    f"n_{foundation.value}_{i}",
                    foundation,
                ))
        train, test = train_test_split(pairs, test_fraction=0.2, stratify=True)
        # Each foundation should have some in both train and test
        train_foundations = {p.foundation for p in train}
        test_foundations = {p.foundation for p in test}
        assert train_foundations == test_foundations == set(MoralFoundation)

    def test_no_overlap_in_split(self):
        pairs = [
            _make_pair(f"m{i}", f"n{i}", MoralFoundation.CARE_HARM)
            for i in range(50)
        ]
        train, test = train_test_split(pairs, test_fraction=0.2)
        train_neutrals = {p.neutral for p in train}
        test_neutrals = {p.neutral for p in test}
        assert not train_neutrals & test_neutrals

    def test_report_distribution(self):
        pairs = [
            _make_pair("m1", "n1", MoralFoundation.CARE_HARM),
            _make_pair("m2", "n2", MoralFoundation.CARE_HARM),
            _make_pair("m3", "n3", MoralFoundation.FAIRNESS_CHEATING),
        ]
        dist = report_distribution(pairs)
        assert dist["care_harm"] == 2
        assert dist["fairness_cheating"] == 1
        assert dist["loyalty_betrayal"] == 0


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_build_probing_dataset_schema(self):
        ds = build_probing_dataset(target_per_foundation=10)
        assert isinstance(ds, ProbingDataset)
        assert isinstance(ds.train, list)
        assert isinstance(ds.test, list)
        assert len(ds.train) > 0
        assert len(ds.test) > 0

    def test_all_foundations_represented(self):
        ds = build_probing_dataset(target_per_foundation=10)
        foundations_in_train = {p.foundation for p in ds.train}
        foundations_in_test = {p.foundation for p in ds.test}
        for foundation in MoralFoundation:
            assert foundation in foundations_in_train, (
                f"{foundation.value} missing from train"
            )
            assert foundation in foundations_in_test, (
                f"{foundation.value} missing from test"
            )

    def test_train_test_no_overlap(self):
        ds = build_probing_dataset(target_per_foundation=10)
        train_neutrals = {p.neutral for p in ds.train}
        test_neutrals = {p.neutral for p in ds.test}
        assert not train_neutrals & test_neutrals

    def test_metadata_populated(self):
        ds = build_probing_dataset(target_per_foundation=10)
        assert ds.metadata.version == "1.0.0"
        assert ds.metadata.generation_method == "minimal_pair"
        assert ds.metadata.total_pairs > 0
        assert ds.metadata.train_pairs == len(ds.train)
        assert ds.metadata.test_pairs == len(ds.test)
        assert len(ds.metadata.foundations) == 6

    def test_legacy_pool_path(self):
        ds = build_probing_dataset(target_per_foundation=10, legacy_pool=True)
        assert ds.metadata.generation_method == "pool"
        assert ds.metadata.total_pairs > 0

    def test_serialization_roundtrip(self):
        ds = build_probing_dataset(target_per_foundation=5)
        d = ds.to_dict()
        assert isinstance(d, dict)
        assert len(d["train"]) == len(ds.train)
        assert len(d["test"]) == len(ds.test)
        # Check nested serialization
        if ds.train:
            assert isinstance(d["train"][0]["foundation"], str)

    def test_deterministic(self):
        ds1 = build_probing_dataset(target_per_foundation=5, seed=99)
        ds2 = build_probing_dataset(target_per_foundation=5, seed=99)
        assert len(ds1.train) == len(ds2.train)
        for p1, p2 in zip(ds1.train, ds2.train):
            assert p1.moral == p2.moral
            assert p1.neutral == p2.neutral


# ---------------------------------------------------------------------------
# Minimal pairs
# ---------------------------------------------------------------------------


class TestMinimalPairs:
    def test_all_foundations_present(self):
        for foundation in MoralFoundation:
            assert foundation in MINIMAL_PAIRS, f"Missing foundation: {foundation}"

    def test_at_least_40_pairs_per_foundation(self):
        for foundation, pairs in MINIMAL_PAIRS.items():
            assert len(pairs) >= 40, (
                f"{foundation.value}: only {len(pairs)} pairs (need ≥40)"
            )

    def test_word_count_ratio(self):
        for foundation, pairs in MINIMAL_PAIRS.items():
            for i, (moral, neutral) in enumerate(pairs):
                mwc = len(moral.split())
                nwc = len(neutral.split())
                ratio = max(mwc, nwc) / max(min(mwc, nwc), 1)
                assert ratio <= 1.15, (
                    f"{foundation.value}[{i}]: word count ratio {ratio:.2f} "
                    f"({mwc} vs {nwc}): {moral[:40]}..."
                )

    def test_word_overlap(self):
        """Mean content-word overlap should be well above the pipeline gate.

        Individual pairs may have low overlap when the moral seed is densely
        packed with moral vocabulary (requiring nearly all content words to
        change), but the dataset average should be solid.
        """
        _strip = str.maketrans("", "", string.punctuation)
        overlaps: list[float] = []
        for foundation, pairs in MINIMAL_PAIRS.items():
            for moral, neutral in pairs:
                moral_words = {
                    w.translate(_strip) for w in moral.lower().split()
                } - STOPWORDS - {""}
                neutral_words = {
                    w.translate(_strip) for w in neutral.lower().split()
                } - STOPWORDS - {""}
                if moral_words and neutral_words:
                    overlaps.append(
                        len(moral_words & neutral_words)
                        / max(len(moral_words), len(neutral_words))
                    )
        mean_overlap = sum(overlaps) / len(overlaps)
        assert mean_overlap >= 0.30, (
            f"Mean content-word overlap {mean_overlap:.2f} is below 0.30"
        )

    def test_no_moral_keywords_in_neutrals(self):
        for foundation, pairs in MINIMAL_PAIRS.items():
            for i, (_, neutral) in enumerate(pairs):
                words = set(neutral.lower().split())
                # Strip punctuation for keyword check
                stripped = {w.strip(".,;:!?\"'()-") for w in words}
                overlap = stripped & MORAL_KEYWORDS
                assert not overlap, (
                    f"{foundation.value}[{i}]: moral keywords {overlap} "
                    f"in neutral: {neutral[:60]}"
                )

    def test_no_duplicate_neutrals(self):
        all_neutrals: set[str] = set()
        for pairs in MINIMAL_PAIRS.values():
            for _, neutral in pairs:
                assert neutral not in all_neutrals, f"Duplicate: {neutral[:50]}"
                all_neutrals.add(neutral)

    def test_get_minimal_pairs_all(self):
        result = get_minimal_pairs()
        assert len(result) == 6
        # Should be a copy, not a reference
        result[MoralFoundation.CARE_HARM].append(("extra", "pair"))
        assert ("extra", "pair") not in MINIMAL_PAIRS[MoralFoundation.CARE_HARM]

    def test_get_minimal_pairs_single(self):
        result = get_minimal_pairs(MoralFoundation.CARE_HARM)
        assert len(result) == 1
        assert MoralFoundation.CARE_HARM in result


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


class TestImports:
    def test_datasets_package_exports(self):
        from deepsteer.datasets import (
            ProbingDataset,
            ProbingPair,
            build_probing_dataset,
            get_minimal_pairs,
        )
        assert callable(build_probing_dataset)
        assert callable(get_minimal_pairs)
        assert ProbingDataset is not None
        assert ProbingPair is not None

    def test_layer_wise_moral_probe_import(self):
        from deepsteer.benchmarks.representational import LayerWiseMoralProbe
        assert LayerWiseMoralProbe is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(
    moral: str,
    neutral: str,
    foundation: MoralFoundation,
) -> ProbingPair:
    return ProbingPair(
        moral=moral,
        neutral=neutral,
        foundation=foundation,
        neutral_domain=NeutralDomain.COOKING,
        generation_method=GenerationMethod.POOL,
        moral_word_count=len(moral.split()),
        neutral_word_count=len(neutral.split()),
    )


def _make_metadata(total, train, test):
    from deepsteer.datasets.types import DatasetMetadata
    return DatasetMetadata(
        version="1.0.0",
        generation_method="pool",
        total_pairs=total,
        train_pairs=train,
        test_pairs=test,
    )
