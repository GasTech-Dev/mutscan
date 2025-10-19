from __future__ import annotations

import random

import pytest
from Bio.Seq import Seq

from Mutation.simulate_mutations import (
    GenomeFeature,
    SelectionModel,
    blosum_score,
    build_feature_lookup,
    build_selection_model,
    choose_mutation_positions,
    compute_feature_codon_lengths,
    compute_position_weights,
    grantham_distance,
    hydrophobicity,
    mutate_sequence,
    simulate_indel_event,
)


def translate(seq: Seq) -> str:
    return str(seq.translate())


class FixedRandom(random.Random):
    """Random façade renvoyant des valeurs fixes pour random()."""

    def __init__(self, *, values: list[float]):
        super().__init__()
        self._values = iter(values)

    def random(self) -> float:  # type: ignore[override]
        try:
            return next(self._values)
        except StopIteration as exc:  # pragma: no cover - robustesse test
            raise AssertionError("Plus de valeurs aléatoires prédéfinies disponibles.") from exc


def test_mutate_sequence_keeps_protein_for_transition_silent():
    seq = Seq("GCTGAA")  # Ala, Glu
    rng = random.Random(0)

    mutated_seq, outcomes = mutate_sequence(
        seq,
        positions=[2],
        rng=rng,
        transition_weight=1.0,
        transversion_weight=0.0,
    )

    assert str(mutated_seq) == "GCCGAA"
    assert translate(mutated_seq) == translate(seq)
    assert outcomes[0].is_silent
    assert outcomes[0].mutated_codon == "GCC"
    assert outcomes[0].mutated_amino_acid == outcomes[0].original_amino_acid == "A"
    assert outcomes[0].severity_score == -1
    assert outcomes[0].blosum_score == pytest.approx(blosum_score("A", "A"))
    assert outcomes[0].acceptance_probability == pytest.approx(1.0)
    assert outcomes[0].codon_preference == pytest.approx(1.0)
    assert outcomes[0].grantham_distance == pytest.approx(0.0)
    assert outcomes[0].hydrophobicity_delta == pytest.approx(0.0)
    assert outcomes[0].flags == ()
    assert outcomes[0].accepted is True


def test_mutate_sequence_transition_changes_protein():
    seq = Seq("TTTGAA")  # Phe, Glu
    rng = FixedRandom(values=[0.8, 0.0])

    mutated_seq, outcomes = mutate_sequence(
        seq,
        positions=[0],
        rng=rng,
        transition_weight=0.0,
        transversion_weight=1.0,
    )

    assert str(mutated_seq) == "GTTGAA"
    assert translate(mutated_seq) != translate(seq)
    assert not outcomes[0].is_silent
    assert outcomes[0].mutated_codon == "GTT"
    assert outcomes[0].original_amino_acid == "F"
    assert outcomes[0].mutated_amino_acid == "V"
    assert outcomes[0].severity_score == 1
    assert outcomes[0].blosum_score == pytest.approx(blosum_score("F", "V"))
    assert outcomes[0].acceptance_probability == pytest.approx(1.0)
    assert outcomes[0].grantham_distance == pytest.approx(grantham_distance("F", "V"))
    expected_hydro = hydrophobicity("V") - hydrophobicity("F")
    assert outcomes[0].hydrophobicity_delta == pytest.approx(expected_hydro)
    assert outcomes[0].flags == ()
    assert outcomes[0].accepted is True


def test_compute_position_weights_marks_cpg_hotspots():
    seq = Seq("ACGTCG")
    feature_lookup = build_feature_lookup([], len(seq))

    weights = compute_position_weights(seq, feature_lookup, cpg_weight=3.0, feature_weight_overrides={})

    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(3.0)
    assert weights[2] == pytest.approx(3.0)
    assert weights[3] == pytest.approx(1.0)
    assert weights[4] == pytest.approx(3.0)
    assert weights[5] == pytest.approx(3.0)


def test_compute_position_weights_applies_feature_multiplier():
    seq = Seq("AAAAAA")
    features = [GenomeFeature(start=2, end=4, name="nspX", product="Prot X")]
    feature_lookup = build_feature_lookup(features, len(seq))

    weights = compute_position_weights(
        seq,
        feature_lookup,
        cpg_weight=1.0,
        feature_weight_overrides={"nspX": 2.5},
    )

    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(2.5)
    assert weights[2] == pytest.approx(2.5)
    assert weights[3] == pytest.approx(2.5)
    assert weights[4] == pytest.approx(1.0)
    assert weights[5] == pytest.approx(1.0)


def test_compute_feature_codon_lengths_counts_features_and_unannotated_region():
    seq_length = 12  # 4 codons
    features = [
        GenomeFeature(start=1, end=6, name="nsp1", product="Prot 1"),
        GenomeFeature(start=7, end=9, name="nsp2", product="Prot 2"),
    ]
    feature_lookup = build_feature_lookup(features, seq_length)

    lengths = compute_feature_codon_lengths(feature_lookup)

    assert lengths["nsp1"] == 2
    assert lengths["nsp2"] == 1
    assert lengths["non_annotée"] == 1


def test_choose_mutation_positions_respects_zero_weights():
    seq = Seq("ACGT")
    rng = random.Random(0)
    position_weights = [0.0, 2.0, 0.0, 0.0]

    positions = choose_mutation_positions(seq, count=1, rng=rng, position_weights=position_weights)

    assert positions == [1]


def test_selection_model_rejects_stop_codon():
    seq = Seq("TGG")  # Trp
    rng = FixedRandom(values=[0.0, 0.5, 0.8, 0.0])
    selection_model = build_selection_model(seq, selection_strength=1.0, codon_usage_weight=0.0)

    mutated_seq, outcomes = mutate_sequence(
        seq,
        positions=[2],
        rng=rng,
        transition_weight=1.0,
        transversion_weight=1.0,
        selection_model=selection_model,
    )

    assert [outcome.accepted for outcome in outcomes] == [False, True]
    assert outcomes[0].mutated_amino_acid == "*"
    assert outcomes[0].acceptance_probability == 0.0
    assert outcomes[1].mutated_amino_acid != "*"
    assert translate(mutated_seq) == outcomes[1].mutated_amino_acid


def test_mutate_sequence_records_rejection_before_acceptance():
    seq = Seq("TTT")  # Phe
    selection_model = build_selection_model(seq, selection_strength=1.0, codon_usage_weight=0.0)
    rng = FixedRandom(values=[0.4, 0.5, 0.4, 0.0])

    mutated_seq, outcomes = mutate_sequence(
        seq,
        positions=[1],
        rng=rng,
        transition_weight=1.0,
        transversion_weight=1.0,
        selection_model=selection_model,
    )

    assert str(mutated_seq) == "TGT"
    assert [outcome.accepted for outcome in outcomes] == [False, True]
    expected_prob, _ = selection_model.acceptance_factors(
        original_codon="TTT",
        mutated_codon="TGT",
        original_amino_acid="F",
        mutated_amino_acid="C",
    )
    assert outcomes[0].acceptance_probability == pytest.approx(expected_prob)
    assert outcomes[1].acceptance_probability == pytest.approx(expected_prob)
    assert outcomes[0].blosum_score == pytest.approx(blosum_score("F", "C"))
    assert outcomes[0].accepted is False
    assert outcomes[1].accepted is True


def test_selection_model_codons_floor_avoids_zero_weights():
    seq = Seq("GCTGCT")
    selection_model = SelectionModel(
        selection_strength=0.0,
        codon_usage_weight=1.0,
        codon_preferences={"GCT": 1.0},
    )
    rng = FixedRandom(values=[0.4, 0.0])

    mutated_seq, _ = mutate_sequence(
        seq,
        positions=[2],
        rng=rng,
        transition_weight=1.0,
        transversion_weight=1.0,
        selection_model=selection_model,
    )

    # Aucune exception et un codon acceptable doit être généré malgré l'absence de préférence explicite.
    assert len(mutated_seq) == len(seq)


def test_simulate_indel_event_deletion_acceptance():
    seq = Seq("ATGATG")
    rng = random.Random(4)

    new_seq, outcomes = simulate_indel_event(
        seq,
        rng,
        probability=1.0,
        max_codon_length=1,
        proofreading_acceptance=1.0,
        codon_library=["ATG"],
        feature_lookup=[None] * len(seq),
    )

    assert len(new_seq) == len(seq) - 3
    assert outcomes and outcomes[0].kind == "deletion"
    assert outcomes[0].accepted


def test_simulate_indel_event_insertion_rejected():
    seq = Seq("ATG")
    rng = random.Random(0)

    new_seq, outcomes = simulate_indel_event(
        seq,
        rng,
        probability=1.0,
        max_codon_length=1,
        proofreading_acceptance=0.0,
        codon_library=["ATG"],
        feature_lookup=[None] * len(seq),
    )

    assert len(new_seq) == len(seq)
    assert outcomes and outcomes[0].kind == "insertion"
    assert not outcomes[0].accepted
