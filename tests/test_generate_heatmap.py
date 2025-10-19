from __future__ import annotations

import math
from pathlib import Path

from Mutation.generate_heatmap import (
    DomainDefinition,
    MutationRecord,
    compute_domain_summary,
    compute_mutation_density_matrices,
    parse_comparison_argument,
    parse_log_lines,
)


def test_parse_log_lines_extracts_positions_and_scores() -> None:
    lines = [
        (
            "  - [acceptée] nt 16: C->A | codon 5: CCA->CAA | aa P->Q | "
            "score 2 (critique) | BLOSUM -1.0 | Grantham 76 | Δhydro -1.90 | "
            "p_accept 0.50 | codon_pref 1.00 | flags proline_perdue | protéine: test"
        )
    ]

    records = parse_log_lines(lines)
    assert len(records) == 1
    record = records[0]

    assert record.nt_position == 16
    assert record.codon_index == 5
    assert record.amino_position == 6
    assert record.severity == 2
    assert record.severity_label == "critique"
    assert record.aa_reference == "P"
    assert record.aa_mutated == "Q"
    assert math.isclose(record.blosum, -1.0)
    assert math.isclose(record.p_accept, 0.50)


def test_compute_mutation_density_matrices_groups_by_bins() -> None:
    records = [
        MutationRecord(
            status="acceptée",
            protein="prot",
            blosum=1.0,
            p_accept=0.5,
            grantham=10.0,
            hydrophobicity_delta=0.1,
            simulation=1,
            nt_position=0,
            codon_index=0,
            severity=2,
            severity_label="critique",
            aa_reference="A",
            aa_mutated="V",
        ),
        MutationRecord(
            status="acceptée",
            protein="prot",
            blosum=0.0,
            p_accept=0.7,
            grantham=5.0,
            hydrophobicity_delta=0.2,
            simulation=1,
            nt_position=15,
            codon_index=5,
            severity=0,
            severity_label="conservateur",
            aa_reference="L",
            aa_mutated="L",
        ),
        MutationRecord(
            status="rejetée",
            protein="prot",
            blosum=-2.0,
            p_accept=0.1,
            grantham=120.0,
            hydrophobicity_delta=-0.4,
            simulation=1,
            nt_position=33,
            codon_index=11,
            severity=3,
            severity_label="drastique",
            aa_reference="G",
            aa_mutated="*",
        ),
    ]

    matrices = compute_mutation_density_matrices(records, {"prot": 12}, bin_size=5)
    assert "prot" in matrices
    matrix = matrices["prot"]
    assert list(matrix.index) == ["acceptées", "rejetées"]
    assert list(matrix.columns) == ["1-5", "6-10", "11-12"]
    assert matrix.loc["acceptées", "1-5"] == 1
    assert matrix.loc["acceptées", "6-10"] == 1
    assert matrix.loc["rejetées", "11-12"] == 1


def test_compute_domain_summary_aggregates_scores() -> None:
    records = [
        MutationRecord(
            status="acceptée",
            protein="prot",
            blosum=-1.0,
            p_accept=0.5,
            grantham=50.0,
            hydrophobicity_delta=0.3,
            simulation=1,
            nt_position=0,
            codon_index=0,
            severity=2,
            severity_label="critique",
            aa_reference="P",
            aa_mutated="Q",
        ),
        MutationRecord(
            status="acceptée",
            protein="prot",
            blosum=3.0,
            p_accept=0.9,
            grantham=10.0,
            hydrophobicity_delta=-0.1,
            simulation=1,
            nt_position=12,
            codon_index=4,
            severity=0,
            severity_label="conservateur",
            aa_reference="L",
            aa_mutated="L",
        ),
    ]
    domains = {
        "prot": [
            DomainDefinition(name="DNA-binding", start=1, end=10, protein="prot", product="binding"),
        ]
    }

    summary = compute_domain_summary(records, domains)
    assert not summary.empty
    row = summary.iloc[0]
    assert row["domain"] == "DNA-binding"
    assert row["accepted"] == 2
    assert math.isclose(row["mutations_per_aa"], 2 / 10)
    assert math.isclose(row["mean_severity"], 1.0)
    assert math.isclose(row["mean_blosum"], 1.0)
    assert math.isclose(row["mean_acceptance_probability"], 0.7)


def test_parse_comparison_argument_supports_optional_fields() -> None:
    arg = "GAPDH|/tmp/gapdh.log|/tmp/gapdh.fasta|/tmp/annot.csv|record1"
    label, log_path, fasta_path, annotation_path, record_id = parse_comparison_argument(arg)
    assert label == "GAPDH"
    assert isinstance(log_path, Path)
    assert isinstance(fasta_path, Path)
    assert isinstance(annotation_path, Path)
    assert record_id == "record1"

    label2, log_path2, fasta_path2, annotation_path2, record_id2 = parse_comparison_argument(
        "ACTB|/tmp/actb.log|/tmp/actb.fasta"
    )
    assert label2 == "ACTB"
    assert isinstance(log_path2, Path)
    assert isinstance(fasta_path2, Path)
    assert annotation_path2 is None
    assert record_id2 is None
