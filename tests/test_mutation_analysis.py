from __future__ import annotations

import pandas as pd
import pytest

from Mutation.analyze_mutations import (
    aggregate_impacts,
    classify_impact,
    compute_oncogene_risk,
)


def test_classify_impact_applies_thresholds():
    assert classify_impact(2, 0.0) == "critique"  # severity >= 2
    assert classify_impact(0, -3.5) == "critique"  # harsh BLOSUM
    assert classify_impact(1, -0.5) == "drastique"  # severity 1
    assert classify_impact(-1, -2.5) == "drastique"  # mild severity but low BLOSUM
    assert classify_impact(0, 1.0) == "conservatrice"
    assert classify_impact(-1, 0.5) == "synonyme"


def test_aggregate_impacts_computes_means():
    df = pd.DataFrame(
        [
            {"impact": "critique", "blosum": -4.0, "grantham": 150.0, "p_accept": 0.1},
            {"impact": "critique", "blosum": -2.0, "grantham": 90.0, "p_accept": 0.2},
            {"impact": "conservatrice", "blosum": 2.0, "grantham": 45.0, "p_accept": 0.9},
        ]
    )

    summary = aggregate_impacts(df)

    impacts = list(summary["impact"].astype(str))
    assert impacts == ["critique", "conservatrice"]

    critique_row = summary[summary["impact"] == "critique"].iloc[0]
    assert critique_row["count"] == 2
    assert critique_row["mean_blosum"] == pytest.approx(-3.0)
    assert critique_row["mean_grantham"] == pytest.approx(120.0)
    assert critique_row["mean_accept"] == pytest.approx(0.15)


def test_compute_oncogene_risk_ranks_proteins():
    df = pd.DataFrame(
        [
            {"protein": "S", "impact": "critique"},
            {"protein": "S", "impact": "drastique"},
            {"protein": "M", "impact": "conservatrice"},
            {"protein": "M", "impact": "synonyme"},
        ]
    )

    risk = compute_oncogene_risk(df)

    assert list(risk.index) == ["S", "M"]
    assert risk.loc["S"] == pytest.approx((3 + 2) / 2)
    assert risk.loc["M"] == pytest.approx((1 + 0) / 2)
