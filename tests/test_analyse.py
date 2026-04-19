"""
Unit tests for src/04_analyse.py

Tests cover the pure analytical functions — no file I/O, no model loading.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import importlib
analyse = importlib.import_module("04_analyse")

table_by_group       = analyse.table_by_group
dominant_framing_share = analyse.dominant_framing_share
aggregate_to_mep     = analyse.aggregate_to_mep
framing_cols         = analyse.framing_cols


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCORE_COLS = ["score_risk_based", "score_rights_based",
              "score_innovation_focused", "score_sovereignty_focused"]


@pytest.fixture
def minimal_cfg():
    return {
        "framings": {
            "risk_based":           {"label": "risk_based"},
            "rights_based":         {"label": "rights_based"},
            "innovation_focused":   {"label": "innovation_focused"},
            "sovereignty_focused":  {"label": "sovereignty_focused"},
        }
    }


@pytest.fixture
def classified_df():
    """Minimal classified speech DataFrame."""
    return pd.DataFrame({
        "firstname":   ["Alice", "Alice", "Bob",   "Bob",   "Carol"],
        "lastname":    ["A",     "A",     "B",     "B",     "C"],
        "party_family": ["Social Democrat", "Social Democrat",
                         "Liberal", "Liberal", "Radical Left"],
        "east_west":   ["West", "West", "East", "East", "West"],
        "north_south": ["North", "North", "South", "South", "Middle"],
        "accession":   ["pre-2004", "pre-2004", "post-2004", "post-2004", "pre-2004"],
        "nationality": ["Germany", "Germany", "Poland", "Poland", "France"],
        "score_risk_based":          [0.4, 0.3, 0.1, 0.2, 0.5],
        "score_rights_based":        [0.3, 0.2, 0.1, 0.1, 0.2],
        "score_innovation_focused":  [0.2, 0.3, 0.5, 0.4, 0.2],
        "score_sovereignty_focused": [0.1, 0.2, 0.3, 0.3, 0.1],
        "dominant_framing": ["risk_based", "innovation_focused",
                             "innovation_focused", "innovation_focused", "risk_based"],
    })


# ---------------------------------------------------------------------------
# framing_cols
# ---------------------------------------------------------------------------

class TestFramingCols:

    def test_returns_correct_column_names(self, minimal_cfg):
        cols = framing_cols(minimal_cfg)
        assert cols == SCORE_COLS

    def test_order_matches_config(self, minimal_cfg):
        cols = framing_cols(minimal_cfg)
        assert cols[0] == "score_risk_based"
        assert cols[-1] == "score_sovereignty_focused"


# ---------------------------------------------------------------------------
# table_by_group
# ---------------------------------------------------------------------------

class TestTableByGroup:

    def test_percentages_sum_to_100(self, classified_df):
        result = table_by_group(classified_df, "party_family", SCORE_COLS)
        pct_cols = [c for c in result.columns if c.endswith("%")]
        row_sums = result[pct_cols].sum(axis=1)
        assert (row_sums - 100).abs().max() < 0.5  # allow rounding

    def test_n_speeches_correct(self, classified_df):
        result = table_by_group(classified_df, "party_family", SCORE_COLS)
        assert result.loc["Social Democrat", "N_speeches"] == 2
        assert result.loc["Liberal", "N_speeches"] == 2

    def test_missing_column_returns_empty(self, classified_df):
        result = table_by_group(classified_df, "nonexistent_col", SCORE_COLS)
        assert result.empty

    def test_dominant_framing_is_highest_score(self, classified_df):
        result = table_by_group(classified_df, "party_family", SCORE_COLS)
        pct_cols = [c for c in result.columns if c.endswith("%")]
        for _, row in result.iterrows():
            dominant = row["dominant_framing"] + "%"
            assert row[dominant] == row[pct_cols].max()

    def test_n_meps_counted_correctly(self, classified_df):
        result = table_by_group(classified_df, "party_family", SCORE_COLS)
        # Alice and Bob are 1 MEP each in their groups
        assert result.loc["Social Democrat", "N_meps"] == 1
        assert result.loc["Liberal", "N_meps"] == 1


# ---------------------------------------------------------------------------
# dominant_framing_share
# ---------------------------------------------------------------------------

class TestDominantFramingShare:

    def test_rows_sum_to_one(self, classified_df):
        result = dominant_framing_share(classified_df, "party_family")
        row_sums = result.sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 0.01

    def test_missing_column_returns_empty(self, classified_df):
        result = dominant_framing_share(classified_df, "nonexistent")
        assert result.empty

    def test_values_between_0_and_1(self, classified_df):
        result = dominant_framing_share(classified_df, "east_west")
        assert result.values.min() >= 0.0
        assert result.values.max() <= 1.0


# ---------------------------------------------------------------------------
# aggregate_to_mep
# ---------------------------------------------------------------------------

class TestAggregateToMep:

    def test_one_row_per_mep(self, classified_df):
        result = aggregate_to_mep(classified_df, SCORE_COLS)
        # 3 unique MEPs: Alice A, Bob B, Carol C
        assert len(result) == 3

    def test_scores_averaged_correctly(self, classified_df):
        result = aggregate_to_mep(classified_df, SCORE_COLS)
        alice = result[result["firstname"] == "Alice"].iloc[0]
        # Alice has scores [0.4, 0.3] → mean 0.35 for risk_based
        assert abs(alice["score_risk_based"] - 0.35) < 0.001

    def test_dominant_framing_from_averaged_scores(self, classified_df):
        result = aggregate_to_mep(classified_df, SCORE_COLS)
        # For each MEP, dominant_framing should match argmax of averaged scores
        for _, row in result.iterrows():
            expected = max(SCORE_COLS, key=lambda c: row[c]).replace("score_", "")
            assert row["dominant_framing"] == expected

    def test_n_speeches_correct(self, classified_df):
        result = aggregate_to_mep(classified_df, SCORE_COLS)
        alice = result[result["firstname"] == "Alice"].iloc[0]
        bob   = result[result["firstname"] == "Bob"].iloc[0]
        carol = result[result["firstname"] == "Carol"].iloc[0]
        assert alice["N_speeches"] == 2
        assert bob["N_speeches"] == 2
        assert carol["N_speeches"] == 1

    def test_missing_name_columns_returns_df(self):
        df = pd.DataFrame({
            "score_risk_based": [0.5],
            "score_rights_based": [0.2],
            "score_innovation_focused": [0.2],
            "score_sovereignty_focused": [0.1],
        })
        result = aggregate_to_mep(df, SCORE_COLS)
        # Should return df unchanged with a warning
        assert len(result) == 1
