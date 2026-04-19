"""
Unit tests for src/01_filter_corpus.py

Tests cover the pure data-transformation functions — no model loading,
no file I/O, no network calls. The NLI relevance filter is not tested here
because it requires a GPU/CPU model; use 03_validate.py for that.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from importlib import import_module
fc = import_module("01_filter_corpus")

resolve_text      = fc.resolve_text
filter_corpus     = fc.filter_corpus
add_cleavage_vars = fc.add_cleavage_vars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_cfg():
    return {
        "corpus": {
            "years": [2019, 2020, 2021, 2022, 2023, 2024],
            "min_speech_length": 50,
        },
        "analysis": {
            "east_countries":   ["Poland", "Czech Republic", "Hungary"],
            "west_countries":   ["Germany", "France", "Italy"],
            "north_countries":  ["Sweden", "Denmark", "Germany"],
            "middle_countries": ["France", "Poland"],
            "south_countries":  ["Italy", "Greece"],
            "pre2004_countries": ["Germany", "France", "Italy"],
            "post2004_countries": ["Poland", "Czech Republic", "Hungary"],
            "party_family": {
                "EPP": "Christian Democrat / Conservative",
                "SOCPESPASD": "Social Democrat",
                "GUENGL": "Radical Left",
            },
        },
    }


@pytest.fixture
def speech_df():
    """Minimal dataframe mimicking EUPDCorp structure."""
    return pd.DataFrame({
        "speech":    ["Hello this is a long enough English speech about AI.",
                      "Ceci est un discours sur l'intelligence artificielle.",
                      "x"],           # too short
        "speech_en": ["Hello this is a long enough English speech about AI.",
                      "This is the English translation of the French speech.",
                      "x"],
        "language":  ["en", "fr", "en"],
        "date":      ["2021-03-01", "2020-06-15", "2021-01-01"],
        "nationality": ["Germany", "France", "Italy"],
        "epg_short": ["EPP", "SOCPESPASD", "GUENGL"],
    })


# ---------------------------------------------------------------------------
# resolve_text
# ---------------------------------------------------------------------------

class TestResolveText:

    def test_english_speech_uses_original(self, speech_df):
        result = resolve_text(speech_df.copy())
        # English speech → uses speech column directly
        assert result.loc[0, "text_resolved"] == speech_df.loc[0, "speech"]

    def test_non_english_uses_translation(self, speech_df):
        result = resolve_text(speech_df.copy())
        # French speech → uses speech_en translation
        assert result.loc[1, "text_resolved"] == speech_df.loc[1, "speech_en"]

    def test_fallback_when_no_translation_column(self):
        df = pd.DataFrame({"speech": ["Only speech column here."]})
        result = resolve_text(df.copy())
        assert result.loc[0, "text_resolved"] == "Only speech column here."

    def test_output_column_always_present(self, speech_df):
        result = resolve_text(speech_df.copy())
        assert "text_resolved" in result.columns


# ---------------------------------------------------------------------------
# filter_corpus
# ---------------------------------------------------------------------------

class TestFilterCorpus:

    def test_year_filter_keeps_correct_years(self, speech_df, minimal_cfg):
        df = resolve_text(speech_df.copy())
        result = filter_corpus(df, minimal_cfg)
        assert all(result["year"].isin(minimal_cfg["corpus"]["years"]))

    def test_year_filter_drops_old_speeches(self, minimal_cfg):
        df = pd.DataFrame({
            "speech": ["A" * 100, "B" * 100],
            "speech_en": ["A" * 100, "B" * 100],
            "language": ["en", "en"],
            "date": ["2015-01-01", "2021-01-01"],
        })
        df = resolve_text(df)
        result = filter_corpus(df, minimal_cfg)
        assert len(result) == 1
        assert result.iloc[0]["year"] == 2021

    def test_length_filter_drops_short_speeches(self, minimal_cfg):
        df = pd.DataFrame({
            "speech": ["short", "A" * 100],
            "speech_en": ["short", "A" * 100],
            "language": ["en", "en"],
            "date": ["2021-01-01", "2021-01-01"],
        })
        df = resolve_text(df)
        result = filter_corpus(df, minimal_cfg)
        assert len(result) == 1

    def test_year_column_added(self, speech_df, minimal_cfg):
        df = resolve_text(speech_df.copy())
        result = filter_corpus(df, minimal_cfg)
        assert "year" in result.columns


# ---------------------------------------------------------------------------
# add_cleavage_vars
# ---------------------------------------------------------------------------

class TestAddCleavageVars:

    def _make_df(self, nationalities, parties=None):
        n = len(nationalities)
        return pd.DataFrame({
            "text_resolved": ["test speech"] * n,
            "nationality": nationalities,
            "epg_short": parties if parties else ["EPP"] * n,
        })

    def test_east_west_coding(self, minimal_cfg):
        df = self._make_df(["Germany", "Poland", "Unknown"])
        result = add_cleavage_vars(df, minimal_cfg)
        assert result.loc[0, "east_west"] == "West"
        assert result.loc[1, "east_west"] == "East"
        assert result.loc[2, "east_west"] == "Other"

    def test_north_south_three_way(self, minimal_cfg):
        df = self._make_df(["Sweden", "France", "Italy", "Unknown"])
        result = add_cleavage_vars(df, minimal_cfg)
        assert result.loc[0, "north_south"] == "North"
        assert result.loc[1, "north_south"] == "Middle"
        assert result.loc[2, "north_south"] == "South"
        assert result.loc[3, "north_south"] == "Other"

    def test_accession_coding(self, minimal_cfg):
        df = self._make_df(["Germany", "Poland", "Unknown"])
        result = add_cleavage_vars(df, minimal_cfg)
        assert result.loc[0, "accession"] == "pre-2004"
        assert result.loc[1, "accession"] == "post-2004"
        assert result.loc[2, "accession"] == "Other"

    def test_party_family_mapping(self, minimal_cfg):
        df = self._make_df(["Germany", "France"], ["EPP", "SOCPESPASD"])
        result = add_cleavage_vars(df, minimal_cfg)
        assert result.loc[0, "party_family"] == "Christian Democrat / Conservative"
        assert result.loc[1, "party_family"] == "Social Democrat"

    def test_unmapped_party_keeps_original(self, minimal_cfg):
        df = self._make_df(["Germany"], ["UNKNOWN_GROUP"])
        result = add_cleavage_vars(df, minimal_cfg)
        # Unrecognised groups keep their original epg_short value
        assert result.loc[0, "party_family"] == "UNKNOWN_GROUP"

    def test_missing_nationality_col_returns_df(self, minimal_cfg):
        df = pd.DataFrame({"text_resolved": ["test"], "epg_short": ["EPP"]})
        result = add_cleavage_vars(df, minimal_cfg)
        assert "east_west" not in result.columns
