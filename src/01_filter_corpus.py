"""
Filter EUPDCorp to AI-relevant EP speeches (9th term, 2019-2024).
Input:  data/raw/EUPDCorp.csv  (download from https://zenodo.org/records/15056399)
Output: data/processed/ep_ai_speeches.csv
"""

import re
import yaml
import pandas as pd
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_eupd(path: str) -> pd.DataFrame:
    print(f"Loading EUPDCorp from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    return df


def resolve_text(df: pd.DataFrame) -> pd.DataFrame:
    """Use original text if English, otherwise fall back to English translation."""
    if "text_en" in df.columns and "language" in df.columns:
        df["text_resolved"] = df.apply(
            lambda r: r["text"] if str(r.get("language", "")).lower() == "en"
                      else r.get("text_en", r["text"]),
            axis=1
        )
    else:
        df["text_resolved"] = df["text"]
    return df


def filter_corpus(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    corpus_cfg = cfg["corpus"]

    # 1. Year range
    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
        df = df[df["year"].isin(corpus_cfg["years"])].copy()
        print(f"  After year filter: {len(df):,} speeches")

    # 2. Minimum length (applied to resolved English text)
    df = df[df["text_resolved"].str.len() >= corpus_cfg["min_speech_length"]].copy()
    print(f"  After length filter: {len(df):,} speeches")

    # 3. Keyword filter on resolved English text
    pattern = "|".join(re.escape(kw) for kw in corpus_cfg["keywords"])
    mask = df["text_resolved"].str.contains(pattern, case=False, na=False)
    df = df[mask].copy()
    print(f"  After keyword filter: {len(df):,} speeches")

    return df


def add_cleavage_vars(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add East/West and North/South dummy variables from nationality."""
    analysis_cfg = cfg["analysis"]
    east = set(analysis_cfg["east_countries"])
    west = set(analysis_cfg["west_countries"])
    north = set(analysis_cfg["north_countries"])
    south = set(analysis_cfg["south_countries"])

    nat_col = next((c for c in ["nationality", "country", "member_state"] if c in df.columns), None)
    if nat_col is None:
        print("  WARNING: no nationality column found, skipping cleavage vars")
        return df

    df["east_west"] = df[nat_col].apply(
        lambda x: "East" if x in east else ("West" if x in west else "Other")
    )
    df["north_south"] = df[nat_col].apply(
        lambda x: "North" if x in north else ("South" if x in south else "Other")
    )

    # Party family mapping
    pf_map = analysis_cfg["party_family"]
    party_col = next((c for c in ["party", "ep_group", "group"] if c in df.columns), None)
    if party_col:
        df["party_family"] = df[party_col].map(pf_map).fillna("Other")

    return df


def main():
    cfg = load_config()
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df = load_eupd(cfg["data"]["raw_path"])
    df = resolve_text(df)
    df = filter_corpus(df, cfg)
    df = add_cleavage_vars(df, cfg)

    # Retain analytically relevant columns
    keep = [c for c in [
        "speaker_name", "nationality", "country", "member_state",
        "party", "ep_group", "group", "party_family",
        "date", "year", "language",
        "east_west", "north_south",
        "text_resolved", "agenda", "debate_title"
    ] if c in df.columns]

    df = df[keep].rename(columns={"text_resolved": "text"}).reset_index(drop=True)
    df.to_csv(cfg["data"]["output_path"], index=False)

    print(f"\nSaved {len(df):,} speeches -> {cfg['data']['output_path']}")

    nat_col = next((c for c in ["nationality", "country", "member_state"] if c in df.columns), None)
    if nat_col:
        print(f"\nNationality distribution (top 15):")
        print(df[nat_col].value_counts().head(15).to_string())
    if "party_family" in df.columns:
        print(f"\nParty family distribution:")
        print(df["party_family"].value_counts().to_string())
    if "east_west" in df.columns:
        print(f"\nEast-West distribution:")
        print(df["east_west"].value_counts().to_string())


if __name__ == "__main__":
    main()
