"""
Analyse framing scores by party family, nationality, East-West, North-South.
Input:  data/processed/ep_ai_classified.csv
Output: data/processed/ep_ai_analysis.csv + printed summary tables
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["results_path"])
    print(f"Loaded {len(df):,} classified speeches.")
    return df


def framing_cols(cfg: dict) -> list:
    return [f"score_{v['label']}" for v in cfg["framings"].values()]


# -------------------------------------------------------------------
# 1. Descriptive tables
# -------------------------------------------------------------------

def table_by_group(df: pd.DataFrame, group_col: str, score_cols: list) -> pd.DataFrame:
    """Mean framing scores by group, plus N speeches, N MEPs, dominant framing."""
    if group_col not in df.columns:
        print(f"  Column '{group_col}' not found, skipping.")
        return pd.DataFrame()

    agg = df.groupby(group_col)[score_cols].mean().round(3)
    agg["N_speeches"] = df.groupby(group_col).size()
    agg["N_meps"] = df.groupby(group_col)["speaker_name"].nunique() \
        if "speaker_name" in df.columns else np.nan
    agg["dominant_framing"] = agg[score_cols].idxmax(axis=1).str.replace("score_", "")
    return agg.sort_values("N_speeches", ascending=False)


def dominant_framing_share(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Share of speeches where each framing is dominant, by group."""
    if group_col not in df.columns:
        return pd.DataFrame()
    cross = pd.crosstab(df[group_col], df["dominant_framing"], normalize="index").round(3)
    return cross


# -------------------------------------------------------------------
# 2. MEP-level aggregation (avoids high-volume speaker bias)
# -------------------------------------------------------------------

def aggregate_to_mep(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    """Average framing scores per MEP, retaining group metadata."""
    if "speaker_name" not in df.columns:
        print("  WARNING: 'speaker_name' column not found, cannot aggregate to MEP level.")
        return df

    meta_cols = [c for c in [
        "speaker_name", "nationality", "country", "member_state",
        "party", "ep_group", "group", "party_family",
        "east_west", "north_south"
    ] if c in df.columns]

    mep_scores = df.groupby("speaker_name")[score_cols].mean().round(4)
    mep_meta = df.groupby("speaker_name")[meta_cols].first()
    mep_df = mep_meta.join(mep_scores).reset_index()
    mep_df["dominant_framing"] = mep_df[score_cols].idxmax(axis=1).str.replace("score_", "")
    mep_df["N_speeches"] = df.groupby("speaker_name").size().values
    return mep_df


# -------------------------------------------------------------------
# 3. OLS with clustered standard errors
# -------------------------------------------------------------------

def run_regressions(mep_df: pd.DataFrame, score_cols: list):
    """
    For each framing score: OLS regressing score on party family +
    east_west + north_south dummies, with country-clustered SEs.
    Reference categories: Social Democrat, West, North.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  statsmodels not installed.")
        print("  Run: pip install statsmodels --break-system-packages")
        return

    regressors = []
    if "party_family" in mep_df.columns:
        regressors.append("C(party_family, Treatment('Social Democrat'))")
    if "east_west" in mep_df.columns:
        regressors.append("C(east_west, Treatment('West'))")
    if "north_south" in mep_df.columns:
        regressors.append("C(north_south, Treatment('North'))")

    if not regressors:
        print("  No regressor columns found.")
        return

    cluster_col = next((c for c in ["nationality", "country", "member_state"]
                        if c in mep_df.columns), None)
    formula_rhs = " + ".join(regressors)

    print("\n" + "=" * 60)
    print("OLS REGRESSION RESULTS (MEP-level, clustered SEs by country)")
    print("Reference: Social Democrat | West | North")
    print("=" * 60)

    for sc in score_cols:
        label = sc.replace("score_", "")
        formula = f"{sc} ~ {formula_rhs}"
        try:
            model = smf.ols(formula, data=mep_df).fit(
                cov_type="cluster",
                cov_kwds={"groups": mep_df[cluster_col]} if cluster_col else {}
            )
            print(f"\n--- Dependent variable: {label} ---")
            coef_df = pd.DataFrame({
                "coef": model.params,
                "se": model.bse,
                "t": model.tvalues,
                "p": model.pvalues
            }).round(4)
            # Show p < 0.15 during exploration; tighten to 0.05 for publication
            print(coef_df[coef_df["p"] < 0.15].to_string())
            print(f"  R2 = {model.rsquared:.4f}  |  N = {int(model.nobs)}")
        except Exception as e:
            print(f"  Could not fit model for {label}: {e}")


# -------------------------------------------------------------------
# 4. Main
# -------------------------------------------------------------------

def main():
    cfg = load_config()
    df = load_data(cfg)
    scols = framing_cols(cfg)

    # Speech-level descriptives
    print("\n" + "=" * 60)
    print("FRAMING BY PARTY FAMILY (speech level)")
    print("=" * 60)
    print(table_by_group(df, "party_family", scols).to_string())

    print("\n" + "=" * 60)
    print("FRAMING BY EAST-WEST (speech level)")
    print("=" * 60)
    print(table_by_group(df, "east_west", scols).to_string())

    print("\n" + "=" * 60)
    print("FRAMING BY NORTH-SOUTH (speech level)")
    print("=" * 60)
    print(table_by_group(df, "north_south", scols).to_string())

    nat_col = next((c for c in ["nationality", "country", "member_state"]
                    if c in df.columns), None)
    if nat_col:
        print("\n" + "=" * 60)
        print("DOMINANT FRAMING SHARE BY COUNTRY")
        print("=" * 60)
        print(dominant_framing_share(df, nat_col).to_string())

    # MEP-level aggregation
    mep_df = aggregate_to_mep(df, scols)

    print("\n" + "=" * 60)
    print("DOMINANT FRAMING SHARE BY PARTY FAMILY (MEP level)")
    print("=" * 60)
    print(dominant_framing_share(mep_df, "party_family").to_string())

    print("\n" + "=" * 60)
    print("DOMINANT FRAMING SHARE BY EAST-WEST (MEP level)")
    print("=" * 60)
    print(dominant_framing_share(mep_df, "east_west").to_string())

    # Regressions
    run_regressions(mep_df, scols)

    # Save MEP-level dataset
    mep_df.to_csv(cfg["data"]["analysis_path"], index=False)
    print(f"\nMEP-level analysis saved -> {cfg['data']['analysis_path']}")


if __name__ == "__main__":
    main()
