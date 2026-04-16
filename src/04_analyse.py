"""
Analyse framing scores by party family, nationality, East-West, North-South cleavages.
Input:  data/processed/ep_ai_classified.csv  (speech-level, from 02_classify.py)
Output: data/processed/ep_ai_analysis.csv    (MEP-level aggregated scores)

Structure
---------
1. Descriptive tables (speech level)
   - Framing share % by party family, East-West, North-South
   - Normalised so the four framing columns sum to 100% per row, making
     cross-group comparisons easy to read

2. MEP-level aggregation
   - Average framing scores per MEP across all their speeches
   - Prevents high-volume speakers (rapporteurs, committee chairs) from
     dominating the analysis — one MEP = one observation in regressions

3. OLS regressions (MEP level)
   - Dependent variable: score for each framing (four separate models)
   - Regressors: party family dummies + East/West dummy + North/South dummy
   - Reference categories: Social Democrat | West | North
   - Standard errors clustered by country (MEPs from the same country share
     institutional context and national media framing)
   - p < 0.15 reported during exploration — tighten to 0.05 for publication

Note on R²
   Low R² (0.03–0.06) is expected and substantively interpretable: party
   family and geography explain little of the variance in individual speech
   framing. MEPs do not strongly toe the party line on AI governance language.
   This is itself a finding, not a model failure.
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["results_path"])
    print(f"Loaded {len(df):,} classified speeches.")
    return df


def framing_cols(cfg: dict) -> list:
    """Return score column names in the order defined in config.yaml."""
    return [f"score_{v['label']}" for v in cfg["framings"].values()]


# -------------------------------------------------------------------
# 1. Descriptive tables
# -------------------------------------------------------------------

def table_by_group(df: pd.DataFrame, group_col: str, score_cols: list) -> pd.DataFrame:
    """
    Framing share (%) by group.

    Raw NLI scores are not directly interpretable as percentages, but their
    relative magnitude within a speech is meaningful (they sum to 1 when
    multi_label=false). We therefore normalise by row sum so each group row
    sums to 100%, showing how attention is distributed across framings.

    Also reports N_speeches, N_meps, and the dominant framing (highest %).
    """
    if group_col not in df.columns:
        print(f"  Column '{group_col}' not found, skipping.")
        return pd.DataFrame()

    agg = df.groupby(group_col)[score_cols].mean()

    # Normalise to % — row sum = 100 regardless of absolute score magnitude
    row_sums = agg.sum(axis=1)
    agg_pct = (agg.div(row_sums, axis=0) * 100).round(1)
    agg_pct.columns = [c.replace("score_", "") + "%" for c in score_cols]

    agg_pct["N_speeches"] = df.groupby(group_col).size()

    # Build speaker ID from whatever name columns are present
    if "speaker_name" in df.columns:
        agg_pct["N_meps"] = df.groupby(group_col)["speaker_name"].nunique()
    elif "firstname" in df.columns and "lastname" in df.columns:
        df["_speaker"] = df["firstname"] + " " + df["lastname"]
        agg_pct["N_meps"] = df.groupby(group_col)["_speaker"].nunique()
        df.drop(columns=["_speaker"], inplace=True)
    else:
        agg_pct["N_meps"] = np.nan

    agg_pct["dominant_framing"] = agg[score_cols].idxmax(axis=1).str.replace("score_", "")
    return agg_pct.sort_values("N_speeches", ascending=False)


def dominant_framing_share(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Share of speeches (or MEPs) where each framing is dominant, by group.
    Uses the pre-computed dominant_framing column rather than re-deriving it,
    so it is consistent with the speech-level classification output.
    """
    if group_col not in df.columns:
        return pd.DataFrame()
    cross = pd.crosstab(df[group_col], df["dominant_framing"], normalize="index").round(3)
    return cross


# -------------------------------------------------------------------
# 2. MEP-level aggregation
# -------------------------------------------------------------------

def aggregate_to_mep(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    """
    Average framing scores per MEP across all their classified speeches.

    Rationale: some MEPs (rapporteurs, committee chairs) contribute dozens of
    speeches; treating each speech as an independent observation would give
    high-volume speakers disproportionate influence in regressions. Averaging
    to the MEP level makes each MEP one unit of analysis.

    The dominant_framing is re-derived from the averaged scores (not majority
    vote over speeches) to reflect the MEP's overall framing tendency.
    N_speeches is retained to flag MEPs with very thin data (1-2 speeches).
    """
    if "speaker_name" in df.columns:
        df = df.copy()
        df["_speaker_id"] = df["speaker_name"]
    elif "firstname" in df.columns and "lastname" in df.columns:
        df = df.copy()
        df["_speaker_id"] = df["firstname"] + " " + df["lastname"]
    else:
        print("  WARNING: no speaker name columns found, cannot aggregate to MEP level.")
        return df

    meta_cols = [c for c in [
        "_speaker_id", "firstname", "lastname",
        "nationality", "country", "member_state",
        "epg_short", "epg_long", "party_name", "party_family",
        "east_west", "north_south"
    ] if c in df.columns]

    mep_scores = df.groupby("_speaker_id")[score_cols].mean().round(4)
    # Use .first() for metadata — assumes these are constant per MEP
    # (a MEP can switch party mid-term; .first() takes the earliest record)
    mep_meta = df.groupby("_speaker_id")[meta_cols].first()
    mep_df = mep_meta.join(mep_scores).reset_index(drop=True)
    mep_df["dominant_framing"] = mep_df[score_cols].idxmax(axis=1).str.replace("score_", "")
    mep_df["N_speeches"] = df.groupby("_speaker_id").size().values
    return mep_df


# -------------------------------------------------------------------
# 3. OLS with clustered standard errors
# -------------------------------------------------------------------

def run_regressions(mep_df: pd.DataFrame, score_cols: list):
    """
    OLS for each framing score on party family + East-West + North-South dummies.

    Reference categories:
      party_family → Social Democrat  (centrist, largest left-of-centre group)
      east_west    → West             (founding member states)
      north_south  → North            (Nordic + Germanic bloc)

    Clustering by country: MEPs from the same country share national political
    context, media framing, and often coordinate within national delegations.
    Ignoring this would understate standard errors.

    "Other" in east_west/north_south (non-EU MEPs, e.g. UK post-Brexit, observers)
    are excluded from regressions — they are not part of the EU cleavage structure
    and their small N inflates coefficients artificially.

    p < 0.15 is shown during exploration to surface weak signals worth
    investigating. Tighten to p < 0.05 before reporting final results.

    Note on R²: values of 0.02–0.07 are expected and substantively meaningful.
    Party family and geography explain little of individual speech framing variance
    — MEPs do not strongly follow party lines on AI governance language. The
    within-group variance dominates. This is a finding, not a model failure.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  statsmodels not installed. Run: pip install statsmodels")
        return

    # Drop "Other" — non-EU MEPs not part of the East/West or North/South cleavage
    reg_df = mep_df[mep_df["east_west"] != "Other"].copy()
    if "north_south" in reg_df.columns:
        reg_df = reg_df[reg_df["north_south"] != "Other"].copy()
    n_dropped = len(mep_df) - len(reg_df)
    if n_dropped > 0:
        print(f"\n  Dropped {n_dropped} MEPs coded 'Other' in east_west/north_south from regressions.")

    regressors = []
    if "party_family" in reg_df.columns:
        regressors.append("C(party_family, Treatment('Social Democrat'))")
    if "east_west" in reg_df.columns and reg_df["east_west"].nunique() > 1:
        regressors.append("C(east_west, Treatment('West'))")
    if "north_south" in reg_df.columns and reg_df["north_south"].nunique() > 1:
        regressors.append("C(north_south, Treatment('North'))")

    if not regressors:
        print("  No regressor columns found.")
        return

    cluster_col = next(
        (c for c in ["nationality", "country", "member_state"] if c in reg_df.columns), None
    )
    formula_rhs = " + ".join(regressors)

    print("\n" + "=" * 60)
    print("OLS REGRESSION RESULTS (MEP-level, clustered SEs by country)")
    print(f"Reference: Social Democrat | West | North  |  N={len(reg_df)} MEPs")
    print("=" * 60)

    results = {}  # collect fitted models for residual plots
    for sc in score_cols:
        label = sc.replace("score_", "")
        formula = f"{sc} ~ {formula_rhs}"
        try:
            result = smf.ols(formula, data=reg_df).fit(
                cov_type="cluster",
                cov_kwds={"groups": reg_df[cluster_col]} if cluster_col else {}
            )
            print(f"\n--- Dependent variable: {label} ---")
            coef_df = pd.DataFrame({
                "coef": result.params,
                "se":   result.bse,
                "t":    result.tvalues,
                "p":    result.pvalues,
            }).round(4)
            print(coef_df[coef_df["p"] < 0.15].to_string())
            print(f"  R2 = {result.rsquared:.4f}  |  N = {int(result.nobs)}")
            results[label] = result
            print_diagnostics(result, label, reg_df, formula_rhs)
        except Exception as e:
            print(f"  Could not fit model for {label}: {e}")

    if results:
        plot_residuals(results, reg_df)
        plot_coefficients(results)


def print_diagnostics(result, label: str, reg_df: pd.DataFrame, formula_rhs: str):
    """
    Print statistical diagnostics for a fitted OLS model:

    F-test (overall significance)
        Tests H0: all coefficients = 0. A significant F means the model as a
        whole explains more variance than a null intercept-only model.

    Jarque-Bera test (normality of residuals)
        Tests H0: residuals are normally distributed. Rejection (p < 0.05) is
        common with bounded [0,1] NLI scores — note this as a limitation but
        OLS is robust to moderate non-normality at N > 100.

    Breusch-Pagan test (heteroscedasticity)
        Tests H0: residual variance is constant across fitted values. Rejection
        means the model is heteroscedastic — clustered SEs already correct for
        this, so it does not invalidate the inference but is worth flagging.

    VIF (multicollinearity)
        Variance Inflation Factor per predictor. VIF > 5 suggests collinearity
        between predictors that inflates standard errors. Common when party
        family and East-West overlap (e.g. ECR is mostly Eastern MEPs).
    """
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.formula.api as smf

    # Re-fit with OLS (no clustering) to get the standard F-test and residual tests
    # Clustered SEs change SEs but not residuals, so diagnostics on unclustered
    # residuals are valid for checking model assumptions
    plain = smf.ols(result.model.formula, data=reg_df).fit()

    print(f"\n  -- Diagnostics: {label} --")

    # Overall F-test
    print(f"  F-statistic: {plain.fvalue:.3f}  (p={plain.f_pvalue:.4f})"
          + ("  ✓ model significant" if plain.f_pvalue < 0.05 else "  ✗ model not significant"))

    # Jarque-Bera
    jb_stat, jb_p, _, _ = jarque_bera(plain.resid)
    print(f"  Jarque-Bera (normality): stat={jb_stat:.3f}  p={jb_p:.4f}"
          + ("  ✓ residuals normal" if jb_p >= 0.05 else "  ✗ non-normal residuals (common with bounded scores)"))

    # Breusch-Pagan
    bp_stat, bp_p, _, _ = het_breuschpagan(plain.resid, plain.model.exog)
    print(f"  Breusch-Pagan (heteroscedasticity): stat={bp_stat:.3f}  p={bp_p:.4f}"
          + ("  ✓ homoscedastic" if bp_p >= 0.05 else "  ✗ heteroscedastic (clustered SEs already correct for this)"))

    # VIF — skip intercept (index 0)
    exog = plain.model.exog
    vif_vals = [variance_inflation_factor(exog, i) for i in range(1, exog.shape[1])]
    vif_names = plain.model.exog_names[1:]
    high_vif = [(n, v) for n, v in zip(vif_names, vif_vals) if v > 5]
    if high_vif:
        print(f"  VIF > 5 (collinearity concern):")
        for name, v in high_vif:
            short = name.replace("C(party_family, Treatment('Social Democrat'))[T.", "").replace("]", "")
            print(f"    {short}: {v:.2f}")


def plot_residuals(results: dict, reg_df: pd.DataFrame):
    """
    Four diagnostic plots per framing model, saved to data/processed/residual_plots.png:

    1. Residuals vs Fitted — checks linearity and homoscedasticity.
       Points should scatter randomly around y=0 with no fan shape.

    2. Q-Q plot — checks normality of residuals.
       Points should fall on the diagonal; heavy tails suggest non-normality,
       which is common with bounded [0,1] NLI scores.

    3. Scale-Location (√|residuals| vs fitted) — second check for
       heteroscedasticity; a flat LOWESS line is ideal.

    4. Residuals vs leverage — identifies influential observations.
       Points outside Cook's distance lines (dashed) may be pulling the
       regression unduly (e.g. a single MEP with many speeches or extreme scores).
    """
    import scipy.stats as stats

    n_models = len(results)
    fig = plt.figure(figsize=(16, 4 * n_models))
    gs = gridspec.GridSpec(n_models, 4, figure=fig, hspace=0.5, wspace=0.35)

    for row, (label, result) in enumerate(results.items()):
        fitted   = result.fittedvalues
        residuals = result.resid
        influence = result.get_influence()
        leverage  = influence.hat_matrix_diag
        cooks_d   = influence.cooks_distance[0]
        std_resid = residuals / residuals.std()

        # 1. Residuals vs Fitted
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.scatter(fitted, residuals, alpha=0.4, s=15, color="steelblue")
        ax1.axhline(0, color="red", linewidth=0.8, linestyle="--")
        ax1.set_xlabel("Fitted values")
        ax1.set_ylabel("Residuals")
        ax1.set_title(f"{label}\nResiduals vs Fitted")

        # 2. Q-Q plot
        ax2 = fig.add_subplot(gs[row, 1])
        (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
        ax2.scatter(osm, osr, alpha=0.4, s=15, color="steelblue")
        ax2.plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=0.8)
        ax2.set_xlabel("Theoretical quantiles")
        ax2.set_ylabel("Sample quantiles")
        ax2.set_title(f"{label}\nQ-Q Plot")

        # 3. Scale-Location
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.4, s=15, color="steelblue")
        ax3.set_xlabel("Fitted values")
        ax3.set_ylabel("√|Standardised residuals|")
        ax3.set_title(f"{label}\nScale-Location")

        # 4. Residuals vs Leverage (with Cook's distance contours)
        ax4 = fig.add_subplot(gs[row, 3])
        ax4.scatter(leverage, std_resid, alpha=0.4, s=15, color="steelblue")
        ax4.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        # Cook's distance = 0.5 and 1.0 contour lines
        x_range = np.linspace(leverage.min(), leverage.max(), 200)
        p = result.df_model + 1
        for cd_level, ls in [(0.5, "--"), (1.0, "-")]:
            boundary = np.sqrt(cd_level * p * (1 - x_range) / x_range)
            ax4.plot(x_range, boundary, color="red", linewidth=0.7, linestyle=ls,
                     label=f"Cook's D={cd_level}")
            ax4.plot(x_range, -boundary, color="red", linewidth=0.7, linestyle=ls)
        ax4.set_xlabel("Leverage")
        ax4.set_ylabel("Standardised residuals")
        ax4.set_title(f"{label}\nResiduals vs Leverage")
        ax4.legend(fontsize=7)

    out_path = Path("data/processed/residual_plots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Residual plots saved -> {out_path}")


# -------------------------------------------------------------------
# 4. Descriptive plots
# -------------------------------------------------------------------

def plot_coefficients(results: dict):
    """
    Forest plot of OLS coefficients with 95% confidence intervals.
    One panel per framing model, all predictors on the y-axis.
    Coefficients to the right of zero = more of that framing relative to
    the reference category (Social Democrat / West / North).
    Only predictors with p < 0.15 are shown to keep the plot readable.
    """
    COLORS = {
        "risk_based": "#d62728",
        "rights_based": "#1f77b4",
        "innovation_focused": "#2ca02c",
        "sovereignty_focused": "#ff7f0e",
    }

    labels = list(results.keys())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        result = results[label]
        coef_df = pd.DataFrame({
            "coef": result.params,
            "lower": result.conf_int()[0],
            "upper": result.conf_int()[1],
            "p": result.pvalues,
        }).drop(index="Intercept", errors="ignore")
        coef_df = coef_df[coef_df["p"] < 0.15].sort_values("coef")

        # Shorten parameter names for readability
        coef_df.index = (
            coef_df.index
            .str.replace(r"C\(party_family.*?\)\[T\.", "", regex=True)
            .str.replace(r"C\(east_west.*?\)\[T\.", "EW: ", regex=True)
            .str.replace(r"C\(north_south.*?\)\[T\.", "NS: ", regex=True)
            .str.replace("]", "", regex=False)
        )

        color = COLORS.get(label, "steelblue")
        y_pos = range(len(coef_df))
        ax.barh(list(y_pos), coef_df["coef"], xerr=[
            coef_df["coef"] - coef_df["lower"],
            coef_df["upper"] - coef_df["coef"],
        ], color=color, alpha=0.7, ecolor="black", capsize=3, height=0.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(coef_df.index, fontsize=8)
        ax.set_title(label.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Coefficient (ref: Social Democrat / West / North)")

    fig.suptitle("OLS Coefficients with 95% CI (p < 0.15, MEP level)", fontsize=12)
    out_path = Path("data/processed/coefficient_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Coefficient plot saved -> {out_path}")


def plot_framing_heatmap(df: pd.DataFrame, group_col: str, score_cols: list,
                         title: str, out_name: str):
    """
    Heatmap of normalised framing shares (%) by group.
    Rows = groups (party family / country), columns = four framings.
    Color intensity = share of that framing within the group.
    Annotated with the % value in each cell.
    """
    if group_col not in df.columns:
        return

    agg = df.groupby(group_col)[score_cols].mean()
    # Normalise to % per row
    agg_pct = (agg.div(agg.sum(axis=1), axis=0) * 100).round(1)
    agg_pct.columns = [c.replace("score_", "").replace("_", " ") for c in score_cols]
    agg_pct = agg_pct.sort_values(agg_pct.columns[0])  # sort by first framing

    fig, ax = plt.subplots(figsize=(8, max(4, len(agg_pct) * 0.45)))
    im = ax.imshow(agg_pct.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(agg_pct.columns)))
    ax.set_xticklabels(agg_pct.columns, fontsize=9)
    ax.set_yticks(range(len(agg_pct)))
    ax.set_yticklabels(agg_pct.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="% of total framing score")

    # Annotate cells
    for i in range(len(agg_pct)):
        for j in range(len(agg_pct.columns)):
            val = agg_pct.iloc[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7, color="black" if val < 60 else "white")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    out_path = Path(f"data/processed/{out_name}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved -> {out_path}")


# -------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------

def main():
    cfg = load_config()
    df = load_data(cfg)
    scols = framing_cols(cfg)

    # --- Speech-level descriptives ---
    print("\n" + "=" * 60)
    print("FRAMING BY PARTY FAMILY (speech level, % of total score)")
    print("=" * 60)
    print(table_by_group(df, "party_family", scols).to_string())

    print("\n" + "=" * 60)
    print("FRAMING BY EAST-WEST (speech level, % of total score)")
    print("=" * 60)
    print(table_by_group(df, "east_west", scols).to_string())

    print("\n" + "=" * 60)
    print("FRAMING BY NORTH-SOUTH (speech level, % of total score)")
    print("=" * 60)
    print(table_by_group(df, "north_south", scols).to_string())

    nat_col = next(
        (c for c in ["nationality", "country", "member_state"] if c in df.columns), None
    )
    if nat_col:
        print("\n" + "=" * 60)
        print("DOMINANT FRAMING SHARE BY COUNTRY (speech level)")
        print("=" * 60)
        print(dominant_framing_share(df, nat_col).to_string())

    # --- MEP-level aggregation ---
    mep_df = aggregate_to_mep(df, scols)

    print("\n" + "=" * 60)
    print("DOMINANT FRAMING SHARE BY PARTY FAMILY (MEP level)")
    print("=" * 60)
    print(dominant_framing_share(mep_df, "party_family").to_string())

    print("\n" + "=" * 60)
    print("DOMINANT FRAMING SHARE BY EAST-WEST (MEP level)")
    print("=" * 60)
    print(dominant_framing_share(mep_df, "east_west").to_string())

    # --- Descriptive heatmaps ---
    plot_framing_heatmap(df, "party_family", scols,
                         "Framing Share by Party Family (speech level)",
                         "heatmap_party_family.png")
    if nat_col:
        plot_framing_heatmap(df, nat_col, scols,
                             "Framing Share by Country (speech level)",
                             "heatmap_country.png")

    # --- Regressions + plots ---
    run_regressions(mep_df, scols)

    # --- Save MEP-level dataset for further analysis ---
    mep_df.to_csv(cfg["data"]["analysis_path"], index=False)
    print(f"\nMEP-level analysis saved -> {cfg['data']['analysis_path']}")


if __name__ == "__main__":
    main()
