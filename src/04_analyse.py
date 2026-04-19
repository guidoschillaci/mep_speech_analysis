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
        "east_west", "north_south", "accession"
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

def _fit_models(reg_df: pd.DataFrame, score_cols: list, formula_rhs: str,
                cluster_col: str, model_label: str) -> dict:
    """Fit OLS for each framing score and return a dict of fitted results."""
    import statsmodels.formula.api as smf

    print("\n" + "=" * 60)
    print(f"OLS — {model_label}  (MEP-level, clustered SEs by country)")
    print(f"Formula RHS: {formula_rhs}")
    print(f"N = {len(reg_df)} MEPs")
    print("=" * 60)

    results = {}
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
    return results


def run_regressions(mep_df: pd.DataFrame, score_cols: list):
    """
    Run two separate OLS specifications to avoid collinearity between
    east_west and accession (both partition countries on the same 2004
    enlargement boundary → VIF ~7 when included together).

    Model A — party_family + east_west
        Tests whether post-enlargement Eastern MEPs differ from Western ones,
        controlling for party family.

    Model B — party_family + north_south + accession
        Tests North/Middle/South geographic gradient AND accession wave
        independently (they are orthogonal — e.g. Estonia is North + post-2004).

    Reference categories:
      party_family → Social Democrat
      east_west    → West
      north_south  → Middle
      accession    → pre-2004

    Clustering by country: MEPs from the same country share national political
    context, media framing, and often coordinate in national delegations.

    p < 0.15 shown during exploration — tighten to 0.05 for publication.
    """
    try:
        import statsmodels.formula.api as smf  # noqa: F401
    except ImportError:
        print("  statsmodels not installed. Run: pip install statsmodels")
        return

    # Drop "Other" — non-EU MEPs not part of any cleavage dimension
    reg_df = mep_df.copy()
    for col in ["east_west", "north_south", "accession"]:
        if col in reg_df.columns:
            reg_df = reg_df[reg_df[col] != "Other"]
    n_dropped = len(mep_df) - len(reg_df)
    if n_dropped > 0:
        print(f"\n  Dropped {n_dropped} MEPs coded 'Other' from regressions.")

    cluster_col = next(
        (c for c in ["nationality", "country", "member_state"] if c in reg_df.columns), None
    )
    pf = "C(party_family, Treatment('Social Democrat'))"

    # --- Model A: party_family + east_west ---
    rhs_a = [pf]
    if "east_west" in reg_df.columns and reg_df["east_west"].nunique() > 1:
        rhs_a.append("C(east_west, Treatment('West'))")
    results_a = _fit_models(reg_df, score_cols, " + ".join(rhs_a), cluster_col,
                            "Model A: party_family + east_west")

    # --- Model B: party_family + north_south + accession ---
    rhs_b = [pf]
    if "north_south" in reg_df.columns and reg_df["north_south"].nunique() > 1:
        rhs_b.append("C(north_south, Treatment('Middle'))")
    if "accession" in reg_df.columns and reg_df["accession"].nunique() > 1:
        rhs_b.append("C(accession, Treatment('pre-2004'))")
    results_b = _fit_models(reg_df, score_cols, " + ".join(rhs_b), cluster_col,
                            "Model B: party_family + north_south + accession")

    if results_a or results_b:
        plot_residuals(results_a, reg_df, suffix="model_a")
        plot_residuals(results_b, reg_df, suffix="model_b")
        plot_coefficients(results_a, results_b)


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


def plot_residuals(results: dict, reg_df: pd.DataFrame, suffix: str = ""):
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

    fname = f"residual_plots{'_' + suffix if suffix else ''}.png"
    out_path = Path("data/processed") / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Residual plots saved -> {out_path}")


# -------------------------------------------------------------------
# 4. Descriptive plots
# -------------------------------------------------------------------

def plot_coefficients(results_a: dict, results_b: dict):
    """
    Forest plot of OLS coefficients with 95% CI for both model specifications.
    Two rows: Model A (party + east_west) on top, Model B (party + north_south
    + accession) on bottom. One column per framing. Only p < 0.15 shown.
    """
    COLORS = {
        "risk_based": "#d62728",
        "rights_based": "#1f77b4",
        "innovation_focused": "#2ca02c",
        "sovereignty_focused": "#ff7f0e",
    }

    def _shorten(idx):
        return (
            idx.str.replace(r"C\(party_family.*?\)\[T\.", "", regex=True)
               .str.replace(r"C\(east_west.*?\)\[T\.", "EW: ", regex=True)
               .str.replace(r"C\(north_south.*?\)\[T\.", "NS: ", regex=True)
               .str.replace(r"C\(accession.*?\)\[T\.", "ACC: ", regex=True)
               .str.replace("]", "", regex=False)
        )

    def _coef_df(result):
        df = pd.DataFrame({
            "coef":  result.params,
            "lower": result.conf_int()[0],
            "upper": result.conf_int()[1],
            "p":     result.pvalues,
        }).drop(index="Intercept", errors="ignore")
        df = df[df["p"] < 0.15].sort_values("coef")
        df.index = _shorten(df.index)
        return df

    labels = list(results_a.keys())
    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, label in enumerate(labels):
        color = COLORS.get(label, "steelblue")
        for row, (results, model_name) in enumerate([
            (results_a, "Model A: party + east/west"),
            (results_b, "Model B: party + N/M/S + accession"),
        ]):
            ax = axes[row, col]
            if label not in results:
                ax.set_visible(False)
                continue
            cdf = _coef_df(results[label])
            if cdf.empty:
                ax.text(0.5, 0.5, "no significant\npredictors (p<0.15)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_title(f"{label.replace('_', ' ').title()}\n{model_name}",
                             fontsize=9, fontweight="bold")
                continue
            y_pos = range(len(cdf))
            ax.barh(list(y_pos), cdf["coef"], xerr=[
                cdf["coef"] - cdf["lower"],
                cdf["upper"] - cdf["coef"],
            ], color=color, alpha=0.7, ecolor="black", capsize=3, height=0.5)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(cdf.index, fontsize=8)
            ax.set_title(f"{label.replace('_', ' ').title()}\n{model_name}",
                         fontsize=9, fontweight="bold")
            ax.set_xlabel("Coefficient", fontsize=8)

    fig.suptitle("OLS Coefficients with 95% CI  (p < 0.15, MEP level)", fontsize=12)
    fig.tight_layout()
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
# 5. Additional descriptive plots
# -------------------------------------------------------------------

def plot_framing_over_time(df: pd.DataFrame, score_cols: list):
    """
    Line chart of mean framing share (%) per year, 2019–2024.

    Two panels:
      Left  — all four framings as lines, showing the full trend.
      Right — stacked area chart for a compositional view (shares sum to 100%).

    Key events to look for:
      2020: first AI White Paper (Feb)
      2021: AI Act proposal (Apr)
      2022: ChatGPT launch (Nov) → expect innovation/sovereignty shift
      2023: AI Act trilogue
      2024: AI Act adopted (Mar)
    """
    if "year" not in df.columns:
        print("  'year' column not found, skipping time trend plot.")
        return

    agg = df.groupby("year")[score_cols].mean()
    # Normalise to % per year so the four lines sum to 100
    agg_pct = (agg.div(agg.sum(axis=1), axis=0) * 100).round(2)
    agg_pct.columns = [c.replace("score_", "").replace("_", " ") for c in score_cols]

    FRAMING_COLORS = {
        "risk based":          "#d62728",
        "rights based":        "#1f77b4",
        "innovation focused":  "#2ca02c",
        "sovereignty focused": "#ff7f0e",
    }
    # Add speech count per year for annotation
    n_per_year = df.groupby("year").size()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: line chart ---
    for col in agg_pct.columns:
        color = FRAMING_COLORS.get(col, "grey")
        ax1.plot(agg_pct.index, agg_pct[col], marker="o", linewidth=2,
                 markersize=5, label=col, color=color)

    # Annotate key policy events
    events = {
        2021: "AI Act\nproposal",
        2022: "ChatGPT\nlaunched",
        2024: "AI Act\nadopted",
    }
    for year, label in events.items():
        if year in agg_pct.index:
            ax1.axvline(year, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
            ax1.text(year + 0.05, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 45,
                     label, fontsize=6.5, color="grey", va="top")

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Framing share (%)")
    ax1.set_title("Framing trends over time", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_xticks(agg_pct.index)
    # Annotate N speeches per year below x-axis
    for year, n in n_per_year.items():
        ax1.annotate(f"n={n}", xy=(year, ax1.get_ylim()[0]),
                     xytext=(0, -18), textcoords="offset points",
                     ha="center", fontsize=6.5, color="grey")

    # --- Right: stacked area ---
    ax2.stackplot(agg_pct.index, agg_pct.T.values,
                  labels=agg_pct.columns,
                  colors=[FRAMING_COLORS.get(c, "grey") for c in agg_pct.columns],
                  alpha=0.8)
    for year, label in events.items():
        if year in agg_pct.index:
            ax2.axvline(year, color="white", linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Framing share (%)")
    ax2.set_title("Framing composition over time", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_xticks(agg_pct.index)
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    out_path = Path("data/processed/framing_over_time.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Time trend plot saved -> {out_path}")


def plot_scatter_innovation_sovereignty(mep_df: pd.DataFrame, score_cols: list):
    """
    Scatter of MEP-level innovation_focused vs sovereignty_focused scores,
    coloured by party family. Shows within- and between-party variance that
    regressions summarise into single coefficients.
    """
    if "score_innovation_focused" not in mep_df.columns or \
       "score_sovereignty_focused" not in mep_df.columns:
        return

    PARTY_COLORS = {
        "Christian Democrat / Conservative": "#1f77b4",
        "Social Democrat":                   "#d62728",
        "Liberal":                           "#ff7f0e",
        "Green / Regionalist":               "#2ca02c",
        "Conservative / Eurosceptic":        "#9467bd",
        "Radical Right":                     "#8c564b",
        "Radical Left":                      "#e377c2",
        "Non-attached":                      "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    parties = mep_df["party_family"].dropna().unique() if "party_family" in mep_df.columns else []
    for party in sorted(parties):
        sub = mep_df[mep_df["party_family"] == party]
        color = PARTY_COLORS.get(party, "grey")
        ax.scatter(sub["score_innovation_focused"], sub["score_sovereignty_focused"],
                   label=party, color=color, alpha=0.6, s=30)

    ax.set_xlabel("Innovation-focused score")
    ax.set_ylabel("Sovereignty-focused score")
    ax.set_title("MEP framing: innovation vs sovereignty (by party family)", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    ax.axline((0.25, 0.25), slope=1, color="grey", linewidth=0.7,
              linestyle="--", label="equal scores")
    fig.tight_layout()
    out_path = Path("data/processed/scatter_innovation_sovereignty.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plot saved -> {out_path}")


def plot_top_meps(mep_df: pd.DataFrame, score_cols: list, top_n: int = 10):
    """
    Horizontal bar charts of the top N MEPs by averaged framing score,
    one panel per framing. Useful for qualitative validation — check whether
    the top-ranked MEPs are those you'd expect (rapporteurs, committee chairs).
    """
    name_col = None
    if "firstname" in mep_df.columns and "lastname" in mep_df.columns:
        mep_df = mep_df.copy()
        mep_df["_name"] = mep_df["firstname"] + " " + mep_df["lastname"]
        name_col = "_name"
    elif "speaker_name" in mep_df.columns:
        name_col = "speaker_name"
    if name_col is None:
        return

    PARTY_COLORS = {
        "Christian Democrat / Conservative": "#1f77b4",
        "Social Democrat":                   "#d62728",
        "Liberal":                           "#ff7f0e",
        "Green / Regionalist":               "#2ca02c",
        "Conservative / Eurosceptic":        "#9467bd",
        "Radical Right":                     "#8c564b",
        "Radical Left":                      "#e377c2",
        "Non-attached":                      "#7f7f7f",
    }

    n = len(score_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, sc in zip(axes, score_cols):
        label = sc.replace("score_", "").replace("_", " ").title()
        top = mep_df.nlargest(top_n, sc)[[name_col, sc, "party_family"]].copy() \
                    if "party_family" in mep_df.columns \
                    else mep_df.nlargest(top_n, sc)[[name_col, sc]].copy()
        top = top.sort_values(sc)

        bar_colors = [PARTY_COLORS.get(pf, "grey")
                      for pf in top.get("party_family", ["grey"] * len(top))]

        ax.barh(top[name_col], top[sc], color=bar_colors, alpha=0.8)
        ax.set_xlabel("Avg framing score")
        ax.set_title(f"Top {top_n} MEPs\n{label}", fontweight="bold", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)

    # Shared legend for party colours, placed below all panels
    from matplotlib.patches import Patch
    present_parties = mep_df["party_family"].dropna().unique() \
        if "party_family" in mep_df.columns else []
    legend_handles = [
        Patch(color=PARTY_COLORS.get(p, "grey"), label=p, alpha=0.8)
        for p in sorted(present_parties)
        if p in PARTY_COLORS
    ]
    fig.legend(handles=legend_handles, title="Party family", fontsize=7,
               title_fontsize=8, loc="lower center",
               ncol=min(len(legend_handles), 4),
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(f"Top {top_n} MEPs per framing (MEP-level avg score)", fontsize=11)
    fig.tight_layout()
    out_path = Path("data/processed/top_meps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Top MEPs plot saved -> {out_path}")


def plot_country_bars(df: pd.DataFrame, score_cols: list):
    """
    Horizontal stacked bar chart of framing share (%) by country,
    sorted by sovereignty_focused score descending. Easier to read
    than the heatmap for country-level comparisons.
    """
    nat_col = next(
        (c for c in ["nationality", "country", "member_state"] if c in df.columns), None
    )
    if nat_col is None:
        return

    agg = df.groupby(nat_col)[score_cols].mean()
    agg_pct = (agg.div(agg.sum(axis=1), axis=0) * 100).round(1)
    agg_pct.columns = [c.replace("score_", "").replace("_", " ") for c in score_cols]

    # Sort by sovereignty_focused descending
    sort_col = next((c for c in agg_pct.columns if "sovereignty" in c), agg_pct.columns[-1])
    agg_pct = agg_pct.sort_values(sort_col, ascending=True)

    FRAMING_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(8, max(5, len(agg_pct) * 0.35)))
    left = np.zeros(len(agg_pct))
    for col, color in zip(agg_pct.columns, FRAMING_COLORS):
        ax.barh(agg_pct.index, agg_pct[col], left=left, label=col,
                color=color, alpha=0.85)
        left += agg_pct[col].values

    ax.set_xlabel("Framing share (%)")
    ax.set_title("Framing distribution by country\n(sorted by sovereignty score)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    out_path = Path("data/processed/country_bars.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Country bar chart saved -> {out_path}")


# -------------------------------------------------------------------
# 6. Main
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
    print("FRAMING BY NORTH-MIDDLE-SOUTH (speech level, % of total score)")
    print("=" * 60)
    print(table_by_group(df, "north_south", scols).to_string())

    print("\n" + "=" * 60)
    print("FRAMING BY ACCESSION WAVE (speech level, % of total score)")
    print("=" * 60)
    print(table_by_group(df, "accession", scols).to_string())

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

    # --- Additional descriptive plots ---
    plot_framing_over_time(df, scols)
    plot_scatter_innovation_sovereignty(mep_df, scols)
    plot_top_meps(mep_df, scols, top_n=10)
    plot_country_bars(df, scols)

    # --- Regressions + plots ---
    run_regressions(mep_df, scols)

    # --- Save MEP-level dataset for further analysis ---
    mep_df.to_csv(cfg["data"]["analysis_path"], index=False)
    print(f"\nMEP-level analysis saved -> {cfg['data']['analysis_path']}")


if __name__ == "__main__":
    main()
