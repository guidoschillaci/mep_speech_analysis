"""
Filter EUPDCorp to AI-relevant EP speeches (9th term, 2019-2024).
Input:  data/raw/EUPDCorp.rds  (download from https://zenodo.org/records/15056399)
Output: data/processed/ep_ai_speeches.csv

Pipeline:
  1. Load raw RDS corpus (~560k speeches, 1999-2024)
  2. Resolve text: use original if English, else use machine translation (text_en)
  3. Filter to 9th EP term (2019-2024) and minimum speech length
  4. NLI relevance filter: keep speeches with entailment score >= threshold for
     the AI-relevance hypothesis (replaces brittle keyword matching)
  5. Add cleavage variables: East/West, North/South, party family
  6. Write trimmed CSV with only analytically relevant columns
"""

import threading
import torch
import yaml
import pandas as pd
import pyreadr
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tqdm.pandas()  # enables df.progress_apply()


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_eupd(path: str) -> pd.DataFrame:
    # pyreadr.read_r is a blocking C call with no progress hooks, so we run it
    # in a background thread and display an elapsed-time spinner while we wait
    holder: dict = {}

    def _read():
        try:
            holder["result"] = pyreadr.read_r(path)
        except Exception as exc:
            holder["error"] = exc

    thread = threading.Thread(target=_read)
    thread.start()

    with tqdm(desc=f"Loading {path}", bar_format="{desc} [{elapsed}]") as pbar:
        while thread.is_alive():
            thread.join(timeout=0.2)
            pbar.refresh()

    if "error" in holder:
        raise holder["error"]

    df = holder["result"][None]  # RDS files (single R object) are keyed as None
    print(f"  Loaded {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    return df


def resolve_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    EUPDCorp stores the original speech in `speech` and an English machine
    translation in `speech_en`. We use the original when the speech was already
    delivered in English (language == 'en') to avoid re-translation artefacts;
    for all other languages we fall back to the English translation so that
    downstream keyword matching and NLI classification work on a common language.
    """
    if "speech_en" in df.columns and "language" in df.columns:
        df["text_resolved"] = df.progress_apply(
            lambda r: r["speech"] if str(r.get("language", "")).lower() == "en"
                      else r.get("speech_en", r["speech"]),
            axis=1
        )
    else:
        # If translation column is absent, use the raw speech column
        df["text_resolved"] = df["speech"]
    return df


def filter_corpus(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    corpus_cfg = cfg["corpus"]

    # 1. Year range — keep only the 9th EP term (2019-2024)
    #    `errors="coerce"` turns unparseable dates into NaT (dropped implicitly)
    #    Note: the corpus already has a `year` column; we re-derive it from `date`
    #    to be safe against encoding differences across corpus versions.
    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
        print(f"  Year distribution in corpus:")
        print(df["year"].value_counts().sort_index().to_string())
        df = df[df["year"].isin(corpus_cfg["years"])].copy()
        print(f"  After year filter: {len(df):,} speeches")

    # 2. Minimum length — drop very short speeches (procedural interventions,
    #    one-line votes, etc.) that carry no substantive content
    df = df[df["text_resolved"].str.len() >= corpus_cfg["min_speech_length"]].copy()
    print(f"  After length filter: {len(df):,} speeches")

    return df


def nli_relevance_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Use the same NLI model as the framing classifier to score each speech
    for AI-relevance. Keeps only speeches where the entailment probability
    for the relevance hypothesis exceeds the configured threshold.

    This replaces keyword matching: the model understands paraphrases and
    context, so it catches speeches that discuss AI without using exact terms
    (e.g. 'algorithmic systems', 'automated tools', translated variants).
    """
    rel_cfg = cfg["relevance_filter"]
    model_cfg = cfg["model"]
    hypothesis = rel_cfg["hypothesis"]
    threshold = rel_cfg["threshold"]
    batch_size = rel_cfg["batch_size"]
    max_length = rel_cfg["max_tokens"]

    device_str = model_cfg["device"]
    if device_str == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch_dtype = torch.float16 if model_cfg.get("fp16") and device != "cpu" else torch.float32
    print(f"\n  Loading NLI model for relevance filter ({device}, {torch_dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"], torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    id2label = {v.lower(): k for k, v in model.config.id2label.items()}
    entailment_idx = id2label.get("entailment", 0)

    texts = df["text_resolved"].tolist()
    scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Relevance filter"):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            [hypothesis] * len(batch),
            padding=True,
            truncation="only_first",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, entailment_idx]
        scores.extend(probs.float().cpu().tolist())

    df = df.copy()
    df["relevance_score"] = scores
    before = len(df)
    df = df[df["relevance_score"] >= threshold].copy()
    print(f"  After NLI relevance filter (threshold={threshold}): "
          f"{len(df):,} / {before:,} speeches retained")
    print(f"  Score distribution (retained): "
          f"min={df['relevance_score'].min():.3f}  "
          f"mean={df['relevance_score'].mean():.3f}  "
          f"max={df['relevance_score'].max():.3f}")

    # Histogram of relevance scores across the full pre-filter sample —
    # helps calibrate the threshold: look for a natural gap or shoulder in
    # the distribution that separates AI-relevant from non-relevant speeches
    df_all = df.copy()
    df_all["_retained"] = df_all["relevance_score"] >= threshold
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df_all[~df_all["_retained"]]["relevance_score"],
                bins=50, alpha=0.6, color="grey", label="excluded")
        ax.hist(df_all[df_all["_retained"]]["relevance_score"],
                bins=50, alpha=0.7, color="steelblue", label="retained")
        ax.axvline(threshold, color="red", linewidth=1.2, linestyle="--",
                   label=f"threshold={threshold}")
        ax.set_xlabel("Relevance score")
        ax.set_ylabel("Number of speeches")
        ax.set_title("NLI relevance score distribution")
        ax.legend()
        out_path = Path("data/processed/relevance_score_distribution.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Score distribution plot saved -> {out_path}")
    except ImportError:
        pass  # matplotlib optional here

    return df


def add_cleavage_vars(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Append three analytical variables used in the regression models:

    east_west   — East (2004+ accession states) / West / Other
    north_south — North / South / Other  (see config.yaml for country lists)
    party_family — collapsed EP group label (EPP, S&D, … → broad family string)

    Country codes follow ISO 3166-1 alpha-2 as used in EUPDCorp metadata.
    Speeches from countries not in either list are coded "Other" rather than
    dropped, so the sample size is preserved for robustness checks.
    """
    analysis_cfg = cfg["analysis"]
    east = set(analysis_cfg["east_countries"])
    west = set(analysis_cfg["west_countries"])
    north = set(analysis_cfg["north_countries"])
    south = set(analysis_cfg["south_countries"])

    # EUPDCorp uses different column names across versions — try all known variants
    nat_col = next((c for c in ["nationality", "country", "member_state"] if c in df.columns), None)
    if nat_col is None:
        print("  WARNING: no nationality column found, skipping cleavage vars")
        return df

    # Cast to str — nationality is a Categorical (R factor) in the RDS
    nat = df[nat_col].astype(str)
    tqdm.pandas(desc="east_west")
    df["east_west"] = nat.progress_apply(
        lambda x: "East" if x in east else ("West" if x in west else "Other")
    )
    tqdm.pandas(desc="north_south")
    df["north_south"] = nat.progress_apply(
        lambda x: "North" if x in north else ("South" if x in south else "Other")
    )

    # Map EP group abbreviation to broad party family; unmapped groups → "Other"
    pf_map = analysis_cfg["party_family"]
    party_col = next((c for c in ["epg_short", "party", "ep_group", "group"] if c in df.columns), None)
    if party_col:
        actual_groups = sorted(df[party_col].dropna().unique())
        print(f"  Actual {party_col} values: {actual_groups}")
        unmatched = [g for g in actual_groups if g not in pf_map]
        if unmatched:
            print(f"  WARNING: no mapping for {unmatched} — update party_family in config.yaml")
        # Unmatched groups keep their original epg_short value instead of
        # being silently collapsed to "Other", so they remain identifiable
        # Cast to str first — epg_short is a Categorical (R factor) in the RDS,
        # and fillna with another Categorical raises TypeError on category mismatch
        col_str = df[party_col].astype(str)
        df["party_family"] = col_str.map(pf_map).fillna(col_str)

    return df


def main():
    cfg = load_config()
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df = load_eupd(cfg["data"]["raw_path"])
    df = resolve_text(df)
    df = filter_corpus(df, cfg)       # year + length filters only
    df = nli_relevance_filter(df, cfg)
    df = add_cleavage_vars(df, cfg)

    # Keep only columns needed for classification and regression;
    # the list is ordered by analytical priority — missing columns are silently skipped
    keep = [c for c in [
        "firstname", "lastname", "nationality", "country", "member_state",
        "epg_short", "epg_long", "party_name", "party_db_code", "party_family",
        "date", "year", "language",
        "east_west", "north_south",
        "text_resolved", "agenda", "mepid"
    ] if c in df.columns]

    df = df[keep].rename(columns={"text_resolved": "text"}).reset_index(drop=True)
    df.to_csv(cfg["data"]["output_path"], index=False)

    print(f"\nSaved {len(df):,} speeches -> {cfg['data']['output_path']}")

    # Spot-check distributions to catch obvious encoding or mapping issues
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
