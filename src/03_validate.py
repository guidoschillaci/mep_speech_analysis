"""
Validate both pipeline stages against manually annotated gold standards.

Stage 1 — Relevance filter:
  Expects: validation/relevance_annotation.csv
  Columns: text, is_relevant  (1 = AI-relevant, 0 = not relevant)
  Evaluates: precision/recall/F1 of the NLI relevance filter at the configured threshold,
             plus a score distribution plot to help calibrate the threshold.

Stage 2 — Framing classifier:
  Expects: validation/annotation_template.csv
  Columns: text, true_label  (one of: risk_based, rights_based, innovation_focused, sovereignty_focused)
  Evaluates: per-class precision/recall/F1 and confusion matrix.

Run:
  python src/03_validate.py              # both stages
  python src/03_validate.py --stage relevance
  python src/03_validate.py --stage framing
"""

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict):
    model_cfg = cfg["model"]
    device_str = model_cfg["device"]
    device = device_str if (device_str == "mps" and torch.backends.mps.is_available()) else "cpu"
    torch_dtype = torch.float16 if model_cfg.get("fp16") and device != "cpu" else torch.float32

    print(f"  Loading model: {model_cfg['name']} ({device}, {torch_dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"], torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    id2label = {v.lower(): k for k, v in model.config.id2label.items()}
    entailment_idx = id2label.get("entailment", 0)
    return tokenizer, model, device, entailment_idx


def run_nli(texts: list, hypothesis: str, tokenizer, model, device: str,
            entailment_idx: int, max_length: int, batch_size: int) -> list:
    """Run a single NLI hypothesis over a list of texts, return entailment probabilities."""
    scores = []
    for i in range(0, len(texts), batch_size):
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
    return scores


# -----------------------------------------------------------------------
# Stage 1: relevance filter validation
# -----------------------------------------------------------------------

def validate_relevance(cfg: dict, tokenizer, model, device: str, entailment_idx: int):
    val_path = Path("validation/relevance_annotation.csv")
    if not val_path.exists():
        print("\n[Stage 1] relevance_annotation.csv not found — skipping.")
        print("  Create validation/relevance_annotation.csv with columns: text, is_relevant")
        print("  is_relevant: 1 = AI-relevant speech, 0 = not relevant")
        print("  Aim for ~50 positive and ~50 negative examples.")
        return

    df = pd.read_csv(val_path).dropna(subset=["text", "is_relevant"])
    df["is_relevant"] = df["is_relevant"].astype(int)
    print(f"\n[Stage 1] Relevance filter validation — {len(df)} examples "
          f"({df['is_relevant'].sum()} relevant, {(~df['is_relevant'].astype(bool)).sum()} not relevant)")

    rel_cfg = cfg["relevance_filter"]
    scores = run_nli(
        df["text"].tolist(),
        rel_cfg["hypothesis"],
        tokenizer, model, device, entailment_idx,
        max_length=rel_cfg["max_tokens"],
        batch_size=rel_cfg["batch_size"],
    )
    df["relevance_score"] = scores

    threshold = rel_cfg["threshold"]
    df["predicted"] = (df["relevance_score"] >= threshold).astype(int)

    print(f"\n  Threshold: {threshold}")
    print(classification_report(df["is_relevant"], df["predicted"],
                                target_names=["not_relevant", "relevant"]))

    if df["is_relevant"].nunique() == 2:
        auc = roc_auc_score(df["is_relevant"], df["relevance_score"])
        print(f"  ROC-AUC: {auc:.3f}  (threshold-independent; >0.80 is good)")

    # Score distributions by true label — helps pick the right threshold
    print("\n  Score distribution by true label:")
    for label, group in df.groupby("is_relevant"):
        name = "relevant" if label == 1 else "not_relevant"
        s = group["relevance_score"]
        print(f"    {name:12s}  n={len(s):3d}  "
              f"min={s.min():.3f}  mean={s.mean():.3f}  "
              f"median={s.median():.3f}  max={s.max():.3f}")

    # Suggest threshold from Youden's J (maximises sensitivity + specificity)
    thresholds = np.linspace(0, 1, 200)
    best_j, best_t = 0, threshold
    for t in thresholds:
        pred = (df["relevance_score"] >= t).astype(int)
        tp = ((pred == 1) & (df["is_relevant"] == 1)).sum()
        tn = ((pred == 0) & (df["is_relevant"] == 0)).sum()
        fp = ((pred == 1) & (df["is_relevant"] == 0)).sum()
        fn = ((pred == 0) & (df["is_relevant"] == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    print(f"\n  Suggested threshold (Youden's J): {best_t:.3f}  "
          f"(current: {threshold})")
    if abs(best_t - threshold) > 0.05:
        print(f"  → Consider updating relevance_filter.threshold to {best_t:.2f} in config.yaml")

    out_path = Path("validation/relevance_validation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"  Detailed results -> {out_path}")


# -----------------------------------------------------------------------
# Stage 2: framing classifier validation
# -----------------------------------------------------------------------

def validate_framing(cfg: dict, tokenizer, model, device: str, entailment_idx: int):
    val_path = Path("validation/annotation_template.csv")
    if not val_path.exists():
        print("\n[Stage 2] annotation_template.csv not found — skipping.")
        return

    df = pd.read_csv(val_path).dropna(subset=["text", "true_label"])
    if len(df) < 4:
        print(f"\n[Stage 2] Only {len(df)} annotated examples — need at least ~30 per class for meaningful validation.")
    else:
        print(f"\n[Stage 2] Framing classifier validation — {len(df)} examples")

    print(f"  Label distribution:\n{df['true_label'].value_counts().to_string()}\n")

    framings = cfg["framings"]
    hypotheses = {v["label"]: v["hypothesis"] for v in framings.values()}
    max_length = cfg["model"].get("max_tokens", 128)
    batch_size = cfg["model"]["batch_size"]

    # Collect entailment score for each hypothesis
    all_scores = {lbl: [] for lbl in hypotheses}
    for lbl, hyp in hypotheses.items():
        scores = run_nli(
            df["text"].tolist(), hyp,
            tokenizer, model, device, entailment_idx,
            max_length=max_length, batch_size=batch_size,
        )
        all_scores[lbl] = scores

    # Predicted label = hypothesis with highest entailment score
    preds = []
    for i in range(len(df)):
        pred = max(hypotheses.keys(), key=lambda lbl: all_scores[lbl][i])
        preds.append(pred)
        for lbl in hypotheses:
            df.loc[df.index[i], f"score_{lbl}"] = round(all_scores[lbl][i], 4)

    df["predicted_label"] = preds

    labels = list(hypotheses.keys())
    print("--- Classification Report ---")
    print(classification_report(df["true_label"], df["predicted_label"],
                                labels=labels, zero_division=0))

    print("--- Confusion Matrix ---")
    cm = pd.DataFrame(
        confusion_matrix(df["true_label"], df["predicted_label"], labels=labels),
        index=labels, columns=labels
    )
    print(cm.to_string())

    # Per-class score distributions — useful to spot systematic mis-calibration
    print("\n--- Mean predicted score by true label ---")
    score_cols = [f"score_{lbl}" for lbl in labels]
    print(df.groupby("true_label")[score_cols].mean().round(3).to_string())

    out_path = Path("validation/validation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Detailed results -> {out_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["relevance", "framing", "both"],
                        default="both", help="Which validation stage to run")
    args = parser.parse_args()

    cfg = load_config()
    tokenizer, model, device, entailment_idx = load_model(cfg)

    if args.stage in ("relevance", "both"):
        validate_relevance(cfg, tokenizer, model, device, entailment_idx)

    if args.stage in ("framing", "both"):
        validate_framing(cfg, tokenizer, model, device, entailment_idx)


if __name__ == "__main__":
    main()
