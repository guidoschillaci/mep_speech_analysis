"""
Validate classifier against manually annotated gold standard.
Expects: validation/annotation_template.csv with columns:
  text, true_label  (true_label is one of the four short framing names)
Prints: per-class precision, recall, F1 and confusion matrix.
Output: validation/validation_results.csv
"""

import yaml
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    val_path = Path("validation/annotation_template.csv")

    if not val_path.exists():
        print("No annotation file found. Create validation/annotation_template.csv first.")
        print("Required columns: text, true_label")
        print("Valid true_label values: risk_based, rights_based, innovation_focused, sovereignty_focused")
        return

    df = pd.read_csv(val_path)
    assert "text" in df.columns and "true_label" in df.columns, \
        "annotation_template.csv must have 'text' and 'true_label' columns."
    df = df.dropna(subset=["text", "true_label"])
    print(f"Loaded {len(df)} annotated examples.")
    print(f"Label distribution:\n{df['true_label'].value_counts().to_string()}\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    classifier = pipeline(
        "zero-shot-classification",
        model=cfg["model"]["name"],
        device=device
    )

    framings = cfg["framings"]
    hypotheses = {v["label"]: v["hypothesis"] for v in framings.values()}
    hyp_to_label = {v: k for k, v in hypotheses.items()}

    preds = []
    for text in df["text"].tolist():
        out = classifier(
            text,
            candidate_labels=list(hypotheses.values()),
            multi_label=False
        )
        pred = hyp_to_label[out["labels"][0]]
        preds.append(pred)

    df["predicted_label"] = preds

    print("--- Classification Report ---")
    print(classification_report(df["true_label"], df["predicted_label"]))

    print("--- Confusion Matrix ---")
    labels = list(hypotheses.keys())
    cm = pd.DataFrame(
        confusion_matrix(df["true_label"], df["predicted_label"], labels=labels),
        index=labels, columns=labels
    )
    print(cm.to_string())

    out_path = Path("validation/validation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nDetailed results -> {out_path}")


if __name__ == "__main__":
    main()
