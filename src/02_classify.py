"""
Classify EP speeches by AI governance framing using DeBERTa-v3 NLI.
Model: MoritzLaurer/deberta-v3-large-zeroshot-v2.0
Input:  data/processed/ep_ai_speeches.csv
Output: data/processed/ep_ai_classified.csv
"""

import yaml
import torch
import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(cfg_device: str) -> str:
    if cfg_device == "mps" and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS backend.")
        return "mps"
    print("MPS not available, falling back to CPU.")
    return "cpu"


def build_classifier(cfg: dict):
    device = get_device(cfg["model"]["device"])
    classifier = pipeline(
        "zero-shot-classification",
        model=cfg["model"]["name"],
        device=device,
    )
    return classifier


def classify_speeches(df: pd.DataFrame, classifier, cfg: dict) -> pd.DataFrame:
    framings = cfg["framings"]
    labels = [v["label"] for v in framings.values()]
    hypotheses = {v["label"]: v["hypothesis"] for v in framings.values()}
    multi_label = cfg["model"]["multi_label"]
    batch_size = cfg["model"]["batch_size"]

    results = []
    texts = df["text"].tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
        batch = texts[i: i + batch_size]
        outputs = classifier(
            batch,
            candidate_labels=list(hypotheses.values()),
            multi_label=multi_label,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]

        for out in outputs:
            hyp_to_label = {v: k for k, v in hypotheses.items()}
            row = {}
            for label_text, score in zip(out["labels"], out["scores"]):
                short = hyp_to_label[label_text]
                row[f"score_{short}"] = round(score, 4)
            row["dominant_framing"] = max(
                [k for k in row if k.startswith("score_")],
                key=lambda k: row[k]
            ).replace("score_", "")
            results.append(row)

    scores_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["output_path"])
    print(f"Loaded {len(df):,} speeches for classification.")

    classifier = build_classifier(cfg)
    df_out = classify_speeches(df, classifier, cfg)

    df_out.to_csv(cfg["data"]["results_path"], index=False)
    print(f"\nSaved classified corpus -> {cfg['data']['results_path']}")
    print("\nDominant framing distribution:")
    print(df_out["dominant_framing"].value_counts().to_string())


if __name__ == "__main__":
    main()
