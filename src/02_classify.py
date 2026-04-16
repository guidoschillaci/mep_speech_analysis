"""
Classify EP speeches by AI governance framing using zero-shot NLI.
Input:  data/processed/ep_ai_speeches.csv
Output: data/processed/ep_ai_classified.csv

Approach
--------
For each speech we run four NLI forward passes — one per framing hypothesis.
Each pass pairs the speech text with a hypothesis string (e.g. "This speech
focuses on risks and safety...") and returns logits over (entailment, neutral,
contradiction). We extract the entailment logit for each framing, then apply
a single softmax across the four values so the scores are mutually exclusive
and sum to 1. The framing with the highest score is the dominant framing.

Why not use the HuggingFace zero-shot pipeline?
  The pipeline ignores max_length at call time and pads to the model's full
  context window (up to 8192 tokens for some models), making it very slow.
  Calling the tokenizer and model directly lets us enforce truncation and
  batch padding to the actual sequence length.

multi_label: false (config.yaml)
  Scores sum to 1 per speech — each speech has one dominant framing.
  Set to true only if you want independent per-framing probabilities
  (useful for exploratory analysis, but breaks the compositional interpretation).
"""

import yaml
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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


def load_model(cfg: dict):
    """
    Load the NLI model and tokenizer. Returns the model, tokenizer, device string,
    and the index of the entailment class in the model's output (varies by model).
    fp16 is used on MPS/CUDA to halve memory and speed up inference.
    """
    model_cfg = cfg["model"]
    device = get_device(model_cfg["device"])
    torch_dtype = torch.float16 if model_cfg.get("fp16") and device != "cpu" else torch.float32

    print(f"  Loading model: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"], torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    # id2label differs between models — look up entailment index at runtime
    # rather than hardcoding, so the script works with any NLI model
    id2label = {v.lower(): k for k, v in model.config.id2label.items()}
    entailment_idx = id2label.get("entailment", 2)
    print(f"  Entailment index: {entailment_idx}  |  dtype: {torch_dtype}  |  device: {device}")

    return tokenizer, model, device, entailment_idx


def classify_speeches(df: pd.DataFrame, tokenizer, model, device: str,
                      entailment_idx: int, cfg: dict) -> pd.DataFrame:
    """
    For each speech, run one NLI forward pass per framing hypothesis and collect
    the entailment logit. Then either:
      - multi_label=false: apply softmax across the four entailment logits
        → scores sum to 1, each speech has one dominant framing
      - multi_label=true:  apply softmax within each pass (entail vs neutral vs contradict)
        → independent probability per framing, scores do NOT sum to 1

    Batching strategy: we loop over hypotheses in the outer loop and speeches
    in the inner loop. This keeps the batch size constant and avoids the memory
    spike of passing all hypothesis×speech combinations at once.

    truncation="only_first": the speech text is truncated if it exceeds max_length;
    the hypothesis is never truncated (it's short and semantically critical).
    padding=True: pads only to the longest sequence in the current batch,
    not to max_length — this is important for speed.
    """
    framings = cfg["framings"]
    hypotheses = {v["label"]: v["hypothesis"] for v in framings.values()}
    batch_size = cfg["model"]["batch_size"]
    max_length = cfg["model"].get("max_tokens", 256)
    multi_label = cfg["model"]["multi_label"]

    texts = df["text"].tolist()
    labels = list(hypotheses.keys())
    hyp_texts = list(hypotheses.values())

    print(f"  {len(texts):,} speeches × {len(labels)} hypotheses | "
          f"max_length={max_length} tokens | batch={batch_size} | "
          f"multi_label={multi_label}")

    # Accumulate raw entailment logits (multi_label=false) or
    # independent entailment probs (multi_label=true) per label
    speech_scores = {lbl: [] for lbl in labels}

    for lbl, hyp in zip(labels, hyp_texts):
        for i in tqdm(range(0, len(texts), batch_size), desc=f"  [{lbl}]"):
            batch = texts[i: i + batch_size]
            enc = tokenizer(
                batch,
                [hyp] * len(batch),   # broadcast single hypothesis across batch
                padding=True,          # pad to longest in batch, not to max_length
                truncation="only_first",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = model(**enc).logits  # shape: (batch_size, 3)

            if multi_label:
                # Independent entailment probability for this hypothesis
                probs = torch.softmax(logits, dim=-1)[:, entailment_idx]
            else:
                # Raw entailment logit — softmax across labels applied below
                probs = logits[:, entailment_idx]

            speech_scores[lbl].extend(probs.float().cpu().tolist())

    # Build one result dict per speech
    all_scores = []
    for j in range(len(texts)):
        if multi_label:
            row = {f"score_{lbl}": round(speech_scores[lbl][j], 4) for lbl in labels}
        else:
            # Joint softmax across framings — forces scores to sum to 1
            raw = torch.tensor([speech_scores[lbl][j] for lbl in labels])
            probs = torch.softmax(raw, dim=0)
            row = {f"score_{lbl}": round(probs[k].item(), 4) for k, lbl in enumerate(labels)}

        row["dominant_framing"] = max(labels, key=lambda lbl: row[f"score_{lbl}"])
        all_scores.append(row)

    scores_df = pd.DataFrame(all_scores)
    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["output_path"])
    print(f"Loaded {len(df):,} speeches for classification.")

    tokenizer, model, device, entailment_idx = load_model(cfg)
    df_out = classify_speeches(df, tokenizer, model, device, entailment_idx, cfg)

    df_out.to_csv(cfg["data"]["results_path"], index=False)
    print(f"\nSaved classified corpus -> {cfg['data']['results_path']}")
    print("\nDominant framing distribution:")
    print(df_out["dominant_framing"].value_counts().to_string())


if __name__ == "__main__":
    main()
