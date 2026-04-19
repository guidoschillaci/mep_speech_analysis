"""
Unit tests for src/02_classify.py

Tests cover classify_speeches logic (score normalisation, dominant framing,
multi_label modes) using lightweight mock objects — no model download required.
get_device is also tested with MPS availability mocked out.
"""

import sys
import pytest
import torch
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import importlib
classify = importlib.import_module("02_classify")

classify_speeches = classify.classify_speeches
get_device        = classify.get_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(multi_label: bool = False, batch_size: int = 4, max_tokens: int = 64):
    return {
        "model": {
            "name":        "mock-model",
            "device":      "cpu",
            "batch_size":  batch_size,
            "max_tokens":  max_tokens,
            "fp16":        False,
            "multi_label": multi_label,
        },
        "framings": {
            "risk_based":          {"label": "risk_based",          "hypothesis": "Risk hypothesis."},
            "rights_based":        {"label": "rights_based",        "hypothesis": "Rights hypothesis."},
            "innovation_focused":  {"label": "innovation_focused",  "hypothesis": "Innovation hypothesis."},
            "sovereignty_focused": {"label": "sovereignty_focused", "hypothesis": "Sovereignty hypothesis."},
        },
    }


def _make_speech_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "text": [f"Speech number {i} about artificial intelligence." for i in range(n)],
        "mepid": list(range(n)),
    })


# ---------------------------------------------------------------------------
# Mock model that returns controlled per-label logits
# ---------------------------------------------------------------------------

class _FakeEncoding(dict):
    """dict subclass that supports .to(device) — mirrors HuggingFace BatchEncoding."""
    def to(self, device):
        return self


class _ControlledModel:
    """
    Callable mock model. Returns fixed entailment logits per label in round-robin order.
    `entailment_logits`: one float per framing label (4), cycled per forward pass.
    """
    def __init__(self, entailment_logits: list, entailment_idx: int = 2):
        self.entailment_logits = entailment_logits
        self.entailment_idx = entailment_idx
        self._call_count = 0

    def __call__(self, **kwargs):
        lbl_idx = self._call_count % len(self.entailment_logits)
        batch_size = kwargs["input_ids"].shape[0]
        logits = torch.zeros(batch_size, 3)
        logits[:, self.entailment_idx] = self.entailment_logits[lbl_idx]
        self._call_count += 1
        out = MagicMock()
        out.logits = logits
        return out

    def eval(self):
        return self


class _ControlledTokenizer:
    """Minimal tokenizer returning a dict-like encoding that supports **-unpacking and .to()."""
    def __call__(self, texts, hypotheses, padding, truncation, max_length, return_tensors):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return _FakeEncoding({
            "input_ids":      torch.zeros(batch_size, 16, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
        })


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:

    def test_returns_cpu_when_mps_unavailable(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            assert get_device("mps") == "cpu"

    def test_returns_mps_when_available(self):
        with patch("torch.backends.mps.is_available", return_value=True):
            assert get_device("mps") == "mps"

    def test_cpu_always_returns_cpu(self):
        assert get_device("cpu") == "cpu"


# ---------------------------------------------------------------------------
# classify_speeches — output shape and columns
# ---------------------------------------------------------------------------

class TestClassifySpeechesOutputShape:

    def test_output_has_same_row_count(self):
        df = _make_speech_df(5)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        model = _ControlledModel([1.0, 0.5, 0.2, 0.1])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        assert len(result) == 5

    def test_output_has_score_columns(self):
        df = _make_speech_df(3)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        model = _ControlledModel([1.0, 0.5, 0.2, 0.1])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        for col in ["score_risk_based", "score_rights_based",
                    "score_innovation_focused", "score_sovereignty_focused"]:
            assert col in result.columns

    def test_output_has_dominant_framing_column(self):
        df = _make_speech_df(3)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        model = _ControlledModel([1.0, 0.5, 0.2, 0.1])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        assert "dominant_framing" in result.columns

    def test_original_columns_preserved(self):
        df = _make_speech_df(3)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        model = _ControlledModel([1.0, 0.5, 0.2, 0.1])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        assert "text" in result.columns
        assert "mepid" in result.columns


# ---------------------------------------------------------------------------
# classify_speeches — score normalisation (multi_label=false)
# ---------------------------------------------------------------------------

class TestScoreNormalisationSingleLabel:

    def test_scores_sum_to_one(self):
        df = _make_speech_df(4)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        model = _ControlledModel([2.0, 1.0, 0.5, 0.1])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        score_cols = ["score_risk_based", "score_rights_based",
                      "score_innovation_focused", "score_sovereignty_focused"]
        row_sums = result[score_cols].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-4

    def test_scores_between_zero_and_one(self):
        df = _make_speech_df(4)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        model = _ControlledModel([2.0, -1.0, 0.5, 3.0])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        score_cols = ["score_risk_based", "score_rights_based",
                      "score_innovation_focused", "score_sovereignty_focused"]
        assert result[score_cols].values.min() >= 0.0
        assert result[score_cols].values.max() <= 1.0


# ---------------------------------------------------------------------------
# classify_speeches — dominant framing
# ---------------------------------------------------------------------------

class TestDominantFraming:

    def test_dominant_framing_is_argmax(self):
        """highest entailment logit among the 4 labels → dominant framing."""
        df = _make_speech_df(2)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        # risk_based gets highest logit (5.0), others lower
        model = _ControlledModel([5.0, 1.0, 1.0, 1.0])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        assert (result["dominant_framing"] == "risk_based").all()

    def test_dominant_framing_innovation(self):
        df = _make_speech_df(2)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        # innovation_focused (index 2) gets highest logit
        model = _ControlledModel([1.0, 1.0, 5.0, 1.0])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        assert (result["dominant_framing"] == "innovation_focused").all()

    def test_dominant_framing_values_are_valid_labels(self):
        df = _make_speech_df(5)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        model = _ControlledModel([2.0, 1.5, 1.0, 0.5])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        valid = {"risk_based", "rights_based", "innovation_focused", "sovereignty_focused"}
        assert set(result["dominant_framing"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# classify_speeches — multi_label=true path
# ---------------------------------------------------------------------------

class TestMultiLabelMode:

    def test_scores_do_not_sum_to_one(self):
        """
        In multi_label=true mode each score is an independent per-hypothesis
        softmax probability — they need not sum to 1.
        """
        df = _make_speech_df(3)
        cfg = _make_cfg(multi_label=True)
        tokenizer = _ControlledTokenizer()
        # Give all labels the same entailment logit so each score ≈ 0.33
        # (softmax([0,0,1.0]) ≈ [0.21, 0.21, 0.58] for entailment_idx=2)
        model = _ControlledModel([1.0, 1.0, 1.0, 1.0])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        score_cols = ["score_risk_based", "score_rights_based",
                      "score_innovation_focused", "score_sovereignty_focused"]
        # With equal logits the row sum will be ~4 × 0.58 ≈ 2.3, definitely not 1.0
        row_sums = result[score_cols].sum(axis=1)
        assert not (row_sums - 1.0).abs().max() < 0.01

    def test_scores_are_valid_probabilities(self):
        df = _make_speech_df(3)
        cfg = _make_cfg(multi_label=True)
        tokenizer = _ControlledTokenizer()
        model = _ControlledModel([2.0, 0.5, -1.0, 3.0])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        score_cols = ["score_risk_based", "score_rights_based",
                      "score_innovation_focused", "score_sovereignty_focused"]
        assert result[score_cols].values.min() >= 0.0
        assert result[score_cols].values.max() <= 1.0


# ---------------------------------------------------------------------------
# classify_speeches — batching
# ---------------------------------------------------------------------------

class TestBatching:

    def test_different_batch_sizes_produce_valid_outputs(self):
        """Both batch_size=1 and batch_size=10 should produce well-formed outputs."""
        df = _make_speech_df(6)
        score_cols = ["score_risk_based", "score_rights_based",
                      "score_innovation_focused", "score_sovereignty_focused"]
        valid_framings = {"risk_based", "rights_based", "innovation_focused", "sovereignty_focused"}

        for batch_size in [1, 10]:
            cfg = _make_cfg(multi_label=False, batch_size=batch_size)
            model = _ControlledModel([3.0, 1.0, 2.0, 0.5])
            result = classify_speeches(df.copy(), _ControlledTokenizer(), model, "cpu", 2, cfg)
            assert len(result) == 6
            assert (result[score_cols].sum(axis=1) - 1.0).abs().max() < 1e-4
            assert set(result["dominant_framing"].unique()).issubset(valid_framings)

    def test_single_speech(self):
        df = _make_speech_df(1)
        cfg = _make_cfg(multi_label=False)
        tokenizer = _ControlledTokenizer()
        model = _ControlledModel([1.0, 0.5, 0.2, 0.1])
        result = classify_speeches(df, tokenizer, model, "cpu", 2, cfg)
        assert len(result) == 1
        assert result.iloc[0]["dominant_framing"] == "risk_based"
