# AI Act Framing of speeches from Members of the European Parliament

Zero-shot transformer classification of AI governance framings in European Parliament debates (2019–2024), mapping risk-based, rights-based, innovation-focused, and sovereignty-focused discourse onto party family and East-West/North-South cleavages.

This repository implements a transformer-based framing analysis of European Parliament debates on artificial intelligence (2019–2024). Using zero-shot NLI classification, it categorises MEP speeches into four governance framings and examines how these map onto party family, national identity, and regional cleavages in EU integration debates.

## Research design

- **Corpus**: EUPDCorp (Zenodo, DOI: 10.5281/zenodo.15056399) — 563,696 EP speeches 1999–2024, with English translations and nationality/party metadata
- **Time window**: 2019–2024 (9th EP term, AI Act period)
- **Classifier**: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` via HuggingFace zero-shot NLI
- **Unit of analysis**: speech level (dominant framing); MEP level for regression
- **Validation**: few-shot — ~30 manually annotated examples per framing class

## Repo structure

```
ep-ai-framing/
├── config.yaml                  # all parameters: paths, keywords, model, framings, cleavage codings
├── requirements.txt
├── .gitignore
├── src/
│   ├── 00_download_data.py      # download EUPDCorp from Zenodo
│   ├── 01_filter_corpus.py      # keyword filter, year filter, cleavage variable construction
│   ├── 02_classify.py           # zero-shot framing classification
│   ├── 03_validate.py           # evaluate classifier against gold standard
│   └── 04_analyse.py            # descriptive tables + OLS regressions
├── validation/
│   └── annotation_template.csv  # seed examples; add ~30 per class before running 03
├── data/
│   ├── raw/                     # EUPDCorp.csv goes here (not tracked by git)
│   └── processed/               # pipeline outputs (not tracked by git)
└── notebooks/
    └── 01_explore.ipynb         # EDA on filtered corpus
```

## Setup

```bash
git clone https://github.com/guidoschillaci/ep-ai-framing.git
cd ep-ai-framing
```

### Python environment

Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

Or with conda:

```bash
conda create -n mep-speech python=3.11
conda activate mep-speech
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.2+. On Apple Silicon (MPS), set `device: mps` in `config.yaml`.

## Pipeline

Run scripts in order from the repo root:

```bash
# 1. Download corpus (~500MB)
python src/00_download_data.py

# 2. Filter to AI-relevant speeches using NLI relevance model
python src/01_filter_corpus.py

# 3. Classify framings (downloads model ~400MB on first run)
python src/02_classify.py

# 4. Validate relevance filter and framing classifier
python src/03_validate.py              # both stages
python src/03_validate.py --stage relevance
python src/03_validate.py --stage framing

# 5. Analyse cleavage patterns
python src/04_analyse.py
```

## Validation

`03_validate.py` validates two pipeline stages independently:

### Stage 1 — Relevance filter

Annotate `validation/relevance_annotation.csv` with columns `text` and `is_relevant` (1 = AI-relevant, 0 = not relevant). Aim for ~50 positive and ~50 negative examples drawn from real EP speeches.

The script reports precision/recall/F1 at the configured threshold, ROC-AUC, and suggests an optimal threshold via Youden's J. Update `relevance_filter.threshold` in `config.yaml` if the suggestion differs substantially.

### Stage 2 — Framing classifier

Annotate `validation/annotation_template.csv` with columns `text` and `true_label`. Valid `true_label` values:

- `risk_based`
- `rights_based`
- `innovation_focused`
- `sovereignty_focused`

Aim for ~30 examples per class (120 total). Target F1 >= 0.70 per class before proceeding to analysis.

## Tests

Unit tests cover the core data-transformation and analysis functions (no model loading required):

```bash
python -m pytest tests/ -v
```

Tests are in `tests/` and cover:
- `test_filter_corpus.py` — text resolution, year/length filters, cleavage variable coding
- `test_analyse.py` — framing share normalisation, MEP aggregation, dominant framing logic

## Configuration

All parameters are in `config.yaml`:

- **relevance_filter**: NLI hypothesis and threshold for the AI-relevance filter in `01_filter_corpus.py`. Raise `threshold` (0–1) to retain fewer, more confidently AI-relevant speeches.
- **framings**: hypothesis strings passed to the NLI framing classifier — edit these to refine classification
- **model**: model name, device, batch size, precision, max token length
- **party_family**: EP group to party family mapping
- **east_countries / west_countries**: East-West cleavage coding
- **north_countries / middle_countries / south_countries**: North-Middle-South cleavage coding
- **pre2004_countries / post2004_countries**: EU accession wave coding

## Notes on validity

- English-only speeches are used directly; non-English speeches use EUPDCorp's machine translations
- Translation quality varies across language communities — noted as a methods limitation
- Regression analysis uses MEP-level aggregated scores to avoid bias from high-volume speakers (rapporteurs)
- Standard errors clustered by country in OLS models

## Citation

If you use this code, please cite:

- EUPDCorp: Zenodo DOI 10.5281/zenodo.15056399
- Classifier: Laurer et al. (2023), arXiv:2312.17543
