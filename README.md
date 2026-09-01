# BERTLens

**BERTLens** is a reproducible Python toolkit for inspecting how BERT moves from subword tokens to contextual representations. It combines tokenization inspection, attention analysis, contextual word-sense clustering, and a layer-wise control probe.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first analysis downloads `bert-base-uncased` (about 440 MB).

## Run

```bash
bertlens tokenize --text "The bank approved the loan." --target-word bank
bertlens attention --text "The bank approved the loan." --layer 8 --head 3
bertlens senses --input data/polysemy_examples.csv --target-word bank --layers 0,4,8,12
bertlens probe --input data/probe_length_examples.csv --layers 0,4,8,12
python scripts/run_demo.py
pytest -q
```

Results are written to `reports/`: tokenization JSON, attention heatmaps, layer-wise PCA projections with clustering metrics, and probe accuracy plots. Target words are aligned by character offsets, so split WordPieces are pooled correctly. The included `bank` data is illustrative, not a benchmark claim. The control probe validates the pipeline only; replace it with a standard probing dataset before making linguistic claims.

The project is independently implemented and conceptually informed by [polysemy-assessment](https://github.com/ksipos/polysemy-assessment) and [interpret_bert](https://github.com/ganeshjawahar/interpret_bert).
