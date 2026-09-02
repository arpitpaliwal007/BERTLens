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
python scripts/download_wic.py --split validation --limit 500
bertlens pairs --input data/wic_validation.csv --layers all --output-dir reports/wic
python scripts/run_demo.py
pytest -q
```

Results are written to `reports/`: tokenization JSON, attention heatmaps, layer-wise PCA projections with clustering metrics, and probe accuracy plots. Target words are aligned by character offsets, so split WordPieces are pooled correctly. The included `bank` data is illustrative, not a benchmark claim. The control probe validates the pipeline only; replace it with a standard probing dataset before making linguistic claims.

The project is independently implemented and conceptually informed by [polysemy-assessment](https://github.com/ksipos/polysemy-assessment) and [interpret_bert](https://github.com/ganeshjawahar/interpret_bert).

## Benchmark upgrade: WiC

`download_wic.py` obtains SuperGLUE WiC as a reproducible CSV. `bertlens pairs` then evaluates whether cosine similarity between two contextual target-word embeddings predicts its gold same-sense label. It reports layer-wise ROC-AUC plus cross-validated, held-out threshold accuracy and saves one summary plot. This is a real held-out benchmark; keep the curated `bank` experiment only as a qualitative visualization.
