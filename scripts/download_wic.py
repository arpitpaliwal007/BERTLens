"""Download SuperGLUE WiC into a stable CSV consumed by `bertlens pairs`."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from datasets import load_dataset

parser = argparse.ArgumentParser(); parser.add_argument("--split", default="validation"); parser.add_argument("--limit", type=int); parser.add_argument("--output", default="data/wic_validation.csv")
args = parser.parse_args()
dataset = load_dataset("aps/super_glue", "wic", split=args.split)
frame = pd.DataFrame(dataset)
if args.limit: frame = frame.head(args.limit)
frame.rename(columns={"sentence1": "sentence1", "sentence2": "sentence2", "word": "target_word", "label": "label"})[["sentence1", "sentence2", "target_word", "label"]].to_csv(Path(args.output), index=False)
print(f"Wrote {len(frame)} examples to {args.output}")
