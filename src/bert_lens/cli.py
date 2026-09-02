"""Command-line interface for BERTLens."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from bert_lens.analysis import attention_matrix, as_dicts, probe_analysis, save_attention, save_probe_plot, save_projection, sense_analysis
from bert_lens.benchmark import evaluate_pairs, save_pair_plot, write_metrics
from bert_lens.embeddings import cls_embeddings, target_embeddings
from bert_lens.model import load_bert
from bert_lens.tokenization import inspect_text, records_as_dicts
from bert_lens.utils import parse_layers

def shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="bert-base-uncased"); parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=42)

def tokenize(args: argparse.Namespace) -> None:
    resources = load_bert(args.model, args.device); path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": args.text, "tokens": records_as_dicts(inspect_text(resources.tokenizer, args.text, args.target_word))}, indent=2)); print(f"Wrote {path}")

def attention(args: argparse.Namespace) -> None:
    tokens, matrix = attention_matrix(load_bert(args.model, args.device), args.text, args.layer, args.head)
    save_attention(tokens, matrix, Path(args.output), f"BERT attention — layer {args.layer}, head {args.head}"); print(f"Wrote {args.output}")

def senses(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input); required = {"sentence", "target_word", "sense"}
    if not required <= set(frame): raise ValueError(f"Input must contain {sorted(required)}.")
    frame = frame[frame.target_word.str.lower() == args.target_word.lower()].reset_index(drop=True)
    if frame.empty: raise ValueError(f"No contexts for {args.target_word!r}.")
    resources = load_bert(args.model, args.device); layers = parse_layers(args.layers, resources.model.config.num_hidden_layers)
    vectors = target_embeddings(resources, frame.sentence.tolist(), frame.target_word.tolist(), layers); labels = frame.sense.to_numpy(); clusters = args.clusters or len(set(labels)); output = Path(args.output_dir)
    metrics = []
    for layer in layers:
        metrics.append(sense_analysis(vectors[layer], labels, layer, clusters, args.seed)); save_projection(vectors[layer], labels, layer, args.target_word, output / f"layer_{layer}_pca.png", args.seed)
    output.mkdir(parents=True, exist_ok=True); (output / "metrics.json").write_text(json.dumps(as_dicts(metrics), indent=2)); print(f"Wrote sense analysis to {output}")

def probe(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input)
    if not {"sentence", "label"} <= set(frame): raise ValueError("Input must contain sentence and label.")
    resources = load_bert(args.model, args.device); layers = parse_layers(args.layers, resources.model.config.num_hidden_layers)
    vectors = cls_embeddings(resources, frame.sentence.tolist(), layers); metrics = [probe_analysis(vectors[x], frame.label.to_numpy(), x, args.seed) for x in layers]; output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(as_dicts(metrics), indent=2)); save_probe_plot(metrics, output / "accuracy_by_layer.png"); print(f"Wrote probe analysis to {output}")

def pairs(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input); required = {"sentence1", "sentence2", "target_word", "label"}
    if not required <= set(frame): raise ValueError(f"Input must contain {sorted(required)}.")
    resources = load_bert(args.model, args.device); layers = parse_layers(args.layers, resources.model.config.num_hidden_layers)
    metrics = evaluate_pairs(resources, frame.sentence1.tolist(), frame.sentence2.tolist(), frame.target_word.tolist(), frame.label.to_numpy(), layers, args.seed)
    output = Path(args.output_dir); write_metrics(metrics, output / "metrics.json"); save_pair_plot(metrics, output / "layer_summary.png"); print(f"Wrote pairwise benchmark to {output}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect BERT representations from tokens to meaning."); subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("tokenize"); p.add_argument("--text", required=True); p.add_argument("--target-word"); p.add_argument("--output", default="reports/tokenization.json"); shared(p); p.set_defaults(handler=tokenize)
    p = subs.add_parser("attention"); p.add_argument("--text", required=True); p.add_argument("--layer", type=int, required=True); p.add_argument("--head", type=int, required=True); p.add_argument("--output", default="reports/attention.png"); shared(p); p.set_defaults(handler=attention)
    p = subs.add_parser("senses"); p.add_argument("--input", default="data/polysemy_examples.csv"); p.add_argument("--target-word", required=True); p.add_argument("--layers", default="0,4,8,12"); p.add_argument("--clusters", type=int); p.add_argument("--output-dir", default="reports/polysemy"); shared(p); p.set_defaults(handler=senses)
    p = subs.add_parser("probe"); p.add_argument("--input", default="data/probe_length_examples.csv"); p.add_argument("--layers", default="0,4,8,12"); p.add_argument("--output-dir", default="reports/probe"); shared(p); p.set_defaults(handler=probe)
    p = subs.add_parser("pairs", help="Evaluate same-sense prediction from contextual similarity."); p.add_argument("--input", required=True); p.add_argument("--layers", default="all"); p.add_argument("--output-dir", default="reports/wic"); shared(p); p.set_defaults(handler=pairs)
    return parser

def main() -> None:
    args = build_parser().parse_args(); args.handler(args)
