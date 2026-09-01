"""Reproduce the complete BERTLens demonstration."""
from __future__ import annotations
import sys
from bert_lens.cli import main

def run(arguments: list[str]) -> None:
    old = sys.argv
    try: sys.argv = ["bertlens", *arguments]; main()
    finally: sys.argv = old

if __name__ == "__main__":
    run(["tokenize", "--text", "The bank approved the loan.", "--target-word", "bank"])
    run(["attention", "--text", "The bank approved the loan.", "--layer", "8", "--head", "3", "--output", "reports/attention_layer8_head3.png"])
    run(["senses", "--target-word", "bank", "--layers", "0,4,8,12"])
    run(["probe", "--layers", "0,4,8,12"])
