"""Pairwise word-in-context evaluation with layer-wise similarity metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bert_lens.embeddings import target_embeddings
from bert_lens.model import BertResources

@dataclass(frozen=True)
class PairMetric:
    layer: int
    roc_auc: float
    threshold_accuracy: float
    threshold_std: float

def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left * right).sum(1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12)

def evaluate_pairs(resources: BertResources, first: list[str], second: list[str], words: list[str], labels: np.ndarray, layers: list[int], seed: int) -> list[PairMetric]:
    """Evaluate whether contextual similarity predicts same-sense labels without training a probe."""
    left = target_embeddings(resources, first, words, layers)
    right = target_embeddings(resources, second, words, layers)
    folds = StratifiedKFold(n_splits=min(5, int(np.bincount(labels).min())), shuffle=True, random_state=seed)
    results = []
    for layer in layers:
        scores = _cosine(left[layer], right[layer])
        held_out = []
        for train, test in folds.split(scores, labels):
            candidates = np.unique(scores[train])
            threshold = max(candidates, key=lambda value: accuracy_score(labels[train], scores[train] >= value))
            held_out.append(accuracy_score(labels[test], scores[test] >= threshold))
        results.append(PairMetric(layer, float(roc_auc_score(labels, scores)), float(np.mean(held_out)), float(np.std(held_out))))
    return results

def save_pair_plot(metrics: list[PairMetric], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5)); layers = [m.layer for m in metrics]
    ax.plot(layers, [m.roc_auc for m in metrics], marker="o", label="ROC-AUC")
    ax.errorbar(layers, [m.threshold_accuracy for m in metrics], yerr=[m.threshold_std for m in metrics], marker="s", capsize=3, label="Held-out threshold accuracy")
    ax.set(xlabel="BERT layer (0 = embeddings)", ylabel="Score", ylim=(0, 1.05), title="WiC: layer-wise same-sense discrimination")
    ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

def write_metrics(metrics: list[PairMetric], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps([asdict(m) for m in metrics], indent=2))
