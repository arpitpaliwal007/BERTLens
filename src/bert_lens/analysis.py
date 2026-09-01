"""Attention, polysemy, probing, and plots."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from bert_lens.model import BertResources

@dataclass(frozen=True)
class SenseMetric:
    layer: int; n_contexts: int; n_clusters: int; silhouette: float; adjusted_rand_index: float

@dataclass(frozen=True)
class ProbeMetric:
    layer: int; mean_accuracy: float; std_accuracy: float; folds: int

def attention_matrix(resources: BertResources, text: str, layer: int, head: int) -> tuple[list[str], np.ndarray]:
    batch = resources.tokenizer(text, return_tensors="pt", truncation=True)
    with torch.inference_mode(): output = resources.model(**{k: v.to(resources.device) for k, v in batch.items()})
    if layer < 0 or layer >= len(output.attentions): raise ValueError("Invalid attention layer.")
    if head < 0 or head >= output.attentions[layer].shape[1]: raise ValueError("Invalid attention head.")
    return resources.tokenizer.convert_ids_to_tokens(batch["input_ids"][0]), output.attentions[layer][0, head].cpu().numpy()

def save_attention(tokens: list[str], matrix: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); size = max(7, len(tokens) * .55)
    fig, ax = plt.subplots(figsize=(size, size*.8)); image = ax.imshow(matrix, cmap="magma")
    ax.set_xticks(range(len(tokens)), tokens, rotation=60, ha="right"); ax.set_yticks(range(len(tokens)), tokens)
    ax.set_xlabel("Key token"); ax.set_ylabel("Query token"); ax.set_title(title); fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

def sense_analysis(vectors: np.ndarray, labels: np.ndarray, layer: int, clusters: int, seed: int) -> SenseMetric:
    predicted = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit_predict(vectors)
    return SenseMetric(layer, len(vectors), clusters, float(silhouette_score(vectors, predicted)), float(adjusted_rand_score(labels, predicted)))

def save_projection(vectors: np.ndarray, labels: np.ndarray, layer: int, target: str, path: Path, seed: int) -> None:
    xy = PCA(n_components=2, random_state=seed).fit_transform(vectors); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label in sorted(set(labels)):
        mask = labels == label; ax.scatter(xy[mask, 0], xy[mask, 1], s=64, alpha=.85, label=str(label))
    ax.set(title=f"{target!r} contextual embeddings — layer {layer}", xlabel="PCA component 1", ylabel="PCA component 2")
    ax.legend(title="Gold sense"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

def probe_analysis(vectors: np.ndarray, labels: np.ndarray, layer: int, seed: int) -> ProbeMetric:
    folds = min(5, int(np.unique(labels, return_counts=True)[1].min()))
    if folds < 2: raise ValueError("Every probe class needs at least two examples.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed), vectors, labels, cv=cv)
    return ProbeMetric(layer, float(scores.mean()), float(scores.std()), folds)

def save_probe_plot(metrics: list[ProbeMetric], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.errorbar([m.layer for m in metrics], [m.mean_accuracy for m in metrics], yerr=[m.std_accuracy for m in metrics], marker="o", capsize=4)
    ax.set(ylim=(0, 1.05), xlabel="BERT layer (0 = embeddings)", ylabel="Cross-validated accuracy", title="Layer-wise control probe: short vs. long sentence")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

def as_dicts(items: list[SenseMetric] | list[ProbeMetric]) -> list[dict]: return [asdict(item) for item in items]
