"""Contextual embedding extraction."""
from __future__ import annotations
import numpy as np
import torch
from bert_lens.model import BertResources
from bert_lens.tokenization import find_target_span, overlapping_token_indices
from bert_lens.utils import parse_layers

def target_embeddings(
    resources: BertResources,
    texts: list[str],
    words: list[str],
    layers: list[int],
    spans: list[tuple[int, int]] | None = None,
) -> dict[int, np.ndarray]:
    if len(texts) != len(words): raise ValueError("texts and words must have equal length.")
    if spans is not None and len(spans) != len(texts): raise ValueError("spans and texts must have equal length.")
    batch = resources.tokenizer(texts, padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offsets = batch.pop("offset_mapping").tolist()
    with torch.inference_mode(): output = resources.model(**{k: v.to(resources.device) for k, v in batch.items()})
    if any(x >= len(output.hidden_states) for x in layers): raise ValueError("A requested layer is unavailable.")
    result = {layer: [] for layer in layers}
    for row, (text, word) in enumerate(zip(texts, words, strict=True)):
        span = spans[row] if spans is not None else find_target_span(text, word)
        indices = overlapping_token_indices(offsets[row], span)
        if not indices: raise ValueError(f"No tokens align with {word!r}.")
        for layer in layers: result[layer].append(output.hidden_states[layer][row, indices].mean(0).cpu().numpy())
    return {layer: np.vstack(vectors) for layer, vectors in result.items()}

def cls_embeddings(resources: BertResources, texts: list[str], layers: list[int]) -> dict[int, np.ndarray]:
    batch = resources.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.inference_mode(): output = resources.model(**{k: v.to(resources.device) for k, v in batch.items()})
    return {layer: output.hidden_states[layer][:, 0].cpu().numpy() for layer in layers}
