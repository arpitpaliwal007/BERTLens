"""Model loading and inference helpers."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

@dataclass(frozen=True)
class BertResources:
    tokenizer: PreTrainedTokenizerFast
    model: PreTrainedModel
    device: torch.device

def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS was requested but is unavailable.")
        return device
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_bert(model_name: str = "bert-base-uncased", device: str = "auto") -> BertResources:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast): raise TypeError("A fast tokenizer is required for offset alignment.")
    target = resolve_device(device)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True).to(target)
    model.eval()
    return BertResources(tokenizer, model, target)
