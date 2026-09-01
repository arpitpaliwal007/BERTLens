"""Dependency-free validation helpers."""
from __future__ import annotations

def parse_layers(value: str, maximum_layer: int) -> list[int]:
    """Parse `all` or comma-separated layer indices."""
    layers = list(range(maximum_layer + 1)) if value == "all" else [int(x) for x in value.split(",")]
    if not layers or any(x < 0 or x > maximum_layer for x in layers):
        raise ValueError(f"Layers must be within 0..{maximum_layer}.")
    return sorted(set(layers))
