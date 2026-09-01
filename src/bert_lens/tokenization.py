"""WordPiece inspection and target-word alignment."""
from __future__ import annotations
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

@dataclass(frozen=True)
class TokenRecord:
    index: int; token: str; token_id: int; offset: tuple[int, int]; is_special: bool; overlaps_target: bool

def find_target_span(text: str, target_word: str) -> tuple[int, int]:
    match = re.search(rf"\b{re.escape(target_word)}\b", text, flags=re.I)
    if not match: raise ValueError(f"Target word {target_word!r} does not occur in {text!r}.")
    return match.span()

def overlapping_token_indices(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    start, end = span
    return [i for i, (left, right) in enumerate(offsets) if left < end and right > start]

def inspect_text(tokenizer: PreTrainedTokenizerFast, text: str, target_word: str | None = None) -> list[TokenRecord]:
    encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    targets = overlapping_token_indices(offsets, find_target_span(text, target_word)) if target_word else []
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    return [TokenRecord(i, token, encoded["input_ids"][i], offsets[i], bool(encoded["special_tokens_mask"][i]), i in targets) for i, token in enumerate(tokens)]

def records_as_dicts(records: list[TokenRecord]) -> list[dict]: return [asdict(record) for record in records]
