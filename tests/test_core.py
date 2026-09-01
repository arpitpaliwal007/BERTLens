import pytest
from bert_lens.tokenization import find_target_span, overlapping_token_indices
from bert_lens.utils import parse_layers

def test_target_span_is_case_insensitive_and_whole_word():
    assert find_target_span("The BANK is near riverbank.", "bank") == (4, 8)

def test_alignment_excludes_special_zero_offsets():
    assert overlapping_token_indices([(0, 0), (0, 3), (4, 8), (0, 0)], (4, 8)) == [2]

def test_layer_parsing():
    assert parse_layers("all", 2) == [0, 1, 2]
    assert parse_layers("2,0,2", 2) == [0, 2]
    with pytest.raises(ValueError): parse_layers("3", 2)
