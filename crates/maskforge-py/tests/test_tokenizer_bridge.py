"""EOS-resolution policy: fake tokenizer/model objects only, no download, no torch/transformers."""

import pytest

maskforge = pytest.importorskip("maskforge")
from maskforge.tokenizers import EosResolutionError, resolve_eos_token_id


class FakeTokenizer:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class FakeGenerationConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class FakeModel:
    def __init__(self, eos_token_id):
        self.generation_config = FakeGenerationConfig(eos_token_id)


def test_explicit_eos_token_id_always_wins():
    tokenizer = FakeTokenizer(eos_token_id=5)
    model = FakeModel(eos_token_id=9)
    assert resolve_eos_token_id(tokenizer, model=model, eos_token_id=42) == 42


def test_falls_back_to_tokenizer_eos_when_no_model_given():
    tokenizer = FakeTokenizer(eos_token_id=7)
    assert resolve_eos_token_id(tokenizer) == 7


def test_model_generation_config_used_when_tokenizer_has_none():
    tokenizer = FakeTokenizer(eos_token_id=None)
    model = FakeModel(eos_token_id=11)
    assert resolve_eos_token_id(tokenizer, model=model) == 11


def test_agreeing_model_and_tokenizer_eos_is_accepted():
    tokenizer = FakeTokenizer(eos_token_id=3)
    model = FakeModel(eos_token_id=3)
    assert resolve_eos_token_id(tokenizer, model=model) == 3


def test_disagreeing_model_and_tokenizer_eos_raises_without_override():
    tokenizer = FakeTokenizer(eos_token_id=3)
    model = FakeModel(eos_token_id=4)
    with pytest.raises(EosResolutionError):
        resolve_eos_token_id(tokenizer, model=model)


def test_disagreeing_eos_is_overridable_explicitly():
    tokenizer = FakeTokenizer(eos_token_id=3)
    model = FakeModel(eos_token_id=4)
    assert resolve_eos_token_id(tokenizer, model=model, eos_token_id=3) == 3


def test_missing_eos_everywhere_raises():
    tokenizer = FakeTokenizer(eos_token_id=None)
    with pytest.raises(EosResolutionError):
        resolve_eos_token_id(tokenizer)


def test_tokenizer_eos_as_single_element_list_normalizes():
    tokenizer = FakeTokenizer(eos_token_id=[8])
    assert resolve_eos_token_id(tokenizer) == 8


def test_tokenizer_eos_as_multiple_distinct_ids_raises_not_silently_truncates():
    tokenizer = FakeTokenizer(eos_token_id=[1, 2, 3])
    with pytest.raises(EosResolutionError):
        resolve_eos_token_id(tokenizer)


def test_tokenizer_eos_as_duplicate_list_normalizes_to_one_id():
    tokenizer = FakeTokenizer(eos_token_id=[6, 6, 6])
    assert resolve_eos_token_id(tokenizer) == 6


def test_missing_backend_tokenizer_raises_clear_error():
    from maskforge.tokenizers import TokenizerBridgeError, from_transformers

    class NoFastTokenizer:
        eos_token_id = 0
        backend_tokenizer = None

    with pytest.raises(TokenizerBridgeError):
        from_transformers(NoFastTokenizer())
