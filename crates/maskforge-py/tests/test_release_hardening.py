"""Release-gate checks that never skip: no model, no network, no optional dependency.

Every test here exercises a guarantee the published wheel must hold on its own.
"""

import json

import pytest

import maskforge
from maskforge import low_level
from maskforge.generator import _reject_unsupported
from maskforge.tokenizers import EosResolutionError, resolve_eos_token_id

EOS = 2
TOKENS = [(b"true", [0]), (b"false", [1])]


class _Tokenizer:
    """The minimal tokenizer surface `resolve_eos_token_id` reads."""

    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _GenerationConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _Model:
    def __init__(self, eos_token_id):
        self.generation_config = _GenerationConfig(eos_token_id)



def test_the_installed_build_exposes_tokenizer_json_construction():
    """from_transformers is unusable without this native method; it must ship in every build."""
    assert hasattr(low_level.PyVocabulary, "from_tokenizer_json")


def test_tokenizer_json_over_the_byte_cap_is_rejected_as_a_resource_limit():
    """An oversized input is a resource limit, not malformed tokenizer metadata."""
    oversized = " " * (64 * 1024 * 1024 + 1)
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        low_level.PyVocabulary.from_tokenizer_json(oversized, EOS, None)
    assert excinfo.value.code == "InternalLimitExceeded"


def test_malformed_tokenizer_json_keeps_its_own_category():
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        low_level.PyVocabulary.from_tokenizer_json("{not json", EOS, None)
    assert excinfo.value.code == "MalformedTokenizer"


# Self-contained tokenizer fixture that requires neither network access nor cached models.
TOKENIZER_JSON = json.dumps(
    {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "WordLevel",
            "vocab": {"true": 0, "false": 1, "<eos>": 2, "{": 3, "}": 4, '"': 5},
            "unk_token": "<eos>",
        },
    }
)


def test_a_vocabulary_builds_from_tokenizer_json_and_drives_a_session():
    from maskforge import _native

    raw = low_level.PyVocabulary.from_tokenizer_json(TOKENIZER_JSON, EOS, None)
    vocabulary = maskforge.Vocabulary(_native.CompiledVocabulary.from_vocabulary(raw))
    bound = maskforge.Compiler().compile({"type": "boolean"}).bind(vocabulary)
    session = bound.start_session()
    buffer = bytearray(4 * bound.mask_word_count)
    session.write_mask(buffer)
    assert any(buffer), "a boolean schema must allow at least one token"



def test_a_bool_is_not_an_eos_token_id():
    with pytest.raises(EosResolutionError, match="bool"):
        resolve_eos_token_id(_Tokenizer(True))


def test_a_float_is_not_silently_truncated_to_an_eos_token_id():
    with pytest.raises(EosResolutionError):
        resolve_eos_token_id(_Tokenizer(1.9))


def test_a_numeric_string_is_not_accepted_as_an_eos_token_id():
    with pytest.raises(EosResolutionError):
        resolve_eos_token_id(_Tokenizer("50256"))


def test_a_numeric_string_inside_a_list_is_not_accepted():
    with pytest.raises(EosResolutionError):
        resolve_eos_token_id(_Tokenizer(["50256"]))


def test_a_negative_eos_token_id_is_rejected():
    with pytest.raises(EosResolutionError, match="non-negative"):
        resolve_eos_token_id(_Tokenizer(-1))


def test_an_explicit_eos_token_id_uses_the_same_strict_rule():
    with pytest.raises(EosResolutionError, match="bool"):
        resolve_eos_token_id(_Tokenizer(5), eos_token_id=True)


def test_two_distinct_eos_ids_are_rejected_rather_than_guessed():
    with pytest.raises(EosResolutionError, match="distinct"):
        resolve_eos_token_id(_Tokenizer([1, 2]))


def test_a_model_tokenizer_disagreement_is_reported():
    with pytest.raises(EosResolutionError, match="disagrees"):
        resolve_eos_token_id(_Tokenizer(1), model=_Model(2))


def test_a_plain_integer_eos_resolves():
    assert resolve_eos_token_id(_Tokenizer(50256)) == 50256



@pytest.mark.parametrize(
    "kwargs",
    [
        {"assistant_model": object()},
        {"prompt_lookup_num_tokens": 3},
        {"assistant_early_exit": 2},
        {"custom_generate": "somewhere/custom"},
        {"num_beams": 2},
        {"num_return_sequences": 2},
        {"eos_token_id": 7},
        {"pad_token_id": 7},
    ],
)
def test_unsupported_generation_modes_are_rejected(kwargs):
    with pytest.raises(NotImplementedError):
        _reject_unsupported(dict(kwargs))


def test_supported_generation_kwargs_pass_through():
    _reject_unsupported({"num_beams": 1, "num_return_sequences": 1, "temperature": 0.7})



def test_bind_policy_is_validated():
    with pytest.raises(ValueError, match="bind_policy"):
        maskforge.Compiler(bind_policy="turbo")


@pytest.mark.parametrize("policy", ["adaptive", "eager", "lazy"])
def test_every_documented_bind_policy_is_accepted(policy):
    compiler = maskforge.Compiler(bind_policy=policy, trie_cache_bytes=1 << 20)
    program = compiler.compile(json.dumps({"type": "boolean"}))
    bound = program.bind(maskforge.Vocabulary.from_tokens(EOS, TOKENS))
    assert bound.mask_vocab_size >= 2



def test_the_root_namespace_exports_only_the_stable_api():
    assert set(maskforge.__all__) == {
        "__version__",
        "BoundSchema",
        "Compiler",
        "GenerationError",
        "Generator",
        "MaskforgeError",
        "OutputSpec",
        "SchemaProgram",
        "Session",
        "TransformersModel",
        "Vocabulary",
        "from_transformers",
        "low_level",
        "tokenizers",
    }


def test_a_legacy_root_name_still_resolves_but_warns():
    with pytest.deprecated_call():
        assert maskforge.PyVocabulary is low_level.PyVocabulary


def test_an_unknown_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError):
        maskforge.definitely_not_an_api


def test_every_public_python_module_is_annotated():
    """py.typed is shipped, so each public callable must carry annotations."""
    import inspect

    from maskforge import compiler, constraints, generator, models, tokenizers, vocabulary

    unannotated = []
    for module in (compiler, constraints, generator, models, tokenizers, vocabulary):
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            members = [(name, obj)] if inspect.isfunction(obj) else []
            if inspect.isclass(obj):
                members = [
                    (f"{name}.{n}", m)
                    for n, m in vars(obj).items()
                    if inspect.isfunction(m) and not n.startswith("_")
                ]
            for label, member in members:
                hints = getattr(member, "__annotations__", {})
                expected = [
                    p
                    for p in inspect.signature(member).parameters
                    if p not in ("self", "cls", "args", "kwargs")
                ]
                if "return" not in hints or any(p not in hints for p in expected):
                    unannotated.append(f"{module.__name__}.{label}")
    assert not unannotated, f"public callables missing annotations: {unannotated}"
