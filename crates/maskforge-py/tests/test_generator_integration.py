"""Real CPU GPT-2 generation through maskforge.Generator; needs a locally cached model.

Not part of ordinary offline CI: requires torch + transformers + a cached "gpt2" checkout.
Run explicitly: uv run pytest crates/maskforge-py/tests/test_generator_integration.py -q
"""

import json
import os

import pytest

from maskforge.generator import (
    _AFTER_ESCAPE,
    _BoundedJsonWhitespaceProcessor,
    _INSIDE_STRING,
    _OUTSIDE_STRING,
    _advance_json_lexical_state,
)

maskforge = pytest.importorskip("maskforge")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

os.environ.setdefault("HF_HUB_OFFLINE", "1")

BOOLEAN = json.dumps({"type": "boolean"})


def test_bounded_json_whitespace_tracks_only_insignificant_runs():
    assert _advance_json_lexical_state(_OUTSIDE_STRING, 0, b" \n", 1)[2] is False
    assert _advance_json_lexical_state(_OUTSIDE_STRING, 0, b'"  "', 1) == (
        _OUTSIDE_STRING,
        0,
        True,
    )
    assert _advance_json_lexical_state(_INSIDE_STRING, 0, b'\\" ', 1) == (
        _INSIDE_STRING,
        0,
        True,
    )
    assert _advance_json_lexical_state(_AFTER_ESCAPE, 0, b'u0063" ', 1) == (
        _OUTSIDE_STRING,
        1,
        True,
    )


def test_bounded_json_whitespace_processor_uses_raw_vocabulary_bytes():
    vocabulary = maskforge.Vocabulary.from_id_ordered_tokens(
        [b" ", b"\n", b"{", b'"', b"x", b"}", None], eos_token_id=6
    )
    processor = _BoundedJsonWhitespaceProcessor(vocabulary, maximum=1, prompt_width=0)

    after_space = torch.zeros((1, 7))
    processor(torch.tensor([[0]]), after_space)
    assert torch.isneginf(after_space[0, 0])
    assert torch.isneginf(after_space[0, 1])
    assert after_space[0, 2] == 0

    inside_string = torch.zeros((1, 7))
    processor(torch.tensor([[3, 0]]), inside_string)
    assert inside_string[0, 0] == 0
    assert inside_string[0, 1] == 0


def _load_gpt2():
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as exc:  # not cached locally, or genuinely offline
        pytest.skip(f"gpt2 not available offline: {exc}")
    return model, tokenizer


def _boolean_generator():
    model, tokenizer = _load_gpt2()
    adapter = maskforge.from_transformers(model, tokenizer)
    return maskforge.Generator(adapter, BOOLEAN), tokenizer


def test_base_import_stays_dependency_free():
    import subprocess
    import sys

    code = (
        "import sys, maskforge; "
        "bad = [m for m in ('torch','transformers','pydantic') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_single_prompt_tight_boolean_schema():
    generate, _ = _boolean_generator()
    result = generate("Return true or false:", max_new_tokens=8, do_sample=False)
    assert result in (True, False)


def test_batch_of_two_prompts():
    generate, _ = _boolean_generator()
    results = generate(
        ["Return true or false:", "Answer with a boolean:"], max_new_tokens=8, do_sample=False
    )
    assert len(results) == 2
    assert all(r in (True, False) for r in results)


def test_batch_rows_match_the_same_prompts_run_singly():
    """Left padding must not change what a row generates."""
    generate, _ = _boolean_generator()
    prompts = ["Return true or false:", "Answer with a boolean:"]
    batched = generate(prompts, max_new_tokens=8, do_sample=False)
    singly = [generate(p, max_new_tokens=8, do_sample=False) for p in prompts]
    assert batched == singly


def test_tight_object_schema_produces_valid_instance():
    model, tokenizer = _load_gpt2()
    adapter = maskforge.from_transformers(model, tokenizer)
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
    )
    generate = maskforge.Generator(adapter, schema)
    result = generate('{"ok":', max_new_tokens=20, do_sample=False)
    assert result in ({"ok": True}, {"ok": False})


def test_the_caller_tokenizer_is_never_written_to_during_a_batch_call():
    """The generator pads locally, so no attribute of the shared tokenizer may be assigned."""
    generate, tokenizer = _boolean_generator()

    writes = []
    tokenizer_class = type(tokenizer)
    original_setattr = tokenizer_class.__setattr__

    def recording_setattr(self, name, value):
        if self is tokenizer:
            writes.append(name)
        original_setattr(self, name, value)

    tokenizer_class.__setattr__ = recording_setattr
    try:
        generate(["a:", "b:"], max_new_tokens=8, do_sample=False)
    finally:
        tokenizer_class.__setattr__ = original_setattr

    assert not writes, f"generator mutated the caller's tokenizer: {sorted(set(writes))}"


def test_beam_search_is_rejected_not_silently_wrong():
    generate, _ = _boolean_generator()
    with pytest.raises(NotImplementedError):
        generate("Return true or false:", max_new_tokens=4, num_beams=2)


def test_prompt_lookup_speculation_is_rejected():
    generate, _ = _boolean_generator()
    with pytest.raises(NotImplementedError, match="prompt_lookup_num_tokens"):
        generate("Return true or false:", max_new_tokens=4, prompt_lookup_num_tokens=3)


def test_second_generator_call_does_not_inherit_first_calls_progress():
    generate, _ = _boolean_generator()
    first = generate("Return true or false:", max_new_tokens=8, do_sample=False)
    second = generate("Return true or false:", max_new_tokens=8, do_sample=False)
    assert first == second, "identical greedy prompt must reproduce identically across calls"


def test_repeated_calls_do_not_recompile_or_rebind():
    """A prebound generator holds one program and one bound artifact for its whole lifetime."""
    model, tokenizer = _load_gpt2()
    adapter = maskforge.from_transformers(model, tokenizer)
    generate = maskforge.Generator(adapter, BOOLEAN)

    program, bound = generate._program, generate._bound
    generate("Return true or false:", max_new_tokens=4, do_sample=False)
    generate("Answer with a boolean:", max_new_tokens=4, do_sample=False)

    assert generate._program is program
    assert generate._bound is bound


def test_whitespace_inside_a_string_survives_decoding():
    """Byte-level BPE plus tokenizer cleanup can rewrite spaces; the constrained bytes must win."""
    model, tokenizer = _load_gpt2()
    adapter = maskforge.from_transformers(model, tokenizer)
    const = "a b  c"
    generate = maskforge.Generator(adapter, json.dumps({"const": const}))
    assert generate("Emit it:", max_new_tokens=32, do_sample=False) == const


def test_a_unicode_escape_decodes_to_the_exact_value():
    model, tokenizer = _load_gpt2()
    adapter = maskforge.from_transformers(model, tokenizer)
    const = "caf\u00e9"
    generate = maskforge.Generator(adapter, json.dumps({"const": const}))
    assert generate("Emit it:", max_new_tokens=32, do_sample=False) == const


def test_unequal_batch_completion_lengths_decode_independently():
    """A row that finishes early is padded; padding must not leak into its value."""
    model, tokenizer = _load_gpt2()
    adapter = maskforge.from_transformers(model, tokenizer)
    schema = json.dumps({"enum": ["a", "a-much-longer-value"]})
    generate = maskforge.Generator(adapter, schema)
    results = generate(["Pick:", "Pick one:", "Choose:"], max_new_tokens=32, do_sample=False)
    assert len(results) == 3
    assert all(r in ("a", "a-much-longer-value") for r in results)


def test_a_value_completing_exactly_at_max_new_tokens_is_returned():
    """No off-by-one: a value that finishes on the final permitted step still parses."""
    model, tokenizer = _load_gpt2()
    adapter = maskforge.from_transformers(model, tokenizer)
    generate = maskforge.Generator(adapter, json.dumps({"const": True}))

    exact = None
    for budget in range(1, 12):
        try:
            if generate("Answer:", max_new_tokens=budget, do_sample=False) is True:
                exact = budget
                break
        except maskforge.GenerationError:
            continue
    assert exact is not None, "`true` never completed within 11 tokens"
    assert generate("Answer:", max_new_tokens=exact, do_sample=False) is True


def test_an_incomplete_value_raises_a_maskforge_error_not_a_json_error():
    model, tokenizer = _load_gpt2()
    adapter = maskforge.from_transformers(model, tokenizer)
    schema = json.dumps({"const": "a-deliberately-long-constant-value-that-cannot-fit"})
    generate = maskforge.Generator(adapter, schema)
    with pytest.raises(maskforge.GenerationError):
        generate("Emit it:", max_new_tokens=2, do_sample=False)
