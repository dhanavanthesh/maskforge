"""The flows the README documents must actually work.

Each test mirrors one README snippet. The snippets use placeholder identifiers for the
vocabulary; here they are filled with a real one so the documented calls are executed.
"""

import json

import pytest

import maskforge

EOS = 2
TOKEN_BYTES = [b"true", b"false", None]


def _vocabulary():
    return maskforge.Vocabulary.from_id_ordered_tokens(TOKEN_BYTES, EOS)


def test_the_low_level_mask_snippet_runs():
    """The 'drive masks yourself, without transformers' snippet."""
    vocabulary = _vocabulary()
    program = maskforge.Compiler().compile({"type": "boolean"})
    bound = program.bind(vocabulary)
    session = bound.start_session()

    buffer = bytearray(4 * bound.mask_word_count)
    session.write_mask(buffer)
    assert any(buffer), "at least one token must be allowed at the start"
    session.advance(0)
    assert session.is_accepting


def test_the_cache_ownership_snippet_runs():
    """The Compiler cache-ownership snippet, with every documented keyword."""
    compiler = maskforge.Compiler(
        executable_cache_bytes=256 << 20,
        trie_cache_bytes=256 << 20,
        bind_policy="adaptive",
    )
    compiler.compile({"type": "boolean"}).bind(_vocabulary())
    assert compiler.cache_stats() is not None
    compiler.clear_caches()


def test_without_cache_is_documented_and_works():
    compiler = maskforge.Compiler.without_cache()
    bound = compiler.compile({"type": "boolean"}).bind(_vocabulary())
    assert bound.mask_word_count >= 1


def test_every_documented_output_type_form_is_accepted():
    """The README promises a schema string, a dict, and Pydantic-compatible types."""
    compiler = maskforge.Compiler()
    vocabulary = _vocabulary()
    for output_type in (json.dumps({"type": "boolean"}), {"type": "boolean"}, bool):
        program = compiler.compile(output_type)
        assert program.bind(vocabulary).mask_word_count >= 1


def test_the_mask_layout_documented_for_rust_matches_python():
    """Token `id` is allowed when bit `id % 32` of word `id / 32` is set."""
    vocabulary = _vocabulary()
    bound = maskforge.Compiler().compile({"const": True}).bind(vocabulary)
    session = bound.start_session()
    buffer = bytearray(4 * bound.mask_word_count)
    session.write_mask(buffer)

    words = [
        int.from_bytes(buffer[i * 4 : i * 4 + 4], "little")
        for i in range(bound.mask_word_count)
    ]
    allowed = [i for i in range(bound.mask_vocab_size) if words[i // 32] >> (i % 32) & 1]
    assert allowed == [0], "only the `true` token may start a `const: true` value"


def test_low_level_cache_controls_are_where_the_readme_says():
    assert callable(maskforge.low_level.executable_cache_stats)
    assert callable(maskforge.low_level.clear_executable_cache)


@pytest.mark.parametrize("policy", ["adaptive", "eager", "lazy"])
def test_every_bind_policy_the_readme_lists_exists(policy):
    compiler = maskforge.Compiler(bind_policy=policy)
    assert compiler.compile({"type": "boolean"}).bind(_vocabulary()).mask_word_count >= 1
