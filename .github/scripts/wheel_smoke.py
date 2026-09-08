"""End-to-end smoke test for an installed maskforge wheel.

Importing proves almost nothing: it does not prove the extension carries tokenizer processing,
that a schema compiles, or that masks and EOS behave. This runs the whole path a caller uses:

    import -> tokenizer JSON -> compile -> bind -> session -> mask -> advance -> EOS

Uses no optional dependency, no network and no cached model, so it runs on every platform.
"""

import json
import sys

import maskforge
from maskforge import low_level

EOS = 2

# A self-contained fast-tokenizer document, so the native ingestion path is exercised
# without transformers or a downloaded model.
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
            "vocab": {"true": 0, "false": 1, "<eos>": EOS},
            "unk_token": "<eos>",
        },
    }
)


def check(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"wheel smoke test failed: {message}")


def main() -> None:
    print(f"maskforge {maskforge.__version__} on {sys.version.split()[0]} ({sys.platform})")

    # No optional dependency may be dragged in by importing the package.
    leaked = [m for m in ("torch", "transformers", "pydantic", "numpy") if m in sys.modules]
    check(not leaked, f"importing maskforge loaded optional dependencies {leaked}")

    # The released build must carry tokenizer-processing.
    check(
        hasattr(low_level.PyVocabulary, "from_tokenizer_json"),
        "wheel lacks tokenizer-processing (from_transformers would fail at runtime)",
    )

    raw = low_level.PyVocabulary.from_tokenizer_json(TOKENIZER_JSON, EOS, None)
    vocabulary = maskforge.Vocabulary(
        maskforge._native.CompiledVocabulary.from_vocabulary(raw)
    )

    bound = maskforge.Compiler().compile({"type": "boolean"}).bind(vocabulary)
    session = bound.start_session()

    buffer = bytearray(4 * bound.mask_word_count)
    session.write_mask(buffer)
    words = [
        int.from_bytes(buffer[i * 4 : i * 4 + 4], "little")
        for i in range(bound.mask_word_count)
    ]
    allowed = {i for i in range(bound.mask_vocab_size) if words[i // 32] >> (i % 32) & 1}
    check(allowed, "the start mask allowed no token at all")
    check(EOS not in allowed, "EOS was allowed before any value was generated")

    token = min(allowed)
    session.advance(token)
    check(session.is_accepting, f"session did not accept after committing token {token}")

    session.write_mask(buffer)
    check(session.eos_token_id == EOS, "session reported the wrong EOS id")

    session.reset()
    check(not session.is_stopped, "reset left the session stopped")

    print(f"end-to-end OK: allowed={sorted(allowed)} committed={token}")


if __name__ == "__main__":
    main()
