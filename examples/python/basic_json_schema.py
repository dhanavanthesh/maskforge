"""Compile a schema, drive one generation sequence, show an illegal-token failure. Dependencies: pip install maskforge[transformers]"""

import json

from transformers import AutoTokenizer

import maskforge

MODEL_ID = "gpt2"


def bit_set(mask, token_id):
    return mask[4 * (token_id // 32) + (token_id % 32) // 8] & (1 << (token_id % 8)) != 0


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    vocabulary = maskforge.tokenizers.from_transformers(tokenizer)

    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name", "age"],
        }
    )
    compiler = maskforge.Compiler()
    session = compiler.compile(schema).bind(vocabulary).start_session()

    # A real LLM samples one legal token at a time; here we replay a known-valid prefix instead.
    for token_id in tokenizer.encode('{"name":"a', add_special_tokens=False):
        mask = bytearray(4 * session.mask_word_count)
        session.write_mask(mask)
        assert bit_set(mask, token_id), f"token {token_id} unexpectedly rejected"
        session.advance(token_id)

    # An unescaped control character is illegal inside a JSON string; raises MaskforgeError.
    illegal_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    try:
        session.advance(illegal_id)
    except maskforge.MaskforgeError as exc:
        print(f"rejected as expected: code={exc.code} stage={exc.stage} message={exc.message}")
    else:
        raise AssertionError("expected an IllegalToken error")


if __name__ == "__main__":
    main()
