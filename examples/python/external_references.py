"""Resolve `$ref` against caller-supplied resources through the Compiler façade. Dependencies: pip install maskforge[transformers]"""

from transformers import AutoTokenizer

import maskforge

MODEL_ID = "gpt2"

SCHEMA = '{"allOf":[{"$ref":"a.json#value"},{"$ref":"b.json#value"}]}'
RESOURCES = {
    "https://example.com/a.json": '{"$anchor":"value","type":"integer"}',
    "https://example.com/b.json": '{"$anchor":"value","minimum":5}',
}


def bit_set(mask, token_id):
    return mask[4 * (token_id // 32) + (token_id % 32) // 8] & (1 << (token_id % 8)) != 0


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    vocabulary = maskforge.tokenizers.from_transformers(tokenizer)

    compiler = maskforge.Compiler()
    # MaskForge never fetches a $ref target itself: RESOURCES is caller-supplied, no network I/O.
    program = compiler.compile(SCHEMA, retrieval_uri="https://example.com/root.json", resources=RESOURCES)
    session = program.bind(vocabulary).start_session()

    mask = bytearray(4 * session.mask_word_count)
    session.write_mask(mask)
    five_id = tokenizer.encode("5", add_special_tokens=False)[0]
    assert bit_set(mask, five_id), "'5' unexpectedly rejected at the start of the value"
    print("the token for '5' is allowed at the start of the value")

    session.advance(five_id)
    print("finished after committing '5':", session.is_accepting)


if __name__ == "__main__":
    main()
