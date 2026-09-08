"""Drive MaskforgeLogitsProcessor from transformers model.generate(). Dependencies: pip install maskforge transformers torch"""

import json

from transformers import AutoModelForCausalLM, AutoTokenizer

import maskforge


def _byte_to_unicode_table():
    # The fixed GPT-2 byte-level BPE alphabet (Radford et al.).
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(
        range(ord("\xae"), ord("\xff") + 1)
    )
    cs = bs.copy()
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


# One unicode char of a GPT-2 token string back to the raw byte it represents.
_UNICODE_TO_BYTE = {v: k for k, v in _byte_to_unicode_table().items()}


def vocabulary_from_tokenizer(tokenizer):
    # MaskForge has no from_pretrained wrapper yet; eos_token_id comes straight from the tokenizer.
    eos_id = tokenizer.eos_token_id
    token_bytes = [
        None
        if token_id == eos_id
        else bytes(_UNICODE_TO_BYTE[ch] for ch in tok)
        for token_id, tok in enumerate(tokenizer.convert_ids_to_tokens(range(len(tokenizer))))
    ]
    return maskforge.low_level.PyVocabulary.from_id_ordered_tokens(
        token_bytes, eos_token_id=eos_id, logits_vocab_size=len(tokenizer)
    )


def main():
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    schema = json.dumps(
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
    )
    vocabulary = vocabulary_from_tokenizer(tokenizer)
    session = maskforge.low_level.ConstraintSession.from_json_schema(vocabulary, schema)
    processor = maskforge.low_level.MaskforgeLogitsProcessor(session)

    prompt = tokenizer("Respond with JSON: ", return_tensors="pt")
    output = model.generate(
        **prompt,
        max_new_tokens=20,
        logits_processor=[processor],
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0, prompt["input_ids"].shape[-1] :], skip_special_tokens=True)
    parsed = json.loads(text)  # the mask only allows a value matching the schema, so this always parses
    assert parsed.keys() == {"ok"} and isinstance(parsed["ok"], bool)
    print(f"generated: {text.strip()!r} -> parsed: {parsed!r}")


if __name__ == "__main__":
    main()
