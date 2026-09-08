"""Constrain a real CPU model's output to one of a fixed set of choices via an enum schema. Dependencies: pip install maskforge[transformers]"""

import json

from transformers import AutoModelForCausalLM, AutoTokenizer

import maskforge

MODEL_ID = "openai-community/gpt2"

SCHEMA = json.dumps({"enum": ["cat", "dog", "fish"]})


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    adapter = maskforge.from_transformers(model, tokenizer)
    generate = maskforge.Generator(adapter, SCHEMA)

    result = generate("My favorite animal is a", max_new_tokens=6, do_sample=False)
    print(f"chosen option: {result!r}")
    assert result in ("cat", "dog", "fish")


if __name__ == "__main__":
    main()
