"""Constrain a real CPU Transformers model to a JSON Schema. Dependencies: pip install maskforge[transformers]"""

import json

from transformers import AutoModelForCausalLM, AutoTokenizer

import maskforge

MODEL_ID = "openai-community/gpt2"

SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    adapter = maskforge.from_transformers(model, tokenizer)
    generate = maskforge.Generator(adapter, SCHEMA)

    result = generate('{"ok":', max_new_tokens=20, do_sample=False)
    print(f"constrained result: {result!r}")


if __name__ == "__main__":
    main()
