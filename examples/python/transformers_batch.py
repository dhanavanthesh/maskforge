"""Batch-generate constrained output for several prompts at once. Dependencies: pip install maskforge[transformers]"""

import json

from transformers import AutoModelForCausalLM, AutoTokenizer

import maskforge

MODEL_ID = "openai-community/gpt2"
SCHEMA = json.dumps({"type": "boolean"})
PROMPTS = ["Return true or false:", "Answer with a boolean:", "True or false, answer:"]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    adapter = maskforge.from_transformers(model, tokenizer)
    generate = maskforge.Generator(adapter, SCHEMA)

    results = generate(PROMPTS, max_new_tokens=8, do_sample=False)
    for prompt, result in zip(PROMPTS, results):
        print(f"{prompt!r} -> {result!r}")


if __name__ == "__main__":
    main()
