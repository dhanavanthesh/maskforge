"""Constrain a real CPU Transformers model to a Pydantic type. Dependencies: pip install maskforge[transformers] pydantic"""

import pydantic
from transformers import AutoModelForCausalLM, AutoTokenizer

import maskforge

MODEL_ID = "openai-community/gpt2"


class Answer(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")  # keeps a small base model from wandering into unrelated fields
    ok: bool


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    adapter = maskforge.from_transformers(model, tokenizer)
    generate = maskforge.Generator(adapter, Answer)

    result = generate('{"ok":', max_new_tokens=20, do_sample=False)
    print(f"parsed into {type(result).__name__}: {result!r}")


if __name__ == "__main__":
    main()
