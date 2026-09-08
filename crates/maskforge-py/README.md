# maskforge

**Schema-guided token masks for reliable structured generation.**

[![PyPI](https://img.shields.io/pypi/v/maskforge.svg)](https://pypi.org/project/maskforge/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A model can only emit a token that keeps the document valid. Constrained while generating, not
validated afterwards.

```bash
pip install maskforge
pip install "maskforge[transformers]"   # CPU Hugging Face generation
pip install "maskforge[pydantic]"       # constrain to a Python type
```

The wheel ships a compiled extension; no Rust toolchain needed.

## Example

```python
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer

import maskforge


class Profile(BaseModel):
    # Pydantic objects are open by default; extra="forbid" makes the constraint as strict as the type looks.
    model_config = ConfigDict(extra="forbid")

    name: str
    age: int


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").cpu()

generate = maskforge.Generator(maskforge.from_transformers(model, tokenizer), Profile)
profile = generate("Create a profile:", max_new_tokens=128)   # -> Profile
```

Also accepts a JSON Schema string or dict, a dataclass, `TypedDict`, `Enum`, `Literal`, a union, or
`list[T]`. The schema compiles and binds once when the generator is built.

## Decoding modes

Sessions are append-only: one token per step, no rewind. Modes that break that are rejected up
front, never silently mis-masked.

| Mode | Status |
| --- | --- |
| Greedy, sampling, batched prompts | supported |
| Beam search, `num_return_sequences > 1` | rejected |
| Assisted / speculative decoding | rejected |
| Encoder-decoder, non-CPU placement | rejected |

## Notes

On Python 3.11 use `typing_extensions.TypedDict`; Pydantic needs the backport before 3.12.

Lower-level `Compiler`, `Vocabulary`, `BoundSchema` and `Session` are available for your own
generation loop, with NumPy and PyTorch adapters for applying the mask to logits.

Limitations are documented, not left to be discovered:
[known limitations](https://github.com/dhanavanthesh/maskforge#known-limitations).

## More

- [Repository](https://github.com/dhanavanthesh/maskforge)
- [Keyword support](https://github.com/dhanavanthesh/maskforge/blob/main/docs/development/json-schema-support.md)
- Rust crate: [`maskforge-core`](https://crates.io/crates/maskforge-core)

## License

Apache-2.0.
