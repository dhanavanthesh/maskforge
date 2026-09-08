# maskforge

Python bindings and Hugging Face integration for MaskForge.

The package compiles a JSON Schema or Pydantic type once, binds it to exact tokenizer bytes, and
maintains one constrained-decoding session per generated sequence. The native extension is backed by
`maskforge-core`.

See the [MaskForge README](https://github.com/dhanavanthesh/maskforge) for installation, verified
model output, current limitations, and the full project overview. Detailed references:

- [JSON Schema support](https://github.com/dhanavanthesh/maskforge/blob/main/docs/development/json-schema-support.md)
- [Architecture](https://github.com/dhanavanthesh/maskforge/blob/main/docs/architecture.md)
- [Development roadmap](https://github.com/dhanavanthesh/maskforge/blob/main/docs/development/to-do.md)

## Install

```bash
pip install maskforge
pip install "maskforge[transformers]"
pip install "maskforge[pydantic]"
```

## Example

```python
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer

import maskforge


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    age: int


model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).cpu()

constrained = maskforge.from_transformers(model, tokenizer)
generate = maskforge.Generator(constrained, Profile)
profile = generate("Create a profile:", max_new_tokens=128, max_json_whitespace=1)
```

`Generator` supports greedy decoding, sampling, and batched prompts on CPU. Modes that need session
branching or rollback are rejected explicitly. The lower-level `Compiler`, `Vocabulary`,
`BoundSchema`, and `Session` APIs can be used with another generation loop.

## License

Apache-2.0.
