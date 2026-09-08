# MaskForge examples

## Python

Requires `pip install maskforge` (add `numpy`/`torch` for the mask-adapter examples, `maskforge[transformers]` for the `transformers_*` ones, `pydantic` for the Pydantic ones).

- `python/basic_json_schema.py` — compile a schema, walk one generation sequence, show one invalid-token failure.
- `python/external_references.py` — resolve `$ref` against caller-supplied resources.
- `python/multiple_sessions.py` — reuse one compiled artifact across independent sequences.
- `python/pydantic_schema.py` — constrain to a Pydantic model's JSON Schema.
- `python/low_level/numpy_mask.py` / `python/low_level/torch_mask.py` - apply the packed mask to a logits array.
- `python/low_level/transformers_logits_processor.py` - drive `MaskforgeLogitsProcessor` from `model.generate()` with a real GPT-2 model (legacy single-sequence path).
- `python/transformers_json_schema.py` — constrain a real CPU model to a JSON Schema via `maskforge.from_transformers` + `maskforge.Generator`.
- `python/transformers_pydantic.py` — same, constrained to a Pydantic type.
- `python/transformers_batch.py` — same, batched across several prompts at once.
- `python/transformers_choice.py` — constrain output to one of a fixed set of choices via an enum schema.
- `python/payment_instruction_showcase.py` — a finite closed payment schema verified with cached Qwen2.5-0.5B and GPT-2.

## Rust

Requires `maskforge-core` as a path or crates.io dependency. The examples live inside the
crate itself, at `crates/maskforge-core/examples/`, so they ship in the published `.crate`
and are auto-discovered by Cargo — no `[[example]]` entries needed.

- `basic_json_schema.rs` — compile a schema, build a synthetic vocabulary, walk a mask.
- `external_references.rs` — resolve `$ref` against caller-supplied resources.
- `cached_multi_vocabulary.rs` — compile one schema once, bind it to two vocabularies.

Run with `cargo run --example basic_json_schema -p maskforge-core` from the repository root.
