<div align="center">

# MaskForge

**Schema-guided token masks for reliable structured generation in Rust and Python.**

[![PyPI](https://img.shields.io/pypi/v/maskforge.svg)](https://pypi.org/project/maskforge/)
[![crates.io](https://img.shields.io/crates/v/maskforge-core.svg)](https://crates.io/crates/maskforge-core)
[![CI](https://github.com/dhanavanthesh/maskforge/actions/workflows/tests.yml/badge.svg)](https://github.com/dhanavanthesh/maskforge/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

Ask a language model for JSON and it will usually give you JSON. Usually. A trailing comma, a field
it invented, a string where a number belongs. The common fix is to generate, validate, and retry.
MaskForge skips the retry: at every decoding step it removes every token that would break the
schema, so the model can only ever choose from tokens that keep the document valid.

```python
generate = maskforge.Generator(maskforge.from_transformers(model, tokenizer), Invoice)
invoice = generate("Extract the invoice:")   # already an Invoice, already valid
```

This is an **alpha release**: the compiler and correctness surface are well tested, the high-level
API below is the supported surface, and raw IR/index handles in `maskforge.low_level` change more
freely. Pin an exact version and read [what's not finished yet](#whats-not-finished-yet) first.

## The idea

A JSON Schema compiles once into a reusable executable. Each schema node routes to one of two
mechanisms:

- **Regular** structure, most of JSON Schema, becomes a byte-level DFA: fixed states, no memory
  beyond the current position.
- **Structured** state becomes a small pushdown matcher with a typed frame stack and an undo log,
  so it can hold real memory: a seen-value set for `uniqueItems`, an open-property list for
  `unevaluatedProperties`, a counter for `contains`. Ordinary schemas never pay for this; only the
  nodes that need it route here.

Either way the caller sees one API: compile, bind to a vocabulary, start a session, and pull a
packed allowed-token bitmask before every sampling step. For prefix $p$ and candidate token $t$,
the mask is exactly:

```math
\text{mask}(p, t) \iff \text{ok}\big(\text{advance}(p, t)\big)
```

`advance` is atomic: it commits every byte of $t$, or none of them. That is the entire contract,
which is why nothing needs a second validation pass afterward. Masks themselves are computed by
walking the vocabulary as a trie *jointly* with the automaton, so a shared token prefix is parsed
once and a dead subtree is pruned whole. Full mechanics and the formal routing rules:
[`docs/architecture.md`](docs/architecture.md).

## What only MaskForge bothers with

Most of JSON Schema compiles to a finite automaton. These keywords do not, and each one runs in the
real decoding loop here, not as a post-hoc validation pass:

| Keyword | Why it is hard |
| --- | --- |
| `unevaluatedProperties` / `unevaluatedItems` | Annotations must flow across in-place applicators |
| `$dynamicRef` / `$dynamicAnchor` | The target subschema depends on the dynamic scope |
| `uniqueItems` | Unbounded value memory, not finite state |
| `contains` with `minContains` / `maxContains` | Counting while the outcome is still undecided |
| `dependentSchemas` | A subschema activates part-way through an object |
| `oneOf` | Kept as true exclusive-or, not weakened to `anyOf` |
| `multipleOf` | A residue automaton, on decimals as well as integers |

Speed numbers are deliberately not published here: a throughput comparison only means something
between engines solving the same problem on the same schema, and that is a benchmark worth doing
properly or not at all.

## Try it

### Python

```bash
pip install maskforge                  # core: compile a schema, drive a session, get masks
pip install "maskforge[transformers]"  # adds the CPU Hugging Face generation integration
pip install "maskforge[pydantic]"      # constrain to a Python type
```

Installing a supported wheel does not require Rust, Cargo, maturin, or a C/C++ compiler.

Constrain a Hugging Face CPU model to a Pydantic type. The schema is compiled and bound once, when
the generator is built, and reused by every call.

```python
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import maskforge


class Profile(BaseModel):
    name: str
    age: int


model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).cpu()

constrained = maskforge.from_transformers(model, tokenizer)
generate = maskforge.Generator(constrained, Profile)

profile = generate(                                             # -> Profile
    "Create a profile:", max_new_tokens=64, max_json_whitespace=1
)
profiles = generate(["Profile one:", "Profile two:"], max_new_tokens=64)  # -> list[Profile]
```

`Generator` also takes a raw JSON Schema, or any Pydantic-compatible type: `BaseModel`, dataclass,
`TypedDict`, `Enum`, `Literal`, a union, `list[T]`. `max_json_whitespace` is optional: it caps
consecutive insignificant whitespace so a small model does not spend its token budget on newlines.

[`examples/python/payment_instruction_showcase.py`](examples/python/payment_instruction_showcase.py)
runs one closed schema through `const`, `enum`, `pattern`, `contains`, `uniqueItems`, and `$ref` in a
real generation loop, against two different models:

```bash
MASKFORGE_MODEL_ID=Qwen/Qwen2.5-0.5B python examples/python/payment_instruction_showcase.py
```

```json
{
  "amountMinor": "500",
  "currency": "EUR",
  "iban": "DE1234",
  "allocation": { "pct": "60" },
  "approvals": ["treasury"],
  "rail": "wire"
}
```

The same script against GPT-2, a different model with a different tokenizer, returns an equally
valid object with different values. Both runs pass every invariant the example checks.

To drive masks yourself, without `transformers`, see [`examples/`](examples/README.md) for the raw
loop, external `$ref` resolution, and multi-session usage.

### Rust

```bash
cargo add maskforge-core
```

`maskforge-core` has no PyO3 dependency and links standalone. Python bindings live in a separate
crate (`maskforge-py`, not published to crates.io) and are consumed only through the Python package.

```rust
use std::sync::Arc;

use maskforge_core::{
    build_vocabulary, BindPolicy, CompiledVocabulary, Compiler, CompilerOptions,
};

let vocabulary =
    CompiledVocabulary::try_from(Arc::new(build_vocabulary(eos_token_id, token_map)?))?;

let options = CompilerOptions::default()
    .with_executable_cache_bytes(256 << 20)
    .with_trie_cache_bytes(256 << 20)
    .with_bind_policy(BindPolicy::Adaptive);

let compiler = Compiler::new(options);
let program = compiler.compile_json_schema(schema_text)?; // vocabulary-independent, reusable
let bound = program.bind(&vocabulary)?; // picks regular or structured internally
let mut session = bound.start_session()?;

let mut mask = vec![0u32; bound.mask_word_count()];
session.write_mask(&mut mask)?;
session.advance(token_id)?;
```

`CompilerOptions` is `#[non_exhaustive]`; build it from `default()` and the `with_*` setters rather
than a struct literal. Token `id` is allowed when bit `id % 32` of word `id / 32` is set in the
written mask. Bind policies and feature flags are covered in
[`docs/architecture.md`](docs/architecture.md); the full compiled-and-run version of this snippet is
`crates/maskforge-core/examples/basic_json_schema.rs`.

## What it won't let you do

`Generator` drives append-only sessions: exactly one token is committed per step, and nothing
rewinds. A generation mode that breaks that contract is rejected before any work is done, never
silently mis-masked.

| Mode | Status |
| --- | --- |
| Greedy and sampling (`do_sample`) | ✓ Supported |
| Batched prompts (one session per row) | ✓ Supported |
| Beam search (`num_beams > 1`) | ✗ Rejected |
| `num_return_sequences > 1` | ✗ Rejected |
| Assisted decoding (`assistant_model`) | ✗ Rejected |
| Prompt-lookup speculation (`prompt_lookup_num_tokens`) | ✗ Rejected |
| `assistant_early_exit`, `custom_generate` | ✗ Rejected |
| Encoder-decoder models | ✗ Rejected |
| Non-CPU, sharded, or meta-device placement | ✗ Rejected |

## What's actually covered

MaskForge follows the selected draft unless a limitation is listed below. Unsupported constructs
return a typed diagnostic; they are never silently dropped from the schema.

Status: **✓ supported**, **◐ partial or limited**, **✗ unsupported**.

| Keyword or feature | Status | Notes |
| --- | :---: | --- |
| `type` | ✓ | `string`, `number`, `integer`, `boolean`, `null`, `array`, and `object` |
| `enum`, `const` | ✓ | Scalar and composite JSON values; strings compare by decoded Unicode value |
| `minLength`, `maxLength` | ✓ | Counts decoded Unicode scalar values, not JSON source bytes |
| `pattern` | ◐ | Common ECMA-style patterns and bounded leading lookaheads; see limits below |
| `format` | ◐ | Annotation-only by default; opt-in assertions for `date`, `date-time`, `time`, `uuid`, `ipv4` |
| `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | ◐ | Complete-instance semantics; see numeric prefix-liveness below |
| `multipleOf` | ◐ | Exact complete-instance validation; see numeric prefix-liveness below |
| `items` | ✓ | Homogeneous arrays and the tail after `prefixItems` |
| `prefixItems` | ✓ | Draft 2020-12 tuple validation |
| `minItems`, `maxItems` | ✓ | Array length bounds |
| `contains`, `minContains`, `maxContains` | ✓ | Incremental contains counters and bounded completion checks |
| `uniqueItems` | ✓ | Semantic equality for strings, numbers, arrays, and objects |
| `properties`, `required` | ✓ | Required and optional properties |
| `additionalProperties` | ✓ | Boolean and schema forms; omission follows the selected draft/profile |
| `propertyNames`, `patternProperties` | ✓ | Decoded keys, including overlapping patterns |
| `minProperties`, `maxProperties` | ✓ | Object property-count bounds |
| `dependentRequired`, `dependentSchemas` | ✓ | Draft 2019-09 and 2020-12 forms |
| `dependencies` | ✓ | Legacy draft-04 through draft-07 form |
| `unevaluatedProperties` | ✓ | Accumulates successful annotations across adjacent applicators |
| `unevaluatedItems` | ◐ | Complete-instance semantics; see prefix-liveness below |
| `allOf`, `anyOf`, `oneOf`, `not` | ✓ | `oneOf` retains exactly-one semantics; not treated as `anyOf` |
| `if`, `then`, `else` | ◐ | Complete-instance semantics; see cross-property prefix-liveness below |
| `$defs`, `$ref`, `$anchor` | ✓ | Internal references; callers may supply external resources |
| `$dynamicRef`, `$dynamicAnchor` | ✓ | Draft 2020-12 dynamic scope with bounded recursive execution |
| `{}` and open object schemas | ✓ | Arbitrary JSON or object properties, subject to the selected profile |
| Legacy array-form `items`, `additionalItems` | ✗ | Use Draft 2020-12 `prefixItems` plus `items` |
| `$recursiveRef`, `$recursiveAnchor` | ✗ | Use Draft 2020-12 `$dynamicRef` and `$dynamicAnchor` |
| Full Format-Assertion vocabulary | ✗ | Only the opt-in subset above is asserted |
| `contentEncoding`, `contentMediaType`, `contentSchema` validation | ✗ | Preserved as annotations; content is not independently decoded or validated |

`pattern`, `patternProperties`, and regex entry points reject numeric and named backreferences,
lookbehind, non-leading or nested lookahead, and malformed expressions. When compilation rejects a
schema, the diagnostic carries a stable error code, the keyword, and its JSON Pointer; unresolved
external references return an error instead of triggering a network request. Draft coverage and
diagnostic detail: [`docs/development/json-schema-support.md`](docs/development/json-schema-support.md).

## What's not finished yet

| Area | Detail |
| --- | --- |
| `unevaluatedItems: false` prefix-liveness | Not guaranteed across all token/chunk boundaries. Complete instances still validate, but an item separator can be admitted before the engine proves the next array position can receive an evaluation annotation, so generation can reach a dead end. Avoid this keyword when every admitted prefix must stay completable. |
| Conditional (`if`/`then`/`else`) prefix-liveness | Properties can be admitted while `if` is undecided, and the selected branch can later make the object unclosable. Complete instances validate correctly; prefer a preselected or discriminated branch for unattended generation. |
| Numeric prefix-liveness | Numeric bounds, `multipleOf`, and numeric `const`/`enum` are enforced on complete values, but some impossible decimal prefixes stay locally admissible until the number ends. Prefer a string `const`/`enum` for fixed numeric-looking choices. |
| `transformers` integration | CPU-only and decoder-only; other placements are rejected rather than silently mis-masked. |
| Performance claims | Measured numbers cover mask-application only; end-to-end throughput against a real model, and any comparison against other constrained-decoding engines, has not been benchmarked. Cost-by-keyword numbers are in [`docs/architecture.md`](docs/architecture.md). |

Schema compilation is separately bounded by `SchemaResourceLimits` (document size, resolved input
bytes, structured-graph size); exceeding a configured limit returns a typed diagnostic, never a
silent truncation.

## Versioning, contributing, license

`0.y.z` releases may change the public API without a major version bump; semantic versioning
guarantees begin at `1.0`. To contribute, see [`CONTRIBUTING.md`](CONTRIBUTING.md) and run
`cargo test -p maskforge-core` and
`uv run --project crates/maskforge-py pytest crates/maskforge-py/tests` before sending a change.

Apache-2.0. Portions of `maskforge-core` derive from
[outlines-core](https://github.com/dottxt-ai/outlines-core) (also Apache-2.0); see
[`crates/maskforge-core/PROVENANCE.md`](crates/maskforge-core/PROVENANCE.md) for the file-by-file
audit trail and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for third-party test-corpus
licensing.
