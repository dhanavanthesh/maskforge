<div align="center">

# MaskForge

**Schema-guided token masks for reliable structured generation in Rust and Python.**

[![PyPI](https://img.shields.io/pypi/v/maskforge.svg)](https://pypi.org/project/maskforge/)
[![crates.io](https://img.shields.io/crates/v/maskforge-core.svg)](https://crates.io/crates/maskforge-core)
[![CI](https://github.com/dhanavanthesh/maskforge/actions/workflows/tests.yml/badge.svg)](https://github.com/dhanavanthesh/maskforge/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

A language model can only emit a token that keeps the document valid. Not checked afterwards, not
retried until it parses: constrained while it is being generated.

```python
generate = maskforge.Generator(maskforge.from_transformers(model, tokenizer), Invoice)
invoice = generate("Extract the invoice:")   # already an Invoice, already valid
```

## How it works

A JSON Schema compiles once into a reusable executable. At every decoding step the engine emits a
packed allowed-token bitmask over the tokenizer vocabulary, so the sampler never sees a token that
would break the schema.

Two backends, chosen by the compiler rather than by the caller:

- **Regular schemas** lower to a byte-level DFA, built with product, shuffle and Kleene state
  elimination over a hash-consed arena IR.
- **Everything else** - nesting, order-independent objects, value memory - runs on an incremental
  pushdown matcher with a typed frame stack and a transactional undo log.

Masks are computed by walking the vocabulary as a compressed-sparse-row trie *jointly* with the
automaton, so a shared token prefix is parsed once and a dead subtree is pruned whole.

## Beyond the regular fragment

Most of JSON Schema compiles to a finite automaton. These keywords do not, and each one is
supported here:

| Keyword | Why it is hard |
| --- | --- |
| `unevaluatedProperties` / `unevaluatedItems` | annotations must flow across in-place applicators |
| `$dynamicRef` / `$dynamicAnchor` | the target subschema depends on the dynamic scope |
| `uniqueItems` | unbounded value memory, not finite state |
| `contains` with `minContains` / `maxContains` | counting while the outcome is still undecided |
| `dependentSchemas` | a subschema activates part-way through an object |
| `oneOf` | kept as true exclusive-or, not weakened to `anyOf` |
| `multipleOf` | a residue automaton, on decimals as well as integers |

Every one of these runs in the real decoding loop, not as a post-hoc validation pass.

Speed numbers are deliberately not published here: a throughput comparison only means something
between engines solving the same problem on the same schema, and that is a benchmark worth doing
properly or not at all.

This is an **alpha release**. The core compiler and correctness surface are well tested. The
high-level API below is the supported surface; raw IR and index handles live in
`maskforge.low_level` and change more freely. Pin an exact version, and read the
[known limitations](#known-limitations) before relying on it unattended.

## Installation

### Python

```bash
pip install maskforge                  # core: compile a schema, drive a session, get masks
pip install "maskforge[transformers]"  # adds the CPU Hugging Face generation integration
pip install "maskforge[pydantic]"      # constrain to a Python type
```

Installing a supported wheel does not require Rust, Cargo, maturin, or a C/C++ compiler.

### Rust

```bash
cargo add maskforge-core
```

`maskforge-core` has no PyO3 dependency and links standalone. Python bindings live in a
separate crate (`maskforge-py`, not published to crates.io) and are consumed only through
the Python package above.

## Python quickstart

Constrain a Hugging Face CPU model to a Pydantic type. The schema is compiled and bound once,
when the generator is built, and reused by every call.

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

`Generator` also accepts a JSON Schema string or dict, and any Pydantic-compatible type
(`BaseModel`, dataclass, `TypedDict`, `Enum`, `Literal`, a union, `list[T]`).
`max_json_whitespace` is an optional generation policy: it caps consecutive insignificant JSON
whitespace outside strings, preventing small models from spending their token budget on legal but
unproductive newlines. Omitting it preserves the full JSON whitespace language.

Small base models may still choose verbose but valid spellings such as `\u0061` instead of `a`.
MaskForge does not rewrite or forbid those spellings because doing so would change the accepted JSON
language. Size `max_new_tokens` for the tokenizer and schema, and prefer finite schemas for unattended
generation; recursive schemas can always choose another recursive branch instead of terminating.

### Verified payment generation

[`examples/python/payment_instruction_showcase.py`](examples/python/payment_instruction_showcase.py)
uses one finite, closed schema and exercises `const`, `enum`, `pattern`, `items`, `contains`,
`uniqueItems`, `minItems`, `maxItems`, `additionalProperties: false`, and `$defs`/`$ref` in a
real token-generation loop.

```bash
MASKFORGE_MODEL_ID=Qwen/Qwen2.5-0.5B \
  python examples/python/payment_instruction_showcase.py
```

Verified Qwen2.5-0.5B output:

```json
{
  "amountMinor": "500",
  "currency": "EUR",
  "iban": "DE1234",
  "allocation": {
    "pct": "60"
  },
  "approvals": [
    "treasury"
  ],
  "rail": "wire"
}
```

Verified GPT-2 output from the same schema and extension:

```json
{
  "amountMinor": "500",
  "currency": "USD",
  "approvals": [
    "treasury",
    "risk",
    "ops"
  ],
  "rail": "wire",
  "allocation": {
    "pct": "60"
  },
  "iban": "WJ9999"
}
```

Both runs returned parsed Python objects and passed the example's invariant assertions.

To drive masks yourself, without `transformers`:

```python
import maskforge

vocabulary = maskforge.Vocabulary.from_id_ordered_tokens(
    token_bytes,              # list[bytes | None], index == token id
    eos_token_id=eos_id,
)
program = maskforge.Compiler().compile({"type": "object", "required": ["ok"]})
bound = program.bind(vocabulary)
session = bound.start_session()

buffer = bytearray(4 * bound.mask_word_count)
session.write_mask(buffer)     # packed little-endian allowed-token bitmask
session.advance(chosen_token_id)
```

See `examples/python/` for the `transformers` integration, external-`$ref` resolution, and
multi-session usage.

## Supported and rejected generation modes

`Generator` drives append-only sessions: exactly one token is committed per step, and nothing
rewinds. Modes that break that contract are rejected before any work is done, never silently
mis-masked.

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

The generator never mutates the tokenizer you pass it: batch prompts are encoded
independently and left-padded into a local tensor, so a tokenizer shared across threads is
never reconfigured underneath them.

## Rust quickstart

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

`CompilerOptions` is `#[non_exhaustive]`; build it from `default()` and the `with_*` setters
rather than a struct literal. The bind policy picks how a regular schema binds:
`Eager` always builds the packed table (fastest per-mask query, higher bind cost), `Lazy`
always uses trie-backed rows cached on first query, and `Adaptive` (the default) takes the
packed table when it fits the budget.

**Mask layout.** `write_mask` fills `mask_word_count() == ceil(mask_vocab_size / 32)` `u32`
words. Token `id` is allowed when bit `id % 32` of word `id / 32` is set. `write_mask_le_bytes`
writes the same bits as `mask_byte_count()` little-endian bytes, which is the representation
the Python and tensor adapters consume; both are correct on big-endian targets.

**Feature flags.** `tokenizer-processing` adds `Vocabulary::from_tokenizer_json`, decoding a
local fast-tokenizer JSON with no network dependency; it is what the published Python wheel
ships. `huggingface-hub` builds on it and adds `Vocabulary::from_pretrained`, pulling from the
Hub over the network. Neither is on by default.

`Compiler` owns configurable, clearable executable and trie caches (`Compiler::without_cache()`
for deterministic tests); `SchemaProgram` is immutable and reusable across every vocabulary
that needs it. See `crates/maskforge-core/examples/basic_json_schema.rs` for the full compiled
and run version of this snippet.

## Architecture

Four things are cached and reused independently, so the expensive work is paid once:

| Stage | Produces | Reusable across |
| --- | --- | --- |
| **Frontend** (`frontend::schema_to_ir`) | frozen, vocabulary-neutral `SchemaIR` | - |
| **Route + compile** (`ExecutableSchema::compile`) | a byte automaton, a structured program, or a hybrid of both | every vocabulary |
| **Vocabulary preparation** | validated, canonicalized `CompiledVocabulary` | every schema |
| **Bind** | immutable `BoundSchema`; a shared vocabulary trie | every generation sequence |

A `Session` is then the only mutable object: it holds the position of exactly one sequence,
producing a packed mask and consuming committed tokens. Concurrent sequences get separate
sessions over one shared `BoundSchema`.

`ExecutableSchema::compile` performs the routing. `compile::compile_ir` compiles regular IR
only, and is one of the three arms it dispatches to.

The Python bindings (`maskforge-py`) wrap the same runtime behind PyO3, adding the
`Compiler`/`Generator` facade and tensor adapters (NumPy, PyTorch) for applying the packed
mask to logits.

**Full detail: [`docs/architecture.md`](docs/architecture.md)**: pipelines, cache identities,
bind policies, structured rollback, mask layout, memory ownership and error flow.

## Supported JSON Schema

MaskForge follows the selected draft unless a limitation is listed below. Unsupported constructs
return a typed diagnostic; they are not silently removed from the schema.

Status: **✓ supported**, **◐ partial or limited**, **✗ unsupported**.

| Keyword or feature | Status | Notes |
| --- | :---: | --- |
| `type` | ✓ | `string`, `number`, `integer`, `boolean`, `null`, `array`, and `object` |
| `enum`, `const` | ✓ | Scalar and composite JSON values; strings compare by decoded Unicode value |
| `minLength`, `maxLength` | ✓ | Counts decoded Unicode scalar values, not JSON source bytes |
| `pattern` | ◐ | Common ECMA-style patterns and bounded leading lookaheads; limits below |
| `format` | ◐ | Annotation-only by default; opt-in assertions for `date`, `date-time`, `time`, `uuid`, and `ipv4` |
| `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | ◐ | Complete-instance semantics; numeric prefix-liveness caveat below |
| `multipleOf` | ◐ | Exact complete-instance validation; numeric prefix-liveness caveat below |
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
| `unevaluatedItems` | ◐ | Complete-instance semantics; prefix-liveness caveat below |
| `allOf`, `anyOf`, `oneOf`, `not` | ✓ | `oneOf` retains exactly-one semantics; it is not treated as `anyOf` |
| `if`, `then`, `else` | ◐ | Complete-instance semantics; cross-property prefix-liveness caveat below |
| `$defs`, `$ref`, `$anchor` | ✓ | Internal references; callers may supply external resources |
| `$dynamicRef`, `$dynamicAnchor` | ✓ | Draft 2020-12 dynamic scope with bounded recursive execution |
| `{}` and open object schemas | ✓ | Arbitrary JSON or object properties, subject to the selected profile |
| Legacy array-form `items`, `additionalItems` | ✗ | Use Draft 2020-12 `prefixItems` plus `items` |
| `$recursiveRef`, `$recursiveAnchor` | ✗ | Use Draft 2020-12 `$dynamicRef` and `$dynamicAnchor` |
| Full Format-Assertion vocabulary | ✗ | Only the opt-in subset listed above is asserted |
| `contentEncoding`, `contentMediaType`, `contentSchema` validation | ✗ | Preserved as annotations; content is not independently decoded or validated |

### Regular-expression limits

`pattern`, `patternProperties`, and regex entry points support ordinary regular constructs and a
bounded set of leading positive and negative lookaheads. The following are rejected:

- numeric and named backreferences
- lookbehind
- non-leading or nested lookahead
- incompatible ECMA group constructs
- malformed expressions, including unmatched groups or classes and trailing escapes

### Unsupported schema behavior

Compilation stops before generation when a schema uses an unsupported construct. The diagnostic
includes a stable error code, the keyword, and its JSON Pointer. Unsupported branches are never
dropped from `anyOf`, `allOf`, or another applicator, and unresolved external references return a
`ReferenceResolution` error instead of triggering a network request.

See [`docs/development/json-schema-support.md`](docs/development/json-schema-support.md) for draft
coverage, backend routing, and diagnostic details.

## Resource limits

Schema compilation is bounded: `SchemaResourceLimits` caps document size, total resolved
input bytes, and structured-graph size. A schema that would exceed a configured limit is
rejected with a typed diagnostic, not silently truncated.

## Cache ownership

In Rust, `Compiler::new(CompilerOptions { executable_cache_bytes, .. })` owns its own
schema-executable cache with a configurable budget; `Compiler::without_cache()` disables
caching entirely for deterministic tests. `Compiler::clear_caches()` /
`Compiler::cache_stats()` give explicit lifecycle control. Nothing requires the process-wide
default (`runtime::global_executable_cache()`).

In Python a `Compiler` owns both caches, sized and clearable per instance:

```python
compiler = maskforge.Compiler(
    executable_cache_bytes=256 << 20,
    trie_cache_bytes=256 << 20,
    bind_policy="adaptive",   # or "eager" / "lazy"
)
compiler.cache_stats()
compiler.clear_caches()
```

`maskforge.Compiler.without_cache()` caches nothing, for deterministic tests. The process-wide
cache used by the low-level `compile_*_with_vocabulary` functions is inspected and cleared with
`maskforge.low_level.executable_cache_stats()` / `maskforge.low_level.clear_executable_cache()`.
A server hosting many models can therefore release retained memory explicitly rather than
waiting on process-wide eviction.

## Thread/session safety

A compiled schema and a `BoundSchema` are immutable and safe to share across threads. Each
generation sequence needs its own `Session`; mutable per-sequence state is never shared.
Batched generation is supported and uses one session per row. `Generator` never writes to the
model or tokenizer it was given, so one adapter can back concurrent requests.

## Backend selection

Regular-vs-structured backend selection is internal to schema compilation
(`requires_structured_backend`); callers use one API (`Compiler.compile` in Python,
`Compiler::compile_json_schema` in Rust) regardless of which backend a given schema needs.

## Mask-generation cost by keyword

Most schema shapes use precomputed vocabulary buckets. Stateful positions may use the exact
transactional walk so duplicate values, annotations, and dynamic branches remain observable.
The regression corpus compares optimized masks with full-trie, record-scan, and direct-transition
routes at selected prefixes.

Representative exact-trie node counts, measured on a 50,257-token GPT-2 vocabulary:

| Position | Nodes walked | Fast path |
| --- | --- | --- |
| Object key under `unevaluatedProperties` + `allOf` + `if`/`then`/`else` + `dependentRequired` | ~107 | Yes |
| Object key under `additionalProperties: false` with the same siblings | ~99 | Yes |
| Enum, pattern, and nested-value string bodies | leaves the full trie | Yes |
| Strings inside an active canonical `uniqueItems` builder | ~97,930 | No |
| Keys constrained by `propertyNames` | ~18,600 | No |

These fallbacks are intentional:

- **`uniqueItems`**: the canonical-value builder must observe every byte to detect duplicate
  items, so no whole-bucket certificate can be admitted while it is active. Correct, not fast.
- **`propertyNames`**: pattern-constrained keys commonly use the exact walk because their decoded
  prefix, escape state, duplicate state, and applicable property patterns interact.

### Current limitation

Prefix-complete generation is not currently guaranteed for `unevaluatedItems: false` across all
token/chunk boundaries. Complete instances are still validated, but an item separator can be
admitted before the engine proves that the next array position can receive an evaluation
annotation; generation may then reach a state with no valid completion. Avoid this keyword when
every admitted generation prefix must remain completable.

Conditional schemas can also admit properties while `if` is undecided and later discover that
the selected `then` or `else` branch makes the object unclosable. Complete instances remain
validated correctly, but cross-property conditional generation is not yet guaranteed dead-end-free.
Prefer a preselected or discriminated branch for unattended generation.

Numeric bounds, integer type, numeric `const`/`enum`, and `multipleOf` are enforced on complete
values, but some impossible decimal prefixes can remain locally admissible until the number ends or
reaches a resource limit. Prefer a string `const`/`enum` for fixed numeric-looking choices in
unattended generation.

## Known limitations

- The `transformers` integration is CPU-only and decoder-only; other placements are rejected
  rather than silently mis-masked.
- Beam search and every assisted/speculative decoding mode are rejected (see the table above).
- Full Draft 2020-12 Format-Assertion vocabulary is not implemented; see above.
- `uniqueItems` and `propertyNames` positions take the exact walk (see the table above);
  schemas that spend most of their tokens in those positions will mask more slowly.
- `unevaluatedItems: false` does not yet carry a prefix-liveness guarantee across every tokenizer
  chunking. Treat successful schema compilation as complete-instance support, not proof that every
  admitted generation prefix remains completable.
- Cross-property `if`/`then`/`else` constraints can reach a non-accepting whitespace-only prefix;
  preselect the branch when generation must always remain completable.
- Numeric constraints, including a numeric `const`, can admit a non-completable decimal prefix; use
  a string `const` for a fixed numeric-looking value until residual-language pruning is complete.
- Measured numbers cover the mask-application path (see
  `integration/program6_processor_latency.py`); end-to-end generation throughput against a
  real model has not been benchmarked separately.
- No comparative performance claim is made against other constrained-decoding engines. A
  matched-expressiveness cross-engine evaluation has not been run, and coverage and
  correctness are the only claims made here.

## Versioning policy

`0.y.z` releases may change the public API without a major version bump. Semantic
versioning guarantees begin at `1.0`.

## Contributing

See `CONTRIBUTING.md`. Run `cargo test -p maskforge-core` and
`uv run --project crates/maskforge-py pytest crates/maskforge-py/tests` before sending a
change.

## License and provenance

Apache-2.0. Portions of `maskforge-core` derive from
[outlines-core](https://github.com/dottxt-ai/outlines-core) (also Apache-2.0); see
`crates/maskforge-core/PROVENANCE.md` for the file-by-file audit trail and
`THIRD_PARTY_NOTICES.md` for third-party test-corpus licensing.
