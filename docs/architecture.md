# Architecture

MaskForge compiles a JSON Schema once, binds it to a tokenizer vocabulary once, and creates one
mutable session per generated sequence. Everything upstream of a session is immutable and shared,
so the expensive work (parsing, routing, vocabulary preparation) is paid once and reused across
every sequence that follows.

## Public object model

| Object | State | Purpose |
| --- | --- | --- |
| `Compiler` | Shared cache state | Compiles schemas and owns bounded executable and trie caches |
| `SchemaProgram` | Immutable | Vocabulary-independent compiled schema |
| `CompiledVocabulary` | Immutable | Validated token IDs and exact raw token bytes |
| `BoundSchema` | Immutable | One schema paired with one vocabulary |
| `Session` | Mutable | Position in one generation sequence |

`SchemaProgram`, `CompiledVocabulary`, and `BoundSchema` are cheap handles over shared storage. A
session must not be shared between sequences.

```text
schema -> Compiler -> SchemaProgram
tokens -> CompiledVocabulary
SchemaProgram + CompiledVocabulary -> BoundSchema
BoundSchema -> Session per sequence
```

## Compilation

```text
JSON Schema text
  -> parse and resource registration
  -> reference resolution
  -> immutable schema IR
  -> route selection
  -> executable schema
```

Unsupported validation constructs become typed diagnostics. Applicator branches are not silently
dropped. External references must be supplied by the caller; MaskForge performs no network fetch.

Routes are selected per schema node:

| Route | Use |
| --- | --- |
| Regular | The subtree can be represented by a byte automaton |
| Structured | Runtime state is needed for objects, arrays, annotations, uniqueness, or references |
| Hybrid | Structured control with reusable regular descendants |

Routing is internal. All routes use the same public API.

## Vocabulary and binding

Vocabulary preparation validates token IDs, records each token's exact raw bytes, builds a reverse
index, and computes a stable fingerprint. A trie shares work across tokens with common byte prefixes.
Tokenizer decoding is never used as a substitute for these raw bytes; that matters for byte-level
BPE vocabularies, aliases, UTF-8 prefixes, and JSON escapes.

Binding combines a schema executable with a compiled vocabulary. A regular schema chooses how its
mask rows are materialized:

| Bind policy | Behavior |
| --- | --- |
| `Eager` | Always builds the packed table. Fastest per-mask query, higher bind cost. |
| `Lazy` | Always uses trie-backed rows, cached on first query. |
| `Adaptive` (default) | Takes the packed table when it fits the configured budget, falls back to `Lazy` otherwise. |

A structured schema instead retains its plan and creates mutable state only when a session starts.

Two Cargo features gate vocabulary sources, both off by default:

| Feature | Adds | Notes |
| --- | --- | --- |
| `tokenizer-processing` | `Vocabulary::from_tokenizer_json` | Decodes a local fast-tokenizer JSON, no network dependency; what the published Python wheel ships |
| `huggingface-hub` | `Vocabulary::from_pretrained`, built on `tokenizer-processing` | Pulls a tokenizer from the Hub over the network |

## Generation contract

Each decoding step has two operations:

```text
session.write_mask(buffer)
model selects an allowed token
session.advance(token_id)
```

The mask is a packed little-endian bitset: token $t$ is allowed exactly when bit $t$ is set. EOS is
set only when the current state is accepting. For prefix $p$ and candidate token $t$:

```math
\text{mask}(p, t) \iff \text{ok}\big(\text{advance}(p, t)\big)
```

`advance` is atomic: it commits every raw byte of $t$, or restores the exact prior state and returns
`IllegalToken`. Calling `write_mask` repeatedly must never change committed state.

This contract proves only that $t$ keeps the *next* step legal. It does not by itself prove that the
resulting prefix can still reach a complete document, prefix liveness, which is a separate property.
Current liveness limits are listed in the README's
["What's not finished yet"](../README.md#whats-not-finished-yet) and in
[`development/json-schema-support.md`](development/json-schema-support.md).

## Structured execution

Structured state tracks the active JSON lexer, schema frames, object properties, array items,
combinator obligations, reference scopes, annotations, and resource accounting.

Mask construction speculatively tests vocabulary paths. Every speculative mutation is recorded in an
undo log and rolled back before the next candidate. Rejected `advance` calls use the same transaction
mechanism, so committed state is unchanged after rejection.

Fast slice certificates may skip exact token walks only when the selected token language is proven
safe for the complete active state. If the proof is unavailable, masking falls back to the exact trie
walk. Unknown must not be converted into rejection or unrestricted admission.

## Caches, threading, and resource bounds

Each `Compiler` owns its own pair of bounded caches (schema executables, vocabulary tries), sized
independently. Entries are evictable, single-flight, and held through reference-counted handles;
clearing a cache never invalidates a program or binding a caller already holds.

| Aspect | Behavior |
| --- | --- |
| Lifecycle control | `clear_caches()` / `cache_stats()` in both Rust and Python; `Compiler::without_cache()` (`maskforge.Compiler.without_cache()`) disables caching entirely for deterministic tests |
| Process-wide cache | The low-level `compile_*_with_vocabulary` functions use one instead, inspected and cleared with `maskforge.low_level.executable_cache_stats()` / `clear_executable_cache()`, so a server hosting many models can release memory explicitly rather than waiting on eviction |
| Thread safety | A compiled schema and a `BoundSchema` are immutable and safe to share across threads; each sequence still needs its own `Session`, and `Generator` never mutates the model or tokenizer it was given |

Resource gates bound schema input, supplied resources, graph construction, automata, vocabulary
construction, active structured state, and retained cache bytes. A limit returns a typed error; data
is never silently truncated.

## Mask-generation cost by keyword

Most schema shapes use precomputed vocabulary buckets. Stateful positions may use the exact
transactional walk so duplicate values, annotations, and dynamic branches remain observable.
Representative exact-trie node counts, measured on a 50,257-token GPT-2 vocabulary:

| Position | Nodes walked | Fast path | Why |
| --- | --- | :---: | --- |
| Object key under `unevaluatedProperties` + `allOf` + `if`/`then`/`else` + `dependentRequired` | ~107 | Yes | - |
| Object key under `additionalProperties: false` with the same siblings | ~99 | Yes | - |
| Enum, pattern, and nested-value string bodies | leaves the full trie | Yes | - |
| Strings inside an active canonical `uniqueItems` builder | ~97,930 | No | Must observe every byte to detect duplicates; no whole-bucket certificate is admissible while active |
| Keys constrained by `propertyNames` | ~18,600 | No | Decoded prefix, escape state, duplicate state, and applicable property patterns interact |

## JSON strings and semantic equality

The JSON lexer decodes escapes and UTF-8 incrementally. String patterns, enums, object keys, and
`uniqueItems` operate on decoded Unicode values, not source spelling: a Unicode escape sequence for
code point U+0061 inside a JSON string decodes to the plain letter `a`, and `uniqueItems` treats
both spellings as the same value.

Partial UTF-8 and `\uXXXX` sequences are checked for lexical and schema viability where the active
constraint supports that proof. High-surrogate handling includes the supplementary Unicode range.

## Formatting policy

Core matching accepts all RFC JSON whitespace. Python generation may opt into
`max_json_whitespace=N`, which caps consecutive insignificant whitespace outside strings. This is a
generation policy, not a schema keyword: it prevents small models from spending the token budget on
newlines, while leaving default validation semantics unchanged. The policy reads exact token bytes
from `CompiledVocabulary`; it never alters string contents or forbids legal Unicode escape spellings.

## Errors

| Situation | Result |
| --- | --- |
| Malformed schema | `Malformed` |
| Unsupported validation construct | `Unsupported` |
| Missing external resource | `ReferenceResolution` |
| Resource budget reached | `InternalLimitExceeded` |
| Unknown vocabulary ID | `UnknownToken` |
| Illegal token for the current prefix | `IllegalToken` |
| Advance after EOS | `SessionStopped` |

Callers should match error code and stage, retain a catch-all case, and treat message text as
diagnostic detail.

## Source map

| Path | Responsibility |
| --- | --- |
| `crates/maskforge-core/src/api.rs` | Public Rust lifecycle |
| `crates/maskforge-core/src/frontend/` | Parsing, resources, and lowering |
| `crates/maskforge-core/src/ir.rs` | Immutable schema IR |
| `crates/maskforge-core/src/routing.rs` | Route selection |
| `crates/maskforge-core/src/automaton/` | Regular byte automata |
| `crates/maskforge-core/src/structured/` | Structured plan, state, and matcher |
| `crates/maskforge-core/src/vocab/` | Vocabulary validation and canonical storage |
| `crates/maskforge-core/src/index/` | Vocabulary trie and mask indexing |
| `crates/maskforge-core/src/runtime/` | Executable cache, binding, and sessions |
| `crates/maskforge-py/python/maskforge/` | Python facade and generation integration |
