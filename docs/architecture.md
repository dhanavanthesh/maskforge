# Architecture

MaskForge compiles a JSON Schema once, binds it to a tokenizer vocabulary once, and creates one
mutable session per generated sequence.

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

## Generation contract

Each decoding step has two operations:

```text
session.write_mask(buffer)
model selects an allowed token
session.advance(token_id)
```

The mask is a packed little-endian bitset. Token `t` is allowed when bit `t` is set. EOS is set only
when the current state is accepting.

`advance` is atomic. It commits every raw byte of the token, or restores the exact prior state and
returns `IllegalToken`. Calling `write_mask` repeatedly must not change committed state.

The immediate transition contract is:

```text
mask(prefix, token) = advance_from_equivalent_state(prefix, token).is_ok()
```

This contract does not by itself prove that the resulting prefix can reach a complete document.
Prefix liveness is a separate property. Current liveness limits are listed in
[`development/json-schema-support.md`](development/json-schema-support.md).

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

Binding combines a schema executable with a compiled vocabulary. Regular schemas may use eager,
lazy, or adaptive mask-row materialization. Structured schemas retain their plan and create mutable
state only when a session starts.

Tokenizer decoding is not used as a substitute for raw vocabulary bytes. This matters for byte-level
BPE vocabularies, aliases, UTF-8 prefixes, and JSON escapes.

## Structured execution

Structured state tracks the active JSON lexer, schema frames, object properties, array items,
combinator obligations, reference scopes, annotations, and resource accounting.

Mask construction speculatively tests vocabulary paths. Every speculative mutation is recorded in an
undo log and rolled back before the next candidate. Rejected `advance` calls use the same transaction
mechanism, so committed state is unchanged after rejection.

Fast slice certificates may skip exact token walks only when the selected token language is proven
safe for the complete active state. If the proof is unavailable, masking falls back to the exact trie
walk. Unknown must not be converted into rejection or unrestricted admission.

## JSON strings and semantic equality

The JSON lexer decodes escapes and UTF-8 incrementally. String patterns, enums, object keys, and
`uniqueItems` operate on decoded Unicode values, not source spelling. For example, `"risk"` and
`"r\u0069sk"` are the same JSON string value.

Partial UTF-8 and `\uXXXX` sequences are checked for lexical and schema viability where the active
constraint supports that proof. High-surrogate handling includes the supplementary Unicode range.

## Formatting policy

Core matching accepts all RFC JSON whitespace. Python generation may opt into
`max_json_whitespace=N`, which caps consecutive insignificant whitespace outside strings. This is a
generation policy, not a schema keyword. It prevents small models from spending the token budget on
newlines while leaving default validation semantics unchanged.

The policy reads exact token bytes from `CompiledVocabulary`. It does not alter string contents or
forbid legal Unicode escape spellings.

## Caches and resource bounds

The compiler owns bounded caches for schema executables and vocabulary tries. Entries are evictable,
single-flight, and held through reference-counted handles. Clearing a cache does not invalidate a
program or binding already held by a caller.

Resource gates bound schema input, supplied resources, graph construction, automata, vocabulary
construction, active structured state, and retained cache bytes. A limit returns a typed error. Data
is not silently truncated.

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
