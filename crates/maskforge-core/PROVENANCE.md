# Provenance and attribution

`maskforge-core` began as a reorganization of [outlines-core](https://github.com/dottxt-ai/outlines-core)
(Apache-2.0). This file records what came from there and how it changed, as Apache-2.0 section 4(b)
requires.

Roughly 98% of the crate is original work. Four files carry upstream code:

| File | Origin | Classification |
| --- | --- | --- |
| `src/vocab/vocabulary.rs` | `src/vocabulary/mod.rs` | modified |
| `src/vocab/processor.rs` | `src/vocabulary/processor.rs` | modified |
| `src/vocab/locator.rs` | `src/vocabulary/locator.rs` | modified |
| `src/mask/bitmask.rs` | `src/python_bindings/mod.rs` | derived |

Everything else - `automaton/`, `compile.rs`, `frontend/`, `index/`, `runtime/`, `structured/`,
`correctness/`, `api.rs`, `ir.rs`, `wire.rs`, `primitives.rs`, `error.rs`, `mask/mod.rs`,
`vocab/{mod,build,error,canonical,prepared}.rs`, `benches/`, `tools/` - is original.

## Changes to the upstream files

**`src/vocab/vocabulary.rs`**

- Rejects empty-byte tokens everywhere; upstream let the raw map constructors bypass that check.
- The two raw `TryFrom` map constructors became `pub(crate)`; only the validating
  `build_vocabulary`/`build_vocabulary_from_str_keys` stayed public.
- `try_insert` is idempotent (same `(token, id)` is a no-op), replacing upstream's
  `entry().or_default().push(id)`, which inflated `len()` and did redundant mask work.
- `from_pretrained` and `from_tokenizer_json` now share one `from_tokenizer` pipeline.
- `from_tokenizer_json` caps input at `MAX_TOKENIZER_JSON_BYTES` before parsing.

**`src/vocab/processor.rs`**

`Mods::apply_bytes` (upstream `apply_default`) skips the allocation when there is nothing to
replace, instead of calling `String::replace` unconditionally. No behavior change.

**`src/vocab/locator.rs`**

`EosTokenLocation::lookup` returns `Result<TokenId, String>` with a distinct reason per failure,
instead of collapsing every failure into `None`. Fallback probing is unchanged. Only reachable
under the optional, off-by-default `huggingface-hub` feature.

**`src/mask/bitmask.rs`**

Same bit-packing contract as upstream (word `id / 32`, bit `id % 32`). The PyO3 coupling and the
raw pointer write are replaced with a safe `Vec<u32>`.

## Behavioral differences from upstream

- **Enum alternation ordering.** Alternatives are emitted longest-first, so leftmost-first DFA
  semantics do not lower `[1, 12]` to `1|12` and reject the valid value `12`.
- **Duplicate token ids.** One id bound to two different byte sequences is rejected
  (`MalformedTokenizer`); multiple ids sharing one byte sequence stay legal.
- **Out-of-range state ids.** Treated as dead rather than causing a panic.
- **Non-ASCII bracket expressions.** Rejected under byte-level `unicode(false)`; upstream builds
  with Unicode mode on.
- **Feature name.** `huggingface-hub` keeps upstream's spelling so the moved modules compile
  unchanged.

## Known constraints

`SchemaIR` validation is recursive with no depth limit. It is `pub(crate)`, and every construction
path today is bounded by this crate's own code, so the stack overflow a debug build hits around
1000 levels of nesting is not currently reachable. A future public untrusted-schema parser must
cap depth before calling into it.

## Vendored data

`tools/json-schema-test-suite/` vendors the official JSON Schema Test Suite (MIT); see its own
`PROVENANCE.md`.

## License

`LICENSE` is the unchanged Apache-2.0 text, copied to the crate root; `Cargo.toml` carries
`license = "Apache-2.0"`.
