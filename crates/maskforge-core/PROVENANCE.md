# Provenance and attribution

`maskforge-core` began as a reorganization of [outlines-core](https://github.com/dottxt-ai/outlines-core)
(Apache-2.0). This file records which code came from there and how it was changed, as
Apache-2.0 section 4(b) requires.

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

### `src/vocab/vocabulary.rs`

- Every construction path rejects empty-byte tokens (`Error::EmptyToken`). Upstream allowed an
  empty token through the raw map constructors, bypassing the choke-point.
- The two public `impl TryFrom<(TokenId, HashMap<..>)>` impls became `pub(crate)`
  `from_raw_bytes_map` / `from_raw_string_map`. `build_vocabulary` and
  `build_vocabulary_from_str_keys` are the only public map constructors, and both validate first.
  A trait impl cannot be less visible than its trait and type, so dropping the trait was the only
  way to keep the raw path non-public.
- `try_insert` is idempotent: repeating the same `(token, id)` is a no-op rather than a duplicate
  entry. Upstream's `entry().or_default().push(id)` inflated `len()` and did redundant mask work.
  A different id under the same bytes is still kept. Both raw constructors dedup the same way,
  preserving first-occurrence order rather than sorting.
- `from_pretrained` and `from_tokenizer_json` share one internal `from_tokenizer` pipeline, so a
  fix cannot land in one path and miss the other.
- `from_tokenizer_json` rejects input above `MAX_TOKENIZER_JSON_BYTES` before parsing, reported as
  a resource limit rather than malformed metadata.

### `src/vocab/processor.rs`

`Mods::apply_bytes` (upstream `apply_default`) called `String::replace` unconditionally, which
allocates even when the space marker is absent - the common case. It now checks `contains` first
and returns `token.as_bytes().to_vec()` directly when there is nothing to replace; the replace
path uses `into_bytes()` to transfer the buffer instead of copying it. No behavior change.

### `src/vocab/locator.rs`

`EosTokenLocation::lookup` returned `Option<TokenId>`, collapsing a network failure, an unreadable
file, an absent field, and an unknown token into the same silent `None`; the resulting error said
only "EOS token id". It now returns `Result<TokenId, String>` with a distinct reason per failure,
and `locate_eos_token_id` returns every attempted location's reason when the search misses
entirely. Fallback probing is unchanged - a miss at one location still tries the next - and the
search makes one pass, not two.

Only reachable under the optional `huggingface-hub` feature, which is off by default.

### `src/mask/bitmask.rs`

The bit-packing contract is preserved exactly: word `id / 32`, bit `id % 32`, a set bit means the
token is allowed, unused high bits zero. The PyO3 coupling and the raw pointer write are replaced
with a safe `Vec<u32>`. Attribution is in the file header.

## Behavioral differences from upstream

- **Enum alternation ordering.** Enum alternatives are emitted longest-first. Under leftmost-first
  DFA semantics, declaration order lowers `[1, 12]` to `1|12`, whose DFA stops after matching `1`
  and rejects the valid value `12`. Verified against `jsonschema`, `re.fullmatch`, and a direct
  byte-language check, and reproduced with a hand-built six-token vocabulary independent of any
  tokenizer.
- **Duplicate token ids.** A vocabulary binding one id to two different byte sequences is rejected
  (`MalformedTokenizer`): the mask is keyed by bytes and the matcher consumes one byte sequence per
  id, so the two would disagree. Multiple ids sharing the same byte sequence stay legal and are set
  together in the mask.
- **Out-of-range state ids.** `RefEngine`'s state predicates treat any out-of-range `StateId` as
  dead rather than panicking, so `is_dead(s) == (!is_accepting(s) && !can_continue(s))` holds for
  forged ids too.
- **Non-ASCII bracket expressions.** A `[...]` class containing a non-ASCII byte cannot compile
  under the byte-level `unicode(false)` configuration and is reported as a structured diagnostic.
  Multi-byte literals outside brackets are unaffected. Upstream builds with Unicode mode on.
- **Feature name.** `huggingface-hub` keeps upstream's spelling so the moved modules compile
  unchanged.

## Known constraints

`SchemaIR` validation is recursive and enforces no depth limit. It is `pub(crate)` and every
construction path today is bounded by this crate's own code, so no reachable input can drive it.
A debug build overflows its stack at roughly 1000 levels of nesting. Any future public
untrusted-schema parser must cap depth before calling into it.

## Vendored data

`tools/json-schema-test-suite/` - the official JSON Schema Test Suite (MIT). See its own
`PROVENANCE.md`.

## License

The Apache-2.0 `LICENSE` is copied to the crate root unchanged, and `Cargo.toml` carries
`license = "Apache-2.0"`.
