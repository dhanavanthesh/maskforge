# maskforge-core

**Schema-guided token masks for reliable structured generation.**

[![crates.io](https://img.shields.io/crates/v/maskforge-core.svg)](https://crates.io/crates/maskforge-core)
[![docs.rs](https://img.shields.io/docsrs/maskforge-core)](https://docs.rs/maskforge-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Compile a JSON Schema once, bind it to a tokenizer, and get a packed allowed-token bitmask at every
decoding step. A model driven through these masks cannot emit a token that breaks the schema.

Links standalone; no PyO3 dependency.

## How it works

JSON Schema lowers into a hash-consed arena IR, then a cost model picks a backend:

- **Regular schemas** become a byte-level DFA, built with product and shuffle constructions and
  Kleene state elimination. `multipleOf` compiles to a residue automaton.
- **Everything else** runs on an incremental pushdown matcher: a typed frame stack, byte at a time,
  with an undo log so a speculative byte costs no clone.

Masks come from walking the vocabulary as a compressed-sparse-row trie jointly with the automaton.
A shared token prefix is parsed once; a dead subtree is pruned whole.

## Example

```rust
use std::sync::Arc;

use maskforge_core::{build_vocabulary, CompiledVocabulary, Compiler, CompilerOptions, TokenMap};

let mut token_map = TokenMap::default();
for byte in 0u8..=u8::MAX {
    token_map.insert(vec![byte], vec![u32::from(byte)]);
}
let vocabulary = CompiledVocabulary::try_from(Arc::new(build_vocabulary(256, token_map)?))?;

let compiler = Compiler::new(CompilerOptions::default());
let program = compiler.compile_json_schema(r#"{"type":"boolean"}"#)?;
let bound = program.bind(&vocabulary)?;
let mut session = bound.start_session()?;

let mut mask = vec![0u32; bound.mask_word_count()];
session.write_mask(&mut mask)?;
session.advance(maskforge_core::TokenId::try_from(b't' as usize)?)?;
```

`SchemaProgram`, `CompiledVocabulary` and `BoundSchema` are immutable and shared. `Session` is the
only mutable object, holding one sequence. Full version in `examples/basic_json_schema.rs`.

## Correctness

Every mask is checked three ways - optimized, full-trie and record-scan - and the routes must agree.
The official JSON Schema Test Suite runs in CI.

Limitations are documented, not left to be discovered:
[what's not finished yet](https://github.com/dhanavanthesh/maskforge#whats-not-finished-yet).

## More

- [Repository](https://github.com/dhanavanthesh/maskforge)
- [Architecture](https://github.com/dhanavanthesh/maskforge/blob/main/docs/architecture.md)
- [Keyword support](https://github.com/dhanavanthesh/maskforge/blob/main/docs/development/json-schema-support.md)
- Python bindings: [`maskforge`](https://pypi.org/project/maskforge/)

## License

Apache-2.0. Portions derive from [outlines-core](https://github.com/dottxt-ai/outlines-core)
(also Apache-2.0); `PROVENANCE.md` has the file-by-file trail.
