# maskforge-core

Schema-guided token masks for reliable structured generation in Rust. The crate compiles JSON
Schema and regular expressions into reusable executables, binds them to exact tokenizer bytes,
and produces a packed allowed-token bitmask at every decoding step.

This crate has no PyO3 dependency and links standalone; the Python bindings live in the
separate `maskforge-py` crate. See the [repository README](https://github.com/dhanavanthesh/maskforge)
for the full project overview, verified model output, current JSON Schema support, and installation
instructions for both Rust and Python.

See the public [architecture](../../docs/architecture.md),
[JSON Schema support](../../docs/development/json-schema-support.md), and
[development roadmap](../../docs/development/to-do.md) for details.

```rust
use std::sync::Arc;

use maskforge_core::{build_vocabulary, CompiledVocabulary, Compiler, CompilerOptions, TokenMap};

let mut token_map = TokenMap::default();
for byte in 0u8..=u8::MAX {
    token_map.insert(vec![byte], vec![u32::from(byte)]);
}
let eos_token_id = 256;
let schema_text = r#"{"type":"boolean"}"#;
let token_id = maskforge_core::TokenId::try_from(b't' as usize)?;

let vocabulary = CompiledVocabulary::try_from(Arc::new(build_vocabulary(eos_token_id, token_map)?))?;

let compiler = Compiler::new(CompilerOptions::default());
let program = compiler.compile_json_schema(schema_text)?; // vocabulary-independent, reusable
let bound = program.bind(&vocabulary)?; // picks regular or structured internally
let mut session = bound.start_session()?;

let mut mask = vec![0u32; bound.mask_word_count()];
session.write_mask(&mut mask)?;
session.advance(token_id)?;
```

The complete runnable version is in `examples/basic_json_schema.rs`. `Compiler` owns a configurable,
clearable executable cache. `SchemaProgram`, `CompiledVocabulary`, and `BoundSchema` are immutable
and reusable; each generated sequence owns one mutable `Session`.

## License

Apache-2.0. Portions derive from [outlines-core](https://github.com/dottxt-ai/outlines-core)
(also Apache-2.0); see `PROVENANCE.md` for the file-by-file audit trail.
