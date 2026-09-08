//! Compile a JSON Schema, build a synthetic vocabulary, walk a mask, via the `Compiler` facade.
//! Run with: cargo run --example basic_json_schema -p maskforge-core

use std::sync::Arc;

use maskforge_core::{build_vocabulary, CompiledVocabulary, Compiler, CompilerOptions, TokenId};
use rustc_hash::FxHashMap;

fn byte_vocabulary() -> FxHashMap<Vec<u8>, Vec<u32>> {
    (0u32..256).map(|b| (vec![b as u8], vec![b])).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = r#"{"type":"boolean"}"#;
    let vocabulary =
        CompiledVocabulary::try_from(Arc::new(build_vocabulary(256, byte_vocabulary())?))?;

    let compiler = Compiler::new(CompilerOptions::default());
    let program = compiler.compile_json_schema(schema)?;
    let bound = program.bind(&vocabulary)?;
    let mut session = bound.start_session()?;

    let mut mask = vec![0u32; bound.mask_word_count()];
    session.write_mask(&mut mask)?;
    println!("mask at start: {mask:?}");

    session.advance(TokenId::try_from(b't' as usize)?)?;
    println!("accepting after 't': {}", session.is_accepting());

    let rejected = session.advance(TokenId::try_from(b'!' as usize)?);
    println!("committing '!' after 't': {rejected:?}");

    Ok(())
}
