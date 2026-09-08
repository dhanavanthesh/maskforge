//! Resolve `$ref` against caller-supplied resources. Run with: cargo run --example external_references -p maskforge-core

use std::sync::Arc;

use maskforge_core::{schema_to_ir_with_external_refs, CompileOptions, StructuredProgram};
use rustc_hash::FxHashMap;

fn byte_vocabulary() -> Vec<(Vec<u8>, Vec<u32>)> {
    (0u32..256).map(|b| (vec![b as u8], vec![b])).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = r#"{"$id":"https://example.com/root.json","allOf":[{"$ref":"a.json#value"},{"$ref":"b.json#value"}]}"#;
    let mut resolver = FxHashMap::default();
    resolver.insert(
        "https://example.com/a.json".to_string(),
        r#"{"$anchor":"value","type":"integer"}"#.to_string(),
    );
    resolver.insert(
        "https://example.com/b.json".to_string(),
        r#"{"$anchor":"value","minimum":5}"#.to_string(),
    );

    let ir = schema_to_ir_with_external_refs(schema, CompileOptions::default(), &resolver)?;
    let program = StructuredProgram::compile(Arc::new(ir))?;
    let mut matcher = program.new_matcher()?;

    let records = byte_vocabulary();
    let vocab_size: usize = 257;
    let mut mask = vec![0u8; 4 * vocab_size.div_ceil(32)];
    matcher.write_record_mask_le_bytes_into(
        vocab_size,
        records
            .iter()
            .map(|(bytes, ids)| (bytes.as_slice(), ids.as_slice())),
        &mut mask,
    )?;
    let digit_allowed =
        |d: u8| (mask[4 * (d as usize / 32) + (d as usize % 32) / 8] >> (d % 8)) & 1 == 1;
    println!(
        "digits allowed at the start of the value: {:?}",
        (b'0'..=b'9')
            .filter(|&d| digit_allowed(d))
            .collect::<Vec<_>>()
    );

    matcher.advance(b"5")?;
    println!("finished after committing '5': {}", matcher.is_accepting());

    Ok(())
}
