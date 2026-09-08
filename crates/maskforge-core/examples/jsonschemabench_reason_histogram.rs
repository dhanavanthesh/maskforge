//! Summarizes unsupported-reason frequencies in the jsonschemabench corpus.
use std::collections::HashMap;
use std::path::PathBuf;

use maskforge_core::{schema_to_ir, CompileOptions, ErrorCode, ObjectClosure, UnsupportedReason};

fn main() {
    let root = format!(
        "{}/../../integration/jsonschemabench/data",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut scanned = 0usize;
    let mut supported = 0usize;
    let mut unsupported_at_least_one = 0usize;
    let mut malformed = 0usize;
    let mut reason_counts: HashMap<UnsupportedReason, usize> = HashMap::new();
    let mut schemas_with_reason: HashMap<UnsupportedReason, usize> = HashMap::new();
    let mut error_code_counts: HashMap<ErrorCode, usize> = HashMap::new();
    let mut error_samples: HashMap<ErrorCode, Vec<String>> = HashMap::new();
    let mut unsupported_keyword_text: HashMap<String, usize> = HashMap::new();

    let mut stack = vec![PathBuf::from(&root)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            scanned += 1;
            let opts = CompileOptions {
                object_closure: ObjectClosure::AssumeClosedProfile,
                ..CompileOptions::default()
            };
            match schema_to_ir(&text, opts) {
                Ok(ir) => {
                    let mut seen_this_schema: HashMap<UnsupportedReason, bool> = HashMap::new();
                    let mut any = false;
                    for d in ir.diagnostics() {
                        any = true;
                        *reason_counts.entry(d.reason).or_insert(0) += 1;
                        seen_this_schema.entry(d.reason).or_insert(true);
                        if d.reason == UnsupportedReason::UnsupportedKeyword {
                            let kw = ir.str_at(d.keyword).unwrap_or("<?>").to_string();
                            *unsupported_keyword_text.entry(kw).or_insert(0) += 1;
                        }
                    }
                    for reason in seen_this_schema.keys() {
                        *schemas_with_reason.entry(*reason).or_insert(0) += 1;
                    }
                    if any {
                        unsupported_at_least_one += 1;
                    } else {
                        supported += 1;
                    }
                }
                Err(e) => {
                    malformed += 1;
                    *error_code_counts.entry(e.code).or_insert(0) += 1;
                    let samples = error_samples.entry(e.code).or_default();
                    if samples.len() < 5 {
                        samples.push(format!(
                            "{}: {} (keyword={:?}, ptr={:?})",
                            path.display(),
                            e.message,
                            e.keyword,
                            e.json_pointer_path
                        ));
                    }
                }
            }
        }
    }

    println!("scanned={scanned} supported={supported} unsupported_at_least_one={unsupported_at_least_one} malformed_or_error={malformed}");
    println!("\n=== schemas affected by each reason (a schema can have multiple) ===");
    let mut by_schema_count: Vec<_> = schemas_with_reason.into_iter().collect();
    by_schema_count.sort_by(|a, b| b.1.cmp(&a.1));
    for (reason, n) in &by_schema_count {
        println!("{n:5}  {reason:?}  -- {}", reason.advisory());
    }
    println!("\n=== raw occurrence counts (one schema can trip a reason many times) ===");
    let mut by_occurrence: Vec<_> = reason_counts.into_iter().collect();
    by_occurrence.sort_by(|a, b| b.1.cmp(&a.1));
    for (reason, n) in &by_occurrence {
        println!("{n:6}  {reason:?}");
    }

    println!("\n=== UnsupportedKeyword: which keyword text (occurrence counts) ===");
    let mut by_kw: Vec<_> = unsupported_keyword_text.into_iter().collect();
    by_kw.sort_by(|a, b| b.1.cmp(&a.1));
    for (kw, n) in &by_kw {
        println!("{n:6}  {kw}");
    }

    println!("\n=== hard schema_to_ir Err() breakdown (the schema never got an IR at all) ===");
    let mut by_error: Vec<_> = error_code_counts.into_iter().collect();
    by_error.sort_by(|a, b| b.1.cmp(&a.1));
    for (code, n) in &by_error {
        println!("{n:5}  {code:?}");
        for s in &error_samples[code] {
            println!("        e.g. {s}");
        }
    }
}
