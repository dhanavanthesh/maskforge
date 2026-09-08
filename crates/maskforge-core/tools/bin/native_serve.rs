//! Native per-token benchmark for the packed `byte_trie`, without a Python FFI boundary.
//! Measures borrowed mask rows, owned copies, and timer overhead in batches.
//!
//! Run: cargo run --release -p maskforge-core --features huggingface-hub,benchmark-tools --bin native_serve
#![cfg(feature = "huggingface-hub")]

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use maskforge_core::error::HashState;
use maskforge_core::index::{BindMode, VocabTrie};
use maskforge_core::{
    compile_ir, schema_to_ir, CompileOptions, CompiledArtifact, Provenance, StateId, Vocabulary,
};

const SCHEMA: &str = r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}},"required":["a","b"],"additionalProperties":false}"#;
const MODELS: &[(&str, &str)] = &[
    ("gpt2", "openai-community/gpt2"),
    ("qwen2.5", "Qwen/Qwen2.5-0.5B"),
];
const BATCH: usize = 256; // ops per clock read; per-op time = batch elapsed / BATCH
const SAMPLES: usize = 4000; // batches per stat (SAMPLES * BATCH ~= 1M ops)

/// Times `op` in batches of `BATCH` (one clock read per batch), returning per-op nanoseconds. `op`
/// receives a running index so it can vary the state without a bounds-checked modulo in the timer.
fn per_op_ns(mut op: impl FnMut(usize)) -> Vec<f64> {
    for i in 0..BATCH {
        op(i); // warm
    }
    let mut out = Vec::with_capacity(SAMPLES);
    for s in 0..SAMPLES {
        let base = s * BATCH;
        let t = Instant::now();
        for i in 0..BATCH {
            op(base + i);
        }
        out.push(t.elapsed().as_nanos() as f64 / BATCH as f64);
    }
    out
}

fn pcts(mut xs: Vec<f64>) -> (f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| xs[((q * (xs.len() as f64 - 1.0)) as usize).min(xs.len() - 1)];
    (p(0.50), p(0.99), p(0.999))
}

fn reachable(a: &CompiledArtifact) -> Vec<StateId> {
    let mut seen = vec![a.start()];
    let mut i = 0;
    while i < seen.len() {
        let s = seen[i];
        i += 1;
        if let Ok(ids) = a.allowed_ids(s) {
            for t in ids {
                if let Some(bytes) = a.token_bytes(t) {
                    if let Some(n) = a.walk(s, bytes) {
                        if !seen.contains(&n) {
                            seen.push(n);
                        }
                    }
                }
            }
        }
    }
    seen
}

fn main() {
    println!(
        "native per-token serve (in-process, NO Python FFI), batched timing: BATCH={BATCH} SAMPLES={SAMPLES}, us"
    );
    // Empty-loop/timer overhead: the same batched loop doing only the index + a black_box, so the
    // clock and loop cost is visible and can be discounted from the serve numbers below.
    let sink = black_box(vec![0u32; 8]);
    let overhead = pcts(per_op_ns(|k| {
        black_box(sink[k & 7]);
    }));
    println!(
        "timer+loop overhead per op: p50={:.4} p99={:.4} us",
        overhead.0 / 1000.0,
        overhead.2 / 1000.0
    );
    println!(
        "{:10} {:>6} {:>26} {:>26}",
        "model", "wpr", "borrow_view p50/p99/p99.9", "owned_copy p50/p99/p99.9"
    );
    for (tag, repo) in MODELS {
        let vocab = match Vocabulary::from_pretrained(repo, None) {
            Ok(v) => Arc::new(v),
            Err(e) => {
                println!("{tag}: skipped (vocab unavailable: {e})");
                continue;
            }
        };
        let engine = compile_ir(&schema_to_ir(SCHEMA, CompileOptions::default()).expect("ir"))
            .expect("engine");
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        let artifact = CompiledArtifact::new_trie(
            engine,
            vocab.clone(),
            Provenance::reference(HashState::Hash([7; 32])),
            BindMode::TrieJointBytePacked,
            &trie,
        )
        .expect("packed artifact");
        let states = reachable(&artifact);
        let wpr = artifact.words_per_row();
        let m = states.len();

        // Borrowed zero-copy row view: what a native consumer reads directly (no copy).
        let mut acc = 0u64;
        let view = pcts(per_op_ns(|k| {
            if let Some(row) = artifact.mask_row_words(states[k % m]) {
                acc = acc.wrapping_add(row.len() as u64 + row[0] as u64);
            }
        }));
        black_box(acc);

        // Copies the mask into a caller-owned buffer.
        // `csum` observes every copy so the optimizer cannot remove it.
        let mut buf = vec![0u32; wpr];
        let mut csum = 0u64;
        let copy = pcts(per_op_ns(|k| {
            artifact
                .write_mask_into(&[states[k % m]], &mut buf, None)
                .expect("write");
            csum = csum.wrapping_add(buf[k % wpr] as u64);
        }));
        black_box(csum);

        let us = |n: f64| n / 1000.0;
        println!(
            "{tag:10} {wpr:>6} {:>8.4}/{:>7.4}/{:>7.4} {:>8.4}/{:>7.4}/{:>7.4}",
            us(view.0),
            us(view.1),
            us(view.2),
            us(copy.0),
            us(copy.1),
            us(copy.2)
        );
    }
}
