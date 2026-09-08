//! Measures concurrent structured matcher sessions sharing compiled state.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use maskforge_core::index::{TrieCache, VocabularyHandle};
use maskforge_core::{schema_to_ir, CompileOptions, StructuredProgram};

fn vocabulary() -> VocabularyHandle {
    VocabularyHandle::from_packed_with_logits_vocab_size(
        b"az\"\\ ",
        &[0, 1, 2, 3, 4, 5],
        &[0, 1, 2, 3, 4],
        &[0, 1, 2, 3, 4, 5],
        6,
        None,
    )
    .expect("benchmark vocabulary")
}

fn queries_per_sec(threads: usize, calls_per_thread: usize) -> f64 {
    let schema = r#"{"allOf":[{"type":"string","pattern":"^a"},{"pattern":"z$"}]}"#;
    let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).expect("schema"));
    let program = StructuredProgram::compile(ir).expect("program");
    let vocabulary = Arc::new(vocabulary());
    let trie = Arc::new(TrieCache::new().bind(&vocabulary).expect("trie"));
    let barrier = Arc::new(Barrier::new(threads));
    let workers: Vec<_> = (0..threads)
        .map(|_| spawn_worker(&program, &vocabulary, &trie, &barrier, calls_per_thread))
        .collect();
    let start = Instant::now();
    for worker in workers {
        worker.join().expect("worker thread");
    }
    (threads * calls_per_thread) as f64 / start.elapsed().as_secs_f64()
}

fn spawn_worker(
    program: &Arc<StructuredProgram>,
    vocabulary: &Arc<VocabularyHandle>,
    trie: &Arc<maskforge_core::index::BoundByteTrie>,
    barrier: &Arc<Barrier>,
    calls: usize,
) -> thread::JoinHandle<()> {
    let program = program.clone();
    let vocabulary = vocabulary.clone();
    let trie = trie.clone();
    let barrier = barrier.clone();
    thread::spawn(move || {
        let mut matcher = program.new_matcher().expect("matcher");
        assert!(matcher.advance(b"\"a").expect("prefix"));
        let mut out = [0u8; 4];
        barrier.wait();
        for _ in 0..calls {
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut out)
                .expect("mask");
        }
        assert!(out.iter().any(|byte| *byte != 0));
    })
}

fn main() {
    println!("== INDEPENDENT STRUCTURED SESSION CONTENTION ==");
    const TOTAL_CALLS: usize = 80_000;
    for threads in [1usize, 8, 32, 128] {
        let calls = (TOTAL_CALLS / threads).max(100);
        let throughput = queries_per_sec(threads, calls);
        println!(
            "sessions={threads:>4} calls_per_session={calls:>6} aggregate_masks_per_sec={throughput:.0}"
        );
    }
}
