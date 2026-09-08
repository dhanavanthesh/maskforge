//! Measures lazy mask-query throughput under concurrent access.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use maskforge_core::correctness::corpus::corpus_cases;
use maskforge_core::index::VocabularyHandle;
use maskforge_core::primitives::StateId;
use maskforge_core::{CompiledArtifact, HashState, Provenance, RefEngine};

fn test_handle() -> VocabularyHandle {
    VocabularyHandle::from_packed_with_logits_vocab_size(
        b"ab",
        &[0, 1, 2],
        &[0, 1],
        &[0, 1, 2],
        2,
        None,
    )
    .expect("handle")
}

fn boolean_engine() -> RefEngine {
    corpus_cases()
        .expect("corpus")
        .into_iter()
        .find(|c| c.name == "suite-type-boolean")
        .expect("boolean case")
        .engine
}

fn lazy_artifact() -> CompiledArtifact {
    let handle = test_handle();
    let bound = maskforge_core::index::TrieCache::new()
        .bind(&handle)
        .expect("bound trie");
    CompiledArtifact::new_from_bound_trie(
        &handle,
        boolean_engine(),
        Provenance::reference(HashState::Hash([1; 32])),
        maskforge_core::index::BindMode::TrieJointByteLazy,
        &bound,
    )
    .expect("lazy artifact")
}

fn queries_per_sec(threads: usize, calls_per_thread: usize, same_state: bool) -> f64 {
    let artifact = Arc::new(lazy_artifact());
    let state_count = artifact.state_count();
    let barrier = Arc::new(Barrier::new(threads));
    let workers: Vec<_> = (0..threads)
        .map(|t| {
            let artifact = artifact.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for i in 0..calls_per_thread {
                    let raw = if same_state { 0 } else { (t + i) % state_count };
                    artifact.allowed_ids(StateId(raw as u32)).expect("mask");
                }
            })
        })
        .collect();
    let start = Instant::now();
    for w in workers {
        w.join().expect("worker thread panicked");
    }
    let wall = start.elapsed();
    (threads * calls_per_thread) as f64 / wall.as_secs_f64()
}

fn main() {
    println!("== LAZY-MODE MASK QUERY CONTENTION (reported, not gated) ==");
    const TOTAL_CALLS: usize = 40_000;
    for &same_state in &[true, false] {
        println!(
            "-- {} --",
            if same_state {
                "same state, every thread"
            } else {
                "different states, round-robin"
            }
        );
        for &threads in &[1usize, 8, 32, 128] {
            let calls_per_thread = (TOTAL_CALLS / threads).max(50);
            let throughput = queries_per_sec(threads, calls_per_thread, same_state);
            println!(
                "threads={threads:>4}  calls_per_thread={calls_per_thread:>7}  aggregate_queries_per_sec={throughput:.0}"
            );
        }
    }
    println!("== END ==");
}
