//! Measures `ArtifactCache` hit throughput under concurrent access.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use maskforge_core::correctness::corpus::corpus_cases;
use maskforge_core::index::{BindMode, VocabularyHandle};
use maskforge_core::runtime::ArtifactCache;
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

fn one_call(cache: &ArtifactCache, handle: &VocabularyHandle) {
    cache
        .get_or_build_json_schema(
            "bench-schema",
            false,
            handle.fingerprint(),
            handle.mask_vocab_size(),
            BindMode::Naive,
            || {
                CompiledArtifact::new_from_handle(
                    handle,
                    boolean_engine(),
                    Provenance::reference(HashState::Hash([1; 32])),
                )
            },
        )
        .expect("cache build");
}

fn hits_per_sec(threads: usize, calls_per_thread: usize) -> f64 {
    let cache = Arc::new(ArtifactCache::new());
    let handle = Arc::new(test_handle());
    one_call(&cache, &handle);

    let barrier = Arc::new(Barrier::new(threads));
    let workers: Vec<_> = (0..threads)
        .map(|_| {
            let cache = cache.clone();
            let handle = handle.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..calls_per_thread {
                    one_call(&cache, &handle);
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
    println!("== ARTIFACT CACHE LOCK CONTENTION (reported, not gated) ==");
    println!(
        "every timed call is a cache HIT on one pre-warmed key - the contention-relevant case"
    );
    const TOTAL_CALLS: usize = 40_000;
    for &threads in &[1usize, 8, 32, 128] {
        let calls_per_thread = (TOTAL_CALLS / threads).max(50);
        let throughput = hits_per_sec(threads, calls_per_thread);
        println!(
            "threads={threads:>4}  calls_per_thread={calls_per_thread:>7}  aggregate_hits_per_sec={throughput:.0}"
        );
    }
    println!("== END ==");
}
