//! Release RSS probe for concurrent lazy walks and packed-table construction.
//! Reports admission outcomes and RSS after dropping held artifacts.
//!
//! Run: cargo run --release -p maskforge-core --features huggingface-hub,benchmark-tools --bin rss_concurrency_probe
#![cfg(all(feature = "huggingface-hub", feature = "benchmark-tools"))]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use maskforge_core::error::HashState;
use maskforge_core::index::{vocab_build_gate_stats, BindMode, TrieCache, VocabularyHandle};
use maskforge_core::{
    build_vocabulary, compile_ir, schema_to_ir, CompileOptions, CompiledArtifact, Provenance,
    Vocabulary,
};
use rustc_hash::FxHashMap;

const SCHEMA: &str = r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}},"required":["a","b"],"additionalProperties":false}"#;
const CONCURRENCY_LEVELS: &[usize] = &[1, 8, 32, 128];

// RSS comes from the Win32 process-status API; other targets report zero.
#[cfg(windows)]
fn peak_rss_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let cb = u32::try_from(std::mem::size_of::<PROCESS_MEMORY_COUNTERS>()).unwrap_or(0);
    // SAFETY: `pmc` is zeroed, sized correctly, and passed as a live local pointer.
    // The API writes within `cb`; its result is checked before reading fields.
    unsafe {
        let mut pmc: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        pmc.cb = cb;
        if GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) == 0 {
            return 0;
        }
        pmc.PeakWorkingSetSize as u64
    }
}

#[cfg(windows)]
fn current_rss_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let cb = u32::try_from(std::mem::size_of::<PROCESS_MEMORY_COUNTERS>()).unwrap_or(0);
    // SAFETY: same as `peak_rss_bytes` above.
    unsafe {
        let mut pmc: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        pmc.cb = cb;
        if GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) == 0 {
            return 0;
        }
        pmc.WorkingSetSize as u64
    }
}

#[cfg(not(windows))]
fn peak_rss_bytes() -> u64 {
    0
}

#[cfg(not(windows))]
fn current_rss_bytes() -> u64 {
    0
}

fn synthetic_near_cap_vocab() -> Vocabulary {
    // Near the mask-width cap (1 << 21): dense single-byte-derived tokens spread across the id
    // space so the canonical id index sizing is genuinely exercised, not a degenerate tiny case.
    const N: u32 = (1 << 21) - 100;
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    map.reserve(N as usize);
    for i in 0..N {
        map.insert(i.to_le_bytes().to_vec(), vec![i]);
    }
    build_vocabulary(N, map).expect("synthetic vocab")
}

fn prov() -> Provenance {
    Provenance::reference(HashState::Hash([3; 32]))
}

fn engine() -> maskforge_core::RefEngine {
    compile_ir(&schema_to_ir(SCHEMA, CompileOptions::default()).expect("ir")).expect("engine")
}

struct Round {
    concurrency: usize,
    admitted: u64,
    rejected: u64,
    baseline_rss: u64,
    peak_rss_during: u64,
    rss_after_join: u64,
    rss_after_drop: u64,
}

fn run_round(tag: &str, handle: &Arc<VocabularyHandle>, cache: &Arc<TrieCache>, n: usize) -> Round {
    let baseline_rss = current_rss_bytes();
    let (_, before_peak_bytes, _, before_rejected) = vocab_build_gate_stats();
    let admitted = Arc::new(AtomicU64::new(0));
    let rejected = Arc::new(AtomicU64::new(0));

    let artifacts: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .map(|i| {
                let handle = handle.clone();
                let cache = cache.clone();
                let admitted = admitted.clone();
                let rejected = rejected.clone();
                scope.spawn(move || {
                    let mode = if i % 2 == 0 {
                        BindMode::TrieJointByteLazy
                    } else {
                        BindMode::TrieJointBytePacked
                    };
                    let bound = match cache.bind(&handle) {
                        Ok(b) => b,
                        Err(_) => {
                            rejected.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };
                    match CompiledArtifact::new_from_bound_trie(
                        &handle,
                        engine(),
                        prov(),
                        mode,
                        &bound,
                    ) {
                        Ok(a) => {
                            admitted.fetch_add(1, Ordering::Relaxed);
                            Some(a)
                        }
                        Err(_) => {
                            rejected.fetch_add(1, Ordering::Relaxed);
                            None
                        }
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().unwrap())
            .collect()
    });

    let peak_rss_during = peak_rss_bytes();
    let rss_after_join = current_rss_bytes();
    println!(
        "{tag} n={n}: kept {} artifacts, admitted={} rejected={} gate_peak_bytes_delta={}",
        artifacts.len(),
        admitted.load(Ordering::Relaxed),
        rejected.load(Ordering::Relaxed),
        vocab_build_gate_stats().1.saturating_sub(before_peak_bytes),
    );
    drop(artifacts);

    let rss_after_drop = current_rss_bytes();
    let (_, _, _, after_rejected) = vocab_build_gate_stats();
    Round {
        concurrency: n,
        admitted: admitted.load(Ordering::Relaxed),
        rejected: after_rejected.saturating_sub(before_rejected),
        baseline_rss,
        peak_rss_during,
        rss_after_join,
        rss_after_drop,
    }
}

fn probe_vocab(tag: &str, vocab: Vocabulary) {
    let vocab = Arc::new(vocab);
    let handle = Arc::new(VocabularyHandle::new(vocab).expect("handle"));
    let cache = Arc::new(TrieCache::new());
    println!("\n=== {tag} ===");
    println!(
        "tag,n,admitted,rejected,baseline_rss_mb,peak_rss_during_mb,rss_after_join_mb,rss_after_drop_mb,returned_mb"
    );
    for &n in CONCURRENCY_LEVELS {
        let r = run_round(tag, &handle, &cache, n);
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        println!(
            "{tag},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.concurrency,
            r.admitted,
            r.rejected,
            mb(r.baseline_rss),
            mb(r.peak_rss_during),
            mb(r.rss_after_join),
            mb(r.rss_after_drop),
            mb(r.rss_after_join.saturating_sub(r.rss_after_drop)),
        );
    }
}

fn main() {
    let gpt2 = Vocabulary::from_pretrained("openai-community/gpt2", None).expect("gpt2");
    probe_vocab("gpt2", gpt2);

    let qwen = Vocabulary::from_pretrained("Qwen/Qwen2.5-0.5B", None).expect("qwen2.5");
    probe_vocab("qwen2.5", qwen);

    let bloom = Vocabulary::from_pretrained("bigscience/bloom", None).expect("bloom");
    probe_vocab("bloom-250k", bloom);

    probe_vocab("synthetic-near-cap", synthetic_near_cap_vocab());
}
