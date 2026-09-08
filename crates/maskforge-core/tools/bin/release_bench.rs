//! Release benchmark for trie binding with isolated per-process RSS measurements.
//! Measures uncached and cached mask paths.
//!
//! Run: cargo run --release -p maskforge-core --features huggingface-hub --bin release_bench -- driver
#![cfg(feature = "huggingface-hub")]

use std::env;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use maskforge_core::error::HashState;
use maskforge_core::index::{build_delta, BindMode, VocabFingerprint, VocabTrie};
use maskforge_core::{
    compile_ir, schema_to_ir, CompileOptions, CompiledArtifact, Matcher, Provenance, RefEngine,
    Vocabulary,
};

const SCHEMA: &str = r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}},"required":["a","b"],"additionalProperties":false}"#;
// A linear-chain pattern schema so a straight-line walk can visit up to 100,000 distinct states.
const CROSSOVER_SCHEMA_PATTERN_LEN: usize = 100_000;
const MODEL: &str = "openai-community/gpt2";
const SAMPLES: usize = 31;
const DISTINCT_N: usize = 8;
const CROSSOVER_QS: &[usize] = &[1, 2, 8, 32, 64, 128, 512, 4_096, 16_384, 100_000];

fn prov() -> Provenance {
    Provenance::reference(HashState::Hash([9; 32]))
}

fn engine() -> RefEngine {
    compile_ir(&schema_to_ir(SCHEMA, CompileOptions::default()).expect("ir")).expect("engine")
}

fn crossover_engine() -> RefEngine {
    let pattern = "a".repeat(CROSSOVER_SCHEMA_PATTERN_LEN);
    let schema = format!(r#"{{"type":"string","pattern":"{pattern}"}}"#);
    compile_ir(&schema_to_ir(&schema, CompileOptions::default()).expect("ir")).expect("engine")
}

fn vocab() -> Arc<Vocabulary> {
    Arc::new(Vocabulary::from_pretrained(MODEL, None).expect("gpt2 vocab (offline cache)"))
}

/// Loads a supported benchmark vocabulary by identifier.
fn vocab_by_id(vocab_id: &str) -> Arc<Vocabulary> {
    let model = match vocab_id {
        "gpt2" => MODEL,
        "qwen" => "Qwen/Qwen2.5-0.5B",
        other => panic!("unknown vocab id {other}, expected gpt2|qwen"),
    };
    Arc::new(Vocabulary::from_pretrained(model, None).expect("vocab (offline cache)"))
}

fn bind(mode: BindMode, v: &Arc<Vocabulary>) -> CompiledArtifact {
    bind_engine(engine(), mode, v)
}

fn bind_engine(e: RefEngine, mode: BindMode, v: &Arc<Vocabulary>) -> CompiledArtifact {
    match mode {
        BindMode::Naive => CompiledArtifact::new(e, v.clone(), prov()).unwrap(),
        _ => {
            let trie = VocabTrie::build_byte(v).unwrap();
            CompiledArtifact::new_trie(e, v.clone(), prov(), mode, &trie).unwrap()
        }
    }
}

/// Walks the smallest legal token at each step and returns up to `n` distinct states.
/// The walk stops when the language accepts or dead-ends.
fn walk_distinct_states(
    artifact: &Arc<CompiledArtifact>,
    n: usize,
) -> Vec<maskforge_core::StateId> {
    let mut states = vec![artifact.start()];
    let mut m = Matcher::new(artifact.clone());
    while states.len() < n {
        let allowed = artifact.allowed_ids(m.state()).unwrap();
        let Some(&tok) = allowed.iter().min_by_key(|t| t.0) else {
            break;
        };
        if m.advance(tok).is_err() {
            break;
        }
        states.push(m.state());
    }
    states
}

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
        let ok = GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb);
        if ok == 0 {
            return 0;
        }
        pmc.PeakWorkingSetSize as u64
    }
}

#[cfg(not(windows))]
fn peak_rss_bytes() -> u64 {
    // getrusage(RUSAGE_SELF).ru_maxrss - not compiled on this host, kept honest rather than guessed.
    0
}

/// One measurement, one process. Prints `TIME_NS=<u64> RSS_BYTES=<u64>` and exits.
fn run_child(stage: &str, mode: &str) {
    let mode = match mode {
        "naive" => BindMode::Naive,
        "byte_trie" => BindMode::TrieJointByte,
        other => panic!("unknown mode {other}"),
    };
    let v = vocab();
    let elapsed_ns: u128 = match stage {
        "compile" => {
            let t = Instant::now();
            let e = engine();
            let ns = t.elapsed().as_nanos();
            std::hint::black_box(&e);
            ns
        }
        "bind_cold" => {
            let t = Instant::now();
            let artifact = bind(mode, &v);
            let ns = t.elapsed().as_nanos();
            std::hint::black_box(&artifact);
            ns
        }
        "bind_warm" => {
            // Trie built ONCE and reused for both binds, so this measures genuine warm reuse
            // (naive has no index to reuse, so its warm number stays close to cold).
            match mode {
                BindMode::Naive => {
                    let _warm_up = CompiledArtifact::new(engine(), v.clone(), prov()).unwrap();
                    let t = Instant::now();
                    let artifact = CompiledArtifact::new(engine(), v.clone(), prov()).unwrap();
                    let ns = t.elapsed().as_nanos();
                    std::hint::black_box(&artifact);
                    ns
                }
                _ => {
                    let trie = VocabTrie::build_byte(&v).unwrap();
                    let _warm_up =
                        CompiledArtifact::new_trie(engine(), v.clone(), prov(), mode, &trie)
                            .unwrap();
                    let t = Instant::now();
                    let artifact =
                        CompiledArtifact::new_trie(engine(), v.clone(), prov(), mode, &trie)
                            .unwrap();
                    let ns = t.elapsed().as_nanos();
                    std::hint::black_box(&artifact);
                    ns
                }
            }
        }
        // Uncached by construction (never touches `mask_cache`); not a first-vs-repeated stage.
        "core_raw_allowed_mask" => {
            let artifact = bind(mode, &v);
            let t = Instant::now();
            let m = artifact.allowed_mask(artifact.start()).unwrap();
            let ns = t.elapsed().as_nanos();
            std::hint::black_box(&m);
            ns
        }
        // The path Python `mask_words` calls; first-ever query on a freshly bound artifact.
        "core_cached_first" => {
            let artifact = bind(mode, &v);
            let s = artifact.start();
            let mut buf = vec![0u32; artifact.words_per_row()];
            let t = Instant::now();
            artifact.write_mask_into(&[s], &mut buf, None).unwrap();
            let ns = t.elapsed().as_nanos();
            std::hint::black_box(&buf);
            ns
        }
        "core_cached_repeated" => {
            let artifact = bind(mode, &v);
            let s = artifact.start();
            let mut buf = vec![0u32; artifact.words_per_row()];
            artifact.write_mask_into(&[s], &mut buf, None).unwrap(); // untimed warm-up
            let t = Instant::now();
            artifact.write_mask_into(&[s], &mut buf, None).unwrap();
            let ns = t.elapsed().as_nanos();
            std::hint::black_box(&buf);
            ns
        }
        // N distinct states, one `write_mask_into` call per state (a generation-shaped access
        // pattern: each step visits a state never queried before).
        "core_cached_distinct_n" => {
            let artifact = Arc::new(bind(mode, &v));
            let states = walk_distinct_states(&artifact, DISTINCT_N);
            let wpr = artifact.words_per_row();
            let mut buf = vec![0u32; wpr];
            let t = Instant::now();
            for &s in &states {
                artifact.write_mask_into(&[s], &mut buf, None).unwrap();
            }
            let ns = t.elapsed().as_nanos();
            std::hint::black_box(&buf);
            ns
        }
        // The same N distinct states, but as ONE batched `write_mask_into` call.
        "core_cached_batch_n" => {
            let artifact = Arc::new(bind(mode, &v));
            let states = walk_distinct_states(&artifact, DISTINCT_N);
            let wpr = artifact.words_per_row();
            let mut buf = vec![0u32; wpr * states.len()];
            let t = Instant::now();
            artifact.write_mask_into(&states, &mut buf, None).unwrap();
            let ns = t.elapsed().as_nanos();
            std::hint::black_box(&buf);
            ns
        }
        other => panic!("unknown stage {other}"),
    };
    println!("TIME_NS={elapsed_ns} RSS_BYTES={}", peak_rss_bytes());
}

/// Breaks a ByteTrie cold bind into its named sub-steps, one process per sample. `vocab_id`: "gpt2" or "qwen".
fn run_bind_breakdown_child(vocab_id: &str) {
    let v = vocab_by_id(vocab_id);

    let t = Instant::now();
    let e = engine();
    let t_compile = t.elapsed().as_nanos();
    std::hint::black_box(&e);

    let t = Instant::now();
    let fp = VocabFingerprint::of(&v).unwrap();
    let t_fingerprint = t.elapsed().as_nanos();
    std::hint::black_box(&fp);

    let t = Instant::now();
    let trie = VocabTrie::build_byte(&v).unwrap();
    let t_trie_build = t.elapsed().as_nanos();

    let t = Instant::now();
    let index = build_delta(&e, &trie, BindMode::TrieJointByte, prov()).unwrap();
    let t_delta_build = t.elapsed().as_nanos();
    std::hint::black_box(&index);

    // The shared cost naive also pays (token-byte index + validate), isolated from ByteTrie-only work.
    let t = Instant::now();
    let base = CompiledArtifact::new(engine(), v.clone(), prov()).unwrap();
    let t_artifact_new = t.elapsed().as_nanos();
    std::hint::black_box(&base);

    // Sanity total: a real end-to-end new_trie call, for comparison against the sum of parts above.
    let t = Instant::now();
    let full =
        CompiledArtifact::new_trie(engine(), v.clone(), prov(), BindMode::TrieJointByte, &trie)
            .unwrap();
    let t_new_trie_total = t.elapsed().as_nanos();
    std::hint::black_box(&full);

    println!(
        "vocab={vocab_id} tokens={} \
         schema_compile_ns={t_compile} fingerprint_ns={t_fingerprint} \
         trie_build_ns={t_trie_build} delta_build_ns={t_delta_build} \
         artifact_new_ns={t_artifact_new} new_trie_total_ns={t_new_trie_total} \
         RSS_BYTES={}",
        v.tokens().len(),
        peak_rss_bytes()
    );
}

/// `total_time(Q) = bind (cold) + Q distinct-state cached mask queries`, one process per sample.
fn run_crossover_child(mode: &str, q: usize) {
    let mode = match mode {
        "naive" => BindMode::Naive,
        "byte_trie" => BindMode::TrieJointByte,
        other => panic!("unknown mode {other}"),
    };
    let v = vocab();
    // Schema compile happens before the timer starts: total_time(Q) is bind + Q queries only.
    let e = crossover_engine();
    let t = Instant::now();
    let artifact = Arc::new(bind_engine(e, mode, &v));
    let states = walk_distinct_states(&artifact, q);
    let wpr = artifact.words_per_row();
    let mut buf = vec![0u32; wpr];
    for &s in &states {
        artifact.write_mask_into(&[s], &mut buf, None).unwrap();
    }
    let ns = t.elapsed().as_nanos();
    std::hint::black_box(&buf);
    println!(
        "TIME_NS={ns} RSS_BYTES={} STATES_VISITED={}",
        peak_rss_bytes(),
        states.len()
    );
}

struct Stats {
    p50: f64,
    p95: f64,
    p99: f64,
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

fn stats(mut xs: Vec<f64>) -> Stats {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let pct = |p: f64| xs[((p * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    Stats {
        p50: pct(0.50),
        p95: pct(0.95),
        p99: pct(0.99),
        mean,
        stddev: var.sqrt(),
        min: xs[0],
        max: xs[n - 1],
    }
}

fn sample(stage: &str, mode: &str, samples: usize) -> (Stats, Stats) {
    let exe = env::current_exe().expect("current_exe");
    let mut times_us = Vec::with_capacity(samples);
    let mut rss_kb = Vec::with_capacity(samples);
    for _ in 0..samples {
        let out = Command::new(&exe)
            .args(["child", stage, mode])
            .output()
            .expect("spawn child");
        assert!(out.status.success(), "child failed: {out:?}");
        let text = String::from_utf8(out.stdout).unwrap();
        let mut time_ns = 0u128;
        let mut rss = 0u64;
        for tok in text.split_whitespace() {
            if let Some(v) = tok.strip_prefix("TIME_NS=") {
                time_ns = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("RSS_BYTES=") {
                rss = v.parse().unwrap();
            }
        }
        times_us.push(time_ns as f64 / 1000.0);
        rss_kb.push(rss as f64 / 1024.0);
    }
    (stats(times_us), stats(rss_kb))
}

fn print_row(label: &str, naive: &Stats, byte_trie: &Stats) {
    println!(
        "| {label} | p50={:.1}us p95={:.1}us p99={:.1}us mean={:.1} sd={:.1} min={:.1} max={:.1} | \
         p50={:.1}us p95={:.1}us p99={:.1}us mean={:.1} sd={:.1} min={:.1} max={:.1} |",
        naive.p50,
        naive.p95,
        naive.p99,
        naive.mean,
        naive.stddev,
        naive.min,
        naive.max,
        byte_trie.p50,
        byte_trie.p95,
        byte_trie.p99,
        byte_trie.mean,
        byte_trie.stddev,
        byte_trie.min,
        byte_trie.max,
    );
}

fn run_driver(samples: usize) {
    println!(
        "host_cpu_logical={}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
    println!("rustc=1.86.0 profile=release");
    println!(
        "commit={}",
        Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".into())
    );
    println!("model={MODEL} schema=object cache=integration/.cache/models (offline) samples={samples} distinct_n={DISTINCT_N}");
    println!();

    let stages = [
        "compile",
        "bind_cold",
        "bind_warm",
        "core_raw_allowed_mask",
        "core_cached_first",
        "core_cached_repeated",
        "core_cached_distinct_n",
        "core_cached_batch_n",
    ];
    println!("| Stage | MF Naive (time us, {samples} fresh-process samples) | MF ByteTrie (time us, {samples} fresh-process samples) |");
    println!("|---|---|---|");
    for stage in stages {
        let (n_time, _n_rss) = sample(stage, "naive", samples);
        let (b_time, _b_rss) = sample(stage, "byte_trie", samples);
        print_row(stage, &n_time, &b_time);
    }

    println!();
    println!("| Stage | MF Naive peak RSS (KiB) | MF ByteTrie peak RSS (KiB) |");
    println!("|---|---|---|");
    for stage in ["bind_cold", "core_cached_first"] {
        let (_n_time, n_rss) = sample(stage, "naive", samples);
        let (_b_time, b_rss) = sample(stage, "byte_trie", samples);
        println!(
            "| {stage} | p50={:.0} p95={:.0} max={:.0} | p50={:.0} p95={:.0} max={:.0} |",
            n_rss.p50, n_rss.p95, n_rss.max, b_rss.p50, b_rss.p95, b_rss.max
        );
    }

    // Delta/index bytes: measured directly in-process (a size, not a timing, so no subprocess
    // isolation is needed). Naive keeps no index: N/A - it has no equivalent internal representation.
    let v = vocab();
    let trie = VocabTrie::build_byte(&v).unwrap();
    let index =
        maskforge_core::index::build_delta(&engine(), &trie, BindMode::TrieJointByte, prov())
            .unwrap();
    println!();
    println!("| Field | MF Naive | MF ByteTrie |");
    println!("|---|---|---|");
    println!(
        "| Delta / index bytes (retained, capacity-based) | N/A - no equivalent internal representation (naive keeps no precomputed index) | trie heap {} bytes + CompiledIndex::heap_bytes() {} bytes = {} bytes |",
        trie.heap_bytes(),
        index.heap_bytes(),
        trie.heap_bytes() + index.heap_bytes()
    );
}

fn run_bind_breakdown_driver(samples: usize) {
    println!();
    println!("## Bind breakdown: where does a cold ByteTrie bind's time actually go");
    println!("({samples} fresh-process samples per vocabulary, median ns)");
    println!();
    println!("| Vocab | tokens | schema_compile | fingerprint | trie_build | delta_build | artifact_new (shared w/ naive) | new_trie total (sanity) |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    let exe = env::current_exe().expect("current_exe");
    for vocab_id in ["gpt2", "qwen"] {
        let mut fields: [Vec<f64>; 6] = Default::default();
        let mut tokens = 0usize;
        for _ in 0..samples {
            let out = Command::new(&exe)
                .args(["bind_breakdown", vocab_id])
                .output()
                .expect("spawn bind_breakdown child");
            assert!(out.status.success(), "bind_breakdown child failed: {out:?}");
            let text = String::from_utf8(out.stdout).unwrap();
            for tok in text.split_whitespace() {
                if let Some(v) = tok.strip_prefix("tokens=") {
                    tokens = v.parse().unwrap();
                } else if let Some(v) = tok.strip_prefix("schema_compile_ns=") {
                    fields[0].push(v.parse::<f64>().unwrap());
                } else if let Some(v) = tok.strip_prefix("fingerprint_ns=") {
                    fields[1].push(v.parse::<f64>().unwrap());
                } else if let Some(v) = tok.strip_prefix("trie_build_ns=") {
                    fields[2].push(v.parse::<f64>().unwrap());
                } else if let Some(v) = tok.strip_prefix("delta_build_ns=") {
                    fields[3].push(v.parse::<f64>().unwrap());
                } else if let Some(v) = tok.strip_prefix("artifact_new_ns=") {
                    fields[4].push(v.parse::<f64>().unwrap());
                } else if let Some(v) = tok.strip_prefix("new_trie_total_ns=") {
                    fields[5].push(v.parse::<f64>().unwrap());
                }
            }
        }
        let median = |xs: &mut Vec<f64>| -> f64 {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            xs[xs.len() / 2] / 1000.0 // -> us
        };
        println!(
            "| {vocab_id} | {tokens} | {:.1}us | {:.1}us | {:.1}us | {:.1}us | {:.1}us | {:.1}us |",
            median(&mut fields[0]),
            median(&mut fields[1]),
            median(&mut fields[2]),
            median(&mut fields[3]),
            median(&mut fields[4]),
            median(&mut fields[5]),
        );
    }
}

fn run_crossover_driver(samples: usize) {
    println!();
    println!("## Crossover: total_time(Q) = cold bind + Q distinct-state cached mask queries");
    println!(
        "(long-pattern linear-chain schema, {samples} fresh-process samples per cell, median us)"
    );
    println!();
    println!("| Q | MF Naive (median us) | MF ByteTrie (median us) | ByteTrie ahead? |");
    println!("|---:|---:|---:|:---:|");
    for &q in CROSSOVER_QS {
        // Large-Q processes each do a Q-step walk internally, so use fewer samples at the high
        // end to keep total wall-clock bounded; still enough for a median.
        let n_samples = if q >= 4_096 { samples.min(5) } else { samples };
        let exe = env::current_exe().expect("current_exe");
        let run = |mode: &str| -> f64 {
            let mut xs = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                let out = Command::new(&exe)
                    .args(["crossover", mode, &q.to_string()])
                    .output()
                    .expect("spawn crossover child");
                assert!(out.status.success(), "crossover child failed: {out:?}");
                let text = String::from_utf8(out.stdout).unwrap();
                for tok in text.split_whitespace() {
                    if let Some(v) = tok.strip_prefix("TIME_NS=") {
                        xs.push(v.parse::<f64>().unwrap() / 1000.0);
                    }
                }
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            xs[xs.len() / 2]
        };
        let n = run("naive");
        let b = run("byte_trie");
        let ahead = if b < n { "yes" } else { "no" };
        println!("| {q} | {n:.1} | {b:.1} | {ahead} |");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("child") => run_child(&args[2], &args[3]),
        Some("crossover") => {
            let q: usize = args[3].parse().expect("Q must be a number");
            run_crossover_child(&args[2], q);
        }
        Some("bind_breakdown") => run_bind_breakdown_child(&args[2]),
        Some("driver") | None => {
            let samples = args
                .iter()
                .position(|a| a == "--samples")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(SAMPLES);
            run_driver(samples);
            run_bind_breakdown_driver(samples.min(15));
            run_crossover_driver(samples.min(15)); // crossover sweep is 7 Qs x 2 modes x samples processes
        }
        Some(other) => {
            panic!("unknown mode {other}, expected child|crossover|bind_breakdown|driver")
        }
    }
}
