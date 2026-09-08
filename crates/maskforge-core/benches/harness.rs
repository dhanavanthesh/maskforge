//! Validates the reference engine against the fixed corpus and JSON Schema oracle.

use std::hint::black_box;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use criterion::Criterion;
use maskforge_core::correctness::corpus::{corpus_cases, CorpusCase};
use maskforge_core::error::HashState;
use maskforge_core::index::{build_delta, BindMode, VocabTrie};
use maskforge_core::{build_vocabulary, Provenance, StateId, Vocabulary};
use rustc_hash::FxHashMap;

// Optional allocator instrumentation for allocation measurements.
#[cfg(feature = "alloc-probe")]
mod alloc_probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingAllocator;
    pub static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                let now = LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
                PEAK_BYTES.fetch_max(now, Ordering::Relaxed);
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            System.dealloc(ptr, layout);
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAllocator = CountingAllocator;

    pub fn reset() -> usize {
        let base = LIVE_BYTES.load(Ordering::Relaxed);
        PEAK_BYTES.store(base, Ordering::Relaxed);
        base
    }

    pub fn peak() -> usize {
        PEAK_BYTES.load(Ordering::Relaxed)
    }
}

const ORACLE_PY: &str = include_str!("../tools/jsonschema_oracle.py");

struct OracleResult {
    python: String,
    jsonschema: String,
    verdicts: Vec<bool>,
}

fn run_oracle(payload: &str) -> Result<OracleResult, String> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let script = std::env::temp_dir().join(format!(
        "maskforge_jsonschema_oracle_{}_{nanos}.py",
        std::process::id()
    ));
    std::fs::write(&script, ORACLE_PY).map_err(|e| format!("write helper: {e}"))?;
    let result = run_oracle_with(&script, payload);
    let _ = std::fs::remove_file(&script);
    result
}

fn run_oracle_with(script: &std::path::Path, payload: &str) -> Result<OracleResult, String> {
    let mut child = Command::new("uv")
        .arg("run")
        .arg("--quiet")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn uv: {e}"))?;
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(payload.as_bytes())
        .map_err(|e| format!("write payload: {e}"))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("wait uv: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "uv exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("parse oracle json: {e}"))?;
    let arr = v["verdicts"]
        .as_array()
        .ok_or("oracle 'verdicts' is not an array")?;
    let mut verdicts = Vec::with_capacity(arr.len());
    for b in arr {
        verdicts.push(
            b.as_bool()
                .ok_or_else(|| format!("non-boolean verdict: {b}"))?,
        );
    }
    Ok(OracleResult {
        python: v["python"].as_str().unwrap_or("?").to_string(),
        jsonschema: v["jsonschema"].as_str().unwrap_or("?").to_string(),
        verdicts,
    })
}

fn percentiles_us(cases: &[CorpusCase], iters: usize) -> (f64, f64, f64, f64) {
    let mut samples: Vec<u128> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        for case in cases {
            for s in &case.samples {
                black_box(case.engine.accepts(black_box(s.instance.as_bytes())));
            }
        }
        samples.push(t.elapsed().as_nanos());
    }
    samples.sort_unstable();
    let at = |p: f64| {
        let idx = ((samples.len() as f64 - 1.0) * p).round() as usize;
        samples[idx] as f64 / 1000.0
    };
    (at(0.50), at(0.95), at(0.99), at(0.999))
}

fn bind_vocab() -> Vocabulary {
    let mut tokens: Vec<Vec<u8>> = vec![b"true".to_vec(), b"false".to_vec(), b"null".to_vec()];
    for &b in b"[]{},:\"".iter() {
        tokens.push(vec![b]);
    }
    for b in b'0'..=b'9' {
        tokens.push(vec![b]);
    }
    for b in b'a'..=b'z' {
        tokens.push(vec![b]);
    }
    for b in b'A'..=b'Z' {
        if tokens.len() >= 64 {
            break;
        }
        tokens.push(vec![b]);
    }
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    for (id, bytes) in tokens.into_iter().enumerate() {
        map.insert(bytes, vec![id as u32]);
    }
    build_vocabulary(64, map).expect("bind vocab")
}

fn median_us(mut samples: Vec<u128>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / 1000.0
}

#[cfg(feature = "alloc-probe")]
fn print_alloc_peak(vocab: &Vocabulary, provenance: Provenance) {
    let base = alloc_probe::reset();
    let cold_trie = VocabTrie::build_byte(vocab).expect("byte trie");
    for case in &corpus_cases().expect("corpus") {
        let idx = build_delta(
            &case.engine,
            &cold_trie,
            BindMode::TrieJointByte,
            provenance,
        )
        .expect("peak build");
        black_box(idx);
    }
    let peak = alloc_probe::peak().saturating_sub(base);
    println!("byte_trie_cold_build_peak_alloc_bytes: {peak}   (allocation instrumentation; timings in THIS run are perturbed)");
}

#[cfg(not(feature = "alloc-probe"))]
fn print_alloc_peak(_vocab: &Vocabulary, _provenance: Provenance) {
    println!("byte_trie_cold_build_peak_alloc_bytes: n/a (run `--features alloc-probe`; that run's timings are not valid)");
}

fn print_bind_report(iters: usize) {
    let vocab = bind_vocab();
    let provenance = Provenance::reference(HashState::Hash([8; 32]));
    let byte_trie = VocabTrie::build_byte(&vocab).expect("byte trie");

    let mut naive_samples = Vec::with_capacity(iters);
    let mut byte_cold_samples = Vec::with_capacity(iters);
    let mut byte_warm_samples = Vec::with_capacity(iters);
    let mut class_cold_samples = Vec::with_capacity(iters);
    let mut class_warm_samples = Vec::with_capacity(iters);
    let mut live_pairs = 0usize;
    let mut class_nodes = 0usize;
    for _ in 0..iters {
        let cases = corpus_cases().expect("corpus");
        let class_tries: Vec<VocabTrie> = cases
            .iter()
            .map(|case| VocabTrie::build_class(&vocab, &case.engine).expect("class trie"))
            .collect();
        class_nodes = class_tries.iter().map(VocabTrie::node_count).sum();

        let t = Instant::now();
        for case in &cases {
            for raw in 0..case.engine.state_count() {
                let s = StateId::try_from(raw).unwrap();
                black_box(case.engine.allowed_tokens(s, &vocab).expect("naive"));
            }
        }
        naive_samples.push(t.elapsed().as_nanos());

        let t = Instant::now();
        let cold_trie = VocabTrie::build_byte(&vocab).expect("byte trie");
        for case in &cases {
            let idx = build_delta(
                &case.engine,
                &cold_trie,
                BindMode::TrieJointByte,
                provenance,
            )
            .expect("byte cold");
            black_box(idx);
        }
        byte_cold_samples.push(t.elapsed().as_nanos());

        let t = Instant::now();
        for case in &cases {
            let idx = build_delta(
                &case.engine,
                &byte_trie,
                BindMode::TrieJointByte,
                provenance,
            )
            .expect("byte warm");
            live_pairs = idx.live_pairs();
        }
        byte_warm_samples.push(t.elapsed().as_nanos());

        let t = Instant::now();
        for case in &cases {
            let trie = VocabTrie::build_class(&vocab, &case.engine).expect("class trie");
            let idx = build_delta(&case.engine, &trie, BindMode::TrieJointClass, provenance)
                .expect("class cold");
            black_box(idx);
        }
        class_cold_samples.push(t.elapsed().as_nanos());

        let t = Instant::now();
        for (case, trie) in cases.iter().zip(&class_tries) {
            let idx = build_delta(&case.engine, trie, BindMode::TrieJointClass, provenance)
                .expect("class warm");
            black_box(idx);
        }
        class_warm_samples.push(t.elapsed().as_nanos());
    }

    let cold_trie = VocabTrie::build_byte(&vocab).expect("byte trie");
    let mut max_row = 0usize;
    for case in &corpus_cases().expect("corpus") {
        let idx = build_delta(
            &case.engine,
            &cold_trie,
            BindMode::TrieJointByte,
            provenance,
        )
        .expect("peak build");
        max_row = max_row.max(idx.max_row_len());
    }

    println!("== VOCAB-BINDING BUILD-TIME REPORT (reported, not gated) ==");
    println!("bind_vocab_size: 64 (+EOS)");
    println!("byte_trie_nodes: {}", byte_trie.node_count());
    println!("class_trie_nodes_total_last: {class_nodes}");
    println!("byte_trie_delta_live_pairs_last: {live_pairs}");
    println!("byte_trie_max_delta_row_len: {max_row}");
    print_alloc_peak(&vocab, provenance);
    println!("naive_bind_us_median: {:.3}", median_us(naive_samples));
    println!(
        "byte_trie_cold_us_median: {:.3}   byte_trie_warm_us_median: {:.3}",
        median_us(byte_cold_samples),
        median_us(byte_warm_samples)
    );
    println!(
        "class_trie_cold_us_median: {:.3}   class_trie_warm_us_median: {:.3}",
        median_us(class_cold_samples),
        median_us(class_warm_samples)
    );
    println!("cold = trie build + delta build; warm = delta build with the trie reused");
    println!("note: build-time bind over the fixed corpus; NOT a per-token generation-speed claim");
    println!("== END BIND REPORT ==");
}

fn main() {
    let cases = corpus_cases().expect("correctness corpus builds");

    let mut payload = String::from("[");
    let mut engine_verdicts: Vec<bool> = Vec::new();
    for case in &cases {
        for sample in &case.samples {
            if !engine_verdicts.is_empty() {
                payload.push(',');
            }
            payload.push_str("{\"schema\":");
            payload.push_str(case.json_schema);
            payload.push_str(",\"instance\":");
            payload.push_str(sample.instance);
            payload.push('}');
            engine_verdicts.push(case.engine.accepts(sample.instance.as_bytes()));
        }
    }
    payload.push(']');

    let total_pairs = engine_verdicts.len();
    let oracle = run_oracle(&payload);

    let mut suite_pairs = 0usize;
    let mut suite_agreements = 0usize;
    {
        let mut i = 0usize;
        for case in &cases {
            for sample in &case.samples {
                if let Some(expected) = sample.suite_expected {
                    suite_pairs += 1;
                    if engine_verdicts[i] == expected {
                        suite_agreements += 1;
                    }
                }
                i += 1;
            }
        }
    }
    let synthetic_pairs = total_pairs - suite_pairs;
    let mut synthetic_agreements = 0usize;

    let (oracle_status, python, jsonschema, agreements, mismatches, mismatch_examples) =
        match &oracle {
            Ok(o) if o.verdicts.len() == total_pairs => {
                let mut agree = 0usize;
                let mut mism = 0usize;
                let mut examples: Vec<String> = Vec::new();
                let mut i = 0usize;
                for case in &cases {
                    for sample in &case.samples {
                        if engine_verdicts[i] == o.verdicts[i] {
                            agree += 1;
                            if sample.suite_expected.is_none() {
                                synthetic_agreements += 1;
                            }
                        } else {
                            mism += 1;
                            if examples.len() < 8 {
                                examples.push(format!(
                                    "{} instance={} engine={} oracle={}",
                                    case.name, sample.instance, engine_verdicts[i], o.verdicts[i]
                                ));
                            }
                        }
                        i += 1;
                    }
                }
                (
                    "ok".to_string(),
                    o.python.clone(),
                    o.jsonschema.clone(),
                    agree,
                    mism,
                    examples,
                )
            }
            Ok(o) => (
                format!(
                    "length_mismatch engine={total_pairs} oracle={}",
                    o.verdicts.len()
                ),
                o.python.clone(),
                o.jsonschema.clone(),
                0,
                total_pairs,
                vec![],
            ),
            Err(e) => (
                format!("error: {e}"),
                "n/a".to_string(),
                "n/a".to_string(),
                0,
                total_pairs,
                vec![],
            ),
        };

    let checks_pass = mismatches == 0
        && agreements == total_pairs
        && suite_agreements == suite_pairs
        && oracle_status == "ok";

    let t = Instant::now();
    let rebuilt = corpus_cases().expect("rebuild");
    let compile_wall_ms = t.elapsed().as_secs_f64() * 1000.0;
    let byte_dfa_states: usize = rebuilt.iter().map(|c| c.engine.state_count()).sum();
    let transitions: usize = rebuilt
        .iter()
        .map(|c| c.engine.live_transition_count())
        .sum();

    let (p50, p95, p99, p999) = percentiles_us(&cases, 2000);

    println!("== MASKFORGE CORRECTNESS SELF-CHECK ==");
    println!("engine_version: maskforge-core 0.0.0 (reference)");
    println!("impl_kind: reference");
    println!("cache_state: fresh_compile");
    println!("provenance_profile: reference (ir_hash=unhashed)");
    println!("oracle_model: external uv-run python jsonschema process");
    println!("oracle_status: {oracle_status}");
    println!("python_version: {python}");
    println!("jsonschema_version: {jsonschema}");
    println!("corpus_cases: {}", cases.len());
    println!("total_pairs: {total_pairs}");
    println!("agreements: {agreements}");
    println!("mismatches: {mismatches}");
    println!("suite_sourced_pairs: {suite_pairs}");
    println!("suite_agreements (engine vs official suite): {suite_agreements}");
    println!("synthetic_pairs: {synthetic_pairs}");
    println!("synthetic_agreements (engine vs jsonschema oracle): {synthetic_agreements}");
    println!("byte_dfa_states_total: {byte_dfa_states}");
    println!("byte_dfa_transitions_total: {transitions}");
    println!("token_states_and_edges: n/a (no token product graph in the reference engine)");
    println!("minterm_count: n/a");
    println!("mu_over_m: n/a");
    println!("tokenizer_id: n/a");
    println!("vocab_size: n/a");
    println!("peak_memory: n/a");
    println!("number_of_generated_tokens: n/a");
    println!("compile_wall_time_ms: {compile_wall_ms:.3}");
    println!(
        "runtime_selfcheck_classify_us: p50={p50:.3} p95={p95:.3} p99={p99:.3} p999={p999:.3}"
    );
    for case in &cases {
        let source = match case.provenance {
            Some(p) => format!("suite:{}::{}", p.file, p.description),
            None => "synthetic".to_string(),
        };
        println!(
            "case: {} schema_id={} source=[{}] byte_dfa_states={} transitions={} samples={}",
            case.name,
            case.schema_id,
            source,
            case.engine.state_count(),
            case.engine.live_transition_count(),
            case.samples.len()
        );
    }
    for ex in &mismatch_examples {
        println!("mismatch: {ex}");
    }
    println!(
        "correctness_result: {}",
        if checks_pass { "PASS" } else { "FAIL" }
    );
    println!("== END SELF-CHECK ==");

    print_bind_report(200);

    let mut crit = Criterion::default().sample_size(30).configure_from_args();
    crit.bench_function("selfcheck_classify_corpus", |b| {
        b.iter(|| {
            for case in &cases {
                for s in &case.samples {
                    black_box(case.engine.accepts(black_box(s.instance.as_bytes())));
                }
            }
        });
    });
    crit.final_summary();

    if !checks_pass {
        eprintln!("SELF-CHECK FAILED: reference engine disagreed with the jsonschema oracle");
        std::process::exit(1);
    }
}
