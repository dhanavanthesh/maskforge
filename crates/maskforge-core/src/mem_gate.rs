//! A process-wide, byte-bounded backpressure gate: bounds both the COUNT of simultaneous callers
//! and their estimated transient bytes, so concurrent callers cannot multiply a sound per-call limit.

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::error::{CompileError, ErrorCode, Stage};

/// Default bound on how long a caller waits for a permit before giving up with a typed timeout.
const DEFAULT_WAIT: Duration = Duration::from_secs(30);

/// A backpressure gate bounding both concurrent callers and their summed byte estimate; `stage` tags every error it raises.
#[derive(Debug)]
pub(crate) struct BuildGate {
    inner: Mutex<GateInner>,
    avail: Condvar,
    byte_budget: usize,
    max_wait: Duration,
    stage: Stage,
}

#[derive(Debug, Default)]
struct GateInner {
    permits: usize,
    bytes_avail: usize,
    in_use: usize,
    active_bytes: usize,
    peak: usize,
    peak_bytes: usize,
    rejected: u64,
}

impl BuildGate {
    pub(crate) fn new(max_concurrent: usize, byte_budget: usize, stage: Stage) -> Self {
        Self::with_max_wait(max_concurrent, byte_budget, DEFAULT_WAIT, stage)
    }

    pub(crate) fn with_max_wait(
        max_concurrent: usize,
        byte_budget: usize,
        max_wait: Duration,
        stage: Stage,
    ) -> Self {
        let byte_budget = byte_budget.max(1);
        Self {
            inner: Mutex::new(GateInner {
                permits: max_concurrent.max(1),
                bytes_avail: byte_budget,
                ..GateInner::default()
            }),
            avail: Condvar::new(),
            byte_budget,
            max_wait,
            stage,
        }
    }

    fn gate_limit(&self, msg: &'static str) -> CompileError {
        CompileError::new(ErrorCode::InternalLimitExceeded, self.stage, msg)
    }

    /// Waits for a count permit and `est` bytes. Rejects immediately if `est` alone exceeds the
    /// whole budget; times out with a typed error rather than blocking a caller forever.
    pub(crate) fn acquire(&self, est: usize) -> Result<PermitGuard<'_>, CompileError> {
        if est > self.byte_budget {
            self.inner
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .rejected += 1;
            return Err(self.gate_limit(
                "estimated transient memory exceeds the configured build-memory budget",
            ));
        }
        let deadline = Instant::now() + self.max_wait;
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        while g.permits == 0 || g.bytes_avail < est {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                g.rejected += 1;
                return Err(self.gate_limit("timed out waiting for a build-memory permit"));
            }
            let (guard, result) = self
                .avail
                .wait_timeout(g, remaining)
                .unwrap_or_else(|p| p.into_inner());
            g = guard;
            if result.timed_out() && (g.permits == 0 || g.bytes_avail < est) {
                g.rejected += 1;
                return Err(self.gate_limit("timed out waiting for a build-memory permit"));
            }
        }
        g.permits -= 1;
        g.bytes_avail -= est;
        g.in_use += 1;
        g.active_bytes += est;
        g.peak = g.peak.max(g.in_use);
        g.peak_bytes = g.peak_bytes.max(g.active_bytes);
        Ok(PermitGuard {
            gate: self,
            charge: est,
        })
    }

    /// Admits `est` immediately if it fits, else fails immediately - NEVER waits. For latency-
    /// sensitive serving callers, which must not queue behind a large build holding the budget.
    pub(crate) fn try_acquire(&self, est: usize) -> Result<PermitGuard<'_>, CompileError> {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if est > self.byte_budget || g.permits == 0 || g.bytes_avail < est {
            g.rejected += 1;
            return Err(
                self.gate_limit("serving memory budget is fully committed; try again shortly")
            );
        }
        g.permits -= 1;
        g.bytes_avail -= est;
        g.in_use += 1;
        g.active_bytes += est;
        g.peak = g.peak.max(g.in_use);
        g.peak_bytes = g.peak_bytes.max(g.active_bytes);
        Ok(PermitGuard {
            gate: self,
            charge: est,
        })
    }

    pub(crate) fn peak_concurrent(&self) -> usize {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).peak
    }

    pub(crate) fn peak_bytes(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .peak_bytes
    }

    /// Bytes currently admitted (in-flight across every live permit).
    pub(crate) fn active_bytes(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .active_bytes
    }

    /// Count of `acquire` calls that failed (over-budget or timed out).
    pub(crate) fn rejected(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .rejected
    }
}

/// Releases a caller's count permit + byte charge and wakes waiters on drop (incl. error or panic).
/// Fields stay private: the only way to obtain one is `acquire`/`acquire_vocab_build_bytes`.
#[derive(Debug)]
pub struct PermitGuard<'a> {
    gate: &'a BuildGate,
    charge: usize,
}

impl Drop for PermitGuard<'_> {
    fn drop(&mut self) {
        let mut g = self.gate.inner.lock().unwrap_or_else(|p| p.into_inner());
        g.permits += 1;
        g.bytes_avail += self.charge;
        g.in_use -= 1;
        g.active_bytes -= self.charge;
        drop(g);
        self.gate.avail.notify_all(); // multiple waiters may now fit
    }
}

impl PermitGuard<'_> {
    /// `Err` unless this permit's charged bytes cover `required` - the check every `*_admitted`
    /// constructor runs before trusting a caller-supplied permit as real admission proof.
    pub fn ensure_covers(&self, required: usize) -> Result<(), CompileError> {
        if required > self.charge {
            return Err(self
                .gate
                .gate_limit("build permit does not cover the projected peak"));
        }
        Ok(())
    }
}

/// Default BUILD gate budget: one process-wide byte pool for packed construction and ingestion.
/// `1 << 31` = 2 GiB.
const DEFAULT_VOCAB_BUILD_BYTES: usize = 1 << 31;

/// Default cap on simultaneous admitted builds.
const DEFAULT_VOCAB_BUILD_CONCURRENCY: usize = 16;

/// Default SERVING gate budget, kept separate so a latency-sensitive query never waits behind a
/// build holding the build budget.
const DEFAULT_VOCAB_SERVING_BYTES: usize = 1 << 28;

/// Default cap on simultaneous admitted lazy queries.
const DEFAULT_VOCAB_SERVING_CONCURRENCY: usize = 64;

/// Default AUTOMATON gate budget, separate from vocabulary build/ingestion: `1 << 31` = 2 GiB.
const DEFAULT_AUTOMATON_BUILD_BYTES: usize = 1 << 31;

/// Default cap on simultaneous admitted automaton (regex/DFA) builds.
const DEFAULT_AUTOMATON_BUILD_CONCURRENCY: usize = 16;

/// One `OnceLock` per gate: `configure_*` and `*_gate()` race on the SAME cell, so whichever wins
/// atomically IS the live policy.
static VOCAB_BUILD_GATE: std::sync::OnceLock<BuildGate> = std::sync::OnceLock::new();
static VOCAB_SERVING_GATE: std::sync::OnceLock<BuildGate> = std::sync::OnceLock::new();
static AUTOMATON_BUILD_GATE: std::sync::OnceLock<BuildGate> = std::sync::OnceLock::new();

/// Why `configure_vocab_{build,serving}_gate` failed - a stable, machine-matchable pair instead of
/// a free-text string, so a caller (or the Python FFI) can branch on WHICH failure it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateConfigError {
    /// `max_concurrent` or `byte_budget` was zero.
    InvalidValue,
    /// The gate already has a policy (an earlier configure call, or a prior default-triggering use).
    AlreadyInitialized,
}

impl GateConfigError {
    /// Advisory text for the human-readable side of an error report.
    #[must_use]
    pub fn message(self) -> &'static str {
        match self {
            Self::InvalidValue => "max_concurrent and byte_budget must both be at least 1",
            Self::AlreadyInitialized => {
                "the gate already has a policy; configure it before the first use"
            }
        }
    }
}

impl std::fmt::Display for GateConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for GateConfigError {}

/// Installs the BUILD gate's `(max_concurrent, byte_budget)`. `Err` if it already has a policy
/// (configured or defaulted by a prior use) or either value is zero.
pub(crate) fn configure_vocab_build_gate(
    max_concurrent: usize,
    byte_budget: usize,
) -> Result<(), GateConfigError> {
    if max_concurrent == 0 || byte_budget == 0 {
        return Err(GateConfigError::InvalidValue);
    }
    VOCAB_BUILD_GATE
        .set(BuildGate::new(max_concurrent, byte_budget, Stage::L4Bind))
        .map_err(|_| GateConfigError::AlreadyInitialized)
}

/// Installs the SERVING gate's `(max_concurrent, byte_budget)`. Same contract as
/// `configure_vocab_build_gate`.
pub(crate) fn configure_vocab_serving_gate(
    max_concurrent: usize,
    byte_budget: usize,
) -> Result<(), GateConfigError> {
    if max_concurrent == 0 || byte_budget == 0 {
        return Err(GateConfigError::InvalidValue);
    }
    VOCAB_SERVING_GATE
        .set(BuildGate::new(max_concurrent, byte_budget, Stage::L4Bind))
        .map_err(|_| GateConfigError::AlreadyInitialized)
}

/// Installs the AUTOMATON gate's `(max_concurrent, byte_budget)`; same contract as `configure_vocab_build_gate`.
pub(crate) fn configure_automaton_build_gate(
    max_concurrent: usize,
    byte_budget: usize,
) -> Result<(), GateConfigError> {
    if max_concurrent == 0 || byte_budget == 0 {
        return Err(GateConfigError::InvalidValue);
    }
    AUTOMATON_BUILD_GATE
        .set(BuildGate::new(max_concurrent, byte_budget, Stage::L3))
        .map_err(|_| GateConfigError::AlreadyInitialized)
}

/// The process-wide BUILD gate (packed table construction, packed-buffer ingestion), separate from
/// `TrieCache`'s own build gate and the serving gate below.
pub(crate) fn vocab_build_gate() -> &'static BuildGate {
    VOCAB_BUILD_GATE.get_or_init(|| {
        BuildGate::new(
            DEFAULT_VOCAB_BUILD_CONCURRENCY,
            DEFAULT_VOCAB_BUILD_BYTES,
            Stage::L4Bind,
        )
    })
}

/// The process-wide SERVING gate (lazy per-state mask queries only).
pub(crate) fn vocab_serving_gate() -> &'static BuildGate {
    VOCAB_SERVING_GATE.get_or_init(|| {
        BuildGate::new(
            DEFAULT_VOCAB_SERVING_CONCURRENCY,
            DEFAULT_VOCAB_SERVING_BYTES,
            Stage::L4Bind,
        )
    })
}

/// The process-wide AUTOMATON gate (regex/DFA compilation), separate from the vocabulary gates.
pub(crate) fn automaton_build_gate() -> &'static BuildGate {
    AUTOMATON_BUILD_GATE.get_or_init(|| {
        BuildGate::new(
            DEFAULT_AUTOMATON_BUILD_CONCURRENCY,
            DEFAULT_AUTOMATON_BUILD_BYTES,
            Stage::L3,
        )
    })
}

/// `(active_bytes, peak_bytes, peak_concurrent, rejected)`, all zero if the BUILD gate has never
/// been used - reading stats must never itself install the default policy.
pub(crate) fn vocab_build_gate_stats() -> (usize, usize, usize, u64) {
    VOCAB_BUILD_GATE.get().map_or((0, 0, 0, 0), |g| {
        (
            g.active_bytes(),
            g.peak_bytes(),
            g.peak_concurrent(),
            g.rejected(),
        )
    })
}

/// Same non-initializing contract as `vocab_build_gate_stats`, for the SERVING gate.
pub(crate) fn vocab_serving_gate_stats() -> (usize, usize, usize, u64) {
    VOCAB_SERVING_GATE.get().map_or((0, 0, 0, 0), |g| {
        (
            g.active_bytes(),
            g.peak_bytes(),
            g.peak_concurrent(),
            g.rejected(),
        )
    })
}

/// Same non-initializing contract as `vocab_build_gate_stats`, for the AUTOMATON gate.
pub(crate) fn automaton_build_gate_stats() -> (usize, usize, usize, u64) {
    AUTOMATON_BUILD_GATE.get().map_or((0, 0, 0, 0), |g| {
        (
            g.active_bytes(),
            g.peak_bytes(),
            g.peak_concurrent(),
            g.rejected(),
        )
    })
}

/// Acquires `estimated_bytes` from the shared BUILD gate, waiting if necessary. The returned guard
/// must be held for as long as the estimated memory stays live; dropping it releases the charge.
pub(crate) fn acquire_vocab_build_bytes(
    estimated_bytes: usize,
) -> Result<PermitGuard<'static>, CompileError> {
    vocab_build_gate().acquire(estimated_bytes)
}

/// Fail-fast admission against the shared SERVING gate: never waits, so a lazy query cannot queue
/// behind a large build.
pub(crate) fn try_acquire_vocab_serving_bytes(
    estimated_bytes: usize,
) -> Result<PermitGuard<'static>, CompileError> {
    vocab_serving_gate().try_acquire(estimated_bytes)
}

/// Acquires `estimated_bytes` from the shared AUTOMATON gate, waiting if necessary.
pub(crate) fn acquire_automaton_build_bytes(
    estimated_bytes: usize,
) -> Result<PermitGuard<'static>, CompileError> {
    automaton_build_gate().acquire(estimated_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permit_releases_on_panic_not_just_normal_drop() {
        let gate = BuildGate::new(4, 100, Stage::L4Bind);
        let result = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _permit = gate.acquire(60).unwrap();
                    panic!("simulated failure while a permit is held");
                })
                .join()
        });
        assert!(result.is_err(), "the spawned thread must have panicked");
        assert_eq!(
            gate.active_bytes(),
            0,
            "PermitGuard::drop must run during unwind and release the charge"
        );
    }

    #[test]
    fn configure_vocab_gates_reject_a_zero_value_without_touching_the_singleton() {
        // Rejected before the OnceLock is ever touched, so this is safe regardless of whether
        // another test in this binary already initialized the real global gate.
        assert_eq!(
            configure_vocab_build_gate(0, 100),
            Err(GateConfigError::InvalidValue)
        );
        assert_eq!(
            configure_vocab_build_gate(4, 0),
            Err(GateConfigError::InvalidValue)
        );
        assert_eq!(
            configure_vocab_serving_gate(0, 100),
            Err(GateConfigError::InvalidValue)
        );
        assert_eq!(
            configure_vocab_serving_gate(4, 0),
            Err(GateConfigError::InvalidValue)
        );
    }

    #[test]
    fn acquire_rejects_an_estimate_over_the_whole_budget() {
        let gate = BuildGate::new(4, 100, Stage::L4Bind);
        assert!(gate.acquire(101).is_err());
        assert_eq!(gate.rejected(), 1);
    }

    #[test]
    fn acquire_admits_up_to_the_budget_and_releases_on_drop() {
        let gate = BuildGate::new(4, 100, Stage::L4Bind);
        let p1 = gate.acquire(60).unwrap();
        assert_eq!(gate.active_bytes(), 60);
        drop(p1);
        assert_eq!(gate.active_bytes(), 0);
        let p2 = gate.acquire(100).unwrap();
        assert_eq!(gate.active_bytes(), 100);
        drop(p2);
    }

    #[test]
    fn a_second_caller_times_out_while_the_budget_is_fully_held() {
        let gate = BuildGate::with_max_wait(4, 100, Duration::from_millis(50), Stage::L4Bind);
        let _held = gate.acquire(100).unwrap();
        let err = gate.acquire(1).expect_err("no bytes are available");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert_eq!(gate.rejected(), 1);
    }

    #[test]
    fn every_rejection_path_reports_the_gate_own_stage_not_a_hardcoded_one() {
        let gate = BuildGate::new(1, 100, Stage::L3);
        assert_eq!(gate.acquire(200).unwrap_err().stage, Stage::L3);
        assert_eq!(gate.try_acquire(200).unwrap_err().stage, Stage::L3);
        let permit = gate.acquire(50).unwrap();
        assert_eq!(permit.ensure_covers(51).unwrap_err().stage, Stage::L3);

        let other = BuildGate::new(1, 100, Stage::L4Bind);
        assert_eq!(other.acquire(200).unwrap_err().stage, Stage::L4Bind);
    }

    #[test]
    fn peak_bytes_and_peak_concurrent_track_the_high_water_mark() {
        let gate = BuildGate::new(4, 1000, Stage::L4Bind);
        let p1 = gate.acquire(100).unwrap();
        let p2 = gate.acquire(200).unwrap();
        assert_eq!(gate.peak_concurrent(), 2);
        assert_eq!(gate.peak_bytes(), 300);
        drop(p1);
        drop(p2);
        assert_eq!(
            gate.peak_concurrent(),
            2,
            "peak is a high-water mark, not current"
        );
        assert_eq!(gate.peak_bytes(), 300);
    }

    #[test]
    #[ignore = "timing sanity check, not a correctness gate; run with --ignored --nocapture"]
    fn uncontended_acquire_release_cost() {
        let gate = BuildGate::new(16, 1 << 31, Stage::L4Bind);
        let n = 100_000;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let _p = gate.acquire(1024).unwrap();
        }
        let elapsed = t.elapsed();
        println!(
            "{n} uncontended acquire+release cycles: {:?} total, {:?}/cycle",
            elapsed,
            elapsed / n
        );
    }

    #[test]
    #[ignore = "timing sanity check, not a correctness gate; run with --release --ignored --nocapture"]
    fn contended_acquire_release_cost_1_8_16_32_threads() {
        fn stats(mut xs: Vec<f64>) -> (f64, f64, f64, f64, f64, f64) {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = xs.len();
            let median = xs[n / 2];
            let p95 = xs[((n as f64 * 0.95) as usize).min(n - 1)];
            let p99 = xs[((n as f64 * 0.99) as usize).min(n - 1)];
            let mean = xs.iter().sum::<f64>() / n as f64;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            (median, p95, p99, var.sqrt(), xs[0], xs[n - 1])
        }
        // A large budget so every acquire admits (no forced waiting) - this isolates lock/atomic
        // contention cost as thread count rises, not admission-refusal behavior.
        const ITERS_PER_THREAD: usize = 5_000;
        for &threads in &[1usize, 8, 16, 32] {
            let gate = BuildGate::new(threads * 4, 1 << 34, Stage::L4Bind);
            let samples: Vec<f64> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let gate = &gate;
                        scope.spawn(move || {
                            let mut xs = Vec::with_capacity(ITERS_PER_THREAD);
                            for _ in 0..ITERS_PER_THREAD {
                                let t = std::time::Instant::now();
                                let _p = gate.acquire(1024).unwrap();
                                xs.push(t.elapsed().as_secs_f64() * 1e9); // ns
                            }
                            xs
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect()
            });
            let (median, p95, p99, stddev, min, max) = stats(samples.clone());
            println!(
                "threads={threads}: n={} median={median:.1} p95={p95:.1} p99={p99:.1} \
                 stddev={stddev:.1} min={min:.1} max={max:.1} ns/acquire",
                samples.len()
            );
        }
    }
}
