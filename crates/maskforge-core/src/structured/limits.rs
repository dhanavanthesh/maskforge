//! Resource limits for the incremental structured engine: every plan-compile and session
//! operation that can grow unboundedly is charged against one of these, checked-arithmetic only.

use crate::error::{CompileError, ErrorCode, LimitKind, Stage};

/// Conservative default caps, overridable per `StructuredPlan::compile` call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StructuredLimits {
    pub(crate) max_plan_bytes: usize,
    pub(crate) max_session_bytes: usize,
    pub(crate) max_document_bytes: usize,
    pub(crate) max_depth: u32,
    pub(crate) max_dynamic_scope_depth: u32,
    pub(crate) max_properties: u32,
    pub(crate) max_items: u32,
    pub(crate) max_key_bytes: usize,
    pub(crate) max_string_bytes: usize,
    pub(crate) max_number_bytes: usize,
    pub(crate) max_active_validators: usize,
    pub(crate) max_unique_canonical_bytes: usize,
    pub(crate) max_undo_entries: usize,
    pub(crate) max_mask_work: u64,
    pub(crate) max_mask_scratch_bytes: usize,
    pub(crate) max_slice_cache_bytes: usize,
}

impl Default for StructuredLimits {
    fn default() -> Self {
        Self {
            max_plan_bytes: 64 << 20,
            max_session_bytes: 8 << 20,
            max_document_bytes: 16 << 20,
            max_depth: 512,
            max_dynamic_scope_depth: 256,
            max_properties: 100_000,
            max_items: 1_000_000,
            max_key_bytes: 1 << 20,
            max_string_bytes: 16 << 20,
            max_number_bytes: 1024,
            max_active_validators: 4096,
            max_unique_canonical_bytes: 32 << 20,
            max_undo_entries: 1 << 20,
            max_mask_work: 50_000_000,
            max_mask_scratch_bytes: 1 << 20,
            max_slice_cache_bytes: 64 << 10,
        }
    }
}

/// A `CompileError` reporting exactly which resource, how much was observed/requested, the
/// configured cap, and the stage - never a bare `Err(())`.
pub(crate) fn resource_limit(
    kind: LimitKind,
    observed: usize,
    limit: usize,
    stage: Stage,
) -> CompileError {
    let mut err = CompileError::new(
        ErrorCode::InternalLimitExceeded,
        stage,
        "structured engine resource limit exceeded",
    );
    err.limit = Some((kind, observed, limit));
    err
}

/// Adds `delta` to `total`, returning a `resource_limit` error on overflow or cap violation -
/// the one checked-accounting primitive every budget in this engine goes through.
pub(crate) fn charge(
    total: &mut usize,
    delta: usize,
    cap: usize,
    kind: LimitKind,
    stage: Stage,
) -> Result<(), CompileError> {
    let next = total
        .checked_add(delta)
        .ok_or_else(|| resource_limit(kind, usize::MAX, cap, stage))?;
    if next > cap {
        return Err(resource_limit(kind, next, cap, stage));
    }
    *total = next;
    Ok(())
}

/// A resource limit hit while an incremental session processes bytes. Distinct from
/// `CompileError`: this is a runtime condition on live input, not a plan-compile failure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StructuredRuntimeError {
    pub(crate) kind: LimitKind,
    pub(crate) observed: usize,
    pub(crate) limit: usize,
}

impl StructuredRuntimeError {
    pub(crate) fn new(kind: LimitKind, observed: usize, limit: usize) -> Self {
        Self {
            kind,
            observed,
            limit,
        }
    }
}

/// Tracks session memory usage, peak usage, and allocation work.
/// A terminal session rejects further growth after an accounting failure.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub(crate) struct SessionMemory {
    live: usize,
    peak: usize,
    allocation_work: usize,
    terminal: bool,
}

impl SessionMemory {
    /// Returns whether additional capacity fits before allocation.
    /// Reusing existing capacity always fits.
    pub(crate) fn would_fit(&self, additional: usize, cap: usize) -> bool {
        if additional == 0 {
            return !self.terminal;
        }
        !self.terminal && self.live.checked_add(additional).is_some_and(|n| n <= cap)
    }

    /// Charges capacity granted by a prior reservation.
    /// Budget overshoot is recoverable unless accounting becomes inconsistent.
    pub(crate) fn charge(
        &mut self,
        delta: usize,
        cap: usize,
        kind: LimitKind,
    ) -> Result<(), StructuredRuntimeError> {
        if self.terminal {
            return Err(StructuredRuntimeError::new(kind, self.live, cap));
        }
        if delta == 0 {
            return Ok(());
        }
        let Some(next) = self.live.checked_add(delta) else {
            self.terminal = true;
            return Err(StructuredRuntimeError::new(kind, usize::MAX, cap));
        };
        let Some(work) = self.allocation_work.checked_add(delta) else {
            self.terminal = true;
            return Err(StructuredRuntimeError::new(kind, usize::MAX, cap));
        };
        self.live = next;
        self.peak = self.peak.max(next);
        self.allocation_work = work;
        if next > cap {
            return Err(StructuredRuntimeError::new(kind, next, cap));
        }
        Ok(())
    }

    /// Charges a previously validated capacity delta.
    /// Unexpected allocator growth marks the session terminal.
    pub(crate) fn charge_after_precheck(
        &mut self,
        delta: usize,
        cap: usize,
        kind: LimitKind,
    ) -> Result<(), StructuredRuntimeError> {
        let result = self.charge(delta, cap, kind);
        if result.is_err() {
            self.terminal = true;
        }
        result
    }

    /// Reduces `live` when a whole arena is discarded. `delta` exceeding `live` means the
    /// accounting itself is corrupt, so the session goes `terminal` instead of clamping.
    pub(crate) fn release(&mut self, delta: usize) {
        match self.live.checked_sub(delta) {
            Some(next) => self.live = next,
            None => self.terminal = true,
        }
    }

    pub(crate) fn live(&self) -> usize {
        self.live
    }

    pub(crate) fn peak(&self) -> usize {
        self.peak
    }

    #[cfg(test)]
    pub(crate) fn allocation_work(&self) -> usize {
        self.allocation_work
    }

    #[cfg(test)]
    pub(crate) fn is_terminal(&self) -> bool {
        self.terminal
    }
}

/// How many more items past `len` to request from `try_reserve_exact`, and its charged cost.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GrowthPlan {
    pub(crate) reserve_additional: usize,
    #[cfg(test)]
    pub(crate) worst_case_bytes: usize,
}

/// Shape of one growth request: a `len`/`capacity` container holding `item_size` bytes per
/// element, asked to fit `additional` more.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GrowthRequest {
    pub(crate) len: usize,
    pub(crate) capacity: usize,
    pub(crate) additional: usize,
    pub(crate) item_size: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GrowthPlanError {
    Shortfall { requested_bytes: usize },
    Overflow,
}

/// Plans a bounded, amortized capacity-growth step.
/// Rejects growth that cannot satisfy the minimum required capacity.
pub(crate) fn plan_growth(
    req: GrowthRequest,
    live: usize,
    cap: usize,
) -> Result<GrowthPlan, GrowthPlanError> {
    let GrowthRequest {
        len,
        capacity,
        additional,
        item_size,
    } = req;
    let required_len = len
        .checked_add(additional)
        .ok_or(GrowthPlanError::Overflow)?;
    if required_len <= capacity {
        return Ok(GrowthPlan {
            reserve_additional: 0,
            #[cfg(test)]
            worst_case_bytes: 0,
        });
    }
    let min_growth_items = required_len - capacity;
    let min_growth_bytes = min_growth_items
        .checked_mul(item_size)
        .ok_or(GrowthPlanError::Overflow)?;
    let Some(required_live) = live.checked_add(min_growth_bytes) else {
        return Err(GrowthPlanError::Overflow);
    };
    if required_live > cap {
        return Err(GrowthPlanError::Shortfall {
            requested_bytes: min_growth_bytes,
        });
    }
    let geometric_target = capacity
        .checked_mul(2)
        .unwrap_or(required_len)
        .max(required_len);
    let budget_left_items = cap.saturating_sub(live) / item_size.max(1);
    let affordable_target = capacity.saturating_add(budget_left_items);
    let target_capacity = geometric_target.min(affordable_target).max(required_len);
    let reserve_additional = target_capacity - len;
    #[cfg(test)]
    let worst_case_bytes = (target_capacity - capacity)
        .checked_mul(item_size)
        .ok_or(GrowthPlanError::Overflow)?;
    Ok(GrowthPlan {
        reserve_additional,
        #[cfg(test)]
        worst_case_bytes,
    })
}

/// Plans and applies one bounded, amortized growth step, then charges the REAL measured
/// capacity delta `reserve` reports - never the plan's own worst-case estimate.
pub(crate) fn grow_bounded(
    mem: &mut SessionMemory,
    cap: usize,
    kind: LimitKind,
    req: GrowthRequest,
    reserve: impl FnOnce(usize) -> Result<usize, ()>,
) -> Result<(), StructuredRuntimeError> {
    let plan = plan_growth(req, mem.live(), cap).map_err(|error| match error {
        GrowthPlanError::Shortfall { requested_bytes } => {
            StructuredRuntimeError::new(kind, requested_bytes, cap)
        }
        GrowthPlanError::Overflow => StructuredRuntimeError::new(kind, usize::MAX, cap),
    })?;
    if plan.reserve_additional == 0 {
        return Ok(());
    }
    let allocation_requested = plan
        .reserve_additional
        .checked_mul(req.item_size)
        .ok_or_else(|| StructuredRuntimeError::new(kind, usize::MAX, cap))?;
    let new_capacity = reserve(plan.reserve_additional).map_err(|_| {
        StructuredRuntimeError::new(LimitKind::AllocationBytes, allocation_requested, cap)
    })?;
    let real_bytes = new_capacity
        .saturating_sub(req.capacity)
        .checked_mul(req.item_size)
        .ok_or_else(|| StructuredRuntimeError::new(kind, usize::MAX, cap))?;
    mem.charge_after_precheck(real_bytes, cap, kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_accumulates_until_the_cap() {
        let mut total = 0usize;
        charge(&mut total, 10, 20, LimitKind::PlanBytes, Stage::L3).unwrap();
        charge(&mut total, 10, 20, LimitKind::PlanBytes, Stage::L3).unwrap();
        assert_eq!(total, 20);
        let err = charge(&mut total, 1, 20, LimitKind::PlanBytes, Stage::L3).unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert_eq!(err.limit, Some((LimitKind::PlanBytes, 21, 20)));
    }

    #[test]
    fn session_memory_tracks_live_peak_and_allocation_work_separately() {
        let mut mem = SessionMemory::default();
        mem.charge(10, 100, LimitKind::SessionBytes).unwrap();
        mem.charge(5, 100, LimitKind::SessionBytes).unwrap();
        assert_eq!(
            (mem.live(), mem.peak(), mem.allocation_work()),
            (15, 15, 15)
        );
        mem.release(5);
        assert_eq!(
            (mem.live(), mem.peak(), mem.allocation_work()),
            (10, 15, 15)
        );
        mem.charge(3, 100, LimitKind::SessionBytes).unwrap();
        assert_eq!(
            (mem.live(), mem.peak(), mem.allocation_work()),
            (13, 15, 18)
        );
    }

    #[test]
    fn release_past_live_goes_terminal_instead_of_clamping() {
        let mut mem = SessionMemory::default();
        mem.charge(10, 100, LimitKind::SessionBytes).unwrap();
        mem.release(11);
        assert!(mem.is_terminal());
        assert!(mem.charge(0, 100, LimitKind::SessionBytes).is_err());
    }

    #[test]
    fn session_memory_rejects_new_growth_past_cap_but_never_bricks_zero_growth_reuse() {
        let mut mem = SessionMemory::default();
        mem.charge(90, 100, LimitKind::SessionBytes).unwrap();
        assert!(mem.charge(20, 100, LimitKind::SessionBytes).is_err());
        assert_eq!(mem.live(), 110);
        assert!(!mem.is_terminal());
        mem.charge(0, 100, LimitKind::SessionBytes).unwrap();
    }

    #[test]
    fn would_fit_rejects_before_any_charge_leaving_live_unchanged() {
        let mut mem = SessionMemory::default();
        mem.charge(90, 100, LimitKind::SessionBytes).unwrap();
        assert!(!mem.would_fit(11, 100));
        assert!(mem.would_fit(10, 100));
        assert_eq!(mem.live(), 90);
    }

    #[test]
    fn charge_after_precheck_going_terminal_rejects_every_later_call() {
        let mut mem = SessionMemory::default();
        assert!(mem.would_fit(100, 100));
        assert!(mem
            .charge_after_precheck(101, 100, LimitKind::SessionBytes)
            .is_err());
        assert!(mem.is_terminal());
        assert!(mem.charge(0, 100, LimitKind::SessionBytes).is_err());
        assert!(!mem.would_fit(0, 100));
    }

    #[test]
    fn charge_reports_overflow_as_a_resource_limit_not_a_panic() {
        let mut total = usize::MAX - 1;
        let err = charge(
            &mut total,
            10,
            usize::MAX,
            LimitKind::SessionBytes,
            Stage::L3,
        )
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    fn req(len: usize, capacity: usize, additional: usize, item_size: usize) -> GrowthRequest {
        GrowthRequest {
            len,
            capacity,
            additional,
            item_size,
        }
    }

    #[test]
    fn plan_growth_is_a_no_op_when_capacity_already_covers_the_request() {
        let plan = plan_growth(req(5, 10, 2, 8), 40, 100).unwrap();
        assert_eq!((plan.reserve_additional, plan.worst_case_bytes), (0, 0));
    }

    #[test]
    fn plan_growth_doubles_capacity_within_budget() {
        let plan = plan_growth(req(4, 4, 1, 8), 32, 1000).unwrap();
        assert!(plan.reserve_additional >= 4);
        assert_eq!(plan.worst_case_bytes, plan.reserve_additional * 8);
    }

    #[test]
    fn plan_growth_rejects_before_allocating_past_the_cap() {
        assert!(matches!(
            plan_growth(req(0, 0, 100, 8), 0, 50),
            Err(GrowthPlanError::Shortfall {
                requested_bytes: 800,
            })
        ));
    }

    #[test]
    fn bounded_growth_shortfall_reports_requested_bytes_not_overflow() {
        let mut memory = SessionMemory::default();
        let error = grow_bounded(
            &mut memory,
            50,
            LimitKind::SessionBytes,
            req(0, 0, 100, 8),
            |_| unreachable!("shortfall must not allocate"),
        )
        .unwrap_err();
        assert_eq!(error.kind, LimitKind::SessionBytes);
        assert_eq!(error.observed, 800);
        assert_ne!(error.observed, usize::MAX);
    }

    #[test]
    fn bounded_growth_allocation_failure_has_a_distinct_limit_kind() {
        let mut memory = SessionMemory::default();
        let error = grow_bounded(
            &mut memory,
            800,
            LimitKind::SessionBytes,
            req(0, 0, 10, 8),
            |_| Err(()),
        )
        .unwrap_err();
        assert_eq!(error.kind, LimitKind::AllocationBytes);
        assert_eq!(error.observed, 80);
        assert_ne!(error.observed, usize::MAX);
    }

    #[test]
    fn plan_growth_clamps_the_amortized_target_to_remaining_budget() {
        let plan = plan_growth(req(0, 0, 1, 8), 0, 40).unwrap();
        assert!(plan.worst_case_bytes <= 40);
    }
}
