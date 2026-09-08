//! Stores compiled per-state token transitions and binding metadata.

use super::build::DeltaRow;
use super::BindMode;
use crate::error::Provenance;
use crate::primitives::StateId;

#[derive(Debug)]
pub struct CompiledIndex {
    delta: Vec<DeltaRow>,
    provenance: Provenance,
    bind_mode: BindMode,
}

impl CompiledIndex {
    pub(crate) fn new(delta: Vec<DeltaRow>, provenance: Provenance, bind_mode: BindMode) -> Self {
        Self {
            delta,
            provenance,
            bind_mode,
        }
    }

    pub(crate) fn row(&self, s: StateId) -> Option<&DeltaRow> {
        self.delta.get(s.get() as usize)
    }

    #[must_use]
    pub fn bind_mode(&self) -> BindMode {
        self.bind_mode
    }

    #[must_use]
    pub fn provenance(&self) -> Provenance {
        self.provenance
    }

    #[must_use]
    pub fn state_count(&self) -> usize {
        self.delta.len()
    }

    #[must_use]
    pub fn live_pairs(&self) -> usize {
        self.delta.iter().map(DeltaRow::len).sum()
    }

    #[must_use]
    pub fn max_row_len(&self) -> usize {
        self.delta.iter().map(DeltaRow::len).max().unwrap_or(0)
    }

    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let outer = self.delta.capacity() * std::mem::size_of::<DeltaRow>();
        let rows: usize = self.delta.iter().map(DeltaRow::heap_bytes).sum();
        outer + rows
    }
}
