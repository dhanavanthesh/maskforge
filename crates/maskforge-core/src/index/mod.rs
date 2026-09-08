//! Binds vocabularies to automata for allowed-token mask generation.

mod artifact;
mod build;
mod cache;
mod slice;
mod trie;

pub use artifact::CompiledIndex;
pub use build::{build_delta, DeltaRow};
pub(crate) use build::{
    fill_packed_rows, packed_scratch_fixed_bytes, packed_stack_bytes, walk_state_mask_into,
};
pub use cache::{BoundByteTrie, TrieCache, VocabFingerprint, VocabularyHandle};
pub(crate) use slice::{BoundedMask, TokenSliceCatalog};
pub use trie::VocabTrie;

use crate::error::CompileError;

#[must_use]
pub fn packed_ingestion_peak_bytes(
    token_bytes_len: usize,
    byte_offsets_len: usize,
    token_ids_len: usize,
    id_offsets_len: usize,
) -> Option<usize> {
    let decoded = token_bytes_len
        .checked_add(byte_offsets_len)?
        .checked_add(token_ids_len)?
        .checked_add(id_offsets_len)?;
    let token_count = (byte_offsets_len / 4).saturating_sub(1);
    let ids_count = token_ids_len / 4;
    let core_peak = crate::vocab::projected_peak_bytes(token_bytes_len, ids_count, token_count);
    decoded.checked_add(core_peak)
}

#[must_use]
pub fn direct_csr_ingestion_peak_bytes(
    total_bytes: usize,
    total_ids: usize,
    token_count: usize,
) -> Option<usize> {
    let offsets_len = token_count.checked_add(1)?.checked_mul(4)?;
    let buffers = total_bytes
        .checked_add(total_ids.checked_mul(4)?)?
        .checked_add(offsets_len)?
        .checked_add(offsets_len)?;
    let core_peak = crate::vocab::projected_peak_bytes(total_bytes, total_ids, token_count);
    buffers.checked_add(core_peak)
}

#[must_use]
pub fn dense_id_ordered_ingestion_peak_bytes(
    total_bytes: usize,
    present_count: usize,
    token_count: usize,
) -> Option<usize> {
    let offsets_len = token_count.checked_add(1)?.checked_mul(4)?;
    let present_len = token_count;
    let buffers = total_bytes
        .checked_add(offsets_len)?
        .checked_add(present_len)?;
    let core_peak = crate::vocab::projected_peak_bytes(total_bytes, present_count, present_count);
    buffers.checked_add(core_peak)
}

pub use crate::mem_gate::PermitGuard as VocabBuildPermit;

pub fn acquire_vocab_build_permit(
    estimated_bytes: usize,
) -> Result<VocabBuildPermit<'static>, CompileError> {
    crate::mem_gate::acquire_vocab_build_bytes(estimated_bytes)
}

#[must_use]
pub fn vocab_build_gate_stats() -> (usize, usize, usize, u64) {
    crate::mem_gate::vocab_build_gate_stats()
}

#[must_use]
pub fn vocab_serving_gate_stats() -> (usize, usize, usize, u64) {
    crate::mem_gate::vocab_serving_gate_stats()
}

#[must_use]
pub fn automaton_build_gate_stats() -> (usize, usize, usize, u64) {
    crate::mem_gate::automaton_build_gate_stats()
}

pub use crate::mem_gate::GateConfigError;

pub fn configure_vocab_build_gate(
    max_concurrent: usize,
    byte_budget: usize,
) -> Result<(), GateConfigError> {
    crate::mem_gate::configure_vocab_build_gate(max_concurrent, byte_budget)
}

pub fn configure_vocab_serving_gate(
    max_concurrent: usize,
    byte_budget: usize,
) -> Result<(), GateConfigError> {
    crate::mem_gate::configure_vocab_serving_gate(max_concurrent, byte_budget)
}

pub fn configure_automaton_build_gate(
    max_concurrent: usize,
    byte_budget: usize,
) -> Result<(), GateConfigError> {
    crate::mem_gate::configure_automaton_build_gate(max_concurrent, byte_budget)
}

#[non_exhaustive]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Selects the strategy used to build allowed-token masks.
pub enum BindMode {
    /// Rewalks every token from each automaton state.
    Naive,
    /// Eagerly builds state rows with a shared byte trie.
    TrieJointByte,
    /// Builds byte-trie rows lazily on first use.
    TrieJointByteLazy,
    /// Precomputes packed mask rows for all states.
    TrieJointBytePacked,
    /// Uses a byte-class trie for per-compile binding.
    TrieJointClass,
}
