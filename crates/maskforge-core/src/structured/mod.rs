//! Structured validation for schemas that need more than one anchored byte-DFA.

mod canonical;
pub(crate) mod lexer;
mod limits;
mod matcher;
mod pattern;
mod plan;
#[cfg(any(test, feature = "bench-internals"))]
mod reference;
mod state;
mod validator;

pub use matcher::{
    try_accepts, try_can_continue, StructuredMatcher, StructuredMatcherError, StructuredProgram,
};
pub(crate) use pattern::{build_search_pattern, leading_fragment_pattern};
#[cfg(test)]
pub(crate) use reference::accepts as oracle_accepts;

/// Test helper that maps checked matcher errors to rejection.
#[cfg(test)]
pub fn accepts(ir: &crate::ir::SchemaIR, bytes: &[u8]) -> bool {
    try_accepts(std::sync::Arc::new(ir.clone()), bytes).unwrap_or(false)
}

/// Test helper that maps checked matcher errors to rejection.
#[cfg(test)]
pub fn can_continue(ir: &crate::ir::SchemaIR, bytes: &[u8]) -> bool {
    try_can_continue(std::sync::Arc::new(ir.clone()), bytes).unwrap_or(false)
}
