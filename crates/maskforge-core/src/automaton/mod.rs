//! Byte-DFA wrapper and canonical-DEAD materialization. External `regex-automata` `StateID`
//! never leaves this module; only [`crate::primitives::StateId`] does.

mod byte_dfa;
#[cfg(test)]
mod elimination;
mod graph;
mod multiple_of;
mod product;
mod shuffle;

#[cfg(test)]
pub(crate) use byte_dfa::assert_exact_equivalence;
#[cfg(test)]
pub(crate) use byte_dfa::mask_width;
pub use byte_dfa::RefEngine;
pub(crate) use byte_dfa::{build_from_regex, build_from_unicode_regex};
pub(crate) use graph::AutomatonGraph;
pub(crate) use graph::{byte_string_set, concatenate, repeat, string_set, union};
pub(crate) use multiple_of::multiple_of_engine;
#[cfg(test)]
pub(crate) use multiple_of::{decimal_multiple_of_regex, multiple_of_regex};
pub(crate) use product::{Combinator, ProductAutomaton};
pub(crate) use shuffle::ShuffleAutomaton;
