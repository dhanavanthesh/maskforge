//! Reference implementations and correctness validation utilities.

pub mod corpus;
pub mod oracle;
mod reference_engine;
mod schema_skeleton;

#[cfg(test)]
mod compile_differential;
#[cfg(test)]
mod ir_reference;
#[cfg(test)]
mod official_suite;
