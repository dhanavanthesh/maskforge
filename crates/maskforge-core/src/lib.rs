//! MaskForge core: bounded schema lowering, frozen IR, incremental matching, and vocabulary masks.
//! Links without PyO3; correctness oracles are test/benchmark-only.
//!
//! Architecture: <https://github.com/dhanavanthesh/maskforge/blob/main/docs/architecture.md>.

#![deny(rust_2018_idioms)]
// `bench-internals` only widens visibility for benchmarks and never ships.
#![cfg_attr(not(feature = "bench-internals"), warn(missing_docs))]

pub mod api;
pub mod automaton;
pub mod compile;
#[cfg(any(test, feature = "test-utils"))]
pub mod correctness;
pub mod diagnostics;
pub mod error;
pub mod frontend;
pub mod index;
pub mod ir;
pub mod mask;
mod mem_gate;
pub mod primitives;
pub mod routing;
pub mod runtime;
pub mod structured;
mod vocab;
mod wire;

pub use api::{
    BindPolicy, BoundSchema, CompiledVocabulary, Compiler, CompilerOptions, SchemaProgram, Session,
    SessionError, TrieCacheStats,
};
pub use automaton::RefEngine;
pub use compile::compile_ir;
pub use diagnostics::{Diagnostic, Span, UnsupportedReason};
pub use error::{
    CompileError, ErrorCode, HashState, LimitKind, MatcherError, Provenance, ProvenanceProfile,
    Stage, VocabError,
};
pub use frontend::{
    regex_to_ir, schema_to_ir, schema_to_ir_with_external_refs, schema_to_ir_with_resources,
    SchemaRegistry, SchemaResourceLimits,
};
// The trie-joint vocabulary-binding types stay namespaced under `index::` rather than re-exported at
// the crate root: they are an opt-in optimization path, not yet a frozen top-level API surface.
pub use ir::{
    Builder, Charset, CompileOptions, Node, ObjectClosure, ScalarLit, SchemaIR, UnsupportedPolicy,
    IR_VERSION,
};
pub use mask::Bitmask;
pub use primitives::{ByteClassId, MintermId, StateId, TokenId};
pub use routing::{
    add_bound, mul_bound, pow2_bound, Bound, CostEstimate, ExecutionKind, NodeRoute, RouteReason,
    RouteTable,
};
pub use runtime::{CompiledArtifact, ExecutableCache, ExecutableSchema, Matcher};
pub use structured::{StructuredMatcher, StructuredMatcherError, StructuredProgram};
#[cfg(feature = "tokenizer-processing")]
pub use vocab::build_vocabulary_from_tokenizer_json;
#[cfg(feature = "huggingface-hub")]
pub use vocab::FromPretrainedParameters;
pub use vocab::{
    build_vocabulary, build_vocabulary_from_str_keys, StringTokenMap, TokenMap, Vocabulary,
};
