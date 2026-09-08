//! Python bindings for MaskForge. Every PyO3 item is gated behind `python-bindings`; the default
//! build compiles no pyo3 and produces an empty library, so the core stays FFI-free.

#![deny(rust_2018_idioms)]

#[cfg(feature = "python-bindings")]
mod ffi;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// The compiled extension module (`maskforge._native`), re-exported by the pure-Python package.
#[cfg(feature = "python-bindings")]
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ffi::PyIndex>()?;
    m.add_class::<ffi::PySchemaIR>()?;
    m.add_class::<ffi::PyVocabulary>()?;
    m.add_class::<ffi::PyMaskTableView>()?;
    m.add_class::<ffi::PyStructuredMatcher>()?;
    m.add_class::<ffi::PyCompiler>()?;
    m.add_class::<ffi::PySchemaProgram>()?;
    m.add_class::<ffi::PyCompiledVocabulary>()?;
    m.add_class::<ffi::PyBoundSchema>()?;
    m.add_class::<ffi::PySession>()?;
    m.add("MaskforgeError", m.py().get_type::<ffi::MaskforgeError>())?;
    m.add_function(wrap_pyfunction!(ffi::schema_to_ir, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::schema_to_ir_with_resources, m)?)?;
    #[cfg(feature = "bench-internals")]
    m.add_function(wrap_pyfunction!(
        ffi::schema_to_ir_with_resources_profile_for_bench,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ffi::schema_to_ir_with_external_refs, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::regex_ir, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::ir_from_wire, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::compile_ir, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::compile_json_schema, m)?)?;
    m.add_function(wrap_pyfunction!(
        ffi::compile_json_schema_with_vocabulary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ffi::compile_ir_with_vocabulary, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::executable_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::clear_executable_cache, m)?)?;
    #[cfg(feature = "bench-internals")]
    m.add_function(wrap_pyfunction!(
        ffi::compile_schema_executable_for_bench,
        m
    )?)?;
    #[cfg(feature = "test-utils")]
    m.add_function(wrap_pyfunction!(ffi::corpus_index, m)?)?;
    #[cfg(feature = "test-utils")]
    m.add_function(wrap_pyfunction!(ffi::corpus_json_schema, m)?)?;
    #[cfg(feature = "test-utils")]
    m.add_function(wrap_pyfunction!(ffi::corpus_names, m)?)?;
    #[cfg(feature = "test-utils")]
    m.add_function(wrap_pyfunction!(ffi::corpus_samples, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::trie_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::configure_vocab_build_budget, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::configure_vocab_serving_budget, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::vocab_build_budget_stats, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::vocab_serving_budget_stats, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::configure_automaton_build_budget, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::automaton_build_budget_stats, m)?)?;
    Ok(())
}
