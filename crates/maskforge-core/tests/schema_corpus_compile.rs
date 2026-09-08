use std::sync::Arc;

use maskforge_core::{compile_ir, schema_to_ir, CompileOptions, ObjectClosure, StructuredProgram};

fn compile_corpus_file(relative: &str) -> Result<(), maskforge_core::CompileError> {
    compile_corpus_file_with_format(relative, false)
}

fn compile_corpus_file_with_format(
    relative: &str,
    format_assertion: bool,
) -> Result<(), maskforge_core::CompileError> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../integration/jsonschemabench/data")
        .join(relative);
    // The corpus is an optional local checkout, not part of this repository.
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("skipping: corpus not present at {}", path.display());
        return Ok(());
    };
    let ir = schema_to_ir(
        &text,
        CompileOptions {
            object_closure: ObjectClosure::AssumeClosedProfile,
            format_assertion,
            ..CompileOptions::default()
        },
    )?;
    if ir.requires_structured_backend() {
        StructuredProgram::compile(Arc::new(ir)).map(|_| ())
    } else {
        compile_ir(&ir).map(|_| ())
    }
}

#[test]
fn former_product_cap_routes_to_a_complete_executable() {
    for path in ["Github_trivial/o41599.json", "Github_trivial/o65455.json"] {
        compile_corpus_file(path).unwrap();
    }
}

#[test]
fn bounded_string_shuffle_failures_route_to_complete_executables() {
    for path in [
        "Github_easy/o21459.json",
        "Github_easy/o40224.json",
        "Github_easy/o90407.json",
        "Github_trivial/o44010.json",
        "Github_trivial/o6269.json",
        "Snowplow/sp_233_Normalized.json",
        "Snowplow/sp_405_Normalized.json",
        "Snowplow/sp_40_Normalized.json",
        "Snowplow/sp_41_Normalized.json",
        "Snowplow/sp_44_Normalized.json",
    ] {
        compile_corpus_file(path).unwrap_or_else(|error| panic!("{path}: {error}"));
    }
}

#[test]
fn measured_child_dfas_route_before_the_shuffle_edge_cap() {
    for path in [
        "Github_easy/o43976.json",
        "Github_easy/o43981.json",
        "Github_easy/o90210.json",
        "Github_easy/o90215.json",
        "Github_medium/o8478.json",
        "Snowplow/sp_107_Normalized.json",
    ] {
        compile_corpus_file(path).unwrap_or_else(|error| panic!("{path}: {error}"));
    }
}

#[test]
fn structured_leaf_dfa_failures_have_typed_current_results() {
    for path in ["Github_hard/o72113.json", "JsonSchemaStore/dein.json"] {
        if let Err(error) = compile_corpus_file(path) {
            assert_ne!(
                error.observed, None,
                "{path}: current error must preserve its cause"
            );
            assert_ne!(
                error.code,
                maskforge_core::ErrorCode::Malformed,
                "{path}: a construction budget is not malformed schema syntax"
            );
        }
    }
}

#[test]
fn ecma_shorthand_property_patterns_compile_through_the_canonical_path() {
    for path in [
        "Github_easy/o21729.json",
        "Github_hard/o12613.json",
        "Github_hard/o366.json",
        "Github_hard/o37789.json",
        "Github_hard/o83132.json",
        "Github_hard/o83133.json",
        "Github_medium/o29393.json",
        "Github_trivial/o76865.json",
        "JsonSchemaStore/bamboo-spec.json",
    ] {
        compile_corpus_file_with_format(path, true)
            .unwrap_or_else(|error| panic!("{path}: {error}; {:?}", error.observed));
    }
}

#[test]
fn former_state_width_failure_routes_or_returns_a_typed_limit() {
    if let Err(error) = compile_corpus_file("Github_medium/o65507.json") {
        assert_eq!(error.code, maskforge_core::ErrorCode::InternalLimitExceeded);
        assert!(
            error.limit.is_some(),
            "resource failures require observed/cap fields"
        );
    }
}

#[test]
fn repeated_non_recursive_refs_do_not_consume_the_expansion_budget() {
    for path in [
        "Github_ultra/o13934.json",
        "Github_ultra/o21764.json",
        "JsonSchemaStore/theme.json",
    ] {
        compile_corpus_file(path).unwrap_or_else(|error| panic!("{path}: {error}"));
    }
}
