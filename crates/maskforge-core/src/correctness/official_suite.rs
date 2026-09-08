//! Validates results against the pinned Draft 2020-12 test suite.

use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use crate::automaton::RefEngine;
use crate::compile::{compile_ir, compile_ir_timed};
use crate::diagnostics::UnsupportedReason;
use crate::error::{CompileError, ErrorCode, Stage};
use crate::frontend::{
    schema_to_ir, schema_to_ir_timed, schema_to_ir_with_resources, SchemaRegistry,
    SchemaResourceLimits,
};
use crate::ir::{CompileOptions, ObjectClosure};
use crate::structured::{StructuredMatcherError, StructuredProgram};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SuiteProfile {
    Draft202012Default,
    AssumeClosedGeneration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupDisposition {
    Executed,
    Unsupported,
    Malformed,
    BackendCompileError,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MismatchKind {
    OverAcceptance,
    UnderAcceptance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecutionBackend {
    Automaton,
    StructuredIncremental,
    StructuredReference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KnownDifferenceReason {
    CanonicalNumericSpelling,
    CanonicalCompositeSpelling,
    AssumeClosedProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CaseId {
    file: &'static str,
    group_index: usize,
    case_index: usize,
    group_description: String,
    case_description: String,
}

#[derive(Clone, Debug)]
struct CaseMismatch {
    id: CaseId,
    expected_valid: bool,
    actual_valid: bool,
    kind: MismatchKind,
    backend: ExecutionBackend,
    data: Value,
    reason: Option<KnownDifferenceReason>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct UnsupportedDetail {
    reason: UnsupportedReason,
    keyword: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GroupFailureReason {
    Unsupported(Vec<UnsupportedDetail>),
    Compile {
        code: ErrorCode,
        stage: Stage,
        keyword: Option<String>,
    },
}

#[derive(Clone, Debug)]
struct GroupFailure {
    file: &'static str,
    group_index: usize,
    description: String,
    disposition: GroupDisposition,
    reason: GroupFailureReason,
    error: Option<CompileError>,
}

#[derive(Clone, Debug)]
struct RuntimeFailure {
    id: CaseId,
    backend: ExecutionBackend,
    error: StructuredMatcherError,
}

#[derive(Default, Debug)]
struct SuiteReport {
    files: usize,
    executed_groups: usize,
    unsupported_groups: usize,
    malformed_groups: usize,
    backend_compile_errors: usize,
    automaton_groups: usize,
    incremental_groups: usize,
    reference_groups: usize,
    cases_passed: usize,
    expected_under_acceptance: usize,
    unexpected_under_acceptance: Vec<CaseMismatch>,
    over_acceptance: Vec<CaseMismatch>,
    unexpected_groups: Vec<GroupFailure>,
    runtime_failures: Vec<RuntimeFailure>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExpectedMismatch {
    file: &'static str,
    group_index: usize,
    case_index: usize,
    expected_valid: bool,
    reason: KnownDifferenceReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedGroupReason {
    Malformed {
        code: ErrorCode,
        stage: Stage,
        keyword: Option<&'static str>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExpectedGroupDisposition {
    file: &'static str,
    group_index: usize,
    disposition: GroupDisposition,
    reason: ExpectedGroupReason,
}

const FILES: &[&str] = &[
    "additionalProperties.json",
    "allOf.json",
    "anchor.json",
    "anyOf.json",
    "boolean_schema.json",
    "const.json",
    "contains.json",
    "content.json",
    "default.json",
    "defs.json",
    "dependentRequired.json",
    "dependentSchemas.json",
    "dynamicRef.json",
    "enum.json",
    "exclusiveMaximum.json",
    "exclusiveMinimum.json",
    "format.json",
    "if-then-else.json",
    "infinite-loop-detection.json",
    "items.json",
    "maxContains.json",
    "maximum.json",
    "maxItems.json",
    "maxLength.json",
    "maxProperties.json",
    "minContains.json",
    "minimum.json",
    "minItems.json",
    "minLength.json",
    "minProperties.json",
    "multipleOf.json",
    "not.json",
    "oneOf.json",
    "pattern.json",
    "patternProperties.json",
    "prefixItems.json",
    "properties.json",
    "propertyNames.json",
    "ref.json",
    "refRemote.json",
    "required.json",
    "type.json",
    "unevaluatedItems.json",
    "unevaluatedProperties.json",
    "uniqueItems.json",
    "vocabulary.json",
];

const EXCLUDED_FILES: &[(&str, &str)] = &[];

const EXPECTED_MISMATCHES: &[ExpectedMismatch] = &[
    expected_mismatch(
        "const.json",
        1,
        1,
        KnownDifferenceReason::CanonicalCompositeSpelling,
    ),
    expected_mismatch(
        "enum.json",
        10,
        2,
        KnownDifferenceReason::CanonicalNumericSpelling,
    ),
    expected_mismatch(
        "enum.json",
        12,
        2,
        KnownDifferenceReason::CanonicalNumericSpelling,
    ),
    expected_mismatch(
        "type.json",
        0,
        1,
        KnownDifferenceReason::CanonicalNumericSpelling,
    ),
];

const EXPECTED_ASSUME_CLOSED_MISMATCHES: &[ExpectedMismatch] = &[
    expected_mismatch(
        "additionalProperties.json",
        4,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "allOf.json",
        0,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "allOf.json",
        1,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "anyOf.json",
        5,
        2,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "const.json",
        1,
        1,
        KnownDifferenceReason::CanonicalCompositeSpelling,
    ),
    expected_mismatch(
        "dependentRequired.json",
        0,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentRequired.json",
        0,
        2,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentRequired.json",
        1,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentRequired.json",
        2,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentRequired.json",
        2,
        2,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentRequired.json",
        3,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentRequired.json",
        3,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentSchemas.json",
        0,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "defs.json",
        0,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dynamicRef.json",
        9,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dynamicRef.json",
        10,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dynamicRef.json",
        11,
        0,
        KnownDifferenceReason::CanonicalNumericSpelling,
    ),
    expected_mismatch(
        "dynamicRef.json",
        11,
        3,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentSchemas.json",
        0,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentSchemas.json",
        1,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentSchemas.json",
        2,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentSchemas.json",
        3,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "dependentSchemas.json",
        3,
        3,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "enum.json",
        10,
        2,
        KnownDifferenceReason::CanonicalNumericSpelling,
    ),
    expected_mismatch(
        "enum.json",
        12,
        2,
        KnownDifferenceReason::CanonicalNumericSpelling,
    ),
    expected_mismatch("not.json", 3, 1, KnownDifferenceReason::AssumeClosedProfile),
    expected_mismatch(
        "oneOf.json",
        8,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "oneOf.json",
        8,
        2,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "properties.json",
        0,
        3,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch("ref.json", 6, 0, KnownDifferenceReason::AssumeClosedProfile),
    expected_mismatch(
        "ref.json",
        15,
        2,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "ref.json",
        16,
        2,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "type.json",
        0,
        1,
        KnownDifferenceReason::CanonicalNumericSpelling,
    ),
    expected_mismatch(
        "type.json",
        9,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "type.json",
        10,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "unevaluatedProperties.json",
        21,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "unevaluatedProperties.json",
        30,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "unevaluatedProperties.json",
        31,
        1,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
    expected_mismatch(
        "unevaluatedProperties.json",
        32,
        2,
        KnownDifferenceReason::AssumeClosedProfile,
    ),
];

const fn expected_mismatch(
    file: &'static str,
    group_index: usize,
    case_index: usize,
    reason: KnownDifferenceReason,
) -> ExpectedMismatch {
    ExpectedMismatch {
        file,
        group_index,
        case_index,
        expected_valid: true,
        reason,
    }
}

const EXPECTED_GROUP_DISPOSITIONS: &[ExpectedGroupDisposition] = &[];

fn options(profile: SuiteProfile) -> CompileOptions {
    match profile {
        SuiteProfile::Draft202012Default => CompileOptions::default(),
        SuiteProfile::AssumeClosedGeneration => CompileOptions {
            object_closure: ObjectClosure::AssumeClosedProfile,
            ..CompileOptions::default()
        },
    }
}

enum CaseExecutor {
    Automaton(Box<RefEngine>),
    Structured {
        program: Arc<StructuredProgram>,
        backend: ExecutionBackend,
    },
}

impl CaseExecutor {
    fn backend(&self) -> ExecutionBackend {
        match self {
            Self::Automaton(_) => ExecutionBackend::Automaton,
            Self::Structured { backend, .. } => *backend,
        }
    }

    fn accepts(&self, data: &[u8]) -> Result<bool, StructuredMatcherError> {
        match self {
            Self::Automaton(engine) => Ok(engine.accepts(data)),
            Self::Structured { program, .. } => {
                let mut matcher = program.new_matcher()?;
                Ok(matcher.advance(data)? && matcher.eos_legal())
            }
        }
    }
}

fn build_executor(ir: Arc<crate::ir::SchemaIR>) -> Result<CaseExecutor, CompileError> {
    let requires_structured = ir.requires_structured_backend();
    match select_executor(
        requires_structured,
        || compile_ir(&ir),
        || StructuredProgram::compile(ir.clone()),
    )? {
        SelectedExecutor::Automaton(engine) => Ok(CaseExecutor::Automaton(Box::new(engine))),
        SelectedExecutor::Structured(program) => {
            let backend = if program.uses_incremental_backend() {
                ExecutionBackend::StructuredIncremental
            } else {
                ExecutionBackend::StructuredReference
            };
            Ok(CaseExecutor::Structured { program, backend })
        }
    }
}

#[derive(Debug)]
enum SelectedExecutor<A, S> {
    Automaton(A),
    Structured(S),
}

fn select_executor<A, S>(
    requires_structured: bool,
    automaton: impl FnOnce() -> Result<A, CompileError>,
    structured: impl FnOnce() -> Result<S, CompileError>,
) -> Result<SelectedExecutor<A, S>, CompileError> {
    if requires_structured {
        structured().map(SelectedExecutor::Structured)
    } else {
        automaton().map(SelectedExecutor::Automaton)
    }
}

fn classify_difference(
    file: &'static str,
    group_index: usize,
    data: &Value,
    profile: SuiteProfile,
) -> Option<KnownDifferenceReason> {
    if file == "const.json" && group_index == 1 && data.is_object() {
        return Some(KnownDifferenceReason::CanonicalCompositeSpelling);
    }
    if contains_integral_float(data) {
        return Some(KnownDifferenceReason::CanonicalNumericSpelling);
    }
    (profile == SuiteProfile::AssumeClosedGeneration)
        .then_some(KnownDifferenceReason::AssumeClosedProfile)
}

fn contains_integral_float(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.is_f64() && number.as_f64().is_some_and(f64::is_finite),
        Value::Array(values) => values.iter().any(contains_integral_float),
        Value::Object(values) => values.values().any(contains_integral_float),
        _ => false,
    }
}

fn group_failure(
    file: &'static str,
    group_index: usize,
    description: String,
    disposition: GroupDisposition,
    error: CompileError,
) -> GroupFailure {
    let reason = GroupFailureReason::Compile {
        code: error.code,
        stage: error.stage,
        keyword: error.keyword.clone(),
    };
    GroupFailure {
        file,
        group_index,
        description,
        disposition,
        reason,
        error: Some(error),
    }
}

fn run_suite(profile: SuiteProfile) -> SuiteReport {
    let mut report = SuiteReport {
        files: FILES.len(),
        ..SuiteReport::default()
    };
    let mut actual_groups = Vec::new();
    let mut actual_under = Vec::new();
    for &file in FILES {
        let path = suite_path(file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("official suite file missing ({path}): {error}"));
        let groups: Vec<Value> =
            serde_json::from_str(&text).unwrap_or_else(|error| panic!("{path}: {error}"));
        for (group_index, group) in groups.iter().enumerate() {
            run_group(
                file,
                group_index,
                group,
                profile,
                &mut report,
                &mut actual_groups,
                &mut actual_under,
            );
        }
    }
    reconcile_groups(actual_groups, &mut report);
    let expected = match profile {
        SuiteProfile::Draft202012Default => EXPECTED_MISMATCHES,
        SuiteProfile::AssumeClosedGeneration => EXPECTED_ASSUME_CLOSED_MISMATCHES,
    };
    reconcile_mismatches(actual_under, expected, &mut report);
    report
}

fn run_group(
    file: &'static str,
    group_index: usize,
    group: &Value,
    profile: SuiteProfile,
    report: &mut SuiteReport,
    actual_groups: &mut Vec<GroupFailure>,
    actual_under: &mut Vec<CaseMismatch>,
) {
    let description = group["description"].as_str().unwrap_or("").to_string();
    let schema_text =
        serde_json::to_string(&group["schema"]).expect("suite schema is serializable");
    let ir = match compile_official_schema(file, &schema_text, options(profile)) {
        Ok(ir) => ir,
        Err(error) => {
            report.malformed_groups += 1;
            actual_groups.push(group_failure(
                file,
                group_index,
                description,
                GroupDisposition::Malformed,
                error,
            ));
            return;
        }
    };
    let diagnostics: Vec<_> = ir
        .diagnostics()
        .map(|diagnostic| UnsupportedDetail {
            reason: diagnostic.reason,
            keyword: ir.str_at(diagnostic.keyword).unwrap_or("").to_string(),
        })
        .collect();
    if !diagnostics.is_empty() {
        report.unsupported_groups += 1;
        actual_groups.push(GroupFailure {
            file,
            group_index,
            description,
            disposition: GroupDisposition::Unsupported,
            reason: GroupFailureReason::Unsupported(diagnostics),
            error: None,
        });
        return;
    }
    let executor = match build_executor(Arc::new(ir)) {
        Ok(executor) => executor,
        Err(error) => {
            report.backend_compile_errors += 1;
            actual_groups.push(group_failure(
                file,
                group_index,
                description,
                GroupDisposition::BackendCompileError,
                error,
            ));
            return;
        }
    };
    let backend = executor.backend();
    record_executed_group(report, GroupDisposition::Executed, backend);
    for (case_index, case) in group["tests"].as_array().into_iter().flatten().enumerate() {
        run_case(
            file,
            group_index,
            case_index,
            &description,
            case,
            profile,
            &executor,
            report,
            actual_under,
        );
    }
}

const REMOTE_2020_12_FIXTURES: &[&str] = &[
    "detached-dynamicref.json",
    "detached-ref.json",
    "different-id-ref-string.json",
    "extendible-dynamic-ref.json",
    "integer.json",
    "locationIndependentIdentifier.json",
    "metaschema-no-validation.json",
    "metaschema-optional-vocabulary.json",
    "name-defs.json",
    "nested-absolute-ref-to-string.json",
    "prefixItems.json",
    "ref-and-defs.json",
    "subSchemas.json",
    "tree.json",
    "urn-ref-string.json",
    "baseUriChange/folderInteger.json",
    "baseUriChangeFolder/folderInteger.json",
    "baseUriChangeFolderInSubschema/folderInteger.json",
    "nested/foo-ref-string.json",
    "nested/string.json",
];

const META_2020_12_FIXTURES: &[(&str, &str)] = &[
    (
        "https://json-schema.org/draft/2020-12/schema",
        "schema.json",
    ),
    (
        "https://json-schema.org/draft/2020-12/meta/core",
        "meta-core.json",
    ),
    (
        "https://json-schema.org/draft/2020-12/meta/applicator",
        "meta-applicator.json",
    ),
    (
        "https://json-schema.org/draft/2020-12/meta/unevaluated",
        "meta-unevaluated.json",
    ),
    (
        "https://json-schema.org/draft/2020-12/meta/validation",
        "meta-validation.json",
    ),
    (
        "https://json-schema.org/draft/2020-12/meta/meta-data",
        "meta-meta-data.json",
    ),
    (
        "https://json-schema.org/draft/2020-12/meta/format-annotation",
        "meta-format-annotation.json",
    ),
    (
        "https://json-schema.org/draft/2020-12/meta/content",
        "meta-content.json",
    ),
];

fn compile_official_schema(
    file: &'static str,
    schema_text: &str,
    options: CompileOptions,
) -> Result<crate::ir::SchemaIR, CompileError> {
    if !matches!(
        file,
        "defs.json" | "ref.json" | "refRemote.json" | "dynamicRef.json" | "vocabulary.json"
    ) {
        return schema_to_ir(schema_text, options);
    }
    let limits = SchemaResourceLimits::default();
    let mut registry = SchemaRegistry::new(limits);
    for relative in REMOTE_2020_12_FIXTURES {
        let path = format!(
            "{}/tools/json-schema-test-suite/remotes/draft2020-12/{relative}",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("official remote fixture missing ({path}): {error}"));
        let uri = format!("http://localhost:1234/draft2020-12/{relative}");
        registry.insert(uri, text)?;
    }
    for (uri, fixture) in META_2020_12_FIXTURES {
        let path = format!(
            "{}/tools/json-schema-test-suite/remotes/draft2020-12/official-meta/{fixture}",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!("official meta-schema fixture missing ({path}): {error}")
        });
        registry.insert(*uri, text)?;
    }
    schema_to_ir_with_resources(schema_text, None, options, limits, registry)
}

fn record_executed_group(
    report: &mut SuiteReport,
    disposition: GroupDisposition,
    backend: ExecutionBackend,
) {
    assert_eq!(disposition, GroupDisposition::Executed);
    report.executed_groups += 1;
    match backend {
        ExecutionBackend::Automaton => report.automaton_groups += 1,
        ExecutionBackend::StructuredIncremental => report.incremental_groups += 1,
        ExecutionBackend::StructuredReference => report.reference_groups += 1,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    file: &'static str,
    group_index: usize,
    case_index: usize,
    group_description: &str,
    case: &Value,
    profile: SuiteProfile,
    executor: &CaseExecutor,
    report: &mut SuiteReport,
    actual_under: &mut Vec<CaseMismatch>,
) {
    let id = CaseId {
        file,
        group_index,
        case_index,
        group_description: group_description.to_string(),
        case_description: case["description"].as_str().unwrap_or("").to_string(),
    };
    let data = &case["data"];
    let bytes = serde_json::to_vec(data).expect("suite instance is serializable");
    let expected_valid = case["valid"].as_bool().expect("suite case has valid");
    let Some(actual_valid) = record_execution_result(
        executor.accepts(&bytes),
        id.clone(),
        executor.backend(),
        report,
    ) else {
        return;
    };
    if actual_valid == expected_valid {
        report.cases_passed += 1;
        return;
    }
    let kind = if actual_valid {
        MismatchKind::OverAcceptance
    } else {
        MismatchKind::UnderAcceptance
    };
    let mismatch = CaseMismatch {
        id,
        expected_valid,
        actual_valid,
        kind,
        backend: executor.backend(),
        data: data.clone(),
        reason: classify_difference(file, group_index, data, profile),
    };
    match kind {
        MismatchKind::OverAcceptance => report.over_acceptance.push(mismatch),
        MismatchKind::UnderAcceptance => actual_under.push(mismatch),
    }
}

fn record_execution_result(
    result: Result<bool, StructuredMatcherError>,
    id: CaseId,
    backend: ExecutionBackend,
    report: &mut SuiteReport,
) -> Option<bool> {
    match result {
        Ok(actual) => Some(actual),
        Err(error) => {
            report
                .runtime_failures
                .push(RuntimeFailure { id, backend, error });
            None
        }
    }
}

fn expected_group_matches(actual: &GroupFailure, expected: ExpectedGroupDisposition) -> bool {
    if actual.file != expected.file
        || actual.group_index != expected.group_index
        || actual.disposition != expected.disposition
    {
        return false;
    }
    match (&actual.reason, expected.reason) {
        (
            GroupFailureReason::Compile {
                code,
                stage,
                keyword,
            },
            ExpectedGroupReason::Malformed {
                code: expected_code,
                stage: expected_stage,
                keyword: expected_keyword,
            },
        ) => {
            *code == expected_code
                && *stage == expected_stage
                && keyword.as_deref() == expected_keyword
        }
        _ => false,
    }
}

fn reconcile_groups(mut actual: Vec<GroupFailure>, report: &mut SuiteReport) {
    reconcile_groups_against(&mut actual, EXPECTED_GROUP_DISPOSITIONS, report);
}

fn reconcile_groups_against(
    actual: &mut [GroupFailure],
    expected_groups: &[ExpectedGroupDisposition],
    report: &mut SuiteReport,
) {
    actual.sort_by_key(|group| (group.file, group.group_index));
    for group in actual.iter() {
        if !expected_groups
            .iter()
            .copied()
            .any(|expected| expected_group_matches(group, expected))
        {
            report.unexpected_groups.push(group.clone());
        }
    }
    for expected in expected_groups {
        if !actual
            .iter()
            .any(|group| expected_group_matches(group, *expected))
        {
            report
                .unexpected_groups
                .push(stale_group_expectation(*expected));
        }
    }
}

fn stale_group_expectation(expected: ExpectedGroupDisposition) -> GroupFailure {
    let reason = match expected.reason {
        ExpectedGroupReason::Malformed {
            code,
            stage,
            keyword,
        } => GroupFailureReason::Compile {
            code,
            stage,
            keyword: keyword.map(str::to_string),
        },
    };
    GroupFailure {
        file: expected.file,
        group_index: expected.group_index,
        description: "stale expected group disposition".to_string(),
        disposition: expected.disposition,
        reason,
        error: None,
    }
}

fn mismatch_matches(actual: &CaseMismatch, expected: ExpectedMismatch) -> bool {
    actual.id.file == expected.file
        && actual.id.group_index == expected.group_index
        && actual.id.case_index == expected.case_index
        && actual.expected_valid == expected.expected_valid
        && actual.actual_valid != expected.expected_valid
        && actual.reason == Some(expected.reason)
}

fn reconcile_mismatches(
    mut actual: Vec<CaseMismatch>,
    expected_mismatches: &[ExpectedMismatch],
    report: &mut SuiteReport,
) {
    actual.sort_by(|left, right| left.id.cmp(&right.id));
    for mismatch in &actual {
        if expected_mismatches
            .iter()
            .copied()
            .any(|expected| mismatch_matches(mismatch, expected))
        {
            report.expected_under_acceptance += 1;
        } else {
            report.unexpected_under_acceptance.push(mismatch.clone());
        }
    }
    for expected in expected_mismatches {
        if !actual
            .iter()
            .any(|mismatch| mismatch_matches(mismatch, *expected))
        {
            report
                .unexpected_under_acceptance
                .push(stale_mismatch_expectation(*expected));
        }
    }
}

fn stale_mismatch_expectation(expected: ExpectedMismatch) -> CaseMismatch {
    CaseMismatch {
        id: CaseId {
            file: expected.file,
            group_index: expected.group_index,
            case_index: expected.case_index,
            group_description: "stale expected mismatch".to_string(),
            case_description: String::new(),
        },
        expected_valid: expected.expected_valid,
        actual_valid: expected.expected_valid,
        kind: MismatchKind::UnderAcceptance,
        backend: ExecutionBackend::Automaton,
        data: Value::Null,
        reason: Some(expected.reason),
    }
}

fn suite_directory() -> String {
    format!(
        "{}/tools/json-schema-test-suite/draft2020-12",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn discovered_suite_files(directory: &Path) -> BTreeSet<String> {
    std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("cannot scan {}: {error}", directory.display()))
        .map(|entry| entry.expect("suite directory entry is readable").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .expect("suite filename is UTF-8")
                .to_string()
        })
        .collect()
}

fn assert_suite_provenance() {
    let directory = suite_directory();
    let discovered = discovered_suite_files(Path::new(&directory));
    let exclusions: Vec<_> = EXCLUDED_FILES
        .iter()
        .map(|(file, reason)| (*file, Some(*reason)))
        .collect();
    assert!(
        provenance_matches(&discovered, FILES, &exclusions),
        "vendored suite file set changed"
    );
    for file in FILES {
        assert!(Path::new(&suite_path(file)).is_file(), "missing {file}");
    }
    let provenance = include_str!("../../tools/json-schema-test-suite/PROVENANCE.md");
    assert!(provenance.contains("draft2020-12"));
    assert!(provenance.contains("refRemote.json"));
}

fn provenance_matches(
    discovered: &BTreeSet<String>,
    included: &[&str],
    exclusions: &[(&str, Option<&str>)],
) -> bool {
    if exclusions.iter().any(|(_, reason)| reason.is_none()) {
        return false;
    }
    let expected: BTreeSet<_> = included
        .iter()
        .copied()
        .chain(exclusions.iter().map(|(file, _)| *file))
        .map(str::to_string)
        .collect();
    discovered == &expected
}

fn print_report(label: &str, report: &SuiteReport) {
    for mismatch in &report.over_acceptance {
        print_mismatch("UNEXPECTED OVER-ACCEPTANCE", mismatch);
    }
    for mismatch in &report.unexpected_under_acceptance {
        print_mismatch("UNEXPECTED UNDER-ACCEPTANCE", mismatch);
    }
    for group in &report.unexpected_groups {
        println!(
            "UNEXPECTED GROUP DISPOSITION\n{}[{}] {:?}\ndescription={}\nreason={:?}\nerror={:?}",
            group.file,
            group.group_index,
            group.disposition,
            group.description,
            group.reason,
            group.error,
        );
    }
    for failure in &report.runtime_failures {
        println!(
            "RUNTIME OR RESOURCE FAILURE\n{}[{}]/{} backend={:?}\nerror={:?}",
            failure.id.file,
            failure.id.group_index,
            failure.id.case_index,
            failure.backend,
            failure.error,
        );
    }
    println!(
        "{label}: files={} executed_groups={} unsupported_groups={} malformed_groups={} \
         backend_compile_errors={} automaton_groups={} incremental_groups={} reference_groups={} \
         cases_passed={} expected_under_acceptance={} unexpected_under_acceptance={} \
         over_acceptance={} runtime_failures={}",
        report.files,
        report.executed_groups,
        report.unsupported_groups,
        report.malformed_groups,
        report.backend_compile_errors,
        report.automaton_groups,
        report.incremental_groups,
        report.reference_groups,
        report.cases_passed,
        report.expected_under_acceptance,
        report.unexpected_under_acceptance.len(),
        report.over_acceptance.len(),
        report.runtime_failures.len(),
    );
}

fn print_mismatch(label: &str, mismatch: &CaseMismatch) {
    println!(
        "{label}\n{}[{}]/{}\nexpected={} actual={}\nkind={:?} backend={:?}\ndata={}\nreason={:?}",
        mismatch.id.file,
        mismatch.id.group_index,
        mismatch.id.case_index,
        mismatch.expected_valid,
        mismatch.actual_valid,
        mismatch.kind,
        mismatch.backend,
        mismatch.data,
        mismatch.reason,
    );
}

fn assert_report_clean(report: &SuiteReport) {
    assert!(
        report_is_clean(report),
        "official suite report is not clean"
    );
}

fn report_is_clean(report: &SuiteReport) -> bool {
    report.over_acceptance.is_empty()
        && report.unexpected_under_acceptance.is_empty()
        && report.unexpected_groups.is_empty()
        && report.runtime_failures.is_empty()
}

fn suite_path(file: &str) -> String {
    format!(
        "{}/tools/json-schema-test-suite/draft2020-12/{file}",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn official_suite_default_profile_has_no_unexpected_results() {
    assert_suite_provenance();
    let report = run_suite(SuiteProfile::Draft202012Default);
    print_report("default standards profile", &report);
    assert_report_clean(&report);
    assert_eq!(report.files, 46);
    assert_eq!(report.executed_groups, 383);
    assert_eq!(report.unsupported_groups, 0);
    assert_eq!(report.malformed_groups, 0);
    assert_eq!(report.backend_compile_errors, 0);
    assert_eq!(report.automaton_groups, 38);
    assert_eq!(report.incremental_groups, 345);
    assert_eq!(report.reference_groups, 0);
    assert_eq!(report.cases_passed, 1_295);
    assert_eq!(report.expected_under_acceptance, 4);
}

#[test]
fn official_ref_remote_uses_only_the_offline_registry() {
    let file = "refRemote.json";
    let text = std::fs::read_to_string(suite_path(file)).expect("refRemote fixture");
    let groups: Vec<Value> = serde_json::from_str(&text).expect("refRemote JSON");
    let mut report = SuiteReport::default();
    let mut failures = Vec::new();
    let mut under = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        run_group(
            file,
            group_index,
            group,
            SuiteProfile::Draft202012Default,
            &mut report,
            &mut failures,
            &mut under,
        );
    }
    print_report("offline refRemote", &report);
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(under.is_empty(), "{under:#?}");
    assert!(report.over_acceptance.is_empty());
    assert!(report.runtime_failures.is_empty());
    assert_eq!(report.reference_groups, 0);
    assert_eq!(report.executed_groups, groups.len());
}

#[test]
fn official_dynamic_ref_uses_incremental_scope() {
    let file = "dynamicRef.json";
    let text = std::fs::read_to_string(suite_path(file)).expect("dynamicRef fixture");
    let groups: Vec<Value> = serde_json::from_str(&text).expect("dynamicRef JSON");
    let mut report = SuiteReport::default();
    let mut failures = Vec::new();
    let mut under = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        run_group(
            file,
            group_index,
            group,
            SuiteProfile::Draft202012Default,
            &mut report,
            &mut failures,
            &mut under,
        );
    }
    print_report("offline dynamicRef", &report);
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(under.is_empty(), "{under:#?}");
    assert!(report.over_acceptance.is_empty());
    assert!(report.runtime_failures.is_empty());
    assert_eq!(report.reference_groups, 0);
    assert_eq!(report.executed_groups, groups.len());
}

#[test]
fn official_suite_assume_closed_generation_profile_inventory() {
    assert_suite_provenance();
    let report = run_suite(SuiteProfile::AssumeClosedGeneration);
    print_report("assume-closed generation profile", &report);
    assert_report_clean(&report);
    assert_eq!(report.files, 46);
    assert_eq!(report.executed_groups, 383);
    assert_eq!(report.unsupported_groups, 0);
    assert_eq!(report.malformed_groups, 0);
    assert_eq!(report.backend_compile_errors, 0);
    assert_eq!(report.automaton_groups, 40);
    assert_eq!(report.incremental_groups, 343);
    assert_eq!(report.reference_groups, 0);
    assert_eq!(report.cases_passed, 1_260);
    assert_eq!(report.expected_under_acceptance, 39);
}

fn synthetic_case(group_index: usize, case_index: usize) -> CaseId {
    CaseId {
        file: "synthetic.json",
        group_index,
        case_index,
        group_description: "group".to_string(),
        case_description: "case".to_string(),
    }
}

fn synthetic_mismatch(
    group_index: usize,
    case_index: usize,
    reason: KnownDifferenceReason,
) -> CaseMismatch {
    CaseMismatch {
        id: synthetic_case(group_index, case_index),
        expected_valid: true,
        actual_valid: false,
        kind: MismatchKind::UnderAcceptance,
        backend: ExecutionBackend::Automaton,
        data: Value::from(1.0),
        reason: Some(reason),
    }
}

fn synthetic_group(disposition: GroupDisposition, reason: GroupFailureReason) -> GroupFailure {
    GroupFailure {
        file: "synthetic.json",
        group_index: 0,
        description: "group".to_string(),
        disposition,
        reason,
        error: None,
    }
}

#[test]
fn official_suite_expectation_exact_sets_pass() {
    let expected = [expected_mismatch(
        "synthetic.json",
        0,
        0,
        KnownDifferenceReason::CanonicalNumericSpelling,
    )];
    let mut report = SuiteReport::default();
    reconcile_mismatches(
        vec![synthetic_mismatch(
            0,
            0,
            KnownDifferenceReason::CanonicalNumericSpelling,
        )],
        &expected,
        &mut report,
    );
    assert!(report_is_clean(&report));
    assert_eq!(report.expected_under_acceptance, 1);
}

#[test]
fn official_suite_expectation_new_mismatch_fails() {
    let mut report = SuiteReport::default();
    reconcile_mismatches(
        vec![synthetic_mismatch(
            0,
            0,
            KnownDifferenceReason::CanonicalNumericSpelling,
        )],
        &[],
        &mut report,
    );
    assert_eq!(report.unexpected_under_acceptance.len(), 1);
}

#[test]
fn official_suite_expectation_stale_mismatch_fails() {
    let expected = [expected_mismatch(
        "synthetic.json",
        0,
        0,
        KnownDifferenceReason::CanonicalNumericSpelling,
    )];
    let mut report = SuiteReport::default();
    reconcile_mismatches(Vec::new(), &expected, &mut report);
    assert_eq!(report.unexpected_under_acceptance.len(), 1);
}

#[test]
fn official_suite_expectation_duplicate_descriptions_use_indices() {
    let expected = [
        expected_mismatch(
            "synthetic.json",
            0,
            0,
            KnownDifferenceReason::CanonicalNumericSpelling,
        ),
        expected_mismatch(
            "synthetic.json",
            0,
            1,
            KnownDifferenceReason::CanonicalNumericSpelling,
        ),
    ];
    let actual = vec![
        synthetic_mismatch(0, 1, KnownDifferenceReason::CanonicalNumericSpelling),
        synthetic_mismatch(0, 0, KnownDifferenceReason::CanonicalNumericSpelling),
    ];
    let mut report = SuiteReport::default();
    reconcile_mismatches(actual, &expected, &mut report);
    assert!(report.unexpected_under_acceptance.is_empty());
}

#[test]
fn official_suite_expectation_over_acceptance_fails() {
    let mut report = SuiteReport::default();
    let mut mismatch = synthetic_mismatch(0, 0, KnownDifferenceReason::CanonicalNumericSpelling);
    mismatch.expected_valid = false;
    mismatch.actual_valid = true;
    mismatch.kind = MismatchKind::OverAcceptance;
    report.over_acceptance.push(mismatch);
    assert!(!report_is_clean(&report));
}

#[test]
fn official_suite_expectation_allowed_under_acceptance_passes() {
    let expected = [expected_mismatch(
        "synthetic.json",
        0,
        0,
        KnownDifferenceReason::CanonicalNumericSpelling,
    )];
    let mut report = SuiteReport::default();
    reconcile_mismatches(
        vec![synthetic_mismatch(
            0,
            0,
            KnownDifferenceReason::CanonicalNumericSpelling,
        )],
        &expected,
        &mut report,
    );
    assert_eq!(report.expected_under_acceptance, 1);
}

#[test]
fn official_suite_expectation_wrong_reason_fails() {
    let expected = [expected_mismatch(
        "synthetic.json",
        0,
        0,
        KnownDifferenceReason::AssumeClosedProfile,
    )];
    let mut report = SuiteReport::default();
    reconcile_mismatches(
        vec![synthetic_mismatch(
            0,
            0,
            KnownDifferenceReason::CanonicalNumericSpelling,
        )],
        &expected,
        &mut report,
    );
    assert_eq!(report.unexpected_under_acceptance.len(), 2);
}

#[test]
fn official_suite_expectation_unsupported_is_not_malformed() {
    let mut actual = vec![synthetic_group(
        GroupDisposition::Unsupported,
        GroupFailureReason::Unsupported(vec![UnsupportedDetail {
            reason: UnsupportedReason::UnsupportedKeyword,
            keyword: "x".to_string(),
        }]),
    )];
    let expected = [ExpectedGroupDisposition {
        file: "synthetic.json",
        group_index: 0,
        disposition: GroupDisposition::Malformed,
        reason: ExpectedGroupReason::Malformed {
            code: ErrorCode::Malformed,
            stage: Stage::L1,
            keyword: Some("x"),
        },
    }];
    let mut report = SuiteReport::default();
    reconcile_groups_against(&mut actual, &expected, &mut report);
    assert_eq!(report.unexpected_groups.len(), 2);
}

#[test]
fn official_suite_expectation_backend_error_is_not_case_rejection() {
    let error = CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L3, "limit");
    let mut actual = vec![group_failure(
        "synthetic.json",
        0,
        "group".to_string(),
        GroupDisposition::BackendCompileError,
        error,
    )];
    let mut report = SuiteReport::default();
    reconcile_groups_against(&mut actual, &[], &mut report);
    assert_eq!(
        report.unexpected_groups[0].disposition,
        GroupDisposition::BackendCompileError
    );
    assert_eq!(report.cases_passed, 0);
}

#[test]
fn official_suite_expectation_insertion_order_is_irrelevant() {
    let expected = [
        expected_mismatch(
            "synthetic.json",
            0,
            0,
            KnownDifferenceReason::CanonicalNumericSpelling,
        ),
        expected_mismatch(
            "synthetic.json",
            1,
            0,
            KnownDifferenceReason::CanonicalNumericSpelling,
        ),
    ];
    let mut left = SuiteReport::default();
    let mut right = SuiteReport::default();
    reconcile_mismatches(
        vec![
            synthetic_mismatch(0, 0, KnownDifferenceReason::CanonicalNumericSpelling),
            synthetic_mismatch(1, 0, KnownDifferenceReason::CanonicalNumericSpelling),
        ],
        &expected,
        &mut left,
    );
    reconcile_mismatches(
        vec![
            synthetic_mismatch(1, 0, KnownDifferenceReason::CanonicalNumericSpelling),
            synthetic_mismatch(0, 0, KnownDifferenceReason::CanonicalNumericSpelling),
        ],
        &expected,
        &mut right,
    );
    assert_eq!(
        left.expected_under_acceptance,
        right.expected_under_acceptance
    );
    assert_eq!(
        left.unexpected_under_acceptance.len(),
        right.unexpected_under_acceptance.len()
    );
}

#[test]
fn official_suite_expectation_new_suite_file_fails_provenance() {
    let discovered = BTreeSet::from(["a.json".to_string(), "new.json".to_string()]);
    assert!(!provenance_matches(&discovered, &["a.json"], &[]));
}

#[test]
fn official_suite_expectation_exclusion_requires_reason() {
    let discovered = BTreeSet::from(["a.json".to_string(), "excluded.json".to_string()]);
    assert!(!provenance_matches(
        &discovered,
        &["a.json"],
        &[("excluded.json", None)],
    ));
}

#[test]
fn official_suite_positive_recursion_routes_incrementally() {
    let schema =
        r##"{"$defs":{"x":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/x"}]}},"$ref":"#/$defs/x"}"##;
    let ir = schema_to_ir(schema, CompileOptions::default()).unwrap();
    let executor = build_executor(Arc::new(ir)).unwrap();
    assert_eq!(executor.backend(), ExecutionBackend::StructuredIncremental);
}

#[test]
fn official_suite_expectation_regular_compile_failure_does_not_route_structured() {
    let error = CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L3, "limit");
    let structured_called = std::cell::Cell::new(false);
    let result = select_executor::<(), ()>(
        false,
        || Err(error.clone()),
        || {
            structured_called.set(true);
            Ok(())
        },
    );
    assert_eq!(result.unwrap_err(), error);
    assert!(!structured_called.get());
}

#[test]
fn official_suite_expectation_runtime_error_is_not_rejection() {
    let error = StructuredMatcherError::ResourceLimit {
        kind: crate::error::LimitKind::DocumentBytes,
        observed: 2,
        limit: 1,
    };
    let mut report = SuiteReport::default();
    let actual = record_execution_result(
        Err(error.clone()),
        synthetic_case(0, 0),
        ExecutionBackend::StructuredIncremental,
        &mut report,
    );
    assert_eq!(actual, None);
    assert_eq!(report.runtime_failures[0].error, error);
    assert!(!report_is_clean(&report));
}

#[test]
fn official_negation_and_conditionals_use_incremental_programs() {
    for file in ["not.json", "if-then-else.json"] {
        let path = suite_path(file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("official suite file missing ({path}): {error}"));
        let groups: Vec<Value> =
            serde_json::from_str(&text).unwrap_or_else(|error| panic!("{path}: {error}"));
        let mut incremental_groups = 0usize;
        let mut cases_passed = 0usize;
        let mut cases_failed = 0usize;
        let mut fallbacks = Vec::new();
        for group in &groups {
            let schema_text = serde_json::to_string(&group["schema"]).unwrap();
            let ir = schema_to_ir(&schema_text, CompileOptions::default()).unwrap();
            if ir.diagnostics().len() != 0 {
                continue;
            }
            let program = match crate::structured::StructuredProgram::compile(Arc::new(ir)) {
                Ok(program) if program.uses_incremental_backend() => program,
                Ok(_) => {
                    fallbacks.push(format!("{schema_text}: reference backend"));
                    continue;
                }
                Err(error) => {
                    fallbacks.push(format!("{schema_text}: {error}"));
                    continue;
                }
            };
            incremental_groups += 1;
            for case in group["tests"].as_array().into_iter().flatten() {
                let data = serde_json::to_vec(&case["data"]).unwrap();
                let expected = case["valid"].as_bool().unwrap_or(false);
                let mut matcher = program.new_matcher().unwrap();
                let actual = matcher.advance(&data).unwrap() && matcher.is_accepting();
                if actual == expected {
                    cases_passed += 1;
                } else {
                    cases_failed += 1;
                }
            }
        }
        println!(
            "{file}: incremental_groups={incremental_groups} cases_passed={cases_passed} cases_failed={cases_failed} fallbacks={fallbacks:?}"
        );
        assert_eq!(
            cases_failed, 0,
            "{file}: incremental backend disagreed with the official suite"
        );
        match file {
            "not.json" => {
                assert_eq!(incremental_groups, 9);
                assert_eq!(cases_passed, 40);
                assert!(fallbacks.is_empty(), "{fallbacks:?}");
            }
            "if-then-else.json" => {
                assert_eq!(incremental_groups, 12);
                assert_eq!(cases_passed, 30);
                assert!(fallbacks.is_empty(), "{fallbacks:?}");
            }
            _ => unreachable!("test only iterates not.json and if-then-else.json"),
        }
    }
}

#[test]
fn official_dependent_schemas_use_incremental_programs() {
    let file = "dependentSchemas.json";
    let path = suite_path(file);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("official suite file missing ({path}): {error}"));
    let groups: Vec<Value> =
        serde_json::from_str(&text).unwrap_or_else(|error| panic!("{path}: {error}"));
    let mut incremental_groups = 0usize;
    let mut cases_passed = 0usize;
    let mut cases_failed = 0usize;
    let mut fallbacks = Vec::new();
    for group in &groups {
        let schema_text = serde_json::to_string(&group["schema"]).unwrap();
        let ir = schema_to_ir(&schema_text, CompileOptions::default()).unwrap();
        let program = match crate::structured::StructuredProgram::compile(Arc::new(ir)) {
            Ok(program) if program.uses_incremental_backend() => program,
            Ok(_) => {
                fallbacks.push(format!("{schema_text}: reference backend"));
                continue;
            }
            Err(error) => {
                fallbacks.push(format!("{schema_text}: {error}"));
                continue;
            }
        };
        incremental_groups += 1;
        for case in group["tests"].as_array().into_iter().flatten() {
            let data = serde_json::to_vec(&case["data"]).unwrap();
            let expected = case["valid"].as_bool().unwrap_or(false);
            let mut matcher = program.new_matcher().unwrap();
            let actual = matcher.advance(&data).unwrap() && matcher.is_accepting();
            if actual == expected {
                cases_passed += 1;
            } else {
                cases_failed += 1;
            }
        }
    }
    println!(
        "{file}: total_groups={} incremental_groups={incremental_groups} cases_passed={cases_passed} cases_failed={cases_failed} fallbacks={fallbacks:?}",
        groups.len()
    );
    assert_eq!(cases_failed, 0, "incremental dependentSchemas disagreement");
    assert_eq!(groups.len(), 4);
    assert_eq!(incremental_groups, 4);
    assert_eq!(cases_passed, 20);
    assert!(fallbacks.is_empty(), "unexpected fallbacks: {fallbacks:?}");
}

#[test]
fn official_dependent_required_is_four_groups_twenty_cases_incremental() {
    let file = "dependentRequired.json";
    let path = suite_path(file);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("official suite file missing ({path}): {error}"));
    let groups: Vec<Value> = serde_json::from_str(&text).unwrap();
    let mut incremental_groups = 0usize;
    let mut cases_passed = 0usize;
    let mut cases_failed = Vec::new();
    let mut fallbacks = Vec::new();
    for group in &groups {
        let schema_text = serde_json::to_string(&group["schema"]).unwrap();
        let ir = schema_to_ir(&schema_text, CompileOptions::default()).unwrap();
        assert_eq!(ir.diagnostics().len(), 0, "{schema_text}");
        let reference_ir = Arc::new(ir);
        let program = match crate::structured::StructuredProgram::compile(reference_ir.clone()) {
            Ok(program) if program.uses_incremental_backend() => program,
            Ok(_) => {
                fallbacks.push(format!("{schema_text}: reference backend"));
                continue;
            }
            Err(error) => {
                fallbacks.push(format!("{schema_text}: {error}"));
                continue;
            }
        };
        incremental_groups += 1;
        for case in group["tests"].as_array().into_iter().flatten() {
            let data = serde_json::to_vec(&case["data"]).unwrap();
            let expected = case["valid"].as_bool().unwrap();
            let mut matcher = program.new_matcher().unwrap();
            let incremental = matcher.advance(&data).unwrap() && matcher.is_accepting();
            let reference = crate::structured::accepts(&reference_ir, &data);
            if incremental == expected && reference == expected {
                cases_passed += 1;
            } else {
                cases_failed.push((schema_text.clone(), data, expected, incremental, reference));
            }
        }
    }
    println!(
        "{file}: incremental_groups={incremental_groups} cases_passed={cases_passed} cases_failed={} fallbacks={fallbacks:?}",
        cases_failed.len()
    );
    assert_eq!(groups.len(), 4);
    assert_eq!(incremental_groups, 4);
    assert_eq!(cases_passed, 20);
    assert!(cases_failed.is_empty(), "{cases_failed:?}");
    assert!(fallbacks.is_empty(), "{fallbacks:?}");
}

fn uses_dynamic_scope(value: &Value) -> bool {
    match value {
        Value::Object(fields) => {
            fields.contains_key("$dynamicRef")
                || fields.contains_key("$dynamicAnchor")
                || fields.values().any(uses_dynamic_scope)
        }
        Value::Array(items) => items.iter().any(uses_dynamic_scope),
        _ => false,
    }
}

#[test]
fn official_unevaluated_non_dynamic_is_incremental() {
    for (file, expected_groups, expected_cases) in [
        ("unevaluatedProperties.json", 43usize, 127usize),
        ("unevaluatedItems.json", 28usize, 69usize),
    ] {
        let path = suite_path(file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("official suite file missing ({path}): {error}"));
        let groups: Vec<Value> = serde_json::from_str(&text).unwrap();
        let mut group_count = 0usize;
        let mut case_count = 0usize;
        let mut failures = Vec::new();
        for group in &groups {
            if uses_dynamic_scope(&group["schema"]) {
                continue;
            }
            let schema_text = serde_json::to_string(&group["schema"]).unwrap();
            let ir = schema_to_ir(&schema_text, CompileOptions::default()).unwrap();
            assert_eq!(ir.diagnostics().len(), 0, "{file}: {schema_text}");
            let program = crate::structured::StructuredProgram::compile(Arc::new(ir)).unwrap();
            assert_eq!(
                program.backend_kind(),
                "incremental",
                "{file}: {schema_text}"
            );
            group_count += 1;
            for case in group["tests"].as_array().into_iter().flatten() {
                let data = serde_json::to_vec(&case["data"]).unwrap();
                let expected = case["valid"].as_bool().unwrap();
                let mut matcher = program.new_matcher().unwrap();
                assert_eq!(
                    matcher.backend_name(),
                    "incremental",
                    "{file}: {schema_text}"
                );
                let actual = matcher.advance(&data).unwrap() && matcher.is_accepting();
                case_count += 1;
                if actual != expected {
                    failures.push(format!(
                        "{} / {}: data={} expected={expected} actual={actual}",
                        group["description"].as_str().unwrap_or(""),
                        case["description"].as_str().unwrap_or(""),
                        case["data"]
                    ));
                }
            }
        }
        assert_eq!(group_count, expected_groups, "{file}");
        assert_eq!(case_count, expected_cases, "{file}");
        assert!(failures.is_empty(), "{file}: {failures:#?}");
    }
}

#[test]
#[ignore = "scans a 9558-file external dataset; run explicitly with --ignored"]
fn jsonschemabench_corpus_never_panics() {
    let root = format!(
        "{}/../../integration/jsonschemabench/data",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut scanned = 0usize;
    let mut supported = 0usize;
    let mut malformed = 0usize;
    let mut reasons: std::collections::HashMap<&'static str, usize> =
        std::collections::HashMap::new();
    let mut keywords: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut malformed_msgs: std::collections::HashMap<&'static str, usize> =
        std::collections::HashMap::new();
    let mut stack = vec![std::path::PathBuf::from(&root)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            scanned += 1;
            let opts = CompileOptions {
                object_closure: ObjectClosure::AssumeClosedProfile,
                ..CompileOptions::default()
            };
            match schema_to_ir(&text, opts) {
                Ok(ir) => match ir.diagnostics().next() {
                    None => supported += 1,
                    Some(d) => {
                        *reasons.entry(d.reason.advisory()).or_insert(0) += 1;
                        if let Some(kw) = ir.str_at(d.keyword) {
                            *keywords.entry(kw.to_string()).or_insert(0) += 1;
                        }
                    }
                },
                Err(e) => {
                    malformed += 1;
                    *malformed_msgs.entry(e.message).or_insert(0) += 1;
                }
            }
        }
    }
    println!(
        "jsonschemabench: {scanned} schemas scanned, {supported} fully supported, {malformed} malformed, zero panics"
    );
    let mut ranked: Vec<(&str, usize)> = reasons.into_iter().collect();
    ranked.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    for (reason, n) in ranked {
        println!("  {n:5} {reason}");
    }
    println!("-- unsupported keywords --");
    let mut kw_ranked: Vec<(String, usize)> = keywords.into_iter().collect();
    kw_ranked.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (kw, n) in kw_ranked.into_iter().take(20) {
        println!("  {n:5} {kw}");
    }
    println!("-- malformed messages --");
    let mut mm_ranked: Vec<(&str, usize)> = malformed_msgs.into_iter().collect();
    mm_ranked.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    for (msg, n) in mm_ranked {
        println!("  {n:5} {msg}");
    }
    assert!(
        scanned > 5000,
        "expected the full corpus, only found {scanned} files"
    );
}

#[test]
#[ignore = "scans a 9558-file external dataset; run explicitly with --ignored"]
fn jsonschemabench_corpus_true_default_profile_support_rate() {
    let root = format!(
        "{}/../../integration/jsonschemabench/data",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut scanned = 0usize;
    let mut supported = 0usize;
    let mut stack = vec![std::path::PathBuf::from(&root)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            scanned += 1;
            if let Ok(ir) = schema_to_ir(&text, CompileOptions::default()) {
                if ir.diagnostics().next().is_none() {
                    supported += 1;
                }
            }
        }
    }
    println!("true-default-profile: {scanned} schemas scanned, {supported} fully supported");
}

#[test]
#[ignore = "diagnostic sweep over an external dataset; run explicitly with --ignored"]
fn jsonschemabench_corpus_slowest_compiles() {
    let root = format!(
        "{}/../../integration/jsonschemabench/data",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut timings: Vec<(std::time::Duration, String)> = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(&root)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let opts = CompileOptions {
                object_closure: ObjectClosure::AssumeClosedProfile,
                ..CompileOptions::default()
            };
            let Ok(ir) = schema_to_ir(&text, opts) else {
                continue;
            };
            if ir.diagnostics().next().is_some() || ir.requires_structured_backend() {
                continue;
            }
            let start = std::time::Instant::now();
            let _ = compile_ir(&ir);
            let elapsed = start.elapsed();
            if elapsed > std::time::Duration::from_millis(50) {
                timings.push((elapsed, path.display().to_string()));
            }
        }
    }
    timings.sort_by(|a, b| b.0.cmp(&a.0));
    for (dur, path) in timings.iter().take(30) {
        println!("{dur:>10?}  {path}");
    }
    println!("{} files took over 50ms", timings.len());
}

#[test]
#[ignore = "diagnostic scan over the full local corpus, not part of CI"]
fn jsonschemabench_corpus_slowest_lowerings() {
    let root = format!(
        "{}/../../integration/jsonschemabench/data",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut timings: Vec<(std::time::Duration, String)> = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(&root)];
    let total_start = std::time::Instant::now();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let opts = CompileOptions {
                object_closure: ObjectClosure::AssumeClosedProfile,
                ..CompileOptions::default()
            };
            let start = std::time::Instant::now();
            let _ = schema_to_ir(&text, opts);
            let elapsed = start.elapsed();
            if elapsed > std::time::Duration::from_millis(20) {
                timings.push((elapsed, path.display().to_string()));
            }
        }
    }
    println!("total lowering wall time: {:?}", total_start.elapsed());
    timings.sort_by(|a, b| b.0.cmp(&a.0));
    for (dur, path) in timings.iter().take(30) {
        println!("{dur:>10?}  {path}");
    }
    println!("{} files took over 20ms", timings.len());
}

#[test]
#[cfg(feature = "bench-internals")]
#[ignore = "diagnostic scan over the full local corpus, not part of CI"]
fn jsonschemabench_corpus_full_compile_census() {
    let root = format!(
        "{}/../../integration/jsonschemabench/data",
        env!("CARGO_MANIFEST_DIR")
    );
    let out_path = format!(
        "{}/../../integration/logs/full_corpus_compile_census.jsonl",
        env!("CARGO_MANIFEST_DIR")
    );
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&out_path).expect("create incremental output file"),
    );
    use std::io::Write;

    let opts = CompileOptions {
        object_closure: ObjectClosure::AssumeClosedProfile,
        ..CompileOptions::default()
    };
    let mut stack = vec![std::path::PathBuf::from(&root)];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "json") {
                files.push(path);
            }
        }
    }
    files.sort();
    println!("scanning {} files...", files.len());

    let mut counts: std::collections::BTreeMap<&'static str, u32> =
        std::collections::BTreeMap::new();
    let total_start = std::time::Instant::now();
    for (i, path) in files.iter().enumerate() {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string()
            .replace('\\', "/");
        let Ok(text) = std::fs::read_to_string(path) else {
            *counts.entry("read_error").or_default() += 1;
            continue;
        };

        let started = std::time::Instant::now();
        let record: serde_json::Value = match schema_to_ir(&text, opts.clone()) {
            Err(e) => {
                *counts.entry("ir_lowering_error").or_default() += 1;
                serde_json::json!({"file": rel, "stage": "ir_lowering", "ok": false, "error": e.to_string()})
            }
            Ok(ir) => {
                if ir.requires_structured_backend() {
                    let ir = Arc::new(ir);
                    match StructuredProgram::compile_timed_for_bench(ir) {
                        Ok((program, plan_ns)) => {
                            *counts.entry("structured_ok").or_default() += 1;
                            let retained = program.retained_plan_bytes();
                            serde_json::json!({
                                "file": rel, "stage": "structured", "ok": true,
                                "plan_ns": plan_ns, "retained_plan_bytes": retained,
                                "wall_ns": started.elapsed().as_nanos() as u64,
                            })
                        }
                        Err(e) => {
                            *counts.entry("structured_error").or_default() += 1;
                            serde_json::json!({"file": rel, "stage": "structured", "ok": false, "error": e.to_string()})
                        }
                    }
                } else {
                    match compile_ir_timed(&ir) {
                        Ok((_engine, _lowering_ns, materialization_ns)) => {
                            *counts.entry("regular_ok").or_default() += 1;
                            serde_json::json!({
                                "file": rel, "stage": "regular", "ok": true,
                                "materialization_ns": materialization_ns,
                                "wall_ns": started.elapsed().as_nanos() as u64,
                            })
                        }
                        Err(e) => {
                            *counts.entry("regular_error").or_default() += 1;
                            serde_json::json!({"file": rel, "stage": "regular", "ok": false, "error": e.to_string()})
                        }
                    }
                }
            }
        };
        writeln!(out, "{record}").expect("write incremental record");
        if (i + 1) % 200 == 0 {
            out.flush().expect("flush incremental output");
            println!(
                "{}/{} done, {:.1}s elapsed, counts so far: {:?}",
                i + 1,
                files.len(),
                total_start.elapsed().as_secs_f64(),
                counts
            );
        }
    }
    out.flush().expect("final flush");
    println!("FINAL counts: {counts:?}");
    println!("total elapsed: {:.1}s", total_start.elapsed().as_secs_f64());
    println!("output written to: {out_path}");
}

#[test]
#[ignore = "diagnostic scan over the full local corpus, not part of CI"]
fn jsonschemabench_corpus_full_automaton_compile_census() {
    let root = format!(
        "{}/../../integration/jsonschemabench/data",
        env!("CARGO_MANIFEST_DIR")
    );
    let opts = CompileOptions {
        object_closure: ObjectClosure::AssumeClosedProfile,
        ..CompileOptions::default()
    };
    let mut ir_ok = 0u32;
    let mut ir_err = 0u32;
    let mut regular_ok = 0u32;
    let mut regular_err = 0u32;
    let mut structured_routed = 0u32;
    let mut err_buckets: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    let mut stack = vec![std::path::PathBuf::from(&root)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let rel = path.display().to_string();
            match schema_to_ir(&text, opts.clone()) {
                Err(e) => {
                    ir_err += 1;
                    err_buckets
                        .entry(format!("ir_lowering:{e}"))
                        .or_default()
                        .push(rel);
                }
                Ok(ir) => {
                    ir_ok += 1;
                    if ir.requires_structured_backend() {
                        structured_routed += 1;
                        continue;
                    }
                    match compile_ir(&ir) {
                        Ok(_) => regular_ok += 1,
                        Err(e) => {
                            regular_err += 1;
                            err_buckets
                                .entry(format!("automaton_compile:{e}"))
                                .or_default()
                                .push(rel);
                        }
                    }
                }
            }
        }
    }
    println!(
        "ir_ok={ir_ok} ir_err={ir_err} structured_routed={structured_routed} \
         regular_ok={regular_ok} regular_err={regular_err}"
    );
    for (bucket, files) in &err_buckets {
        println!("{bucket}: {}", files.len());
        for f in files.iter().take(5) {
            println!("    {f}");
        }
    }
}

#[test]
#[ignore = "diagnostic scan over the full local corpus, not part of CI"]
fn jsonschemabench_corpus_stage_breakdown() {
    let root = format!(
        "{}/../../integration/jsonschemabench/data",
        env!("CARGO_MANIFEST_DIR")
    );
    let opts = CompileOptions {
        object_closure: ObjectClosure::AssumeClosedProfile,
        ..CompileOptions::default()
    };
    let mut rows: Vec<(u64, u64, u64, u64, u64, String)> = Vec::new();
    let mut totals = (0u64, 0u64, 0u64, 0u64);
    let mut stack = vec![std::path::PathBuf::from(&root)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok((ir, parse_ns, lower_ns)) = schema_to_ir_timed(&text, opts.clone()) else {
                continue;
            };
            if ir.diagnostics().next().is_some() || ir.requires_structured_backend() {
                continue;
            }
            let Ok((_, regex_build_ns, dfa_build_ns)) = compile_ir_timed(&ir) else {
                continue;
            };
            totals.0 += parse_ns;
            totals.1 += lower_ns;
            totals.2 += regex_build_ns;
            totals.3 += dfa_build_ns;
            let total = parse_ns + lower_ns + regex_build_ns + dfa_build_ns;
            if total > 5_000_000 {
                rows.push((
                    total,
                    parse_ns,
                    lower_ns,
                    regex_build_ns,
                    dfa_build_ns,
                    path.display().to_string(),
                ));
            }
        }
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10}  file",
        "total", "parse", "lower", "regex", "dfa"
    );
    for (total, parse_ns, lower_ns, regex_build_ns, dfa_build_ns, path) in rows.iter().take(30) {
        println!(
            "{:>8.2}ms {:>8.2}ms {:>8.2}ms {:>8.2}ms {:>8.2}ms  {path}",
            *total as f64 / 1e6,
            *parse_ns as f64 / 1e6,
            *lower_ns as f64 / 1e6,
            *regex_build_ns as f64 / 1e6,
            *dfa_build_ns as f64 / 1e6,
        );
    }
    let grand_total = (totals.0 + totals.1 + totals.2 + totals.3).max(1) as f64;
    println!(
        "corpus totals: parse={:.1}% lower={:.1}% regex_build={:.1}% dfa_build={:.1}%",
        100.0 * totals.0 as f64 / grand_total,
        100.0 * totals.1 as f64 / grand_total,
        100.0 * totals.2 as f64 / grand_total,
        100.0 * totals.3 as f64 / grand_total,
    );
}

#[test]
#[ignore = "reads a vendored corpus file, run explicitly with --ignored"]
fn sp49_shuffle_weight_routes_to_structured_backend_instead_of_wasted_dfa_attempt() {
    let path = format!(
        "{}/../../integration/jsonschemabench/data/Snowplow/sp_49_Normalized.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap();
    let opts = CompileOptions {
        object_closure: ObjectClosure::AssumeClosedProfile,
        ..CompileOptions::default()
    };
    let (ir, ..) = schema_to_ir_timed(&text, opts).unwrap();
    assert_eq!(ir.diagnostics().count(), 0);
    assert!(
        ir.requires_structured_backend(),
        "4 fields at exactly MAX_SAFE_SHUFFLE_FIELDS, but their summed maxLength exceeds the weight cap"
    );
    let t0 = std::time::Instant::now();
    let ok = crate::structured::accepts(
        &ir,
        br#"{"documentHostName":"example.com","documentLocationUrl":null,"documentPath":"/x","documentTitle":null}"#,
    );
    let elapsed = t0.elapsed();
    assert!(ok);
    println!("structured accepts took {elapsed:?}");
    assert!(elapsed < std::time::Duration::from_secs(2));
}
