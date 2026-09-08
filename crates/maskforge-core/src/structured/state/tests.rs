use super::*;
use crate::ir::{Builder, CompileOptions, ScalarLit, MAX_UNROLLED_ENUM};
use crate::structured::plan::object_plan_of;
use std::sync::Arc;

fn build(schema: &str) -> StructuredPlan {
    let ir = Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
    StructuredPlan::compile(ir.clone(), StructuredLimits::default())
        .unwrap_or_else(|error| panic!("{error:?}; nodes={:?}", ir.nodes().collect::<Vec<_>>()))
}

#[test]
fn slice_certificate_bounds_fuel_and_detects_active_identity_cycles() {
    let plan = Arc::new(build(r#"{"type":"string"}"#));
    let mut state = StructuredState::try_new(plan).unwrap();
    assert!(feed(&mut state, b"\""));
    state.reserve_undo(2).unwrap();
    let mut proofs = Vec::new();
    let mut context = ProofContext::new(1);
    assert_eq!(
        state.certify_slice(&mut proofs, &mut context),
        SliceCertificate::AllNoBoundary
    );
    assert_eq!(context.depth, 0);
    let identity = &state as *const StructuredState<'_> as usize;
    context.visited[0] = identity;
    context.depth = 1;
    assert_eq!(
        state.certify_slice(&mut proofs, &mut context),
        SliceCertificate::Unknown
    );
    assert_eq!(context.depth, 1);
    context.depth = 0;
    context.fuel = 0;
    assert_eq!(
        state.certify_slice(&mut proofs, &mut context),
        SliceCertificate::Unknown
    );
}

#[test]
fn slice_preflight_session_shortfalls_preserve_retained_capacity() {
    for offset in [-1isize, 0, 1] {
        let plan = Arc::new(build(r#"{"type":"object","additionalProperties":true}"#));
        let mut state = StructuredState::try_new(plan).unwrap();
        assert!(feed(&mut state, br#"{"a"#));
        let retained = state.retained_bytes();
        let undo_capacity = state.undo.capacity();
        state.limits.max_session_bytes = retained.checked_add_signed(offset).unwrap();
        assert_eq!(
            state.prepare_local_slice(128).unwrap(),
            SlicePreparation::Ineligible
        );
        assert_eq!(state.retained_bytes(), retained);
        assert_eq!(state.undo.capacity(), undo_capacity);
        assert!(!state.session_memory.is_terminal());
    }
}

#[test]
fn second_wrapper_key_lease_failure_unwinds_the_first_acquisition() {
    let plan = Arc::new(build(
        r#"{"anyOf":[
            {"type":"object","properties":{"a":{"type":"integer"}}},
            {"type":"object","properties":{"b":{"type":"string"}}}
        ]}"#,
    ));
    let mut state = StructuredState::try_new(plan).unwrap();
    assert!(feed(&mut state, br#"{""#));
    state
        .prepare_certified_slice(&mut ProofContext::new(600))
        .unwrap();
    state.discard_prepared_key_leases_tree();
    let retained = state.retained_bytes();
    let mut context = ProofContext::new(600);
    context.fail_key_lease_at(1);
    let error = state.prepare_certified_slice(&mut context).unwrap_err();
    assert_eq!(error.kind, LimitKind::AllocationBytes);
    assert_eq!(state.key_lease_tree_debug(), (0, 0));
    assert_eq!(state.retained_bytes(), retained);
    assert_eq!(state.retained_bytes(), state.recomputed_retained_bytes());
    assert!(!state.session_memory.is_terminal());
}

#[test]
fn leased_key_content_reaches_property_dependency_and_duplicate_consumers() {
    let plan = Arc::new(build(
        r#"{
            "type":"object",
            "properties":{"leased":{"type":"integer"},"need":{"type":"integer"}},
            "patternProperties":{"^leased$":{"minimum":0}},
            "propertyNames":{"pattern":"^(leased|need)$"},
            "dependentSchemas":{"leased":{"required":["need"]}},
            "additionalProperties":false
        }"#,
    ));
    let mut state = StructuredState::try_new(plan).unwrap();
    assert!(feed(&mut state, br#"{""#));
    state
        .prepare_certified_slice(&mut ProofContext::new(6))
        .unwrap();
    let mark = state.checkpoint();
    assert!(feed(&mut state, b"leased\""));
    state.rollback(mark);
    assert_eq!(state.candidate_key(state.frames.len() - 1), "");
    assert_eq!(state.retained_bytes(), state.recomputed_retained_bytes());

    assert!(feed(&mut state, br#"leased":1,"need":2}"#));
    state.commit_checkpoint().unwrap();
    assert!(state.is_accepting());
    assert_eq!(state.key_lease_tree_debug(), (0, 0));

    let plan = Arc::new(build(r#"{"type":"object","additionalProperties":true}"#));
    let mut state = StructuredState::try_new(plan).unwrap();
    assert!(feed(&mut state, br#"{""#));
    assert_eq!(
        state.prepare_local_slice(7).unwrap(),
        SlicePreparation::Ready
    );
    assert!(feed(&mut state, br#"dynamic":1,""#));
    state.commit_checkpoint().unwrap();
    assert_eq!(
        state.prepare_local_slice(7).unwrap(),
        SlicePreparation::Ready
    );
    let duplicate = state.checkpoint();
    assert!(!feed(&mut state, b"dynamic\""));
    state.rollback(duplicate);
    assert_eq!(state.candidate_key(state.frames.len() - 1), "");
    assert_eq!(state.retained_bytes(), state.recomputed_retained_bytes());
}

#[test]
fn any_json_dynamic_keys_use_leased_content_and_reject_duplicates() {
    let plan = Arc::new(build(r#"{"type":"object"}"#));
    let mut state = StructuredState::try_new(plan).unwrap();
    assert!(feed(&mut state, br#"{"payload":{""#));
    state
        .prepare_certified_slice(&mut ProofContext::new(600))
        .unwrap();
    assert_eq!(state.key_lease_tree_debug().0, 1);
    let depth = state
        .frames
        .iter()
        .rposition(|frame| matches!(frame, Frame::Any(AnyFrame::Object(_))))
        .expect("an unrestricted property value owns an AnyJson object frame");

    let mark = state.checkpoint();
    assert!(feed(&mut state, b"dynamic\""));
    state.rollback(mark);
    assert_eq!(state.candidate_key(depth), "");
    assert_eq!(state.retained_bytes(), state.recomputed_retained_bytes());

    assert!(feed(&mut state, br#"dynamic":1,""#));
    state.commit_checkpoint().unwrap();
    state
        .prepare_certified_slice(&mut ProofContext::new(600))
        .unwrap();
    let duplicate = state.checkpoint();
    assert!(feed(&mut state, b"dynamic\""));
    assert!(!feed(&mut state, b":"));
    state.rollback(duplicate);
    assert_eq!(state.candidate_key(depth), "");
    assert_eq!(state.retained_bytes(), state.recomputed_retained_bytes());
    assert!(feed(&mut state, br#"other":2}}"#));
    state.commit_checkpoint().unwrap();
    assert!(state.is_accepting());
    assert_eq!(state.key_lease_tree_debug(), (0, 0));
    assert_eq!(state.retained_bytes(), state.recomputed_retained_bytes());
}

#[test]
fn prepared_key_leases_release_on_reset_and_object_frame_destruction() {
    let plan = Arc::new(build(r#"{"type":"object","additionalProperties":true}"#));
    let mut reset = StructuredState::try_new(plan.clone()).unwrap();
    assert!(feed(&mut reset, br#"{""#));
    assert_eq!(
        reset.prepare_local_slice(600).unwrap(),
        SlicePreparation::Ready
    );
    assert_eq!(reset.key_lease_tree_debug().0, 1);
    assert!(reset.reset_for_reuse());
    assert_eq!(reset.key_lease_tree_debug(), (0, 0));
    assert_eq!(reset.retained_bytes(), reset.recomputed_retained_bytes());

    let mut destroyed = StructuredState::try_new(plan).unwrap();
    assert!(feed(&mut destroyed, br#"{""#));
    assert_eq!(
        destroyed.prepare_local_slice(600).unwrap(),
        SlicePreparation::Ready
    );
    assert!(feed(&mut destroyed, br#"long":0}"#));
    destroyed.commit_checkpoint().unwrap();
    assert!(destroyed.is_accepting());
    assert_eq!(destroyed.key_lease_tree_debug(), (0, 0));
    assert_eq!(
        destroyed.retained_bytes(),
        destroyed.recomputed_retained_bytes()
    );
}

#[test]
fn surviving_unevaluated_branch_contributes_a_mandatory_scalar_bound() {
    let plan = Arc::new(build(
        r#"{"type":"object","properties":{"owner":{"type":"string","maxLength":8}},"allOf":[{"properties":{"team":{"type":"string","maxLength":8}}}],"unevaluatedProperties":false}"#,
    ));
    let mut state = StructuredState::try_new(plan).unwrap();
    assert!(feed(&mut state, br#"{"owner":"ops","team":""#));
    let mut context = ProofContext::for_preparation(
        8,
        state.retained_bytes(),
        state.limits.max_session_bytes,
        state.certificate_fuel().unwrap(),
    );
    let mut proofs = Vec::new();
    assert_eq!(
        state.certify_slice(&mut proofs, &mut context),
        SliceCertificate::AllNoBoundary
    );
    assert_eq!(context.remaining_scalars, Some(8));
}

fn build_resources(
    root: &str,
    retrieval_uri: &str,
    resources: &[(&str, &str)],
    structured_limits: StructuredLimits,
) -> StructuredPlan {
    let ir = lower_resources(root, retrieval_uri, resources);
    StructuredPlan::compile(ir, structured_limits).expect("structured resource plan")
}

fn lower_resources(
    root: &str,
    retrieval_uri: &str,
    resources: &[(&str, &str)],
) -> Arc<crate::ir::SchemaIR> {
    let resource_limits = crate::frontend::SchemaResourceLimits::default();
    let mut registry = crate::frontend::SchemaRegistry::new(resource_limits);
    for &(uri, schema) in resources {
        registry.insert(uri, schema).expect("registered resource");
    }
    Arc::new(
        crate::frontend::schema_to_ir_with_resources(
            root,
            Some(retrieval_uri),
            CompileOptions::default(),
            resource_limits,
            registry,
        )
        .expect("resource graph"),
    )
}

const DYNAMIC_TREE: &str = r##"{
    "$id":"https://example.com/tree",
    "$dynamicAnchor":"node",
    "type":"object",
    "properties":{
        "data":true,
        "children":{"type":"array","items":{"$dynamicRef":"#node"}}
    }
}"##;

const STRICT_DYNAMIC_TREE: &str = r##"{
    "$id":"https://example.com/strict-tree",
    "$dynamicAnchor":"node",
    "$ref":"https://example.com/tree",
    "unevaluatedProperties":false
}"##;

fn feed(state: &mut StructuredState<'_>, bytes: &[u8]) -> bool {
    for &b in bytes {
        match state.try_push_byte(b) {
            Ok(true) => {}
            _ => return false,
        }
    }
    true
}

fn large_string_enum_plan(mut values: Vec<String>) -> StructuredPlan {
    values.extend((0..=MAX_UNROLLED_ENUM).map(|index| format!("filler-{index:04}")));
    values.sort();
    values.dedup();
    let mut builder = Builder::new(CompileOptions::default());
    let root = builder
        .enum_values(values.into_iter().map(ScalarLit::Str).collect())
        .expect("enum");
    let ir = Arc::new(builder.finish(root).expect("ir"));
    StructuredPlan::compile(ir, StructuredLimits::default()).expect("plan")
}

fn enum_accepts(plan: &StructuredPlan, document: &[u8]) -> bool {
    let mut state = StructuredState::new(plan);
    feed(&mut state, document) && state.is_accepting()
}

#[test]
fn exact_decimal_constraints_accept_without_binary_float_rounding() {
    type DecimalCase<'a> = (&'a str, &'a [&'a [u8]], &'a [&'a [u8]]);
    let cases: &[DecimalCase<'_>] = &[
        (
            r#"{"type":"number","exclusiveMinimum":1.1}"#,
            &[b"1.1000000000000001", b"1.2", b"1e1"],
            &[b"1.1", b"1.09", b"-2"],
        ),
        (
            r#"{"type":"number","multipleOf":1.5}"#,
            &[b"0", b"1.5", b"3", b"-1.5", b"1.5e2"],
            &[b"1", b"1.6", b"3.0000000000000001"],
        ),
        (
            r#"{"type":"integer","multipleOf":0.123456789}"#,
            &[b"0", b"123456789"],
            &[b"1", b"123456788", b"1e308"],
        ),
        (
            r#"{"type":"integer","multipleOf":1e-8}"#,
            &[b"0", b"1", b"-42", b"1e308"],
            &[],
        ),
        (
            r#"{"type":"number","minimum":9223372036854776000,"maximum":36893488147419103000}"#,
            &[b"9223372036854776000", b"1e19", b"36893488147419103000"],
            &[b"9223372036854775999", b"36893488147419103001"],
        ),
        (
            r#"{"type":"number","minimum":1e-9}"#,
            &[b"0.000000001", b"1e-9", b"1e308"],
            &[b"0", b"9e-10", b"-1e-9"],
        ),
        (
            r#"{"type":"integer","minimum":9223372036854776000,"maximum":18446744073709551615}"#,
            &[
                b"9223372036854776000",
                b"1e19",
                b"1.5e19",
                b"18446744073709551615",
            ],
            &[b"9223372036854775999", b"1.5e-1", b"18446744073709551616"],
        ),
        (
            r#"{"type":"integer","exclusiveMinimum":1.5,"exclusiveMaximum":4.5}"#,
            &[b"2", b"3", b"4"],
            &[b"1", b"1.5", b"5"],
        ),
    ];
    for (schema, accepted, rejected) in cases {
        let plan = build(schema);
        for document in *accepted {
            assert!(enum_accepts(&plan, document), "{schema}: {document:?}");
        }
        for document in *rejected {
            assert!(!enum_accepts(&plan, document), "{schema}: {document:?}");
        }
    }
}

#[test]
fn exact_decimal_prefix_rejects_an_impossible_sign_transactionally() {
    for (schema, byte) in [
        (r#"{"const":1}"#, b'-'),
        (r#"{"type":"number","maximum":-1}"#, b'0'),
    ] {
        let plan = build(schema);
        let mut state = StructuredState::new(&plan);
        let before = full_snapshot(&state);
        assert_eq!(state.try_push_byte(byte), Ok(false), "{schema}");
        assert_eq!(full_snapshot(&state), before, "{schema}");
    }
}

#[test]
fn exact_decimal_number_probe_rolls_back_and_enforces_byte_limit() {
    let plan = build(r#"{"type":"number","multipleOf":0.125}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"1.2"));
    let checkpoint = state.checkpoint();
    assert!(!state.try_push_byte(b'x').unwrap());
    assert_eq!(state.checkpoint(), checkpoint);
    assert!(feed(&mut state, b"5"));
    assert!(state.is_accepting());

    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r#"{"type":"number","multipleOf":0.125}"#,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    let limits = StructuredLimits {
        max_number_bytes: 3,
        ..StructuredLimits::default()
    };
    let limited = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&limited);
    assert!(feed(&mut state, b"1.2"));
    let error = state.try_push_byte(b'5').unwrap_err();
    assert_eq!(error.kind, LimitKind::NumberBytes);
}

#[test]
fn large_string_enum_matches_decoded_strings_and_prefix_members() {
    let plan = large_string_enum_plan(vec![
        String::new(),
        "\u{8}\u{c}\n\r\t".to_owned(),
        "\"".to_owned(),
        "/".to_owned(),
        "\\".to_owned(),
        "a".to_owned(),
        "ab".to_owned(),
        "abc".to_owned(),
        "é".to_owned(),
        "😀".to_owned(),
    ]);
    for document in [
        b"\"\"".as_slice(),
        b"\"a\"",
        b"\"ab\"",
        b"\"abc\"",
        b"\"\\u0061\"",
        b"\"\\u00E9\"",
        b"\"\\u00e9\"",
        b"\"\\\"\"",
        b"\"\\/\"",
        b"\"\\\\\"",
        b"\"\\b\\f\\n\\r\\t\"",
        "\"é\"".as_bytes(),
        "\"😀\"".as_bytes(),
        b"\"\\uD83D\\uDE00\"",
    ] {
        assert!(enum_accepts(&plan, document), "{document:?}");
    }
    for document in [
        b"\"ac\"".as_slice(),
        b"\"abcd\"",
        b"\"\\uD800\"",
        b"\"\\uDC00\"",
        b"\"\\uD800\\u0061\"",
        b"\"\xc0\xaf\"",
        b"\"\xe2(\xa1\"",
        b"\"\x01\"",
        b"\"\\u00",
    ] {
        assert!(!enum_accepts(&plan, document), "{document:?}");
    }
}

#[test]
fn large_string_enum_speculation_rolls_back_without_drift() {
    let plan = large_string_enum_plan(vec!["a".to_owned(), "ab".to_owned(), "abc".to_owned()]);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"\"ab"));
    let baseline = full_snapshot(&state);
    for _ in 0..10_000 {
        let mark = state.checkpoint();
        assert!(!state.try_push_byte(b'z').unwrap());
        state.rollback(mark);
        assert_eq!(full_snapshot(&state), baseline);
    }
    assert!(feed(&mut state, b"c\""));
    assert!(state.is_accepting());
}

#[test]
fn large_string_enum_rejects_at_the_first_diverging_decoded_byte() {
    let values = (0..=MAX_UNROLLED_ENUM)
        .map(|index| format!("aa{}-{index:04}", "x".repeat(47)))
        .collect();
    let plan = large_string_enum_plan(values);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"\"aa"));
    let before = full_snapshot(&state);
    assert!(!state.try_push_byte(b'z').unwrap());
    assert_eq!(full_snapshot(&state), before);
    assert!(state.try_push_byte(b'x').unwrap());
}

#[test]
fn large_string_enum_session_memory_is_independent_of_member_count() {
    let small = large_string_enum_plan(vec!["target".to_owned()]);
    let many = large_string_enum_plan(
        (0..10_000)
            .map(|index| format!("member-{index:05}"))
            .collect(),
    );
    let mut small_state = StructuredState::new(&small);
    let mut many_state = StructuredState::new(&many);
    assert_eq!(small_state.retained_bytes(), many_state.retained_bytes());
    assert!(feed(&mut small_state, b"\"t"));
    assert!(feed(&mut many_state, b"\"m"));
    assert_eq!(small_state.retained_bytes(), many_state.retained_bytes());
    fn assert_copy<T: Copy>() {}
    assert_copy::<StringEnumFrame>();
}

#[test]
fn dependent_schemas_cover_order_decoding_dedup_and_dead_late_triggers() {
    let schema = r#"{
        "type":"object",
        "dependentSchemas":{
            "a":{"type":"object","required":["z"],"additionalProperties":true},
            "alias":{"type":"object","required":["z"],"additionalProperties":true},
            "never":false
        },
        "additionalProperties":true
    }"#;
    let plan = build(schema);
    let object = object_plan_of(&plan, plan.root);
    assert_eq!(object.dependent_schemas.len(), 2);
    assert_eq!(
        object.dependent_by_name["a"],
        object.dependent_by_name["alias"]
    );
    for (document, expected) in [
        (br#"{}"#.as_slice(), true),
        (br#"{"z":1}"#.as_slice(), true),
        (br#"{"a":1}"#.as_slice(), false),
        (br#"{"a":1,"z":2}"#.as_slice(), true),
        (br#"{"z":2,"a":1}"#.as_slice(), true),
        (br#"{"\u0061":1,"z":2}"#.as_slice(), true),
        (br#"{"alias":1,"z":2}"#.as_slice(), true),
        (br#"{"never":1}"#.as_slice(), false),
    ] {
        let mut state = StructuredState::new(&plan);
        let actual = feed(&mut state, document) && state.is_accepting();
        assert_eq!(actual, expected, "{}", String::from_utf8_lossy(document));
    }

    let mut state = StructuredState::new(&plan);
    for &byte in br#"{"z":2,"a":1}"# {
        let warm = state.checkpoint();
        let _ = state.try_push_byte(byte);
        state.rollback(warm);
        let mark = state.checkpoint();
        let before = full_snapshot(&state);
        let _ = state.try_push_byte(byte);
        state.rollback(mark);
        assert_eq!(full_snapshot(&state), before);
        assert!(state.try_push_byte(byte).unwrap_or(false));
        state.commit_checkpoint().unwrap();
        assert_exact_session_ledger(&state);
    }
}

#[test]
fn dependent_schema_construction_limits_are_exact_and_atomic() {
    let schema =
        r#"{"type":"object","dependentSchemas":{"a":true,"b":false},"additionalProperties":true}"#;
    let source =
        Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).expect("schema"));
    let baseline = StructuredPlan::compile(source.clone(), StructuredLimits::default()).unwrap();
    let baseline_ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: baseline.limits.max_active_validators,
    });
    let baseline_state = StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&baseline),
        baseline.root,
        0,
        baseline_ledger.clone(),
    )
    .unwrap();
    let exact_validators = baseline_ledger.live.load(Ordering::Relaxed);
    let exact_bytes = baseline_state.session_memory.live();
    drop(baseline_state);
    assert_eq!(baseline_ledger.live.load(Ordering::Relaxed), 0);

    for (limit, succeeds) in [(exact_validators, true), (exact_validators - 1, false)] {
        let plan = StructuredPlan::compile(
            source.clone(),
            StructuredLimits {
                max_active_validators: limit,
                ..StructuredLimits::default()
            },
        )
        .expect("plan");
        let ledger = Arc::new(ValidatorLedger {
            live: AtomicUsize::new(0),
            limit,
        });
        let result = StructuredState::new_at_with_context(
            PlanHandle::Borrowed(&plan),
            plan.root,
            0,
            ledger.clone(),
        );
        assert_eq!(
            result.is_ok(),
            succeeds,
            "validator limit {limit}: {:?}",
            result.as_ref().err()
        );
        if let Ok(state) = result {
            assert_eq!(ledger.live.load(Ordering::Relaxed), exact_validators);
            assert_exact_session_ledger(&state);
            drop(state);
        }
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
    }

    let exact_depth = (1u32..=32)
        .find(|&depth| {
            let plan = StructuredPlan::compile(
                source.clone(),
                StructuredLimits {
                    max_depth: depth,
                    ..StructuredLimits::default()
                },
            )
            .unwrap();
            let succeeds = StructuredState::new_at(PlanHandle::Borrowed(&plan), plan.root).is_ok();
            succeeds
        })
        .expect("finite dependency depth");
    for (depth, succeeds) in [(exact_depth, true), (exact_depth - 1, false)] {
        let plan = StructuredPlan::compile(
            source.clone(),
            StructuredLimits {
                max_depth: depth,
                ..StructuredLimits::default()
            },
        )
        .expect("plan");
        let ledger = Arc::new(ValidatorLedger {
            live: AtomicUsize::new(0),
            limit: plan.limits.max_active_validators,
        });
        let result = StructuredState::new_at_with_context(
            PlanHandle::Borrowed(&plan),
            plan.root,
            0,
            ledger.clone(),
        );
        assert_eq!(result.is_ok(), succeeds);
        drop(result);
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
    }

    let can_construct = |limit| {
        let plan = StructuredPlan::compile(
            source.clone(),
            StructuredLimits {
                max_session_bytes: limit,
                ..StructuredLimits::default()
            },
        )
        .unwrap();
        let succeeds = StructuredState::new_at(PlanHandle::Borrowed(&plan), plan.root).is_ok();
        succeeds
    };
    let mut low = exact_bytes;
    let mut high = baseline.limits.max_session_bytes;
    while low < high {
        let middle = low + (high - low) / 2;
        if can_construct(middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    let exact_session_limit = low;
    for (limit, succeeds) in [
        (exact_session_limit, true),
        (exact_session_limit - 1, false),
    ] {
        let plan = StructuredPlan::compile(
            source.clone(),
            StructuredLimits {
                max_session_bytes: limit,
                ..StructuredLimits::default()
            },
        )
        .unwrap();
        let ledger = Arc::new(ValidatorLedger {
            live: AtomicUsize::new(0),
            limit: plan.limits.max_active_validators,
        });
        let result = StructuredState::new_at_with_context(
            PlanHandle::Borrowed(&plan),
            plan.root,
            0,
            ledger.clone(),
        );
        assert_eq!(result.is_ok(), succeeds, "session limit {limit}");
        if let Ok(state) = result {
            assert_eq!(state.session_memory.live(), exact_bytes);
            assert_exact_session_ledger(&state);
            drop(state);
        }
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn dependent_schema_cursor_death_trigger_and_frame_ownership_are_transactional() {
    let plan =
        build(r#"{"type":"object","dependentSchemas":{"a":false},"additionalProperties":true}"#);

    let mut state = StructuredState::new(&plan);
    let ledger = state._validator_ledger.clone();
    assert_eq!(ledger.live.load(Ordering::Relaxed), 1);
    let opening = state.checkpoint();
    assert!(state.try_push_byte(b'{').unwrap());
    assert_eq!(ledger.live.load(Ordering::Relaxed), 1);
    state.rollback(opening);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 1);
    assert!(state.try_push_byte(b'{').unwrap());
    state.commit_checkpoint().unwrap();
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);

    assert!(feed(&mut state, br#""a"#));
    let before_trigger = full_snapshot(&state);
    assert!(!state.try_push_byte(b'"').unwrap());
    assert_eq!(full_snapshot(&state), before_trigger);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
    drop(state);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);

    let mut popped = StructuredState::new(&plan);
    let popped_ledger = popped._validator_ledger.clone();
    assert!(popped.try_push_byte(b'{').unwrap());
    let close = popped.checkpoint();
    assert!(popped.try_push_byte(b'}').unwrap());
    assert_eq!(popped_ledger.live.load(Ordering::Relaxed), 1);
    popped.rollback(close);
    assert!(matches!(popped.frames.as_slice(), [Frame::Object(_)]));
    assert_eq!(popped_ledger.live.load(Ordering::Relaxed), 1);
    assert!(popped.try_push_byte(b'}').unwrap());
    popped.commit_checkpoint().unwrap();
    assert_eq!(popped_ledger.live.load(Ordering::Relaxed), 0);
    drop(popped);
    assert_eq!(popped_ledger.live.load(Ordering::Relaxed), 0);

    let mut dead_owned = StructuredState::new(&plan);
    let dead_owned_ledger = dead_owned._validator_ledger.clone();
    assert!(dead_owned.try_push_byte(b'{').unwrap());
    assert_eq!(dead_owned_ledger.live.load(Ordering::Relaxed), 1);
    drop(dead_owned);
    assert_eq!(dead_owned_ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn dependent_schema_undo_preflight_and_ten_thousand_rollbacks_have_no_drift() {
    let plan = build(
        r#"{"type":"object","dependentSchemas":{"a":{"required":["z"]}},"additionalProperties":true}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'{').unwrap());
    state.commit_checkpoint().unwrap();
    let logical = full_snapshot(&state);
    let pool = cursor_pool_snapshot(&state.cursors);
    let session = state.session_memory.live();
    let validators = state._validator_ledger.live.load(Ordering::Relaxed);
    state.limits.max_undo_entries = state.undo.len();
    assert!(matches!(
        state.try_push_byte(b' '),
        Err(StructuredRuntimeError {
            kind: LimitKind::UndoEntries,
            ..
        })
    ));
    assert_eq!(full_snapshot(&state), logical);
    assert_eq!(cursor_pool_snapshot(&state.cursors), pool);
    assert_eq!(state.session_memory.live(), session);
    assert_eq!(
        state._validator_ledger.live.load(Ordering::Relaxed),
        validators
    );

    state.limits.max_undo_entries = usize::MAX;
    for iteration in 0..10_000 {
        let mark = state.checkpoint();
        assert!(state.try_push_byte(b' ').unwrap());
        state.rollback(mark);
        if iteration % 1_000 == 0 {
            assert_eq!(full_snapshot(&state), logical);
            assert_eq!(cursor_pool_snapshot(&state.cursors), pool);
            assert_eq!(state.session_memory.live(), session);
            assert_eq!(
                state._validator_ledger.live.load(Ordering::Relaxed),
                validators
            );
        }
    }
    assert_exact_session_ledger(&state);
}

#[test]
fn ten_thousand_dependent_schema_commit_construction_cycles_have_no_drift() {
    let plan = build(
        r#"{"type":"object","dependentSchemas":{"a":{"required":["z"]}},"additionalProperties":true}"#,
    );
    for iteration in 0..10_000 {
        let mut state = StructuredState::new(&plan);
        let ledger = state._validator_ledger.clone();
        assert!(feed(&mut state, br#"{}"#));
        state.commit_checkpoint().unwrap();
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0, "{iteration}");
        assert_exact_session_ledger(&state);
        drop(state);
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0, "{iteration}");
    }
}

#[test]
fn dependent_required_presence_undo_close_and_ten_thousand_rollbacks_have_no_drift() {
    let plan =
        build(r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}"#);
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'{').unwrap());
    state.commit_checkpoint().unwrap();
    let warm = state.checkpoint();
    assert!(feed(&mut state, br#""a""#));
    state.rollback(warm);
    let logical = full_snapshot(&state);
    let pool = cursor_pool_snapshot(&state.cursors);
    let session = state.session_memory.live();
    let validators = state._validator_ledger.live.load(Ordering::Relaxed);
    for iteration in 0..10_000 {
        let mark = state.checkpoint();
        assert!(feed(&mut state, br#""a""#));
        state.rollback(mark);
        if iteration % 1_000 == 0 {
            assert_eq!(full_snapshot(&state), logical);
            assert_eq!(cursor_pool_snapshot(&state.cursors), pool);
            assert_eq!(state.session_memory.live(), session);
            assert_eq!(
                state._validator_ledger.live.load(Ordering::Relaxed),
                validators
            );
        }
    }

    let before_key_close = {
        assert!(feed(&mut state, br#""a"#));
        full_snapshot(&state)
    };
    state.limits.max_undo_entries = state.undo.len();
    assert!(matches!(
        state.try_push_byte(b'"'),
        Err(StructuredRuntimeError {
            kind: LimitKind::UndoEntries,
            ..
        })
    ));
    assert_eq!(full_snapshot(&state), before_key_close);
    state.limits.max_undo_entries = usize::MAX;
    assert!(feed(&mut state, br#"":1"#));
    let before_invalid_close = full_snapshot(&state);
    assert!(!state.try_push_byte(b'}').unwrap());
    assert_eq!(full_snapshot(&state), before_invalid_close);
    assert!(feed(&mut state, br#", "b":2}"#));
    assert!(state.is_accepting());
    state.commit_checkpoint().unwrap();
    assert_exact_session_ledger(&state);
}

#[test]
fn ten_thousand_dependent_required_construct_commit_drop_cycles_have_no_drift() {
    let plan =
        build(r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}"#);
    for iteration in 0..10_000 {
        let mut state = StructuredState::new(&plan);
        let ledger = state._validator_ledger.clone();
        assert!(feed(&mut state, br#"{"a":1,"b":2}"#));
        state.commit_checkpoint().unwrap();
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0, "{iteration}");
        assert_exact_session_ledger(&state);
        drop(state);
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0, "{iteration}");
    }
}

#[test]
fn dependent_required_presence_bits_inline_heap_and_exact_session_budget() {
    let inline =
        build(r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}"#);
    let inline_state = StructuredState::new(&inline);
    let Frame::Object(inline_object) = &inline_state.frames[0] else {
        panic!("object frame");
    };
    assert!(matches!(
        inline_object.dependent_required_seen,
        PresenceBits::Inline(0)
    ));
    assert_eq!(inline_object.dependent_required_seen.allocation_charge(), 0);

    let required = (0..65)
        .map(|index| format!(r#""r{index}""#))
        .collect::<Vec<_>>()
        .join(",");
    let schema = format!(
        r#"{{"type":"object","dependentRequired":{{"trigger":[{required}]}},"additionalProperties":true}}"#
    );
    let source =
        Arc::new(crate::frontend::schema_to_ir(&schema, CompileOptions::default()).unwrap());
    let baseline = StructuredPlan::compile(source.clone(), StructuredLimits::default()).unwrap();
    let baseline_state =
        StructuredState::new_at(PlanHandle::Borrowed(&baseline), baseline.root).unwrap();
    let Frame::Object(heap_object) = &baseline_state.frames[0] else {
        panic!("object frame");
    };
    assert!(matches!(
        heap_object.dependent_required_seen,
        PresenceBits::Heap(_)
    ));
    assert_eq!(heap_object.dependent_required_seen.allocation_charge(), 16);
    let exact_live = baseline_state.session_memory.live();
    drop(baseline_state);

    let can_construct = |limit| {
        let plan = StructuredPlan::compile(
            source.clone(),
            StructuredLimits {
                max_session_bytes: limit,
                ..StructuredLimits::default()
            },
        )
        .unwrap();
        let succeeds = StructuredState::new_at(PlanHandle::Borrowed(&plan), plan.root).is_ok();
        succeeds
    };
    let mut low = exact_live;
    let mut high = baseline.limits.max_session_bytes;
    while low < high {
        let middle = low + (high - low) / 2;
        if can_construct(middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    println!(
        "dependentRequired heap presence live_session_bytes={exact_live} exact_session_budget={low}"
    );
    assert!(can_construct(low));
    assert!(!can_construct(low - 1));
}

#[test]
#[ignore = "release-only Rust transition latency profile"]
fn dependent_schema_transition_latency_profile() {
    use std::hint::black_box;
    use std::time::Instant;

    fn percentile(samples: &mut [u128], fraction: f64) -> u128 {
        samples.sort_unstable();
        samples[((samples.len() - 1) as f64 * fraction) as usize]
    }

    fn prefix(case: &str, target: usize) -> Vec<u8> {
        let mut fields: Vec<String> = match case {
            "early" => vec![r#""a":1"#.into(), r#""z":1"#.into()],
            "eight" => (0..8)
                .flat_map(|index| [format!(r#""z{index}":1"#), format!(r#""t{index}":1"#)])
                .collect(),
            _ => Vec::new(),
        };
        let current = 1usize
            .checked_add(fields.iter().map(String::len).sum::<usize>())
            .and_then(|bytes| bytes.checked_add(fields.len().saturating_sub(1)))
            .unwrap();
        if current < target {
            let padding = target.saturating_sub(current.saturating_add(9));
            fields.push(format!(r#""pad":"{}""#, "x".repeat(padding)));
        }
        if case == "late" {
            fields.push(r#""z":1"#.into());
            fields.push(r#""a":1"#.into());
        }
        format!("{{{},", fields.join(",")).into_bytes()
    }

    let one = r#"{"type":"object","dependentSchemas":{"a":{"type":"object","required":["z"],"additionalProperties":true}},"additionalProperties":true}"#;
    let eight = r#"{"type":"object","dependentSchemas":{"t0":{"type":"object","required":["z0"],"additionalProperties":true},"t1":{"type":"object","required":["z1"],"additionalProperties":true},"t2":{"type":"object","required":["z2"],"additionalProperties":true},"t3":{"type":"object","required":["z3"],"additionalProperties":true},"t4":{"type":"object","required":["z4"],"additionalProperties":true},"t5":{"type":"object","required":["z5"],"additionalProperties":true},"t6":{"type":"object","required":["z6"],"additionalProperties":true},"t7":{"type":"object","required":["z7"],"additionalProperties":true}},"additionalProperties":true}"#;
    let mut phase_rows = Vec::new();
    for (case, schema) in [("one", one), ("eight", eight)] {
        let mut parse_samples = Vec::with_capacity(32);
        for _ in 0..32 {
            let started = Instant::now();
            black_box(crate::frontend::schema_to_ir(
                black_box(schema),
                CompileOptions::default(),
            ))
            .unwrap();
            parse_samples.push(started.elapsed().as_nanos());
        }
        let ir =
            Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
        let mut plan_samples = Vec::with_capacity(32);
        for _ in 0..32 {
            let started = Instant::now();
            black_box(StructuredPlan::compile(
                ir.clone(),
                StructuredLimits::default(),
            ))
            .unwrap();
            plan_samples.push(started.elapsed().as_nanos());
        }
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        let mut construction_samples = Vec::with_capacity(32);
        for _ in 0..32 {
            let started = Instant::now();
            black_box(StructuredState::new(&plan));
            construction_samples.push(started.elapsed().as_nanos());
        }
        let state = StructuredState::new(&plan);
        phase_rows.push(serde_json::json!({
            "case": case,
            "samples": 32,
            "schema_parse_p50_ns": percentile(&mut parse_samples, 0.50),
            "schema_parse_p95_ns": percentile(&mut parse_samples, 0.95),
            "plan_build_p50_ns": percentile(&mut plan_samples, 0.50),
            "plan_build_p95_ns": percentile(&mut plan_samples, 0.95),
            "matcher_construction_p50_ns": percentile(&mut construction_samples, 0.50),
            "matcher_construction_p95_ns": percentile(&mut construction_samples, 0.95),
            "plan_retained_bytes": plan.retained_bytes,
            "initial_session_bytes": state.session_memory.live(),
            "initial_active_validators": state._validator_ledger.live.load(Ordering::Relaxed),
        }));
    }
    let mut rows = Vec::new();
    for case in ["absent", "early", "late", "eight"] {
        let plan = build(if case == "eight" { eight } else { one });
        for target in [8usize, 128, 512, 4096] {
            let bytes = prefix(case, target);
            let mut state = StructuredState::new(&plan);
            let mut active_validator_peak = state._validator_ledger.live.load(Ordering::Relaxed);
            let mut retained_byte_peak = state.retained_bytes();
            let mut session_byte_peak = state.session_memory.live();
            for &byte in &bytes {
                assert!(state.try_push_byte(byte).unwrap(), "{case} {target}");
                state.commit_checkpoint().unwrap();
                active_validator_peak =
                    active_validator_peak.max(state._validator_ledger.live.load(Ordering::Relaxed));
                retained_byte_peak = retained_byte_peak.max(state.retained_bytes());
                session_byte_peak = session_byte_peak.max(state.session_memory.live());
            }
            let warm = state.checkpoint();
            assert!(state.try_push_byte(b' ').unwrap());
            state.rollback(warm);
            let mut samples = Vec::with_capacity(10_000);
            for _ in 0..10_000 {
                let mark = state.checkpoint();
                let started = Instant::now();
                black_box(state.try_push_byte(black_box(b' ')).unwrap());
                samples.push(started.elapsed().as_nanos());
                state.rollback(mark);
            }
            let p50 = percentile(&mut samples, 0.50);
            let p95 = percentile(&mut samples, 0.95);
            rows.push(serde_json::json!({
                "case": case,
                "prefix_target": target,
                "prefix_bytes": bytes.len(),
                "samples": samples.len(),
                "rust_try_push_byte_p50_ns": p50,
                "rust_try_push_byte_p95_ns": p95,
                "active_validator_peak": active_validator_peak,
                "retained_byte_peak": retained_byte_peak,
                "session_byte_peak": session_byte_peak,
            }));
        }
    }
    let report = serde_json::json!({
        "profile": "Rust release StructuredState::try_push_byte; one speculative space and rollback",
        "phase_rows": phase_rows,
        "rows": rows,
    });
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../integration/logs/program3_dependent_schemas_rust_transition.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("{}", report);
}

fn overlapping_object_schema(count: usize) -> String {
    let patterns = ["^x$", "x$", "^x", ".*x", "x.*", "^.{1}$", "^[x]$", "^(x)$"];
    let entries = patterns[..count]
        .iter()
        .enumerate()
        .map(|(index, pattern)| format!(r#""{pattern}":{{"type":"integer","minimum":{index}}}"#))
        .collect::<Vec<_>>()
        .join(",");
    format!(r#"{{"type":"object","patternProperties":{{{entries}}},"additionalProperties":false}}"#)
}

/// Full logical-state comparison via `Frame`'s derived `PartialEq`. `session_bytes` is
/// excluded: like `Vec` capacity after `truncate`, rollback never frees it, so it's monotonic.
#[derive(PartialEq, Debug)]
struct FullSnapshot {
    frames: Vec<Frame>,
    scope_paths: Vec<Vec<(u32, u32)>>,
    live_scope_slots: usize,
    scope_refs: Vec<usize>,
    validator_live: usize,
    root_annotations: RootAnnotations,
    key_arena: KeyArena,
    undo_len: usize,
    document_bytes: usize,
    accepting: bool,
    dead: bool,
    root_closed: bool,
}

fn full_snapshot(s: &StructuredState<'_>) -> FullSnapshot {
    let scope_paths = s
        .frame_scopes
        .iter()
        .map(|scope| {
            let mut path = Vec::new();
            let mut cursor = Some(scope.id);
            while let Some(id) = cursor {
                path.push((
                    scope.arena.resource(id).expect("scope resource").get(),
                    scope.arena.depth(id).expect("scope depth"),
                ));
                cursor = scope.arena.parent(id).expect("scope parent");
            }
            path
        })
        .collect();
    let live_scope_slots = s.frame_scopes.first().map_or(0, |scope| {
        scope
            .arena
            .slots
            .iter()
            .filter(|slot| slot.refs.load(Ordering::Relaxed) != 0)
            .count()
    });
    let scope_refs = s.frame_scopes.first().map_or_else(Vec::new, |scope| {
        scope
            .arena
            .slots
            .iter()
            .map(|slot| slot.refs.load(Ordering::Relaxed))
            .collect()
    });
    FullSnapshot {
        frames: s.frames.clone(),
        scope_paths,
        live_scope_slots,
        scope_refs,
        validator_live: s._validator_ledger.live.load(Ordering::Relaxed),
        root_annotations: s.root_annotations.clone(),
        key_arena: s.key_arena.clone(),
        undo_len: s.undo.len(),
        document_bytes: s.document_bytes,
        accepting: s.accepting,
        dead: s.dead,
        root_closed: s.root_closed,
    }
}

fn assert_exact_session_ledger(state: &StructuredState<'_>) {
    assert_eq!(
        state.cursors.retained_bytes(),
        state.cursors.recomputed_retained_bytes(),
        "cursor retained total differs from recursive capacities"
    );
    assert_eq!(
        state.recomputed_session_bytes(),
        state.session_memory.live(),
        "session ledger differs from owned capacities"
    );
}

#[test]
fn cursor_retained_total_matches_recursive_capacities_across_speculation() {
    for schema in [
        r#"{"type":"object","additionalProperties":{"type":"string","maxLength":8},"unevaluatedProperties":false}"#,
        r#"{"type":"array","items":{"type":"object"},"uniqueItems":true,"contains":{"type":"object"}}"#,
        r#"{"type":"object","dependentSchemas":{"a":{"properties":{"b":{"type":"string"}}}}}"#,
    ] {
        let plan = build(schema);
        let mut state = StructuredState::new(&plan);
        let document: &[u8] = if schema.contains("uniqueItems") {
            br#"[{"a":"text"},{"b":"more"}]"#
        } else {
            br#"{"a":"text","b":"more"}"#
        };
        for &byte in document {
            for candidate in [b'"', b'\\', b'x', b'}', b']', 0xff] {
                let mark = state.checkpoint();
                let _ = state.try_push_byte(candidate);
                assert_exact_session_ledger(&state);
                state.rollback(mark);
                assert_exact_session_ledger(&state);
            }
            assert!(state.try_push_byte(byte).unwrap(), "{schema}: {byte}");
            assert_exact_session_ledger(&state);
            state.commit_checkpoint().unwrap();
            assert_exact_session_ledger(&state);
        }
        assert!(state.is_accepting());
    }
}

#[test]
fn dynamic_scope_depth_is_exact_and_consecutive_resources_reuse_the_scope() {
    let (arena, _) = ScopeArena::new(4).unwrap();
    let root = arena.root(crate::ir::ResourceId(0)).unwrap();
    let same = arena.enter(&root, crate::ir::ResourceId(0), 3).unwrap();
    assert_eq!(same.id, root.id);
    assert_eq!(arena.depth(same.id).unwrap(), 1);
    let second = arena.enter(&same, crate::ir::ResourceId(1), 3).unwrap();
    let third = arena.enter(&second, crate::ir::ResourceId(2), 3).unwrap();
    assert_eq!(arena.depth(third.id).unwrap(), 3);
    let error = match arena.enter(&third, crate::ir::ResourceId(3), 3) {
        Ok(_) => panic!("scope cap+1 must fail"),
        Err(error) => error,
    };
    assert_eq!(error.kind, LimitKind::DynamicScopeDepth);
    assert_eq!(error.observed, 4);
    assert_eq!(error.limit, 3);
}

#[test]
fn dynamic_memo_rejects_reused_scope_slots_and_different_initial_targets() {
    let (arena, _) = ScopeArena::new_with_anchor_capacity(2, 1).unwrap();
    let first = arena.root(crate::ir::ResourceId(0)).unwrap();
    let first_key = DynamicResolutionKey {
        scope_index: first.id.0,
        scope_generation: arena.generation(first.id).unwrap(),
        anchor: crate::ir::AnchorId(0),
        initial_target: NodeId(1),
    };
    arena.dynamic_memo_store(first_key, NodeId(7));
    assert_eq!(arena.dynamic_memo_get(first_key), Some(NodeId(7)));
    let different_target = DynamicResolutionKey {
        initial_target: NodeId(2),
        ..first_key
    };
    assert_eq!(arena.dynamic_memo_get(different_target), None);
    let reused_index = first.id;
    let old_generation = arena.generation(first.id).unwrap();
    drop(first);

    let second = arena.root(crate::ir::ResourceId(1)).unwrap();
    assert_eq!(second.id, reused_index);
    let second_generation = arena.generation(second.id).unwrap();
    assert_ne!(second_generation, old_generation);
    let second_key = DynamicResolutionKey {
        scope_index: second.id.0,
        scope_generation: second_generation,
        anchor: crate::ir::AnchorId(0),
        initial_target: NodeId(1),
    };
    assert_eq!(arena.dynamic_memo_get(second_key), None);
}

#[test]
fn exhausted_scope_generation_retires_the_slot() {
    let (arena, _) = ScopeArena::new(2).unwrap();
    let first = arena.root(crate::ir::ResourceId(0)).unwrap();
    let retired = first.id;
    drop(first);
    arena.generations[retired.0 as usize].store(u64::MAX, Ordering::Relaxed);
    let next = arena.root(crate::ir::ResourceId(1)).unwrap();
    assert_ne!(next.id, retired);
}

#[test]
fn dynamic_resource_chain_reports_the_exact_runtime_scope_cap() {
    const ROOT: &str = r##"{
        "$dynamicAnchor":"node",
        "$ref":"https://example.com/scope-one"
    }"##;
    const ONE: &str = r##"{
        "$id":"https://example.com/scope-one",
        "$dynamicAnchor":"node",
        "$ref":"https://example.com/scope-two"
    }"##;
    const TWO: &str = r##"{
        "$id":"https://example.com/scope-two",
        "$dynamicAnchor":"node",
        "type":"object",
        "properties":{"next":{"$dynamicRef":"#node"}},
        "additionalProperties":false
    }"##;
    let resources = [
        ("https://example.com/scope-one", ONE),
        ("https://example.com/scope-two", TWO),
    ];

    let exact_limits = StructuredLimits {
        max_dynamic_scope_depth: 3,
        ..StructuredLimits::default()
    };
    let exact = Arc::new(build_resources(
        ROOT,
        "https://example.com/scope-root",
        &resources,
        exact_limits,
    ));
    let mut state = StructuredState::try_new(exact).expect("exact scope cap");
    assert!(feed(&mut state, b"{}"));
    assert!(state.is_accepting());

    let short_limits = StructuredLimits {
        max_dynamic_scope_depth: 2,
        ..StructuredLimits::default()
    };
    let short = Arc::new(build_resources(
        ROOT,
        "https://example.com/scope-root",
        &resources,
        short_limits,
    ));
    let error = match StructuredState::try_new(short) {
        Ok(_) => panic!("scope cap+1 must be typed"),
        Err(error) => error,
    };
    assert_eq!(error.kind, LimitKind::DynamicScopeDepth);
    assert_eq!(error.observed, 3);
    assert_eq!(error.limit, 2);
}

#[test]
fn default_dynamic_scope_cap_is_reachable_without_hitting_validator_depth_first() {
    let root_uri = "https://example.com/default-scope/0";
    let mut defs = serde_json::Map::new();
    for index in 1..256usize {
        let uri = format!("https://example.com/default-scope/{index}");
        let body = if index < 255 {
            serde_json::json!({
                "$id": uri,
                "$dynamicAnchor": "node",
                "$ref": format!("https://example.com/default-scope/{}", index + 1),
            })
        } else {
            serde_json::json!({
                "$id": uri,
                "$dynamicAnchor": "node",
                "type": "object",
                "properties": {"child": {"$dynamicRef": "#node"}},
                "additionalProperties": false,
            })
        };
        defs.insert(format!("scope{index}"), body);
    }
    let root = serde_json::json!({
        "$id": root_uri,
        "$dynamicAnchor": "node",
        "$ref": "https://example.com/default-scope/1",
        "$defs": defs,
    })
    .to_string();
    let ir = lower_resources(&root, root_uri, &[]);
    let plan = Arc::new(
        StructuredPlan::compile(ir, StructuredLimits::default()).expect("exact scope plan"),
    );
    assert!(plan.dynamic_scope_enabled);
    let state = StructuredState::try_new(plan).expect("exact default dynamic-scope cap");
    let deepest = state
        .frame_scopes
        .iter()
        .filter_map(|scope| scope.arena.depth(scope.id).ok())
        .max();
    assert_eq!(deepest, Some(256));
}

#[test]
fn dynamic_strict_tree_uses_outer_scope_and_rolls_back_without_scope_leaks() {
    let plan = build_resources(
        STRICT_DYNAMIC_TREE,
        "https://example.com/strict-tree",
        &[("https://example.com/tree", DYNAMIC_TREE)],
        StructuredLimits::default(),
    );
    assert!(plan.dynamic_scope_enabled);
    assert!(enum_accepts(
        &plan,
        br#"{"data":1,"children":[{"data":2}]}"#
    ));
    assert!(enum_accepts(&plan, br#"{"children":[{"data":"ok"}]}"#));
    assert!(!enum_accepts(
        &plan,
        br#"{"children":[{"misspelled":true}]}"#
    ));

    for probe in 0u8..=u8::MAX {
        let mut candidate = StructuredState::new(&plan);
        assert!(feed(&mut candidate, br##"{"children":[{"data":""##));
        let before_probe = full_snapshot(&candidate);
        let mark = candidate.checkpoint();
        if candidate.try_push_byte(probe).unwrap_or(false) {
            candidate.rollback(mark);
        }
        let after_probe = full_snapshot(&candidate);
        assert_eq!(after_probe, before_probe, "probe={probe}");
        for (index, &byte) in br##"ok"}]}"##.iter().enumerate() {
            let result = candidate.try_push_byte(byte);
            assert_eq!(result, Ok(true), "probe={probe} suffix={index}");
        }
        assert!(candidate.is_accepting(), "probe={probe}");
    }

    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br##"{"children":[{"data":""##));
    let baseline = full_snapshot(&state);
    for byte in 0u8..=u8::MAX {
        let mark = state.checkpoint();
        let _ = state.try_push_byte(byte);
        state.rollback(mark);
        assert_eq!(full_snapshot(&state), baseline, "byte={byte}");
    }
    for _ in 0..1_000 {
        let mark = state.checkpoint();
        assert!(!state.try_push_byte(0).unwrap());
        state.rollback(mark);
        assert_eq!(full_snapshot(&state), baseline);
    }
    for (index, &byte) in br##"ok"}]}"##.iter().enumerate() {
        assert_eq!(
            state.try_push_byte(byte),
            Ok(true),
            "suffix index={index} byte={byte}"
        );
    }
    assert!(state.is_accepting());
}

#[test]
fn dynamic_strict_tree_matches_the_independent_oracle_for_prefixes_and_bytes() {
    use crate::structured::matcher::StructuredProgram;

    let ir = lower_resources(
        STRICT_DYNAMIC_TREE,
        "https://example.com/strict-tree",
        &[("https://example.com/tree", DYNAMIC_TREE)],
    );
    let program = StructuredProgram::compile(ir.clone()).expect("program");
    let documents: &[(&[u8], bool)] = &[
        (br#"{"children":[{"data":"ok"}]}"#, true),
        (br#"{"data":1,"children":[{"data":2}]}"#, true),
        (br#"{"children":[{"misspelled":true}]}"#, false),
        (br#"{"children":[{"children":[{"extra":0}]}]}"#, false),
    ];
    for &(document, expected) in documents {
        let production = crate::structured::try_accepts(ir.clone(), document).expect("production");
        let oracle = crate::structured::reference::accepts(&ir, document);
        assert_eq!(production, expected, "production document={document:?}");
        assert_eq!(oracle, expected, "oracle document={document:?}");
    }

    let valid = documents[0].0;
    for end in 0..=valid.len() {
        let prefix = &valid[..end];
        let mut production = program.new_matcher().expect("matcher");
        let production_viable = if prefix.is_empty() {
            true
        } else {
            production.advance(prefix).expect("advance")
        };
        let oracle_viable = crate::structured::reference::accepts(&ir, prefix)
            || crate::structured::reference::can_continue(&ir, prefix);
        assert_eq!(production_viable, oracle_viable, "prefix={prefix:?}");
        if production_viable {
            assert_eq!(
                production.is_accepting(),
                crate::structured::reference::accepts(&ir, prefix),
                "accepting prefix={prefix:?}"
            );
            assert_eq!(production.eos_legal(), production.is_accepting());
        }
    }

    let probe_prefixes: &[&[u8]] = &[
        b"".as_slice(),
        br#"{"children":["#,
        br#"{"children":[{"data":"#,
    ];
    for &prefix in probe_prefixes {
        for byte in 0u8..=127 {
            let mut candidate = prefix.to_vec();
            candidate.push(byte);
            let mut production = program.new_matcher().expect("matcher");
            let production_viable = production.advance(&candidate).expect("advance");
            let oracle_viable = crate::structured::reference::accepts(&ir, &candidate)
                || crate::structured::reference::can_continue(&ir, &candidate);
            assert_eq!(
                production_viable, oracle_viable,
                "prefix={prefix:?} byte={byte}"
            );
        }
    }
}

fn assert_resource_oracle_matrix(
    ir: Arc<crate::ir::SchemaIR>,
    documents: &[(&[u8], bool)],
    probe_prefixes: &[&[u8]],
) {
    use crate::index::{TrieCache, VocabularyHandle};
    use crate::primitives::TokenId;
    use crate::structured::matcher::StructuredProgram;
    use crate::vocab::build_vocabulary;
    use rustc_hash::FxHashMap;

    let program = StructuredProgram::compile(ir.clone()).expect("program");
    for &(document, expected) in documents {
        let production = crate::structured::try_accepts(ir.clone(), document).expect("production");
        let oracle = crate::structured::reference::accepts(&ir, document);
        assert_eq!(production, expected, "production document={document:?}");
        assert_eq!(oracle, expected, "oracle document={document:?}");
        if expected {
            for end in 0..=document.len() {
                let prefix = &document[..end];
                let mut matcher = program.new_matcher().expect("matcher");
                let viable = prefix.is_empty() || matcher.advance(prefix).expect("advance");
                let oracle_viable = crate::structured::reference::accepts(&ir, prefix)
                    || crate::structured::reference::can_continue(&ir, prefix);
                assert_eq!(viable, oracle_viable, "prefix={prefix:?}");
                assert!(viable, "valid document prefix={prefix:?}");
                assert_eq!(
                    matcher.is_accepting(),
                    crate::structured::reference::accepts(&ir, prefix),
                    "accepting prefix={prefix:?}"
                );
                assert_eq!(matcher.eos_legal(), matcher.is_accepting());
                assert!(!matcher.is_dead());
            }
        }
    }

    let mut tokens = FxHashMap::default();
    for byte in 0u8..=127 {
        tokens.insert(vec![byte], vec![u32::from(byte)]);
    }
    let vocabulary = VocabularyHandle::new(Arc::new(
        build_vocabulary(128, tokens).expect("ASCII vocabulary"),
    ))
    .expect("vocabulary handle");
    let trie = TrieCache::new().bind(&vocabulary).expect("trie");
    let output_bytes = vocabulary.mask_vocab_size().div_ceil(32) * 4;

    for &prefix in probe_prefixes {
        let mut direct = program.new_matcher().expect("direct matcher");
        let mut preallocated = program.new_matcher().expect("preallocated matcher");
        let mut walked = program.new_matcher().expect("trie matcher");
        if !prefix.is_empty() {
            assert!(direct.advance(prefix).expect("direct prefix"), "{prefix:?}");
            assert!(
                preallocated.advance(prefix).expect("preallocated prefix"),
                "{prefix:?}"
            );
            assert!(walked.advance(prefix).expect("trie prefix"), "{prefix:?}");
        }
        let before = (
            direct.is_accepting(),
            direct.is_dead(),
            preallocated.is_accepting(),
            preallocated.is_dead(),
            walked.is_accepting(),
            walked.is_dead(),
        );
        let direct_mask = direct
            .allowed_mask_from_records(vocabulary.mask_vocab_size(), vocabulary.iter_records())
            .expect("direct mask");
        let mut preallocated_mask = vec![0xa5; output_bytes];
        preallocated
            .write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut preallocated_mask,
            )
            .expect("preallocated mask");
        let mut trie_mask = vec![0x5a; output_bytes];
        walked
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut trie_mask)
            .expect("trie mask");
        let after = (
            direct.is_accepting(),
            direct.is_dead(),
            preallocated.is_accepting(),
            preallocated.is_dead(),
            walked.is_accepting(),
            walked.is_dead(),
        );
        assert_eq!(after, before, "mask rollback prefix={prefix:?}");

        for byte in 0u8..=127 {
            let mut candidate = prefix.to_vec();
            candidate.push(byte);
            let expected = crate::structured::reference::accepts(&ir, &candidate)
                || crate::structured::reference::can_continue(&ir, &candidate);
            let id = usize::from(byte);
            let direct_bit = direct_mask.get(TokenId(u32::from(byte)));
            let preallocated_bit = preallocated_mask[id / 8] & (1u8 << (id % 8)) != 0;
            let trie_bit = trie_mask[id / 8] & (1u8 << (id % 8)) != 0;
            assert_eq!(direct_bit, expected, "direct prefix={prefix:?} byte={byte}");
            assert_eq!(
                preallocated_bit, expected,
                "record prefix={prefix:?} byte={byte}"
            );
            assert_eq!(trie_bit, expected, "trie prefix={prefix:?} byte={byte}");
        }
    }
}

#[test]
fn external_dynamic_recursive_hard_combinations_match_the_independent_oracle() {
    let external_not = lower_resources(
        r#"{"$ref":"https://example.com/not"}"#,
        "https://example.com/root",
        &[(
            "https://example.com/not",
            r#"{"$id":"https://example.com/not","not":{"const":1}}"#,
        )],
    );
    assert!(
        crate::structured::reference::can_continue(&external_not, b"-"),
        "a negative JSON number remains a viable prefix under not(const: 1)"
    );
    assert_resource_oracle_matrix(
        external_not,
        &[(b"2", true), (b"null", true), (b"1", false)],
        &[b"", b"-"],
    );

    let external_one_of = lower_resources(
        r#"{"$ref":"https://example.com/choice"}"#,
        "https://example.com/root",
        &[(
            "https://example.com/choice",
            r#"{"$id":"https://example.com/choice","oneOf":[{"type":"string"},{"const":1}]}"#,
        )],
    );
    assert_resource_oracle_matrix(
        external_one_of,
        &[(br#""ok""#, true), (b"1", true), (b"false", false)],
        &[b"", b"\""],
    );

    let referenced_intersection = lower_resources(
        r#"{"$ref":"https://example.com/intersection"}"#,
        "https://example.com/root",
        &[(
            "https://example.com/intersection",
            r#"{"$id":"https://example.com/intersection","allOf":[{"type":"string","pattern":"^a"},{"type":"string","pattern":"z$"}]}"#,
        )],
    );
    assert_resource_oracle_matrix(
        referenced_intersection,
        &[(br#""az""#, true), (br#""ab""#, false)],
        &[b"", b"\"", b"\"a"],
    );

    const DEPENDENT_TREE: &str = r##"{
        "$id":"https://example.com/dependent-tree",
        "$dynamicAnchor":"node",
        "type":"object",
        "properties":{
            "flag":true,
            "data":true,
            "children":{"type":"array","items":{"$dynamicRef":"#node"}}
        },
        "dependentSchemas":{"flag":{"required":["data"]}}
    }"##;
    let dynamic_dependent = lower_resources(
        r##"{
            "$id":"https://example.com/strict-dependent-tree",
            "$dynamicAnchor":"node",
            "$ref":"https://example.com/dependent-tree",
            "unevaluatedProperties":false
        }"##,
        "https://example.com/strict-dependent-tree",
        &[("https://example.com/dependent-tree", DEPENDENT_TREE)],
    );
    assert_resource_oracle_matrix(
        dynamic_dependent,
        &[
            (br#"{"flag":true,"data":1}"#, true),
            (br#"{"children":[{"flag":true,"data":2}]}"#, true),
            (br#"{"flag":true}"#, false),
            (br#"{"children":[{"flag":true}]}"#, false),
        ],
        &[b"", b"{", br#"{"children":["#],
    );

    let recursive_array = lower_resources(
        r#"{"$ref":"https://example.com/recursive-array"}"#,
        "https://example.com/root",
        &[(
            "https://example.com/recursive-array",
            r##"{
                "$id":"https://example.com/recursive-array",
                "$anchor":"node",
                "type":"array",
                "items":{"anyOf":[{"type":"integer"},{"$ref":"#node"}]},
                "uniqueItems":true,
                "contains":{"type":"integer"}
            }"##,
        )],
    );
    assert_resource_oracle_matrix(
        recursive_array,
        &[(b"[1,[2]]", true), (b"[1,1]", false), (b"[[2]]", false)],
        &[b"", b"[", b"[1,"],
    );

    let enum_values = (0..=MAX_UNROLLED_ENUM)
        .map(|index| format!(r#""v{index}""#))
        .collect::<Vec<_>>()
        .join(",");
    let recursive_enum_schema = format!(
        r##"{{
            "$id":"https://example.com/recursive-enum",
            "$anchor":"node",
            "type":"object",
            "properties":{{
                "tag":{{"enum":[{enum_values}]}},
                "next":{{"$ref":"#node"}}
            }},
            "required":["tag"],
            "additionalProperties":false
        }}"##
    );
    let recursive_enum = lower_resources(
        r#"{"$ref":"https://example.com/recursive-enum"}"#,
        "https://example.com/root",
        &[("https://example.com/recursive-enum", &recursive_enum_schema)],
    );
    assert_resource_oracle_matrix(
        recursive_enum,
        &[
            (br#"{"tag":"v0"}"#, true),
            (br#"{"tag":"v1","next":{"tag":"v2"}}"#, true),
            (br#"{"tag":"missing"}"#, false),
        ],
        &[b"", b"{", br#"{"tag":"#],
    );
}

#[derive(PartialEq, Debug)]
struct CursorPoolSnapshot {
    slots: Vec<CursorSlot>,
    generations: Vec<u32>,
    epochs: Vec<u32>,
    structured_occupied: Vec<bool>,
    free_slots: Vec<u32>,
    free_structured: Vec<u32>,
}

fn cursor_pool_snapshot(machine: &CursorMachine<'_>) -> CursorPoolSnapshot {
    CursorPoolSnapshot {
        slots: machine.slots.clone(),
        generations: machine.generations.clone(),
        epochs: machine.epochs.clone(),
        structured_occupied: machine.structured.iter().map(Option::is_some).collect(),
        free_slots: machine.free_slots.clone(),
        free_structured: machine.free_structured.clone(),
    }
}

fn feed_value_cursor(
    machine: &mut CursorMachine<'_>,
    cursor: CursorId,
    bytes: &[u8],
) -> Result<CursorStep, StructuredRuntimeError> {
    let mut step = CursorStep::Alive;
    for &byte in bytes {
        step = machine.try_push_byte(cursor, byte)?;
        if step == CursorStep::Dead {
            break;
        }
    }
    Ok(step)
}

#[test]
fn value_cursor_validates_regular_scalar() {
    let plan = build(r#"{"type":"boolean"}"#);
    let mut machine = CursorMachine::new(&plan);
    let cursor = machine.start(plan.root).unwrap();
    assert_eq!(
        feed_value_cursor(&mut machine, cursor, b"true").unwrap(),
        CursorStep::Complete
    );
    assert!(machine.is_accepting(cursor));
}

#[test]
fn value_cursor_accepts_negative_zero_in_a_bounded_number_schema() {
    let plan = build(r#"{"type":"number","minimum":-1,"maximum":2}"#);
    let mut machine = CursorMachine::new(&plan);
    let cursor = machine.start(plan.root).unwrap();
    assert_eq!(
        feed_value_cursor(&mut machine, cursor, b"-0").unwrap(),
        CursorStep::Complete
    );
    assert!(machine.is_accepting(cursor));
}

#[test]
fn value_cursor_validates_any_nested_value() {
    let plan = build(r#"{"type":"boolean"}"#);
    let mut machine = CursorMachine::new(&plan);
    let cursor = machine.start_any().unwrap();
    assert_ne!(
        feed_value_cursor(&mut machine, cursor, br#"{"a":[null,true,{"b":"x"}]}"#).unwrap(),
        CursorStep::Dead
    );
    assert!(machine.is_accepting(cursor));
}

#[test]
fn value_cursor_validates_structured_object() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"],"additionalProperties":false}"#,
    );
    let mut machine = CursorMachine::new(&plan);
    let cursor = machine.start(plan.root).unwrap();
    assert_ne!(
        feed_value_cursor(&mut machine, cursor, br#"{"a":true}"#).unwrap(),
        CursorStep::Dead
    );
    assert!(machine.is_accepting(cursor));
}

#[test]
fn value_cursor_rollback_restores_accepted_and_rejected_bytes() {
    let plan = build(r#"{"type":"boolean"}"#);
    let mut machine = CursorMachine::new(&plan);
    let cursor = machine.start(plan.root).unwrap();
    assert_ne!(
        feed_value_cursor(&mut machine, cursor, b"tru").unwrap(),
        CursorStep::Dead
    );
    let mark = machine.checkpoint(cursor).unwrap();
    assert_eq!(
        machine.try_push_byte(cursor, b'e').unwrap(),
        CursorStep::Complete
    );
    machine.rollback(cursor, mark).unwrap();
    assert!(!machine.is_accepting(cursor));
    assert_eq!(
        machine.try_push_byte(cursor, b'x').unwrap(),
        CursorStep::Dead
    );
    machine.rollback(cursor, mark).unwrap();
    assert_eq!(
        machine.try_push_byte(cursor, b'e').unwrap(),
        CursorStep::Complete
    );
}

#[test]
fn value_cursor_commit_invalidates_older_marks() {
    let plan = build(r#"{"type":"boolean"}"#);
    let mut machine = CursorMachine::new(&plan);
    let cursor = machine.start(plan.root).unwrap();
    let mark = machine.checkpoint(cursor).unwrap();
    assert_ne!(
        feed_value_cursor(&mut machine, cursor, b"true").unwrap(),
        CursorStep::Dead
    );
    machine.commit(cursor).unwrap();
    assert!(machine.rollback(cursor, mark).is_err());
    assert!(machine.is_accepting(cursor));
}

#[test]
fn value_cursor_reuse_rejects_stale_ids() {
    let plan = build(r#"{"type":"boolean"}"#);
    let mut machine = CursorMachine::new(&plan);
    let old = machine.start(plan.root).unwrap();
    machine.release(old).unwrap();
    let fresh = machine.start(plan.root).unwrap();
    assert_eq!(old.index, fresh.index);
    assert_ne!(old.generation, fresh.generation);
    let live = machine.validator_ledger.live.load(Ordering::Relaxed);
    assert!(machine.checkpoint(old).is_err());
    assert!(machine.release(old).is_err());
    assert_eq!(machine.validator_ledger.live.load(Ordering::Relaxed), live);
    assert_ne!(
        feed_value_cursor(&mut machine, fresh, b"false").unwrap(),
        CursorStep::Dead
    );
    assert!(machine.is_accepting(fresh));
    machine.release(fresh).unwrap();
    assert_eq!(machine.validator_ledger.live.load(Ordering::Relaxed), 0);
    assert!(machine.release(fresh).is_err());
    assert_eq!(machine.validator_ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn failed_cursor_construction_and_cursor_tree_drop_balance_the_validator_ledger() {
    let plan = build(r#"{"type":"object","additionalProperties":{"type":"integer"}}"#);
    let denied = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: 0,
    });
    let mut machine = CursorMachine::from_handle(PlanHandle::Borrowed(&plan), denied.clone());
    assert!(machine.start(plan.root).is_err());
    assert_eq!(denied.live.load(Ordering::Relaxed), 0);

    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: 8,
    });
    {
        let mut machine = CursorMachine::from_handle(PlanHandle::Borrowed(&plan), ledger.clone());
        machine.start(plan.root).unwrap();
        machine.start(plan.root).unwrap();
        assert_eq!(ledger.live.load(Ordering::Relaxed), 2);
    }
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn armed_validator_reservation_releases_on_drop() {
    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(1),
        limit: 1,
    });
    {
        let _reservation = ValidatorReservation::new(ledger.clone());
    }
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn negation_preserves_prefix_and_final_acceptance_semantics() {
    let plan = build(r#"{"not":{"const":1}}"#);
    for (document, viable, accepted) in [
        (b"1".as_slice(), true, false),
        (b"12".as_slice(), true, true),
        (b"1 ".as_slice(), false, false),
        (b"2".as_slice(), true, true),
    ] {
        let mut state = StructuredState::new(&plan);
        assert_eq!(feed(&mut state, document), viable, "{document:?}");
        assert_eq!(state.is_accepting(), accepted, "{document:?}");
    }
}

#[test]
fn negation_rejects_a_terminal_literal_without_mutation() {
    let plan = build(r#"{"not":{"enum":[null,true,"x"]}}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"nul"));
    let before = full_snapshot(&state);
    assert!(!state.try_push_byte(b'l').unwrap());
    assert_eq!(full_snapshot(&state), before);
    let [Frame::Negation(frame)] = state.frames.as_slice() else {
        panic!("expected negation frame");
    };
    assert_eq!(frame.inner_state, NegationInnerState::Active);
    assert!(!state.cursors.is_accepting(frame.syntax));
    assert!(!state.cursors.is_accepting(frame.inner));
    assert!(!state.is_accepting());
}

#[test]
fn negation_accepts_non_arrays_and_rejects_arrays() {
    let plan = build(r#"{"not":{"type":"array"}}"#);
    for (document, accepted) in [
        (b"[]".as_slice(), false),
        (b"[1]".as_slice(), false),
        (b"{}".as_slice(), true),
        (b"1".as_slice(), true),
        (br#""x""#.as_slice(), true),
    ] {
        let mut state = StructuredState::new(&plan);
        let advanced = feed(&mut state, document);
        assert!(advanced || !accepted, "{document:?}");
        assert_eq!(state.is_accepting(), accepted, "{document:?}");
    }
}

#[test]
fn negation_as_an_array_item_replays_delimiters_and_tracks_unique_items() {
    let schema = r#"{"type":"array","items":{"not":{"const":1}},"uniqueItems":true}"#;
    let plan = build(schema);
    for (document, accepted) in [
        (b"[2,3]".as_slice(), true),
        (b"[2,2]".as_slice(), false),
        (b"[1]".as_slice(), false),
        (b"[12, 2]".as_slice(), true),
    ] {
        let mut state = StructuredState::new(&plan);
        let advanced = feed(&mut state, document);
        assert!(advanced || !accepted, "{document:?}");
        assert_eq!(state.is_accepting(), accepted, "{document:?}");
    }
}

#[test]
fn negation_inside_contains_reports_the_final_item_verdict() {
    let schema = r#"{"type":"array","items":true,"contains":{"not":{"const":1}},"minContains":1}"#;
    let plan = build(schema);
    for (document, accepted) in [
        (b"[1]".as_slice(), false),
        (b"[1,2]".as_slice(), true),
        (b"[2]".as_slice(), true),
    ] {
        let mut state = StructuredState::new(&plan);
        let advanced = feed(&mut state, document);
        assert!(advanced || !accepted, "{document:?}");
        assert_eq!(state.is_accepting(), accepted, "{document:?}");
    }
}

#[test]
fn negation_uses_exactly_two_child_validators_and_constructs_atomically() {
    let ir = Arc::new(
        crate::frontend::schema_to_ir(r#"{"not":{"const":1}}"#, CompileOptions::default()).unwrap(),
    );
    let exact = StructuredPlan::compile(
        ir.clone(),
        StructuredLimits {
            max_active_validators: 2,
            ..StructuredLimits::default()
        },
    )
    .unwrap();
    let state = StructuredState::new_at(PlanHandle::Borrowed(&exact), exact.root).unwrap();
    assert_eq!(state._validator_ledger.live.load(Ordering::Relaxed), 2);
    let exact_ledger = state._validator_ledger.clone();
    drop(state);
    assert_eq!(exact_ledger.live.load(Ordering::Relaxed), 0);

    let short = StructuredPlan::compile(
        ir,
        StructuredLimits {
            max_active_validators: 1,
            ..StructuredLimits::default()
        },
    )
    .unwrap();
    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: 1,
    });
    let result = StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&short),
        short.root,
        0,
        ledger.clone(),
    );
    assert!(result.is_err());
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn negation_pruning_rolls_back_or_commits_with_exact_ownership() {
    let plan = build(r#"{"not":{"const":"a"}}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"\"a"));
    let mark = state.checkpoint();
    assert!(state.try_push_byte(b'b').unwrap());
    assert!(matches!(
        state.frames.as_slice(),
        [Frame::Negation(NegationFrame {
            inner_state: NegationInnerState::PrunedOwned,
            ..
        })]
    ));
    assert_eq!(state._validator_ledger.live.load(Ordering::Relaxed), 2);
    state.rollback(mark);
    assert!(matches!(
        state.frames.as_slice(),
        [Frame::Negation(NegationFrame {
            inner_state: NegationInnerState::Active,
            ..
        })]
    ));
    assert!(!state.is_accepting());

    assert!(state.try_push_byte(b'b').unwrap());
    state.commit_checkpoint().unwrap();
    assert!(matches!(
        state.frames.as_slice(),
        [Frame::Negation(NegationFrame {
            inner_state: NegationInnerState::Released,
            ..
        })]
    ));
    assert_eq!(state._validator_ledger.live.load(Ordering::Relaxed), 1);
    let ledger = state._validator_ledger.clone();
    drop(state);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn dropping_a_pruned_owned_negation_releases_both_cursors() {
    let plan = build(r#"{"not":{"const":"a"}}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"\"ab"));
    assert!(matches!(
        state.frames.as_slice(),
        [Frame::Negation(NegationFrame {
            inner_state: NegationInnerState::PrunedOwned,
            ..
        })]
    ));
    let ledger = state._validator_ledger.clone();
    assert_eq!(ledger.live.load(Ordering::Relaxed), 2);
    drop(state);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn negation_undo_limit_failure_precedes_cursor_mutation() {
    let plan = build(r#"{"not":{"const":1}}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"1"));
    let logical = full_snapshot(&state);
    let pool = cursor_pool_snapshot(&state.cursors);
    let session = state.session_memory.live();
    let validators = state._validator_ledger.live.load(Ordering::Relaxed);
    state.limits.max_undo_entries = state.undo.len();
    assert!(matches!(
        state.try_push_byte(b'2'),
        Err(StructuredRuntimeError {
            kind: LimitKind::UndoEntries,
            ..
        })
    ));
    assert_eq!(full_snapshot(&state), logical);
    assert_eq!(cursor_pool_snapshot(&state.cursors), pool);
    assert_eq!(state.session_memory.live(), session);
    assert_eq!(
        state._validator_ledger.live.load(Ordering::Relaxed),
        validators
    );
    state.limits.max_undo_entries = usize::MAX;
    assert!(state.try_push_byte(b'2').unwrap());
}

#[test]
fn negation_child_depth_limit_is_exact_and_atomic() {
    let ir = Arc::new(
        crate::frontend::schema_to_ir(r#"{"not":{"const":1}}"#, CompileOptions::default()).unwrap(),
    );
    let exact = StructuredPlan::compile(
        ir.clone(),
        StructuredLimits {
            max_depth: 2,
            ..StructuredLimits::default()
        },
    )
    .unwrap();
    assert!(StructuredState::new_at(PlanHandle::Borrowed(&exact), exact.root).is_ok());

    let over = StructuredPlan::compile(
        ir,
        StructuredLimits {
            max_depth: 1,
            ..StructuredLimits::default()
        },
    )
    .unwrap();
    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: over.limits.max_active_validators,
    });
    let result = StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&over),
        over.root,
        0,
        ledger.clone(),
    );
    assert!(matches!(
        result,
        Err(StructuredRuntimeError {
            kind: LimitKind::RecursionDepth,
            ..
        })
    ));
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn nested_negation_construction_failure_restores_cursor_pools() {
    let ir = Arc::new(
        crate::frontend::schema_to_ir(r#"{"not":{"not":{"const":1}}}"#, CompileOptions::default())
            .unwrap(),
    );
    let plan = StructuredPlan::compile(
        ir,
        StructuredLimits {
            max_active_validators: 2,
            ..StructuredLimits::default()
        },
    )
    .unwrap();
    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: plan.limits.max_active_validators,
    });
    let mut state = StructuredState::empty_for_test(&plan, ledger.clone()).unwrap();
    let before = cursor_pool_snapshot(&state.cursors);
    assert!(state.prepare_negation_frame(plan.root).is_err());
    assert_eq!(cursor_pool_snapshot(&state.cursors), before);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
    assert_exact_session_ledger(&state);
}

#[test]
fn negation_first_and_second_cursor_session_failures_are_atomic() {
    let plan = build(r#"{"not":{"const":1}}"#);
    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: plan.limits.max_active_validators,
    });

    let mut first = StructuredState::empty_for_test(&plan, ledger.clone()).unwrap();
    let first_floor = first.cursors.start_any_growth_floor().unwrap();
    let first_observed = first.retained_bytes().checked_add(first_floor).unwrap();
    first.limits.max_session_bytes = first_observed - 1;
    let before = cursor_pool_snapshot(&first.cursors);
    assert!(matches!(
        first.prepare_negation_frame(plan.root),
        Err(StructuredRuntimeError {
            kind: LimitKind::SessionBytes,
            ..
        })
    ));
    assert_eq!(cursor_pool_snapshot(&first.cursors), before);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);

    let mut second = StructuredState::empty_for_test(&plan, ledger.clone()).unwrap();
    let pool_mark = second.cursors.pool_mark().unwrap();
    let syntax = second.start_any_cursor_at_depth(1).unwrap();
    let NodePlan::Negation { inner } = plan.node(plan.root) else {
        panic!("expected negation plan");
    };
    let second_floor = second.cursors.start_growth_floor(*inner).unwrap();
    let second_limit = second
        .retained_bytes()
        .checked_add(second_floor)
        .unwrap()
        .checked_sub(1)
        .unwrap();
    second
        .cursors
        .rollback_unpublished(syntax, pool_mark)
        .unwrap();
    second.limits.max_session_bytes = second_limit;
    let before = cursor_pool_snapshot(&second.cursors);
    assert!(matches!(
        second.prepare_negation_frame(plan.root),
        Err(StructuredRuntimeError {
            kind: LimitKind::SessionBytes,
            ..
        })
    ));
    assert_eq!(cursor_pool_snapshot(&second.cursors), before);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn repeated_negation_construct_rollback_and_commit_has_no_ledger_drift() {
    let plan = build(r#"{"not":{"const":"a"}}"#);
    for iteration in 0..10_000 {
        let mut state = StructuredState::new_at(PlanHandle::Borrowed(&plan), plan.root).unwrap();
        let ledger = state._validator_ledger.clone();
        let mark = state.checkpoint();
        assert!(feed(&mut state, b"\"ab"));
        if iteration % 2 == 0 {
            state.rollback(mark);
            assert_eq!(ledger.live.load(Ordering::Relaxed), 2);
        } else {
            state.commit_checkpoint().unwrap();
            assert_eq!(ledger.live.load(Ordering::Relaxed), 1);
        }
        drop(state);
        assert_eq!(
            ledger.live.load(Ordering::Relaxed),
            0,
            "iteration {iteration}"
        );
    }
}

#[test]
fn committed_negation_prune_rejects_stale_inner_generation() {
    let plan = build(r#"{"not":{"const":"a"}}"#);
    let mut state = StructuredState::new(&plan);
    let [Frame::Negation(frame)] = state.frames.as_slice() else {
        panic!("expected negation frame");
    };
    let stale = frame.inner;
    assert!(feed(&mut state, b"\"ab"));
    state.commit_checkpoint().unwrap();
    let fresh = state.cursors.start(plan.root).unwrap();
    assert_eq!(stale.index, fresh.index);
    assert_ne!(stale.generation, fresh.generation);
    let live = state._validator_ledger.live.load(Ordering::Relaxed);
    assert!(state.cursors.try_push_byte(stale, b'b').is_err());
    assert_eq!(state._validator_ledger.live.load(Ordering::Relaxed), live);
    state.cursors.release(fresh).unwrap();
}

#[test]
fn value_cursor_runtime_types_remain_compact() {
    assert!(std::mem::size_of::<CursorId>() <= 8);
    assert!(std::mem::size_of::<CursorSlot>() <= 16);
    assert!(std::mem::size_of::<CursorMark>() <= 40);
}

#[test]
fn cursor_retained_capacity_stays_within_the_session_budget() {
    let schema = r#"{"type":"array","items":true,"contains":{"type":"object","properties":{"a":{"type":"array","items":true}},"additionalProperties":true},"minContains":0}"#;
    for budget in [512usize, 1024, 2048, 4096, 8192] {
        let ir =
            Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
        let limits = StructuredLimits {
            max_session_bytes: budget,
            ..StructuredLimits::default()
        };
        let plan = StructuredPlan::compile(ir, limits).unwrap();
        let mut state = StructuredState::new(&plan);
        for &byte in br#"[{"a":[{"b":[1,2,3]}]}]"# {
            if state.try_push_byte(byte).is_err() {
                break;
            }
        }
        let retained = state.retained_bytes();
        // `retained_bytes` is a capacity recompute; amortized structured-cursor growth can overshoot
        // the budget-checked floor by up to one slot, so allow that bounded, non-growing slack.
        let slack = std::mem::size_of::<Option<super::StructuredState<'_>>>();
        assert!(
            retained <= budget.saturating_add(slack),
            "budget={budget} retained={retained}"
        );
    }
}

#[test]
fn decoded_string_pattern_treats_plain_and_escaped_scalars_equally() {
    let plan = build(r#"{"type":"string","pattern":"^a$","minLength":1,"maxLength":1}"#);
    for value in [br#""a""#.as_slice(), br#""\u0061""#.as_slice()] {
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, value));
        assert!(state.is_accepting());
    }
}

#[test]
fn decoded_string_pattern_uses_search_semantics() {
    let plan = build(r#"{"type":"string","pattern":"a"}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#""ba""#));
    assert!(state.is_accepting());
}

#[test]
fn intersected_string_patterns_share_one_compiled_transition_path() {
    let plan =
        build(r#"{"allOf":[{"type":"string","pattern":"a"},{"type":"string","pattern":"b"}]}"#);
    for value in [br#""ab""#.as_slice(), br#""ba""#.as_slice()] {
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, value));
        assert!(state.is_accepting());
    }
    for value in [br#""a""#.as_slice(), br#""b""#.as_slice()] {
        let mut state = StructuredState::new(&plan);
        assert!(!feed(&mut state, value) || !state.is_accepting());
    }
}

#[test]
fn decoded_string_length_counts_non_bmp_scalar_once() {
    let plan = build(r#"{"type":"string","minLength":1,"maxLength":1}"#);
    for value in ["\"😀\"".as_bytes(), br#""\uD83D\uDE00""#.as_slice()] {
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, value));
        assert!(state.is_accepting());
    }
}

#[test]
fn decoded_string_rejects_malformed_surrogates_and_utf8() {
    let plan = build(r#"{"type":"string","maxLength":8}"#);
    for value in [
        br#""\uDE00""#.as_slice(),
        br#""\uD83D""#.as_slice(),
        &[b'"', 0xed, 0xa0, 0x80, b'"'],
    ] {
        let mut state = StructuredState::new(&plan);
        assert!(!feed(&mut state, value));
    }
}

#[test]
fn property_names_validates_decoded_key_spelling() {
    let plan = build(
        r#"{"type":"object","propertyNames":{"type":"string","pattern":"^a$","minLength":1,"maxLength":1},"additionalProperties":true}"#,
    );
    for value in [br#"{"a":1}"#.as_slice(), br#"{"\u0061":1}"#.as_slice()] {
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, value));
        assert!(state.is_accepting());
    }
    let mut rejected = StructuredState::new(&plan);
    assert!(!feed(&mut rejected, br#"{"b":1}"#));
}

#[test]
fn property_names_false_rejects_every_key() {
    let plan = build(r#"{"type":"object","propertyNames":false,"additionalProperties":true}"#);
    let mut empty = StructuredState::new(&plan);
    assert!(feed(&mut empty, b"{}"));
    assert!(empty.is_accepting());
    let mut populated = StructuredState::new(&plan);
    assert!(!feed(&mut populated, br#"{"a":1}"#));
}

#[test]
fn property_name_cursor_is_released_when_installation_fails() {
    let plan =
        build(r#"{"type":"object","propertyNames":{"type":"string"},"additionalProperties":true}"#);
    let mut state = StructuredState::new(&plan);
    state.limits.max_undo_entries = 0;
    let before = state.cursors.retained_bytes();
    assert!(state.start_key(0).is_err());
    assert!(state
        .cursors
        .slots
        .iter()
        .all(|slot| matches!(slot, CursorSlot::Vacant)));
    assert!(state.cursors.retained_bytes() >= before);
}

#[test]
fn contains_cursor_is_released_when_item_installation_fails() {
    let plan = build(r#"{"type":"array","items":true,"contains":{"type":"string"}}"#);
    let mut state = StructuredState::new(&plan);
    state.limits.max_undo_entries = 0;
    let before = state.cursors.retained_bytes();
    assert!(state.start_item_tracking(0).is_err());
    assert!(state.array_ref(0).item.is_none());
    assert!(state
        .cursors
        .slots
        .iter()
        .all(|slot| matches!(slot, CursorSlot::Vacant)));
    assert!(state.cursors.retained_bytes() >= before);
}

#[test]
fn child_frame_preparation_failure_is_transactional() {
    let plan = build(
        r#"{"type":"array","items":{"type":"object","properties":{"x":{"type":"string"}},"additionalProperties":true}}"#,
    );
    let mut state = StructuredState::new(&plan);
    let before = full_snapshot(&state);
    let live = state.session_memory.live();
    state.limits.max_undo_entries = 0;
    let child = match plan.node(plan.root) {
        NodePlan::Array(array) => match array.tail {
            super::super::plan::TailPlan::Schema(id) => id,
            _ => panic!("schema item expected"),
        },
        _ => panic!("array root expected"),
    };
    assert!(state.push_child(child).is_err());
    assert_eq!(full_snapshot(&state), before);
    assert_eq!(state.session_memory.live(), live);
}

#[test]
fn commit_validation_rejects_owned_cursor_inside_pushed_frame_without_mutation() {
    let plan = build(r#"{"type":"array","items":true}"#);
    let mut state = StructuredState::new(&plan);
    let before = full_snapshot(&state);
    let invalid = CursorId {
        index: 7,
        generation: 1,
    };
    let scope = state.current_scope().unwrap().clone();
    state.undo.push(Undo::PushFrame(
        Frame::Obligation(ObligationFrame {
            cursors: vec![invalid],
            marks: Vec::new(),
            allocation_charge: 0,
        }),
        scope,
    ));
    let expected = full_snapshot(&state);
    assert!(state.commit_checkpoint().is_err());
    assert_eq!(full_snapshot(&state), expected);
    assert_eq!(state.undo.len(), before.undo_len + 1);
}

#[test]
fn regular_obligation_commit_does_not_require_structured_free_capacity() {
    let plan = build(r#"{"type":"boolean"}"#);
    let mut state = StructuredState::new(&plan);
    let cursor = state.cursors.start(plan.root).unwrap();
    assert_eq!(state.cursors.free_structured.capacity(), 0);
    let scope = state.current_scope().unwrap().clone();
    state.undo.push(Undo::PushFrame(
        Frame::Obligation(ObligationFrame {
            cursors: vec![cursor],
            marks: Vec::new(),
            allocation_charge: 0,
        }),
        scope,
    ));
    state.commit_checkpoint().unwrap();
    assert!(matches!(
        state.cursors.slots[cursor.index as usize],
        CursorSlot::Vacant
    ));
}

#[test]
fn mixed_obligation_commit_counts_regular_and_structured_slots_separately() {
    let plan = build(r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#);
    let mut state = StructuredState::new(&plan);
    let regular_plan = build(r#"{"type":"boolean"}"#);
    let regular = state.cursors.start(regular_plan.root).unwrap();
    let structured = state.cursors.start(plan.root).unwrap();
    let slots_before = state.cursors.free_slots.len();
    let structured_before = state.cursors.free_structured.len();
    let scope = state.current_scope().unwrap().clone();
    state.undo.push(Undo::PushFrame(
        Frame::Obligation(ObligationFrame {
            cursors: vec![regular, structured],
            marks: Vec::new(),
            allocation_charge: 0,
        }),
        scope,
    ));
    state.commit_checkpoint().unwrap();
    assert_eq!(state.cursors.free_slots.len(), slots_before + 2);
    assert_eq!(state.cursors.free_structured.len(), structured_before + 1);
}

#[test]
fn any_object_commit_rejects_stale_id_before_mutation() {
    let plan = build(r#"{"type":"array","items":true}"#);
    let mut state = StructuredState::new(&plan);
    let scope = state.current_scope().unwrap().clone();
    state.undo.push(Undo::PushFrame(
        Frame::Any(AnyFrame::Object(AnyObjectId(u32::MAX))),
        scope,
    ));
    let before = full_snapshot(&state);
    assert!(state.commit_checkpoint().is_err());
    assert_eq!(full_snapshot(&state), before);
}

#[test]
fn any_object_commit_returns_an_arena_slot_without_growing_the_free_list() {
    let plan = build(r#"{"type":"array","items":true}"#);
    let mut state = StructuredState::new(&plan);
    let id = state.alloc_any_object().unwrap();
    let capacity = state.any_object_free.capacity();
    let scope = state.current_scope().unwrap().clone();
    state
        .undo
        .push(Undo::PushFrame(Frame::Any(AnyFrame::Object(id)), scope));
    state.commit_checkpoint().unwrap();
    assert_eq!(state.any_object_free.capacity(), capacity);
    assert!(state.any_object_free.contains(&id.0));
}

#[test]
fn sequential_nested_objects_reuse_session_capacity() {
    let plan = build(
        r#"{"type":"array","items":{"type":"object","properties":{"x":{"type":"integer"}},"additionalProperties":false}}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'[').unwrap());
    for index in 0..2000 {
        if index != 0 {
            assert!(state.try_push_byte(b',').unwrap());
        }
        for byte in br#"{"x":1}"# {
            assert!(state.try_push_byte(*byte).unwrap());
        }
        assert!(state.retained_bytes() <= state.limits.max_session_bytes);
    }
    assert!(state.try_push_byte(b']').unwrap());
    assert!(state.is_accepting());
}

#[test]
fn rollback_releases_a_successfully_installed_contains_cursor() {
    let plan = build(r#"{"type":"array","items":true,"contains":{"type":"string"}}"#);
    let mut state = StructuredState::new(&plan);
    let mark = state.checkpoint();
    state.start_item_tracking(0).unwrap();
    assert!(state
        .cursors
        .slots
        .iter()
        .any(|slot| !matches!(slot, CursorSlot::Vacant)));
    state.rollback(mark);
    assert!(state.array_ref(0).item.is_none());
    assert!(state
        .cursors
        .slots
        .iter()
        .all(|slot| matches!(slot, CursorSlot::Vacant)));
}

#[test]
fn rollback_releases_a_successfully_installed_obligation_frame() {
    let plan = build(
        r#"{"type":"object","patternProperties":{"^a.$":{"type":"integer","minimum":0,"maximum":99},"^.b$":{"type":"integer","minimum":0,"maximum":9}},"additionalProperties":false}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"ab":"#));
    let mark = state.checkpoint();
    assert!(state.try_push_byte(b'1').unwrap());
    assert!(matches!(state.frames[1], Frame::Obligation(_)));
    state.rollback(mark);
    assert_eq!(state.frames.len(), 1);
    assert!(state
        .cursors
        .slots
        .iter()
        .all(|slot| matches!(slot, CursorSlot::Vacant)));
}

#[test]
#[ignore = "release-only latency profile"]
fn release_latency_profile() {
    use std::hint::black_box;
    use std::time::Instant;

    struct Case {
        name: &'static str,
        schema: &'static str,
        prefix: &'static [u8],
        accepted: u8,
        rejected: u8,
    }

    let cases = [
        Case {
            name: "regular_scalar",
            schema: r#"{"type":"boolean"}"#,
            prefix: b"tru",
            accepted: b'e',
            rejected: b'x',
        },
        Case {
            name: "decoded_pattern",
            schema: r#"{"type":"string","pattern":"^a+$","minLength":1,"maxLength":8}"#,
            prefix: b"\"a",
            accepted: b'a',
            rejected: 0,
        },
        Case {
            name: "property_names",
            schema: r#"{"type":"object","propertyNames":{"type":"string","pattern":"^a+$"},"additionalProperties":true}"#,
            prefix: b"{\"a",
            accepted: b'"',
            rejected: 0,
        },
        Case {
            name: "one_obligation",
            schema: r#"{"type":"object","properties":{"x":{"type":"boolean"}},"additionalProperties":false}"#,
            prefix: b"{\"x\":tru",
            accepted: b'e',
            rejected: b'x',
        },
        Case {
            name: "three_obligations",
            schema: r#"{"type":"object","patternProperties":{"^x$":{"type":"boolean"},"^.$":{"type":"boolean"},"x":{"type":"boolean"}},"additionalProperties":false}"#,
            prefix: b"{\"x\":tru",
            accepted: b'e',
            rejected: b'x',
        },
        Case {
            name: "scalar_contains",
            schema: r#"{"type":"array","items":true,"contains":{"const":1}}"#,
            prefix: b"[",
            accepted: b'1',
            rejected: b'}',
        },
        Case {
            name: "structured_contains",
            schema: r#"{"type":"array","items":true,"contains":{"type":"object","required":["a"],"properties":{"a":{"type":"boolean"}},"additionalProperties":false}}"#,
            prefix: b"[{\"a\":tru",
            accepted: b'e',
            rejected: b'x',
        },
        Case {
            name: "unique_and_contains",
            schema: r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true,"contains":{"const":1}}"#,
            prefix: b"[1,",
            accepted: b'2',
            rejected: b']',
        },
    ];

    fn nanos(start: Instant) -> u64 {
        u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn percentiles(samples: &mut [u64]) -> (u64, u64) {
        samples.sort_unstable();
        let last = samples.len() - 1;
        (samples[last / 2], samples[last * 95 / 100])
    }

    eprintln!(
        "sizes frame={} undo={} item_tracking={} canonical_builder={} array_frame={} obligation={}",
        std::mem::size_of::<Frame>(),
        std::mem::size_of::<Undo>(),
        std::mem::size_of::<ItemTracking>(),
        std::mem::size_of::<CanonicalBuilder>(),
        std::mem::size_of::<ArrayFrame>(),
        std::mem::size_of::<ObligationFrame>()
    );
    for case in cases {
        let mut compile = Vec::with_capacity(40);
        for _ in 0..40 {
            let start = Instant::now();
            black_box(build(case.schema));
            compile.push(nanos(start));
        }
        let plan = build(case.schema);
        let mut construction = Vec::with_capacity(500);
        let mut first = Vec::with_capacity(500);
        for _ in 0..500 {
            let start = Instant::now();
            let mut state = StructuredState::new(&plan);
            construction.push(nanos(start));
            let start = Instant::now();
            black_box(state.try_push_byte(case.prefix[0]).unwrap());
            first.push(nanos(start));
        }
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, case.prefix));
        let warm = state.checkpoint();
        assert!(state.try_push_byte(case.accepted).unwrap());
        state.rollback(warm);
        let mut accepted = Vec::with_capacity(2_000);
        let mut rejected = Vec::with_capacity(2_000);
        for _ in 0..2_000 {
            let mark = state.checkpoint();
            let start = Instant::now();
            black_box(state.try_push_byte(case.accepted).unwrap());
            accepted.push(nanos(start));
            state.rollback(mark);

            let mark = state.checkpoint();
            let start = Instant::now();
            black_box(state.try_push_byte(case.rejected).unwrap());
            rejected.push(nanos(start));
            state.rollback(mark);
        }
        let compile = percentiles(&mut compile);
        let construction = percentiles(&mut construction);
        let first = percentiles(&mut first);
        let accepted = percentiles(&mut accepted);
        let rejected = percentiles(&mut rejected);
        eprintln!(
            "{} compile_ns={:?} construct_ns={:?} first_ns={:?} accepted_ns={:?} rejected_ns={:?} retained={}",
            case.name,
            compile,
            construction,
            first,
            accepted,
            rejected,
            state.session_memory.live()
        );
    }
}

#[test]
#[ignore = "release-only negation latency profile"]
fn negation_latency_profile() {
    use serde_json::json;
    use std::hint::black_box;
    use std::time::Instant;

    fn percentiles(samples: &mut [u64]) -> (u64, u64) {
        samples.sort_unstable();
        let last = samples.len() - 1;
        (samples[last / 2], samples[last * 95 / 100])
    }

    fn timed(samples: usize, mut operation: impl FnMut()) -> (u64, u64) {
        let mut values = Vec::with_capacity(samples);
        for _ in 0..samples {
            let started = Instant::now();
            operation();
            values.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        }
        percentiles(&mut values)
    }

    let cases: &[(&str, &str, &[u8], u8)] = &[
        ("simple_not", r#"{"not":{"const":1}}"#, b"2", b'2'),
        ("nested_not", r#"{"not":{"not":{"const":1}}}"#, b"1", b'2'),
        (
            "not_object",
            r#"{"not":{"type":"object","additionalProperties":{"type":"integer"}}}"#,
            br#"{"x":1"#,
            b'2',
        ),
        (
            "conditional_object",
            r#"{"if":{"type":"object","required":["kind"],"properties":{"kind":{"const":"a"}},"additionalProperties":true},"then":{"required":["a"]},"else":{"required":["b"]}}"#,
            br#"{"kind":"a","a":1"#,
            b'2',
        ),
    ];
    let mut rows = Vec::new();
    for &(name, schema, prefix, byte) in cases {
        let parse = timed(100, || {
            black_box(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
        });
        let ir =
            Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
        let plan_build = timed(100, || {
            black_box(StructuredPlan::compile(ir.clone(), StructuredLimits::default()).unwrap());
        });
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        let construction = timed(1_000, || {
            black_box(StructuredState::new(&plan));
        });
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, prefix));
        let mark = state.checkpoint();
        let transition = timed(2_000, || {
            black_box(state.try_push_byte(byte).unwrap());
            state.rollback(mark);
        });
        rows.push(json!({
            "case": name,
            "schema_parse_ns": {"p50": parse.0, "p95": parse.1, "n": 100},
            "plan_build_ns": {"p50": plan_build.0, "p95": plan_build.1, "n": 100},
            "matcher_creation_ns": {"p50": construction.0, "p95": construction.1, "n": 1000},
            "incremental_transition_ns": {"p50": transition.0, "p95": transition.1, "n": 2000},
            "active_validator_peak_observed": state._validator_ledger.live.load(Ordering::Relaxed),
            "retained_bytes_observed": state.retained_bytes(),
            "session_bytes_observed": state.session_memory.live(),
        }));
    }

    let schema = r#"{"not":{"const":1}}"#;
    let ir = Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
    let plan = StructuredPlan::compile(ir.clone(), StructuredLimits::default()).unwrap();
    let mut prefix = vec![b'2'; 512];
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, &prefix));
    let mark = state.checkpoint();
    let long_incremental = timed(2_000, || {
        black_box(state.try_push_byte(b'2').unwrap());
        state.rollback(mark);
    });
    let long_reference = timed(2_000, || {
        prefix.push(b'2');
        black_box(crate::structured::reference::can_continue(&ir, &prefix));
        prefix.pop();
    });

    let report = json!({
        "profile": "release",
        "rows": rows,
        "prefix_scaling": {
            "prefix_bytes": 512,
            "incremental_transition_ns": {"p50": long_incremental.0, "p95": long_incremental.1, "n": 2000},
            "reference_can_continue_ns": {"p50": long_reference.0, "p95": long_reference.1, "n": 2000}
        }
    });
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/logs");
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("program3_negation_rust_profile.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{}", path.display());
}

#[test]
fn structured_state_is_send_for_gil_released_worker_threads() {
    fn assert_send<T: Send>() {}
    assert_send::<StructuredState<'static>>();
}

#[test]
fn an_unsupported_root_is_dead_and_never_accepting() {
    let plan = StructuredPlan::unsupported_for_test();
    let state = StructuredState::new(&plan);
    assert!(state.is_dead());
    assert!(!state.is_accepting());
}

#[test]
fn accepts_a_valid_open_object_instance_byte_by_byte() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"string"}},"required":["a"],"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"a":true,"b":"x"}"#));
    assert!(state.is_accepting());
}

#[test]
fn unevaluated_properties_runs_incrementally_for_known_and_unknown_members() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"unevaluatedProperties":false}"#,
    );
    assert!(matches!(plan.node(plan.root), NodePlan::Unevaluated(_)));
    let mut accepted = StructuredState::new(&plan);
    for (index, &byte) in br#"{"a":true}"#.iter().enumerate() {
        assert_eq!(
            accepted.try_push_byte(byte),
            Ok(true),
            "byte {index}: {byte:?}"
        );
    }
    assert!(accepted.is_accepting());

    let mut rejected = StructuredState::new(&plan);
    assert!(!feed(&mut rejected, br#"{"b":true}"#));
    assert!(!rejected.is_accepting());
}

#[test]
fn absent_and_explicit_additional_properties_annotate_differently() {
    let absent = build(r#"{"type":"object","unevaluatedProperties":false}"#);
    let explicit =
        build(r#"{"type":"object","additionalProperties":true,"unevaluatedProperties":false}"#);
    let mut absent_state = StructuredState::new(&absent);
    assert!(!feed(&mut absent_state, br#"{"x":1}"#));
    let mut explicit_state = StructuredState::new(&explicit);
    assert!(feed(&mut explicit_state, br#"{"x":1}"#));
    assert!(explicit_state.is_accepting());
}

#[test]
fn unevaluated_items_distinguishes_prefix_and_absent_tail_items() {
    let plan = build(r#"{"prefixItems":[{"type":"boolean"}],"unevaluatedItems":false}"#);
    let mut accepted = StructuredState::new(&plan);
    for (index, &byte) in br#"[true]"#.iter().enumerate() {
        assert_eq!(
            accepted.try_push_byte(byte),
            Ok(true),
            "byte {index}: {byte:?}"
        );
    }
    assert!(accepted.is_accepting());
    let mut rejected = StructuredState::new(&plan);
    assert!(!feed(&mut rejected, br#"[true,1]"#));
}

#[test]
fn contains_annotations_apply_even_when_min_contains_is_zero() {
    let plan = build(r#"{"contains":{"type":"integer"},"minContains":0,"unevaluatedItems":false}"#);
    let mut accepted = StructuredState::new(&plan);
    assert!(feed(&mut accepted, br#"[1,2]"#));
    assert!(accepted.is_accepting());
    let mut rejected = StructuredState::new(&plan);
    assert!(!feed(&mut rejected, br#"[1,true]"#));
}

#[test]
fn adjacent_contains_annotation_marks_the_matching_second_item() {
    let plan =
        build(r#"{"prefixItems":[true],"contains":{"type":"string"},"unevaluatedItems":false}"#);
    let mut state = StructuredState::new(&plan);
    for (index, &byte) in br#"[1,"foo"]"#.iter().enumerate() {
        assert_eq!(
            state.try_push_byte(byte),
            Ok(true),
            "byte {index}: {byte:?}"
        );
    }
    assert!(state.is_accepting());
}

#[test]
fn adjacent_contains_scope_accepts_matching_second_item() {
    let plan = build(r#"{"prefixItems":[true],"contains":{"type":"string"}}"#);
    let mut state = StructuredState::new(&plan);
    for (index, &byte) in br#"[1,"foo"]"#.iter().enumerate() {
        assert_eq!(
            state.try_push_byte(byte),
            Ok(true),
            "byte {index}: {byte:?}"
        );
    }
    assert!(state.is_accepting());
}

#[test]
fn unevaluated_candidate_heap_and_annotations_roll_back_exactly() {
    let plan = build(r#"{"unevaluatedProperties":true}"#);
    let mut state = StructuredState::new(&plan);
    assert_eq!(state.try_push_byte(b'{'), Ok(true));
    for index in 0..65 {
        let member = format!(r#"{}"p{index}":{index}"#, if index == 0 { "" } else { "," });
        assert!(feed(&mut state, member.as_bytes()), "member {index}");
    }
    state.commit_checkpoint().unwrap();
    assert_exact_session_ledger(&state);
    let warm = state.checkpoint();
    assert!(feed(&mut state, br#","fresh":1,"#));
    assert_eq!(state.try_push_byte(b']'), Ok(false));
    state.rollback(warm);
    let before = full_snapshot(&state);
    let retained = state.retained_bytes();
    let mark = state.checkpoint();
    assert!(feed(&mut state, br#","fresh":1,"#));
    assert_eq!(state.try_push_byte(b']'), Ok(false));
    state.rollback(mark);
    assert_eq!(full_snapshot(&state), before);
    assert_eq!(state.retained_bytes(), retained);
    assert_exact_session_ledger(&state);
}

#[test]
fn unevaluated_rejected_speculative_token_restores_root_annotations() {
    let plan = build(r#"{"properties":{"a":true},"unevaluatedProperties":false}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"a":1"#));
    let before = full_snapshot(&state);
    let mark = state.checkpoint();
    assert_eq!(state.try_push_byte(b'}'), Ok(true));
    assert!(state.is_accepting());
    assert_eq!(state.try_push_byte(b'x'), Ok(false));
    state.rollback(mark);
    assert_eq!(full_snapshot(&state), before);
    assert_exact_session_ledger(&state);
}

#[test]
fn unevaluated_annotation_semantics_cover_producers_and_composition() {
    let cases = [
        (
            "overlapping properties and patternProperties",
            r#"{"properties":{"a":true},"patternProperties":{"^a$":true},"unevaluatedProperties":false}"#,
            br#"{"a":1}"#.as_slice(),
            true,
        ),
        (
            "schema additionalProperties annotates successes",
            r#"{"additionalProperties":{"type":"integer"},"unevaluatedProperties":false}"#,
            br#"{"x":1}"#.as_slice(),
            true,
        ),
        (
            "schema additionalProperties still asserts",
            r#"{"additionalProperties":{"type":"integer"},"unevaluatedProperties":false}"#,
            br#"{"x":true}"#.as_slice(),
            false,
        ),
        (
            "propertyNames does not annotate",
            r#"{"propertyNames":true,"unevaluatedProperties":false}"#,
            br#"{"x":1}"#.as_slice(),
            false,
        ),
        (
            "dependentRequired does not annotate",
            r#"{"dependentRequired":{"a":["b"]},"unevaluatedProperties":false}"#,
            br#"{"a":1,"b":2}"#.as_slice(),
            false,
        ),
        (
            "triggered dependentSchemas propagates",
            r#"{"properties":{"a":true},"dependentSchemas":{"a":{"properties":{"b":true}}},"unevaluatedProperties":false}"#,
            br#"{"a":1,"b":2}"#.as_slice(),
            true,
        ),
        (
            "untriggered dependentSchemas contributes nothing",
            r#"{"dependentSchemas":{"a":{"properties":{"b":true}}},"unevaluatedProperties":false}"#,
            br#"{"b":2}"#.as_slice(),
            false,
        ),
        (
            "allOf unions successful annotations",
            r#"{"allOf":[{"properties":{"a":true}},{"properties":{"b":true}}],"unevaluatedProperties":false}"#,
            br#"{"a":1,"b":2}"#.as_slice(),
            true,
        ),
        (
            "anyOf unions every successful branch",
            r#"{"anyOf":[{"properties":{"a":true}},{"properties":{"b":true}}],"unevaluatedProperties":false}"#,
            br#"{"a":1,"b":2}"#.as_slice(),
            true,
        ),
        (
            "failed anyOf branch contributes nothing",
            r#"{"anyOf":[{"properties":{"a":true}},false],"unevaluatedProperties":false}"#,
            br#"{"a":1}"#.as_slice(),
            true,
        ),
        (
            "oneOf contributes its unique success",
            r#"{"oneOf":[{"required":["a"],"properties":{"a":true}},{"required":["b"],"properties":{"b":true}}],"unevaluatedProperties":false}"#,
            br#"{"a":1}"#.as_slice(),
            true,
        ),
        (
            "not contributes no annotations",
            r#"{"not":false,"unevaluatedProperties":false}"#,
            br#"{"x":1}"#.as_slice(),
            false,
        ),
        (
            "conditional true path",
            r#"{"properties":{"flag":true},"if":{"required":["flag"]},"then":{"properties":{"a":true}},"else":{"properties":{"b":true}},"unevaluatedProperties":false}"#,
            br#"{"flag":1,"a":2}"#.as_slice(),
            true,
        ),
        (
            "conditional false path discards failed if",
            r#"{"if":{"required":["flag"]},"then":{"properties":{"a":true}},"else":{"properties":{"b":true}},"unevaluatedProperties":false}"#,
            br#"{"b":2}"#.as_slice(),
            true,
        ),
        (
            "if-only successful annotations",
            r#"{"if":{"required":["a"],"properties":{"a":true}},"unevaluatedProperties":false}"#,
            br#"{"a":1}"#.as_slice(),
            true,
        ),
        (
            "$ref propagates annotations",
            r##"{"$defs":{"shape":{"properties":{"a":true}}},"$ref":"#/$defs/shape","unevaluatedProperties":false}"##,
            br#"{"a":1}"#.as_slice(),
            true,
        ),
        (
            "explicit items annotates the tail",
            r#"{"prefixItems":[true],"items":true,"unevaluatedItems":false}"#,
            br#"[1,2,3]"#.as_slice(),
            true,
        ),
        (
            "multiple contains matches annotate every match",
            r#"{"contains":{"type":"integer"},"unevaluatedItems":false}"#,
            br#"[1,2,3]"#.as_slice(),
            true,
        ),
        (
            "boolean true unevaluated schema",
            r#"{"unevaluatedProperties":true}"#,
            br#"{"x":1}"#.as_slice(),
            true,
        ),
        (
            "boolean false unevaluated schema",
            r#"{"unevaluatedItems":false}"#,
            br#"[1]"#.as_slice(),
            false,
        ),
        (
            "escaped key keeps encounter ordinal",
            r#"{"properties":{"a":true},"unevaluatedProperties":false}"#,
            br#"{"\u0061":1}"#.as_slice(),
            true,
        ),
        (
            "deep candidate container streams without buffering",
            r#"{"unevaluatedProperties":{"type":"array"}}"#,
            br#"{"x":[{"a":[1,{"b":2}]}]}"#.as_slice(),
            true,
        ),
        (
            "empty object",
            r#"{"unevaluatedProperties":false}"#,
            br#"{}"#.as_slice(),
            true,
        ),
        (
            "empty array",
            r#"{"unevaluatedItems":false}"#,
            br#"[]"#.as_slice(),
            true,
        ),
    ];
    for (name, schema, document, expected) in cases {
        let plan = build(schema);
        assert!(
            matches!(plan.node(plan.root), NodePlan::Unevaluated(_)),
            "{name}"
        );
        let mut state = StructuredState::new(&plan);
        let actual = feed(&mut state, document) && state.is_accepting();
        assert_eq!(
            actual,
            expected,
            "{name}: {}",
            String::from_utf8_lossy(document)
        );
    }
}

#[test]
fn nested_unevaluated_scopes_and_productive_refs_keep_locations_separate() {
    let cases = [
        (
            r##"{"$defs":{"node":{"type":"object","properties":{"child":{"$ref":"#/$defs/node"}},"unevaluatedProperties":false}},"$ref":"#/$defs/node"}"##,
            br#"{"child":{"child":{}}}"#.as_slice(),
            true,
        ),
        (
            r#"{"properties":{"inner":{"properties":{"a":true},"unevaluatedProperties":false}},"unevaluatedProperties":false}"#,
            br#"{"inner":{"a":1}}"#.as_slice(),
            true,
        ),
        (
            r#"{"prefixItems":[{"prefixItems":[true],"unevaluatedItems":false}],"unevaluatedItems":false}"#,
            br#"[[1]]"#.as_slice(),
            true,
        ),
        (
            r#"{"properties":{"inner":{"properties":{"a":true},"unevaluatedProperties":false}},"unevaluatedProperties":false}"#,
            br#"{"inner":{"a":1,"leak":2}}"#.as_slice(),
            false,
        ),
        (
            r#"{"prefixItems":[{"prefixItems":[true],"unevaluatedItems":false}],"unevaluatedItems":false}"#,
            br#"[[1,2]]"#.as_slice(),
            false,
        ),
    ];
    for (schema, document, expected) in cases {
        let plan = build(schema);
        let mut state = StructuredState::new(&plan);
        assert_eq!(
            feed(&mut state, document) && state.is_accepting(),
            expected,
            "{schema}"
        );
    }
}

#[test]
fn unevaluated_candidate_limit_errors_and_duplicate_keys_roll_back_exactly() {
    let plan = build(r#"{"additionalProperties":true,"unevaluatedProperties":false}"#);

    let mut memory = StructuredState::new(&plan);
    assert!(feed(&mut memory, br#"{"x":"#));
    memory.commit_checkpoint().unwrap();
    let before = full_snapshot(&memory);
    let live = memory.session_memory.live();
    memory.limits.max_session_bytes = memory.retained_bytes();
    let error = memory.try_push_byte(b'[').unwrap_err();
    assert_eq!(error.kind, LimitKind::SessionBytes);
    assert_eq!(full_snapshot(&memory), before);
    assert_eq!(memory.session_memory.live(), live);
    assert_exact_session_ledger(&memory);

    let validator_plan = build(
        r#"{"additionalProperties":true,"unevaluatedProperties":{"anyOf":[{"type":"null"},{"type":"boolean"},{"type":"number"},{"type":"string"},{"type":"array"},{"type":"object"},true,false]}}"#,
    );
    let baseline_ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: validator_plan.limits.max_active_validators,
    });
    let baseline = StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&validator_plan),
        validator_plan.root,
        0,
        baseline_ledger.clone(),
    )
    .unwrap();
    let initial_validators = baseline_ledger.live.load(Ordering::Relaxed);
    drop(baseline);
    assert_eq!(baseline_ledger.live.load(Ordering::Relaxed), 0);
    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: initial_validators,
    });
    let mut validators = StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&validator_plan),
        validator_plan.root,
        0,
        ledger.clone(),
    )
    .unwrap();
    assert_eq!(ledger.live.load(Ordering::Relaxed), initial_validators);
    assert!(feed(&mut validators, br#"{"x":"#));
    validators.commit_checkpoint().unwrap();
    let before = full_snapshot(&validators);
    let live_validators = ledger.live.load(Ordering::Relaxed);
    let error = validators.try_push_byte(b'1').unwrap_err();
    assert_eq!(error.kind, LimitKind::ActiveValidators);
    assert_eq!(full_snapshot(&validators), before);
    assert_eq!(ledger.live.load(Ordering::Relaxed), live_validators);
    drop(validators);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);

    let mut duplicate = StructuredState::new(&plan);
    assert!(feed(&mut duplicate, br#"{"a":1,"#));
    duplicate.commit_checkpoint().unwrap();
    let before = full_snapshot(&duplicate);
    let live = duplicate.session_memory.live();
    let mark = duplicate.checkpoint();
    assert!(!feed(&mut duplicate, br#""a":2"#));
    duplicate.rollback(mark);
    assert_eq!(full_snapshot(&duplicate), before);
    assert_eq!(duplicate.session_memory.live(), live);
    assert_exact_session_ledger(&duplicate);
}

/// Tests the candidate lifecycle's independent location-count limit.
/// Fabricated ordinals verify the reported `LimitKind` for this defensive path.
#[test]
fn unevaluated_property_count_limit_uses_property_count_kind_and_rolls_back_exactly() {
    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r#"{"unevaluatedProperties":{"type":"integer"}}"#,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    let limits = StructuredLimits {
        max_properties: 5,
        ..StructuredLimits::default()
    };
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: plan.limits.max_active_validators,
    });
    let mut state = StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&plan),
        plan.root,
        0,
        ledger.clone(),
    )
    .unwrap();
    assert!(state.try_push_byte(b'{').unwrap());
    state.start_unevaluated_candidate(0, 5).unwrap();
    state.commit_checkpoint().unwrap();
    let before = full_snapshot(&state);
    let live = state.session_memory.live();
    let live_validators = ledger.live.load(Ordering::Relaxed);
    let mark = state.checkpoint();
    let error = state.finish_unevaluated_candidate(0, true).unwrap_err();
    assert_eq!(error.kind, LimitKind::PropertyCount);
    assert_eq!(error.observed, 6);
    assert_eq!(error.limit, 5);
    state.rollback(mark);
    assert_eq!(full_snapshot(&state), before);
    assert_eq!(state.session_memory.live(), live);
    assert_eq!(ledger.live.load(Ordering::Relaxed), live_validators);
    assert_exact_session_ledger(&state);

    // The matcher is not bricked by the rejected speculative token: a fresh session sharing
    // the same ledger still accepts an in-budget ordinal (new_count=5, exactly at the cap).
    let mut fresh = StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&plan),
        plan.root,
        0,
        ledger.clone(),
    )
    .unwrap();
    assert!(fresh.try_push_byte(b'{').unwrap());
    fresh.start_unevaluated_candidate(0, 4).unwrap();
    fresh.finish_unevaluated_candidate(0, true).unwrap();
    assert_exact_session_ledger(&fresh);
}

#[test]
fn unevaluated_item_count_limit_uses_array_length_kind_and_rolls_back_exactly() {
    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r#"{"unevaluatedItems":{"type":"integer"}}"#,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    let limits = StructuredLimits {
        max_items: 5,
        ..StructuredLimits::default()
    };
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: plan.limits.max_active_validators,
    });
    let mut state = StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&plan),
        plan.root,
        0,
        ledger.clone(),
    )
    .unwrap();
    assert!(state.try_push_byte(b'[').unwrap());
    state.start_unevaluated_candidate(0, 5).unwrap();
    state.commit_checkpoint().unwrap();
    let before = full_snapshot(&state);
    let live = state.session_memory.live();
    let live_validators = ledger.live.load(Ordering::Relaxed);
    let mark = state.checkpoint();
    let error = state.finish_unevaluated_candidate(0, true).unwrap_err();
    assert_eq!(error.kind, LimitKind::ArrayLength);
    assert_eq!(error.observed, 6);
    assert_eq!(error.limit, 5);
    state.rollback(mark);
    assert_eq!(full_snapshot(&state), before);
    assert_eq!(state.session_memory.live(), live);
    assert_eq!(ledger.live.load(Ordering::Relaxed), live_validators);
    assert_exact_session_ledger(&state);

    let mut fresh = StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&plan),
        plan.root,
        0,
        ledger.clone(),
    )
    .unwrap();
    assert!(fresh.try_push_byte(b'[').unwrap());
    fresh.start_unevaluated_candidate(0, 4).unwrap();
    fresh.finish_unevaluated_candidate(0, true).unwrap();
    assert_exact_session_ledger(&fresh);
}

#[test]
fn unevaluated_large_ordinal_and_branch_stress_restores_validator_baseline() {
    fn feed_committed(state: &mut StructuredState<'_>, bytes: &[u8]) -> bool {
        for (index, &byte) in bytes.iter().enumerate() {
            let result = state.try_push_byte(byte);
            if result != Ok(true) {
                panic!("stress byte {index} ({byte:?}) failed: {result:?}");
            }
            if index % 8 == 7 && state.commit_checkpoint().is_err() {
                return false;
            }
        }
        true
    }

    let mut object_document = String::from("{");
    for index in 0..10_000usize {
        if index != 0 {
            object_document.push(',');
        }
        object_document.push_str(&format!(r#""p{index}":{index}"#));
    }
    object_document.push('}');
    let object_plan = build(r#"{"unevaluatedProperties":true}"#);
    let mut object = StructuredState::new(&object_plan);
    assert!(feed_committed(&mut object, object_document.as_bytes()));
    assert!(object.is_accepting());
    object.commit_checkpoint().unwrap();
    assert_exact_session_ledger(&object);

    let array_document = format!(
        "[{}]",
        std::iter::repeat_n("0", 10_000)
            .collect::<Vec<_>>()
            .join(",")
    );
    let array_plan = build(r#"{"unevaluatedItems":true}"#);
    let mut array = StructuredState::new(&array_plan);
    assert!(feed_committed(&mut array, array_document.as_bytes()));
    assert!(array.is_accepting());
    array.commit_checkpoint().unwrap();
    assert_exact_session_ledger(&array);

    let contains_document = format!(
        "[{}]",
        std::iter::repeat_n("1", 4_096)
            .collect::<Vec<_>>()
            .join(",")
    );
    let contains_plan =
        build(r#"{"contains":{"type":"integer"},"minContains":0,"unevaluatedItems":false}"#);
    let mut contains = StructuredState::new(&contains_plan);
    assert!(feed_committed(&mut contains, contains_document.as_bytes()));
    assert!(contains.is_accepting());

    let branches = (0..128usize)
        .map(|index| format!(r#"{{"properties":{{"p{index}":true}}}}"#))
        .collect::<Vec<_>>()
        .join(",");
    let schema = format!(r#"{{"anyOf":[{branches}],"unevaluatedProperties":false}}"#);
    let branch_plan = build(&schema);
    let mut branch_state = StructuredState::new(&branch_plan);
    let branch_document = format!(
        "{{{}}}",
        (0..128usize)
            .map(|index| format!(r#""p{index}":{index}"#))
            .collect::<Vec<_>>()
            .join(",")
    );
    assert!(feed_committed(
        &mut branch_state,
        branch_document.as_bytes()
    ));
    assert!(branch_state.is_accepting());

    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: object_plan.limits.max_active_validators,
    });
    for _ in 0..64 {
        let mut state = StructuredState::new_at_with_context(
            PlanHandle::Borrowed(&object_plan),
            object_plan.root,
            0,
            ledger.clone(),
        )
        .unwrap();
        assert!(feed(&mut state, br#"{"x":{"deep":[1,2,3]}}"#));
        drop(state);
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn rejects_a_document_missing_a_required_property() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"],"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"{"));
    assert!(!feed(&mut state, b"}"));
}

#[test]
fn property_order_does_not_matter() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"boolean"}},"required":["a","b"],"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut a = StructuredState::new(&plan);
    assert!(feed(&mut a, br#"{"a":true,"b":false}"#));
    assert!(a.is_accepting());
    let mut b = StructuredState::new(&plan);
    assert!(feed(&mut b, br#"{"b":false,"a":true}"#));
    assert!(b.is_accepting());
}

#[test]
fn rejects_a_known_property_with_the_wrong_type() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(!feed(&mut state, br#"{"a":1}"#));
}

#[test]
fn a_key_outside_known_and_pattern_uses_the_additional_schema() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":{"type":"integer"}}"#,
    );
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"{"z":1}"#));
    assert!(ok.is_accepting());
    let mut bad = StructuredState::new(&plan);
    assert!(!feed(&mut bad, br#"{"z":"x"}"#));
}

#[test]
fn forbidden_additional_properties_rejects_an_unknown_key() {
    let plan = build(
        r#"{"type":"object","patternProperties":{"^x[0-9]$":{"type":"boolean"}},"additionalProperties":false}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(!feed(&mut state, br#"{"z":true}"#));
}

#[test]
fn a_nested_object_property_is_walked_incrementally() {
    let plan = build(
        r#"{"type":"object","additionalProperties":{"type":"boolean"},"properties":{"a":{"type":"object","additionalProperties":{"type":"boolean"},"properties":{"b":{"type":"boolean"}},"required":["b"]}},"required":["a"]}"#,
    );
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"{"a":{"b":true}}"#));
    assert!(ok.is_accepting());
    let mut missing_inner = StructuredState::new(&plan);
    assert!(!feed(&mut missing_inner, br#"{"a":{}}"#));
}

#[test]
fn a_single_pattern_property_matches_by_regex() {
    let plan = build(
        r#"{"type":"object","patternProperties":{"^x[0-9]$":{"type":"boolean"}},"additionalProperties":false}"#,
    );
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"{"x1":true}"#));
    assert!(ok.is_accepting());
    let mut bad_key = StructuredState::new(&plan);
    assert!(!feed(&mut bad_key, br#"{"y1":true}"#));
}

#[test]
fn reject_then_accept_leaves_frames_byte_for_byte_equal_to_a_clean_walk() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut probe = StructuredState::new(&plan);
    assert!(feed(&mut probe, br#"{"a":tru"#));
    let mark = probe.checkpoint();
    assert_eq!(probe.try_push_byte(b'x'), Ok(false));
    probe.rollback(mark);

    let mut clean = StructuredState::new(&plan);
    assert!(feed(&mut clean, br#"{"a":tru"#));
    assert_eq!(probe.frames, clean.frames);
    assert_eq!(probe.document_bytes, clean.document_bytes);

    assert!(feed(&mut probe, b"e}"));
    assert!(probe.is_accepting());
}

#[test]
fn invalid_first_value_byte_after_colon_restores_phase_and_property_count() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut probe = StructuredState::new(&plan);
    assert!(feed(&mut probe, br#"{"a":"#));
    let before = full_snapshot(&probe);
    assert_eq!(probe.try_push_byte(b'x'), Ok(false));
    assert_eq!(full_snapshot(&probe), before);
}

#[test]
fn a_thousand_speculative_bytes_then_rollback_restores_document_bytes() {
    let plan = build(r#"{"type":"string"}"#);
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'"').unwrap());
    let mark = state.checkpoint();
    let before = state.document_bytes;
    for _ in 0..1000 {
        state.try_push_byte(b'a').unwrap();
    }
    assert_eq!(state.document_bytes, before + 1000);
    state.rollback(mark);
    assert_eq!(state.document_bytes, before);
}

#[test]
fn unknown_key_colon_under_forbidden_additional_leaves_state_unchanged() {
    let plan = build(
        r#"{"type":"object","patternProperties":{"^x[0-9]$":{"type":"boolean"}},"additionalProperties":false}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"#));
    let before = full_snapshot(&state);
    assert_eq!(state.try_push_byte(b'z'), Ok(false));
    assert_eq!(full_snapshot(&state), before);
    assert!(feed(&mut state, br#""x0":true}"#));
    assert!(state.is_accepting());
}

#[test]
fn duplicate_key_raw_and_escaped_spelling_is_rejected() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"a":true,"#));
    assert!(!feed(&mut state, b"\"\\u0061\":false}"));
}

#[test]
fn duplicate_dynamic_key_is_rejected() {
    let plan = build(r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"z":true,"#));
    assert!(!feed(&mut state, br#""z":false}"#));
}

#[test]
fn duplicate_dynamic_key_raw_and_escaped_spelling_is_rejected() {
    let plan = build(r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"z":true,"#));
    assert!(!feed(&mut state, b"\"\\u007a\":false}"));
}

#[test]
fn dynamic_map_growth_over_session_budget_leaves_arena_and_map_consistent() {
    let limits = StructuredLimits {
        max_session_bytes: 4096,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"{"));
    let mut accepted = 0;
    loop {
        let key = format!("\"k{accepted}\":true,");
        let before = full_snapshot(&state);
        let mark = state.checkpoint();
        match feed_checked(&mut state, key.as_bytes()) {
            Ok(true) => accepted += 1,
            Ok(false) => panic!("a plain rejection, not a resource limit, at key {accepted}"),
            Err(_) => {
                state.rollback(mark);
                assert_eq!(full_snapshot(&state), before);
                break;
            }
        }
        assert!(accepted < 10_000, "budget never triggered");
    }
    // Every retained KeyId is still valid, and the last accepted key is still there.
    let Frame::Object(obj) = &state.frames[0] else {
        panic!("expected the root object frame")
    };
    for slot in obj.seen_dynamic.values() {
        assert!(slot.contains(|id| state.key_arena.get(id).is_some()));
    }
    let after_limit = full_snapshot(&state);
    let mark = state.checkpoint();
    match feed_checked(&mut state, br#""ok":true}"#) {
        Ok(true) => assert!(state.is_accepting()),
        Ok(false) => panic!("a plain rejection, not a resource limit, after the budget break"),
        Err(_) => {
            state.rollback(mark);
            assert_eq!(full_snapshot(&state), after_limit);
        }
    }
}

/// Runs every byte via `try_push_byte`, returning the first resource error (if any) instead
/// of collapsing it into `false` the way `feed` does.
fn feed_checked(
    state: &mut StructuredState<'_>,
    bytes: &[u8],
) -> Result<bool, StructuredRuntimeError> {
    for &b in bytes {
        match state.try_push_byte(b) {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

#[test]
fn known_property_intersecting_a_pattern_validates_both_bounds() {
    let plan = build(
        r#"{"type":"object","properties":{"ab":{"type":"integer","minimum":0,"maximum":9}},"patternProperties":{"^a.$":{"type":"integer","minimum":5,"maximum":20}},"additionalProperties":false}"#,
    );
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"{"ab":7}"#));
    assert!(ok.is_accepting());
    let mut too_low_for_pattern = StructuredState::new(&plan);
    assert!(!feed(&mut too_low_for_pattern, br#"{"ab":3}"#));
    let mut too_high_for_known = StructuredState::new(&plan);
    assert!(!feed(&mut too_high_for_known, br#"{"ab":15}"#));
}

#[test]
fn two_overlapping_patterns_validate_the_intersection() {
    let plan = build(
        r#"{"type":"object","patternProperties":{"^a.$":{"type":"integer","minimum":0,"maximum":9},"^.b$":{"type":"integer","minimum":5,"maximum":20}},"additionalProperties":false}"#,
    );
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"{"ab":7}"#));
    assert!(ok.is_accepting());
    let mut fails_first = StructuredState::new(&plan);
    assert!(!feed(&mut fails_first, br#"{"ab":15}"#));
    let mut fails_second = StructuredState::new(&plan);
    assert!(!feed(&mut fails_second, br#"{"ab":2}"#));
}

#[test]
fn one_obligation_still_uses_the_single_target_path_with_no_obligation_frame() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"a":tru"#));
    assert!(matches!(state.frames[1], Frame::Regular { .. }));
}

/// Builds intersecting `patternProperties` constraints for one matching key.
fn n_obligation_schema(n: usize) -> String {
    let mut patterns = String::new();
    for i in 0..n {
        if i > 0 {
            patterns.push(',');
        }
        let mut regex = String::from("^");
        for j in 0..n {
            regex.push(if j == i { 'a' } else { '.' });
        }
        regex.push('$');
        patterns.push_str(&format!(
            r#""{regex}":{{"type":"integer","minimum":0,"maximum":{}}}"#,
            9 - i
        ));
    }
    format!(
        r#"{{"type":"object","patternProperties":{{{patterns}}},"additionalProperties":false}}"#
    )
}

fn n_obligation_key(n: usize) -> String {
    format!("\"{}\"", "a".repeat(n))
}

#[test]
fn one_two_four_five_eight_obligations_intersect_correctly() {
    for n in [1usize, 2, 4, 5, 8] {
        let schema = n_obligation_schema(n);
        let plan = build(&schema);
        let key = n_obligation_key(n);
        let accept_bound = 9 - (n as i32 - 1);
        let doc_ok = format!("{{{key}:{accept_bound}}}");
        let mut ok = StructuredState::new(&plan);
        assert!(feed(&mut ok, doc_ok.as_bytes()), "n={n} doc={doc_ok}");
        assert!(ok.is_accepting(), "n={n}");
        if accept_bound < 9 {
            let doc_bad = format!("{{{key}:9}}");
            let mut bad = StructuredState::new(&plan);
            assert!(!feed(&mut bad, doc_bad.as_bytes()), "n={n} doc={doc_bad}");
        }
    }
}

#[test]
fn validator_zero_rejects_while_later_validators_would_advance() {
    let plan = build(&n_obligation_schema(2));
    let key = n_obligation_key(2);
    // Any value failing one validator must reject the whole obligation.
    let mut state = StructuredState::new(&plan);
    let doc = format!("{{{key}:99}}");
    assert!(!feed(&mut state, doc.as_bytes()));
}

#[test]
fn undo_limit_failure_inside_an_obligation_frame_leaves_every_validator_unchanged() {
    // Two distinct wide-bound integer obligations: many digits stay within bounds, so the
    // undo cap - not a semantic bound - is what eventually fires.
    let schema = r#"{"type":"object","patternProperties":{"^a$":{"type":"integer","minimum":0,"maximum":999999999999999999},"^.$":{"type":"integer","minimum":0,"maximum":999999999999999998}},"additionalProperties":false}"#;
    let limits = StructuredLimits {
        max_undo_entries: 30,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    let prefix_result = feed_checked(&mut state, br#"{"a":1"#);
    let mut last_good = match prefix_result {
        Ok(true) => full_snapshot(&state),
        other => panic!("prefix must fit the undo budget: {other:?}"),
    };
    loop {
        match feed_checked(&mut state, b"1") {
            Ok(true) => last_good = full_snapshot(&state),
            Err(_) => break,
            Ok(false) => panic!("expected a resource error, not a plain rejection"),
        }
    }
    assert_eq!(full_snapshot(&state), last_good);
}

#[test]
fn two_obligations_push_an_obligation_frame_and_rollback_restores_every_validator() {
    let plan = build(
        r#"{"type":"object","patternProperties":{"^a.$":{"type":"integer","minimum":0,"maximum":99},"^.b$":{"type":"integer","minimum":0,"maximum":9}},"additionalProperties":false}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"ab":"#));
    assert!(state.try_push_byte(b'1').unwrap());
    assert!(matches!(state.frames[1], Frame::Obligation(_)));
    let mark = state.checkpoint();
    let before = full_snapshot(&state);
    // '9' advances the wide obligation (0..99) but kills the narrow one (0..9, already at
    // "1" -> "19" exceeds 9): the whole obligation frame must reject and roll back both.
    assert_eq!(state.try_push_byte(b'9'), Ok(false));
    assert_eq!(full_snapshot(&state), before);
    state.rollback(mark);
    assert_eq!(full_snapshot(&state), before);
    assert!(feed(&mut state, b"}"));
    assert!(state.is_accepting());
}

#[test]
fn obligation_cursors_intersect_exact_and_pattern_object_schemas() {
    let plan = build(
        r#"{"type":"object","properties":{"x":{"type":"object","required":["a"],"properties":{"a":{"type":"boolean"}},"additionalProperties":true}},"patternProperties":{"^x$":{"type":"object","required":["b"],"properties":{"b":{"type":"boolean"}},"additionalProperties":true}},"additionalProperties":false}"#,
    );
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"{"x":{"a":true,"b":false}}"#));
    assert!(ok.is_accepting());
    let mut missing = StructuredState::new(&plan);
    assert!(!feed(&mut missing, br#"{"x":{"a":true}}"#));
}

#[test]
fn obligation_cursors_intersect_overlapping_array_schemas() {
    let plan = build(
        r#"{"type":"object","patternProperties":{"^a":{"type":"array","items":true,"minItems":2},"a$":{"type":"array","items":true,"contains":{"const":1},"minContains":1}},"additionalProperties":false}"#,
    );
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"{"a":[0,1]}"#));
    assert!(ok.is_accepting());
    let mut missing = StructuredState::new(&plan);
    assert!(!feed(&mut missing, br#"{"a":[0,2]}"#));
}

#[test]
fn duplicate_node_id_obligations_are_deduplicated() {
    let mut set = SchemaObligationSet::default();
    assert!(set.insert_uncounted(NodeId(3)));
    assert!(!set.insert_uncounted(NodeId(3)));
    assert_eq!(set.len(), 1);
}

#[test]
fn insert_obligation_spills_past_four_and_charges_the_real_delta() {
    let plan = build(r#"{"type":"boolean"}"#);
    let mut state = StructuredState::new(&plan);
    let mut set = SchemaObligationSet::default();
    for i in 0..8u32 {
        state.insert_obligation(&mut set, NodeId(i)).unwrap();
    }
    assert_eq!(set.len(), 8);
    assert_eq!(set.inline_len as usize, MAX_INLINE_OBLIGATIONS);
    assert_eq!(set.spill.len(), 4);
    assert_eq!(
        set.iter().collect::<Vec<_>>(),
        (0..8).map(NodeId).collect::<Vec<_>>()
    );
}

#[test]
fn large_uniform_array_beyond_unroll_threshold_uses_the_structured_array_frame() {
    let plan = build(r#"{"type":"array","items":{"type":"boolean"},"minItems":2,"maxItems":100}"#);
    assert!(matches!(plan.node(plan.root), NodePlan::Array(_)));
    let mut too_short = StructuredState::new(&plan);
    assert!(!feed(&mut too_short, br#"[true]"#));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"[true,false,true]"#));
    assert!(ok.is_accepting());
    let mut wrong_item = StructuredState::new(&plan);
    assert!(!feed(&mut wrong_item, br#"[true,1]"#));
}

#[test]
fn large_array_max_items_rejects_one_past_the_bound() {
    let plan = build(r#"{"type":"array","items":{"type":"boolean"},"maxItems":100}"#);
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'[').unwrap());
    for i in 0..100 {
        if i > 0 {
            assert!(state.try_push_byte(b',').unwrap());
        }
        assert!(feed(&mut state, b"true"), "item {i}");
    }
    // The comma itself is syntactically legal; the 101st ITEM is what the cap rejects.
    let mark = state.checkpoint();
    assert!(state.try_push_byte(b',').unwrap());
    let before = full_snapshot(&state);
    assert_eq!(state.try_push_byte(b't'), Ok(false));
    assert_eq!(full_snapshot(&state), before);
    state.rollback(mark);
    assert!(feed(&mut state, b"]"));
    assert!(state.is_accepting());
}

#[test]
fn tuple_with_a_non_regular_prefix_element_uses_the_structured_array_frame() {
    // An OpenObject prefix element makes the whole tuple non-regular (subtree_is_regular
    // requires every prefix/tail element to be regular too), reaching NodePlan::Array.
    let plan = build(
        r#"{"type":"array","prefixItems":[{"type":"object","additionalProperties":{"type":"boolean"}}],"items":{"type":"integer"}}"#,
    );
    assert!(matches!(plan.node(plan.root), NodePlan::Array(_)));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"[{"a":true},1,2,3]"#));
    assert!(ok.is_accepting());
    let mut wrong_prefix = StructuredState::new(&plan);
    assert!(!feed(&mut wrong_prefix, br#"[1,2,3]"#));
}

#[test]
fn any_json_accepts_every_scalar_kind_for_additional_properties_true() {
    let plan = build(r#"{"type":"object","additionalProperties":true}"#);
    for (label, value) in [
        ("null", "null"),
        ("true", "true"),
        ("false", "false"),
        ("int", "42"),
        ("negative", "-7"),
        ("frac", "3.25"),
        ("exp", "1e10"),
        ("string", "\"hello\""),
    ] {
        let doc = format!(r#"{{"k":{value}}}"#);
        let mut state = StructuredState::new(&plan);
        for &b in doc.as_bytes() {
            let r = state.try_push_byte(b);
            assert_eq!(r, Ok(true), "{label}: {doc} failed at byte {}", b as char);
        }
        assert!(state.is_accepting(), "{label}");
    }
}

#[test]
fn any_json_accepts_nested_arrays_and_objects() {
    let plan = build(r#"{"type":"object","additionalProperties":true}"#);
    let doc = br#"{"k":[1,{"a":[true,null,"x"]},{"b":2}]}"#;
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, doc));
    assert!(state.is_accepting());
}

#[test]
fn items_true_accepts_any_element_and_still_enforces_min_max_items() {
    let plan = build(r#"{"type":"array","items":true,"minItems":1,"maxItems":2}"#);
    let mut too_few = StructuredState::new(&plan);
    assert!(!feed(&mut too_few, b"[]"));

    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"[1,{"a":[true,null]}]"#));
    assert!(ok.is_accepting());

    let mut too_many = StructuredState::new(&plan);
    assert!(!feed(&mut too_many, b"[1,2,3]"));
}

#[test]
fn any_json_rejects_malformed_syntax() {
    let plan = build(r#"{"type":"object","additionalProperties":true}"#);
    for bad in ["tru", "01", "[1,]", "{\"a\":}"] {
        let doc = format!(r#"{{"k":{bad}}}"#);
        let mut state = StructuredState::new(&plan);
        assert!(!feed(&mut state, doc.as_bytes()), "should reject: {doc}");
    }
    // An unterminated string never rejects a byte (every byte is valid string content); it
    // never reaches a complete, accepting state.
    let mut unterminated = StructuredState::new(&plan);
    assert!(feed(&mut unterminated, b"{\"k\":\"unterminated}"));
    assert!(!unterminated.is_accepting());
}

#[test]
fn any_json_object_rejects_duplicate_keys_including_escaped_equivalent() {
    let plan = build(r#"{"type":"object","additionalProperties":true}"#);
    let mut raw_dup = StructuredState::new(&plan);
    assert!(!feed(&mut raw_dup, br#"{"k":{"a":1,"a":2}}"#));
    let mut escaped_dup = StructuredState::new(&plan);
    assert!(!feed(&mut escaped_dup, b"{\"k\":{\"a\":1,\"\\u0061\":2}}"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"{"k":{"a":1,"b":2}}"#));
    assert!(ok.is_accepting());
}

#[test]
fn any_json_object_arena_slot_is_reused_across_sequential_objects_at_one_depth() {
    let plan = build(r#"{"type":"array","items":true}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"["));
    state.commit_checkpoint().unwrap();
    for i in 0..50 {
        let doc = format!("{{\"a\":{i}}}");
        assert!(feed(&mut state, doc.as_bytes()), "item {i}");
        assert!(feed(&mut state, b","), "comma after item {i}");
        state.commit_checkpoint().unwrap();
    }
    assert!(
        state.any_object_arena.len() <= 2,
        "expected slot reuse, not growth per object"
    );
}

#[test]
fn any_json_rollback_restores_state_at_every_structural_boundary() {
    let plan = build(r#"{"type":"object","additionalProperties":true}"#);
    let cases: &[&[u8]] = &[
        br#"{"k":"#,
        br#"{"k":["#,
        br#"{"k":[1,"#,
        br#"{"k":{"#,
        br#"{"k":{"a""#,
        br#"{"k":{"a":"#,
        br#"{"k":{"a":1"#,
        br#"{"k":"\"#,
        br#"{"k":1e"#,
    ];
    for prefix in cases {
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, prefix), "prefix must be valid: {prefix:?}");
        let mark = state.checkpoint();
        let before = full_snapshot(&state);
        for byte in 0u16..=255 {
            let _ = state.try_push_byte(byte as u8);
            state.rollback(mark);
            assert_eq!(
                full_snapshot(&state),
                before,
                "byte {byte:#04x} at {prefix:?}"
            );
        }
    }
}

#[test]
fn any_json_array_undo_limit_failure_leaves_state_byte_for_byte_unchanged() {
    // A small cap forces failure while adding one unconstrained array item.
    let limits = StructuredLimits {
        max_undo_entries: 40,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r#"{"type":"object","additionalProperties":true}"#,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    let mut last_good = match feed_checked(&mut state, br#"{"k":["#) {
        Ok(true) => full_snapshot(&state),
        other => panic!("prefix must fit the undo budget: {other:?}"),
    };
    loop {
        match feed_checked(&mut state, b"1,") {
            Ok(true) => last_good = full_snapshot(&state),
            Err(_) => break,
            Ok(false) => panic!("expected a resource error, not a plain rejection"),
        }
    }
    assert_eq!(full_snapshot(&state), last_good);
    assert!(state.undo.len() <= state.limits.max_undo_entries);
}

#[test]
fn any_json_enforces_depth_items_properties_and_number_bytes_limits() {
    let deep_limits = StructuredLimits {
        max_depth: 4,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r#"{"type":"object","additionalProperties":true}"#,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    let plan = StructuredPlan::compile(ir, deep_limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"k":[[["#));
    let before = full_snapshot(&state);
    assert!(state.try_push_byte(b'[').is_err());
    assert_eq!(full_snapshot(&state), before);

    let item_limits = StructuredLimits {
        max_items: 2,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r#"{"type":"object","additionalProperties":true}"#,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    let plan = StructuredPlan::compile(ir, item_limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"k":[1,2,"#));
    let before = full_snapshot(&state);
    assert!(state.try_push_byte(b'3').is_err());
    assert_eq!(full_snapshot(&state), before);

    let number_limits = StructuredLimits {
        max_number_bytes: 3,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r#"{"type":"object","additionalProperties":true}"#,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    let plan = StructuredPlan::compile(ir, number_limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"k":123"#));
    let before = full_snapshot(&state);
    assert!(state.try_push_byte(b'4').is_err());
    assert_eq!(full_snapshot(&state), before);
}

#[test]
fn structured_array_uniqueitems_with_a_container_item_schema_compiles_and_dedupes() {
    let plan =
        build(r#"{"type":"array","items":{"type":"object"},"maxItems":100,"uniqueItems":true}"#);
    let mut dup = StructuredState::new(&plan);
    assert!(!feed(&mut dup, b"[{\"a\":1},{\"a\":1}]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[{\"a\":1},{\"a\":2}]"));
    assert!(ok.is_accepting());
}

#[test]
fn unique_items_with_items_true_dedupes_arbitrary_json() {
    let plan = build(r#"{"type":"array","items":true,"uniqueItems":true}"#);
    let mut dup = StructuredState::new(&plan);
    assert!(!feed(&mut dup, b"[1,1.0]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[1,\"1\",[1],{\"a\":1},true,null]"));
    assert!(ok.is_accepting());
}

#[test]
fn unique_items_rejects_duplicate_nested_arrays_but_accepts_reordered_elements() {
    let plan =
        build(r#"{"type":"array","items":{"type":"array","items":true},"uniqueItems":true}"#);
    let mut dup = StructuredState::new(&plan);
    assert!(!feed(&mut dup, b"[[1,[2,3]],[1,[2,3]]]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[[1,[2,3]],[[2,3],1]]"));
    assert!(ok.is_accepting());
}

#[test]
fn unique_items_rejects_reordered_object_keys_as_duplicate() {
    let plan = build(r#"{"type":"array","items":{"type":"object"},"uniqueItems":true}"#);
    let mut dup = StructuredState::new(&plan);
    assert!(!feed(&mut dup, b"[{\"a\":1,\"b\":2},{\"b\":2,\"a\":1}]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[{\"a\":1,\"b\":2},{\"a\":1,\"b\":3}]"));
    assert!(ok.is_accepting());
}

#[test]
fn unique_items_with_prefix_items_and_a_schema_tail() {
    let plan = build(
        r#"{"type":"array","prefixItems":[{"type":"integer"}],"items":{"type":"string"},"uniqueItems":true}"#,
    );
    let mut dup = StructuredState::new(&plan);
    assert!(!feed(&mut dup, b"[1,\"x\",\"x\"]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[1,\"x\",\"y\"]"));
    assert!(ok.is_accepting());
}

#[test]
fn unique_items_treats_empty_array_and_empty_object_as_single_distinct_values() {
    let plan = build(r#"{"type":"array","items":true,"uniqueItems":true}"#);
    let mut dup_arr = StructuredState::new(&plan);
    assert!(!feed(&mut dup_arr, b"[[],[]]"));
    let mut dup_obj = StructuredState::new(&plan);
    assert!(!feed(&mut dup_obj, b"[{},{}]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[[],{}]"));
    assert!(ok.is_accepting());
}

#[test]
fn unique_items_rejects_decimal_equal_numbers_and_accepts_distinct_ones() {
    let plan = build(r#"{"type":"array","items":{"type":"number"},"uniqueItems":true}"#);
    let mut dup = StructuredState::new(&plan);
    assert!(!feed(&mut dup, b"[1,1.0]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[1,2,1.5]"));
    assert!(ok.is_accepting());
}

#[test]
fn unique_items_rejects_escaped_and_raw_equivalent_strings() {
    let plan = build(r#"{"type":"array","items":{"type":"string"},"uniqueItems":true}"#);
    let mut dup = StructuredState::new(&plan);
    assert!(!feed(&mut dup, b"[\"A\",\"\\u0041\"]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[\"A\",\"B\"]"));
    assert!(ok.is_accepting());
}

#[test]
fn unique_items_duplicate_rejection_leaves_state_byte_for_byte_unchanged() {
    let plan = build(r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"[1,2,"));
    let before = full_snapshot(&state);
    let mark = state.checkpoint();
    assert_eq!(state.try_push_byte(b'1'), Ok(true));
    assert_eq!(state.try_push_byte(b']'), Ok(false));
    state.rollback(mark);
    assert_eq!(full_snapshot(&state), before);
}

#[test]
fn contains_enforces_min_and_max_matching_items() {
    let plan = build(
        r#"{"type":"array","items":{"type":"integer"},"contains":{"const":0},"minContains":2,"maxContains":2}"#,
    );
    let mut too_few = StructuredState::new(&plan);
    assert!(!feed(&mut too_few, b"[0,1,2]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[0,1,0]"));
    assert!(ok.is_accepting());
    let mut too_many = StructuredState::new(&plan);
    assert!(!feed(&mut too_many, b"[0,0,0]"));
}

#[test]
fn contains_true_counts_every_completed_item() {
    let plan =
        build(r#"{"type":"array","items":{"type":"integer"},"contains":true,"minContains":2}"#);
    let mut too_few = StructuredState::new(&plan);
    assert!(!feed(&mut too_few, b"[1]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[1,2]"));
    assert!(ok.is_accepting());
}

#[test]
fn contains_cursor_handles_every_json_value_with_items_true() {
    let plan = build(
        r#"{"type":"array","items":true,"contains":{"type":"object","required":["a"],"properties":{"a":{"const":1}},"additionalProperties":true},"minContains":1}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"[null,true,1,"x",[1,2],{"a":1}]"#));
    assert!(state.is_accepting());
}

#[test]
fn contains_cursor_validates_nested_arrays_and_objects() {
    let plan = build(
        r#"{"type":"array","items":true,"contains":{"type":"array","items":{"type":"object","required":["ok"],"properties":{"ok":{"const":true}},"additionalProperties":false},"minItems":1},"minContains":1}"#,
    );
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, br#"[0,[{"ok":true}]]"#));
    assert!(ok.is_accepting());
    let mut no_match = StructuredState::new(&plan);
    assert!(!feed(&mut no_match, br#"[0,[{"ok":false}]]"#));
}

#[test]
fn contains_cursor_death_does_not_reject_the_primary_item() {
    let plan =
        build(r#"{"type":"array","items":true,"contains":{"type":"object"},"minContains":0}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"["not an object",17,false]"#));
    assert!(state.is_accepting());
}

#[test]
fn max_contains_zero_rejects_the_first_match_transactionally() {
    let plan = build(
        r#"{"type":"array","items":true,"contains":{"const":1},"minContains":0,"maxContains":0}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"["));
    let warm = state.checkpoint();
    assert!(state.try_push_byte(b'2').unwrap());
    assert!(state.try_push_byte(b']').unwrap());
    state.rollback(warm);
    let before = full_snapshot(&state);
    let mark = state.checkpoint();
    assert!(state.try_push_byte(b'1').unwrap());
    assert_eq!(state.try_push_byte(b']'), Ok(false));
    state.rollback(mark);
    assert_eq!(full_snapshot(&state), before);

    let mut no_match = StructuredState::new(&plan);
    assert!(feed(&mut no_match, b"[2]"));
    assert!(no_match.is_accepting());
}

#[test]
fn unique_items_applies_to_tuple_tail_items_too() {
    let plan = build(
        r#"{"type":"array","prefixItems":[{"type":"boolean"}],"items":{"type":"integer"},"uniqueItems":true}"#,
    );
    let mut dup = StructuredState::new(&plan);
    assert!(!feed(&mut dup, b"[true,1,1]"));
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[true,1,2]"));
    assert!(ok.is_accepting());
}

#[test]
fn contains_and_unique_items_combine_on_the_same_array() {
    let plan = build(
        r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true,"contains":{"const":0},"minContains":1}"#,
    );
    let mut ok = StructuredState::new(&plan);
    assert!(feed(&mut ok, b"[0,1,2]"));
    assert!(ok.is_accepting());
    let mut dup = StructuredState::new(&plan);
    assert!(!feed(&mut dup, b"[0,1,1]"));
    let mut no_match = StructuredState::new(&plan);
    assert!(!feed(&mut no_match, b"[1,2,3]"));
}

#[test]
fn structured_array_rollback_restores_index_and_phase_on_a_rejected_item() {
    let plan = build(r#"{"type":"array","items":{"type":"boolean"},"maxItems":100}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"[true,"#));
    let mark = state.checkpoint();
    let before = full_snapshot(&state);
    assert_eq!(state.try_push_byte(b'1'), Ok(false));
    assert_eq!(full_snapshot(&state), before);
    state.rollback(mark);
    assert_eq!(full_snapshot(&state), before);
    assert!(feed(&mut state, b"false]"));
    assert!(state.is_accepting());
}

#[test]
fn insert_obligation_enforces_max_active_validators_on_the_first_spill() {
    let limits = StructuredLimits {
        max_active_validators: 4,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(r#"{"type":"boolean"}"#, CompileOptions::default()).unwrap(),
    );
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    let mut set = SchemaObligationSet::default();
    for i in 0..4u32 {
        state.insert_obligation(&mut set, NodeId(i)).unwrap();
    }
    let before_bytes = state.session_memory.live();
    assert!(state.insert_obligation(&mut set, NodeId(99)).is_err());
    assert_eq!(
        set.len(),
        4,
        "the rejected 5th obligation must not be recorded"
    );
    assert_eq!(state.session_memory.live(), before_bytes);
}

#[test]
fn insert_obligation_spill_growth_never_leaves_live_over_the_session_cap() {
    let limits = StructuredLimits {
        max_session_bytes: 64,
        max_active_validators: 100_000,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(r#"{"type":"boolean"}"#, CompileOptions::default()).unwrap(),
    );
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    let mut set = SchemaObligationSet::default();
    for i in 0..4u32 {
        state.insert_obligation(&mut set, NodeId(i)).unwrap();
    }
    let mut i = 4u32;
    loop {
        let before = set.len();
        match state.insert_obligation(&mut set, NodeId(i)) {
            Ok(()) => {
                assert_eq!(set.len(), before + 1);
                assert!(state.session_memory.live() <= 64 || state.session_memory.is_terminal());
            }
            Err(_) => {
                assert_eq!(set.len(), before, "a rejected spill must not grow the set");
                break;
            }
        }
        i += 1;
        assert!(i < 10_000, "budget never triggered");
    }
}

#[test]
fn max_active_validators_cap_rejects_before_pushing_the_obligation_frame() {
    let limits = StructuredLimits {
        max_active_validators: 1,
        ..StructuredLimits::default()
    };
    // Two DISTINCT value schemas: identical schemas would intern to the same NodeId and
    // collapse to a single (deduplicated) obligation, never reaching the obligation frame.
    let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r#"{"type":"object","patternProperties":{"^a.$":{"type":"integer","minimum":0,"maximum":9},"^.b$":{"type":"integer","minimum":0,"maximum":99}},"additionalProperties":false}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"ab":"#));
    let before = full_snapshot(&state);
    assert!(state.try_push_byte(b'1').is_err());
    assert_eq!(full_snapshot(&state), before);
}

#[test]
fn slot_transition_classification_matches_bucket_shape() {
    assert!(matches!(
        classify_slot_transition(None),
        SlotTransition::ToOne
    ));
    let one = DynamicSlot::One(KeyId(7));
    assert!(matches!(
        classify_slot_transition(Some(&one)),
        SlotTransition::ToMany2 { existing: KeyId(7) }
    ));
    let many = DynamicSlot::Many(vec![KeyId(1), KeyId(2), KeyId(3)]);
    assert!(matches!(
        classify_slot_transition(Some(&many)),
        SlotTransition::ToManyPush { old_len: 3 }
    ));
}

#[test]
fn a_mocked_hash_collision_upgrades_one_to_many_without_losing_either_key() {
    // Forces two values into one bucket to exercise collision handling.
    let plan = build(r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"{"));
    let a = intern_for_test(&mut state, 0, "alpha");
    let b = intern_for_test(&mut state, 0, "beta");
    let c = intern_for_test(&mut state, 0, "gamma");
    let forced_hash = 0xDEAD_BEEFu64;
    {
        let Frame::Object(obj) = &mut state.frames[0] else {
            unreachable!()
        };
        obj.seen_dynamic.insert(forced_hash, DynamicSlot::One(a));
    }
    assert!(matches!(
        classify_slot_transition(Some(&DynamicSlot::One(a))),
        SlotTransition::ToMany2 { existing } if existing == a
    ));
    {
        let Frame::Object(obj) = &mut state.frames[0] else {
            unreachable!()
        };
        obj.seen_dynamic
            .insert(forced_hash, DynamicSlot::Many(vec![a, b]));
    }
    assert!(matches!(
        classify_slot_transition(Some(&DynamicSlot::Many(vec![a, b]))),
        SlotTransition::ToManyPush { old_len: 2 }
    ));
    {
        let Frame::Object(obj) = &mut state.frames[0] else {
            unreachable!()
        };
        let Some(DynamicSlot::Many(ids)) = obj.seen_dynamic.get_mut(&forced_hash) else {
            unreachable!()
        };
        ids.push(c);
    }
    let Frame::Object(obj) = &state.frames[0] else {
        unreachable!()
    };
    let DynamicSlot::Many(ids) = &obj.seen_dynamic[&forced_hash] else {
        panic!("expected Many after two upgrades")
    };
    assert_eq!(ids, &vec![a, b, c]);
}

fn intern_for_test(state: &mut StructuredState<'_>, depth: usize, text: &str) -> KeyId {
    let mut schemas = SchemaObligationSet::default();
    schemas.insert_uncounted(NodeId(0));
    state.object_mut(depth).key.clear();
    state.object_mut(depth).key.push_str(text);
    match state
        .intern_dynamic_key(depth, ValueObligations::Schemas(schemas))
        .unwrap()
    {
        Some(_) => {}
        None => panic!("expected a fresh key, got a duplicate"),
    }
    let Frame::Object(obj) = &state.frames[depth] else {
        unreachable!()
    };
    match &obj.seen_dynamic[&state.key_hasher.hash_one(text.as_bytes())] {
        DynamicSlot::One(id) => *id,
        DynamicSlot::Many(ids) => *ids.last().unwrap(),
    }
}

#[test]
fn equal_decoded_keys_hash_equal_within_one_session() {
    let plan = build(r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#);
    let state = StructuredState::new(&plan);
    let h1 = state.key_hasher.hash_one(b"same-key");
    let h2 = state.key_hasher.hash_one(b"same-key");
    assert_eq!(h1, h2);
}

#[test]
fn key_arena_get_on_a_stale_or_corrupt_id_returns_none_not_a_panic() {
    let arena = KeyArena::default();
    assert_eq!(arena.get(KeyId(0)), None);
    assert_eq!(arena.get(KeyId(u32::MAX)), None);
}

#[test]
fn key_arena_reclaims_per_object_lifetime_not_total_document_history() {
    // Closed inner-object keys must be reclaimed while outer keys remain live.
    let plan = build(
        r#"{"type":"object","additionalProperties":{"type":"object","additionalProperties":{"type":"boolean"}}}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'{').unwrap());
    let inner_key = "y".repeat(100);
    for i in 0..5000 {
        if i > 0 {
            assert!(state.try_push_byte(b',').unwrap());
        }
        let prop = format!("\"p{i}\":{{\"{inner_key}\":true}}");
        assert!(feed(&mut state, prop.as_bytes()), "property {i}");
        state.commit_checkpoint().unwrap();
    }
    assert!(
        state.key_arena.bytes.len() < 100_000,
        "arena retained {} bytes across 5000 closed inner objects",
        state.key_arena.bytes.len()
    );
    assert!(feed(&mut state, b"}"));
    assert!(state.is_accepting());
}

#[test]
fn min_properties_rejects_a_nonempty_object_below_the_minimum() {
    let plan =
        build(r#"{"type":"object","additionalProperties":{"type":"boolean"},"minProperties":2}"#);
    let mut state = StructuredState::new(&plan);
    assert!(!feed(&mut state, br#"{"a":true}"#));
}

#[test]
fn completed_root_object_accepts_all_json_whitespace_and_rejects_anything_else() {
    let plan = build(r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"{}"));
    assert!(state.is_accepting());
    assert!(feed(&mut state, b" \t\r\n"));
    assert!(state.is_accepting());
    assert_eq!(state.try_push_byte(b'x'), Ok(false));
}

#[test]
fn completed_root_regular_value_accepts_trailing_whitespace() {
    let plan = build(r#"{"type":"boolean"}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"true"));
    assert!(state.is_accepting());
    assert!(feed(&mut state, b"  \n"));
    assert!(state.is_accepting());
    assert_eq!(state.try_push_byte(b'x'), Ok(false));
}

#[test]
fn resource_limit_error_leaves_state_byte_for_byte_equivalent() {
    let limits = StructuredLimits {
        max_document_bytes: 3,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(r#"{"type":"string"}"#, CompileOptions::default()).unwrap(),
    );
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'"').unwrap());
    assert!(state.try_push_byte(b'a').unwrap());
    assert!(state.try_push_byte(b'a').unwrap());
    let before = full_snapshot(&state);
    assert!(state.try_push_byte(b'a').is_err());
    assert_eq!(full_snapshot(&state), before);
}

#[test]
fn key_bytes_limit_rejects_exactly_at_the_boundary_for_every_scalar_width() {
    for (scalar, width) in [(b"a".as_slice(), 1usize), ("\u{20AC}".as_bytes(), 3)] {
        let limits = StructuredLimits {
            max_key_bytes: width,
            ..StructuredLimits::default()
        };
        let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let plan = StructuredPlan::compile(ir, limits).unwrap();
        let mut state = StructuredState::new(&plan);
        assert!(state.try_push_byte(b'{').unwrap());
        assert!(state.try_push_byte(b'"').unwrap());
        for &b in scalar {
            assert!(state.try_push_byte(b).unwrap());
        }
        // The key sits exactly at the cap. Feeding the scalar again must be rejected at
        // the byte that completes it (continuation bytes grow no allocation on their own).
        let mut last_good = full_snapshot(&state);
        let mut rejected = false;
        for &b in scalar {
            match state.try_push_byte(b) {
                Ok(true) => last_good = full_snapshot(&state),
                Err(_) => {
                    rejected = true;
                    break;
                }
                Ok(false) => panic!("expected a resource error, not a plain rejection"),
            }
        }
        assert!(rejected, "scalar completion must hit the key-bytes cap");
        assert_eq!(full_snapshot(&state), last_good);
    }
}

#[test]
fn a_four_byte_scalar_is_rejected_the_instant_it_would_cross_the_key_cap() {
    let limits = StructuredLimits {
        max_key_bytes: 3,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'{').unwrap());
    assert!(state.try_push_byte(b'"').unwrap());
    // U+1F600, a 4-byte scalar: the first 3 bytes only advance the UTF-8 decoder (no key
    // growth yet), and the 4th byte - which would complete the scalar at 4 > 3 - errors.
    let bytes = "\u{1F600}".as_bytes();
    for &b in &bytes[..3] {
        assert!(state.try_push_byte(b).unwrap());
    }
    let before = full_snapshot(&state);
    assert!(state.try_push_byte(bytes[3]).is_err());
    assert_eq!(full_snapshot(&state), before);
}

#[test]
fn property_pattern_work_uses_a_checked_linear_bound() {
    let schema = r#"{"type":"object","patternProperties":{"a":true,"z":true}}"#;
    let key = "a".repeat(32);
    for (cap, accepted) in [(64, true), (63, false)] {
        let limits = StructuredLimits {
            max_mask_work: cap,
            ..StructuredLimits::default()
        };
        let ir = Arc::new(
            crate::frontend::schema_to_ir(schema, CompileOptions::default()).expect("schema"),
        );
        let plan = StructuredPlan::compile(ir, limits).expect("plan");
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, format!("{{\"{key}\"").as_bytes()));
        let before = full_snapshot(&state);
        let result = state.try_push_byte(b':');
        if accepted {
            assert_eq!(result, Ok(true));
        } else {
            let error = result.expect_err("work cap plus one");
            assert_eq!(error.kind, LimitKind::MaskWork);
            assert_eq!(error.observed, 64);
            assert_eq!(error.limit, 63);
            assert_eq!(full_snapshot(&state), before);
        }
    }
}

#[test]
fn depth_limit_rejects_a_nested_object_beyond_max_depth() {
    let limits = StructuredLimits {
        max_depth: 1,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r#"{"type":"object","additionalProperties":{"type":"object","additionalProperties":{"type":"boolean"}}}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'{').unwrap());
    assert!(state.try_push_byte(b'"').unwrap());
    assert!(state.try_push_byte(b'a').unwrap());
    assert!(state.try_push_byte(b'"').unwrap());
    assert!(state.try_push_byte(b':').unwrap());
    let before = full_snapshot(&state);
    assert!(state.try_push_byte(b'{').is_err());
    assert_eq!(full_snapshot(&state), before);
}

fn recursive_object_plan(limits: StructuredLimits) -> StructuredPlan {
    let ir = Arc::new(
        crate::frontend::schema_to_ir(
            r##"{
                "$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"additionalProperties":false}},
                "$ref":"#/$defs/node"
            }"##,
            CompileOptions::default(),
        )
        .unwrap(),
    );
    StructuredPlan::compile(ir, limits).unwrap()
}

fn recursive_object_document(depth: usize) -> Vec<u8> {
    let mut document = Vec::new();
    for _ in 1..depth {
        document.extend_from_slice(br#"{"next":"#);
    }
    document.extend_from_slice(b"{}");
    document.extend(std::iter::repeat_n(b'}', depth.saturating_sub(1)));
    document
}

#[test]
fn recursive_references_honor_exact_global_depth_limits() {
    for depth in [1usize, 8, 64] {
        let limits = StructuredLimits {
            max_depth: u32::try_from(depth).unwrap(),
            ..StructuredLimits::default()
        };
        let plan = recursive_object_plan(limits);
        let mut exact = StructuredState::new(&plan);
        assert!(
            feed(&mut exact, &recursive_object_document(depth)),
            "depth {depth}"
        );
        assert!(exact.is_accepting(), "depth {depth}");

        let mut over = StructuredState::new(&plan);
        for _ in 0..depth {
            assert!(feed(&mut over, br#"{"next":"#));
        }
        let before = full_snapshot(&over);
        let live = over.session_memory.live();
        let retained = over.retained_bytes();
        let error = over.try_push_byte(b'{').unwrap_err();
        assert_eq!(error.kind, LimitKind::RecursionDepth);
        assert_eq!(error.observed, depth + 1);
        assert_eq!(error.limit, depth);
        assert_eq!(full_snapshot(&over), before);
        assert_eq!(over.session_memory.live(), live);
        assert_eq!(over.retained_bytes(), retained);
    }
}

#[test]
fn recursive_reference_validator_limit_is_global_and_atomic() {
    let schema = r##"{
        "$defs":{
            "a":{"type":"object","properties":{"next":{"$ref":"#/$defs/a"}}},
            "b":{"type":"object","properties":{"next":{"$ref":"#/$defs/b"}}},
            "c":{"type":"object","properties":{"next":{"$ref":"#/$defs/c"}}}
        },
        "type":"object",
        "properties":{"x":{"$ref":"#/$defs/a"}},
        "patternProperties":{"^x$":{"$ref":"#/$defs/b"},"x$":{"$ref":"#/$defs/c"}},
        "additionalProperties":false
    }"##;
    let limits = StructuredLimits {
        max_active_validators: 2,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"x":"#));
    let before = full_snapshot(&state);
    let live = state.session_memory.live();
    let error = state.try_push_byte(b'{').unwrap_err();
    assert_eq!(error.kind, LimitKind::ActiveValidators);
    assert_eq!(error.observed, 3);
    assert_eq!(error.limit, 2);
    assert_eq!(full_snapshot(&state), before);
    assert_eq!(state.session_memory.live(), live);
    let limits = StructuredLimits {
        max_active_validators: 3,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut permitted = StructuredState::new(&plan);
    assert!(feed(&mut permitted, br#"{"x":{}"#));
}

#[test]
fn sibling_combinator_trees_share_one_live_validator_limit() {
    let schema = r#"{
        "anyOf":[
            {"anyOf":[
                {"type":"object","additionalProperties":{"type":"integer"}},
                {"type":"array","items":{"type":"integer"},"uniqueItems":true}
            ]},
            {"anyOf":[
                {"type":"object","additionalProperties":{"type":"string"}},
                {"type":"array","items":{"type":"string"},"contains":{"type":"string"}}
            ]}
        ]
    }"#;
    let limits = StructuredLimits {
        max_active_validators: 5,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let ledger = Arc::new(ValidatorLedger {
        live: AtomicUsize::new(0),
        limit: 5,
    });
    let error = match StructuredState::new_at_with_context(
        PlanHandle::Borrowed(&plan),
        plan.root,
        0,
        ledger.clone(),
    ) {
        Ok(_) => panic!("combined sibling trees exceeded the global limit"),
        Err(error) => error,
    };
    assert_eq!(error.kind, LimitKind::ActiveValidators);
    assert_eq!(error.observed, 6);
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
}

#[test]
fn combinator_branch_limit_is_exact_and_failed_installation_releases_every_reservation() {
    let make_schema = |count: usize| {
        let branches = (0..count)
            .map(|index| {
                format!(
                    r#"{{"type":"object","properties":{{"x{index}":{{"type":"integer"}}}},"additionalProperties":true}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(r#"{{"anyOf":[{branches}]}}"#)
    };
    for (count, succeeds) in [(4usize, true), (5, false)] {
        let limits = StructuredLimits {
            // Keep the plan compilable to exercise the capped validator ledger.
            max_active_validators: count,
            ..StructuredLimits::default()
        };
        let schema = make_schema(count);
        let ir =
            Arc::new(crate::frontend::schema_to_ir(&schema, CompileOptions::default()).unwrap());
        let plan = StructuredPlan::compile(ir, limits).unwrap();
        let ledger = Arc::new(ValidatorLedger {
            live: AtomicUsize::new(0),
            limit: 4,
        });
        let result = StructuredState::new_at_with_context(
            PlanHandle::Borrowed(&plan),
            plan.root,
            0,
            ledger.clone(),
        );
        if succeeds {
            let state = result.expect("exact validator limit");
            assert_eq!(ledger.live.load(Ordering::Relaxed), 4);
            drop(state);
        } else {
            let error = match result {
                Ok(_) => panic!("one branch beyond limit"),
                Err(error) => error,
            };
            assert_eq!(error.kind, LimitKind::ActiveValidators);
            assert_eq!(error.observed, 5);
        }
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn combinator_middle_installation_session_failure_is_atomic() {
    let plan = build(
        r#"{"anyOf":[{"type":"null"},{"type":"object","additionalProperties":{"type":"integer"}},{"type":"array","items":{"type":"integer"},"uniqueItems":true}]}"#,
    );
    let mut found_middle_failure = false;
    for budget in 1..16_384usize {
        let ledger = Arc::new(ValidatorLedger {
            live: AtomicUsize::new(0),
            limit: plan.limits.max_active_validators,
        });
        let mut state = StructuredState::empty_for_test(&plan, ledger.clone()).unwrap();
        state.limits.max_session_bytes = budget;
        let before_session = state.session_memory.live();
        let result = state.prepare_combinator_frame(plan.root);
        if matches!(result, Err(ref error) if error.kind == LimitKind::SessionBytes)
            && !state.cursors.slots.is_empty()
        {
            assert!(state
                .cursors
                .slots
                .iter()
                .all(|slot| matches!(slot, CursorSlot::Vacant)));
            assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
            assert_eq!(state.session_memory.live(), before_session);
            assert_eq!(state.cursors.free_slots.len(), state.cursors.slots.len());
            found_middle_failure = true;
            break;
        }
        if let Ok(prepared) = result {
            state.discard_new_frame(prepared.frame);
        }
    }
    assert!(
        found_middle_failure,
        "no budget failed after installing a branch"
    );
}

#[test]
fn combinator_undo_limit_is_preflighted_before_branch_mutation() {
    let plan = build(
        r#"{"anyOf":[{"type":"object","additionalProperties":{"type":"integer"}},{"type":"array","items":{"type":"integer"},"uniqueItems":true}]}"#,
    );
    let mut state = StructuredState::new(&plan);
    let before = full_snapshot(&state);
    let slots = state.cursors.slots.clone();
    let generations = state.cursors.generations.clone();
    let epochs = state.cursors.epochs.clone();
    let free_slots = state.cursors.free_slots.clone();
    let free_structured = state.cursors.free_structured.clone();
    let session = state.session_memory.live();
    let validators = state._validator_ledger.live.load(Ordering::Relaxed);
    state.limits.max_undo_entries = state.undo.len();
    let error = state.try_push_byte(b'{').unwrap_err();
    assert_eq!(error.kind, LimitKind::UndoEntries);
    assert_eq!(full_snapshot(&state), before);
    assert_eq!(state.cursors.slots, slots);
    assert_eq!(state.cursors.generations, generations);
    assert_eq!(state.cursors.epochs, epochs);
    assert_eq!(state.cursors.free_slots, free_slots);
    assert_eq!(state.cursors.free_structured, free_structured);
    assert_eq!(state.session_memory.live(), session);
    assert_eq!(
        state._validator_ledger.live.load(Ordering::Relaxed),
        validators
    );
    state.limits.max_undo_entries = usize::MAX;
    assert!(state.try_push_byte(b'[').unwrap());
}

#[test]
fn committed_pruned_combinator_cursor_generation_cannot_touch_reused_slot() {
    let plan = build(
        r#"{"anyOf":[{"type":"object","additionalProperties":{"type":"integer"}},{"type":"array","items":{"type":"integer"},"uniqueItems":true}]}"#,
    );
    let ledger;
    {
        let mut state = StructuredState::new(&plan);
        ledger = state._validator_ledger.clone();
        let (old, branch) = match &state.frames[0] {
            Frame::Combinator(frame) => {
                let NodePlan::Combinator(combinator) = state.plan.as_ref().node(frame.node) else {
                    panic!("expected combinator plan");
                };
                (frame.cursors[0], combinator.branches[0])
            }
            _ => panic!("expected combinator frame"),
        };
        assert!(state.try_push_byte(b'[').unwrap());
        state.commit_checkpoint().unwrap();
        let live_before = ledger.live.load(Ordering::Relaxed);
        let fresh = state.start_cursor(branch).unwrap();
        assert_eq!(old.index, fresh.index);
        assert_ne!(old.generation, fresh.generation);
        assert_eq!(ledger.live.load(Ordering::Relaxed), live_before + 1);
        assert!(state.cursors.try_push_byte(old, b'{').is_err());
        assert!(state.cursors.release(old).is_err());
        assert_eq!(ledger.live.load(Ordering::Relaxed), live_before + 1);
        state.cursors.release(fresh).unwrap();
        assert_eq!(ledger.live.load(Ordering::Relaxed), live_before);
    }
    assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
}

fn nested_non_regular_combinator_schema(levels: usize) -> String {
    let mut schema = String::from(r#"{"type":"object","additionalProperties":{"type":"integer"}}"#);
    for _ in 0..levels {
        schema = format!(
            r#"{{"anyOf":[{schema},{{"type":"array","items":{{"type":"integer"}},"uniqueItems":true}}]}}"#
        );
    }
    schema
}

#[test]
fn nested_non_regular_combinators_honor_the_exact_depth_limit_atomically() {
    let levels = 16usize;
    let schema = nested_non_regular_combinator_schema(levels);
    for (limit, succeeds) in [(levels + 1, true), (levels, false)] {
        let limits = StructuredLimits {
            max_depth: u32::try_from(limit).unwrap(),
            ..StructuredLimits::default()
        };
        let ir =
            Arc::new(crate::frontend::schema_to_ir(&schema, CompileOptions::default()).unwrap());
        let plan = StructuredPlan::compile(ir, limits).unwrap();
        let ledger = Arc::new(ValidatorLedger {
            live: AtomicUsize::new(0),
            limit: plan.limits.max_active_validators,
        });
        let result = StructuredState::new_at_with_context(
            PlanHandle::Borrowed(&plan),
            plan.root,
            0,
            ledger.clone(),
        );
        if succeeds {
            drop(result.expect("exact combinator depth"));
        } else {
            let error = match result {
                Ok(_) => panic!("one level beyond depth limit constructed"),
                Err(error) => error,
            };
            assert_eq!(error.kind, LimitKind::RecursionDepth);
            assert_eq!(error.observed, limit + 1);
            assert_eq!(error.limit, limit);
        }
        assert_eq!(ledger.live.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn recursive_reference_cursor_installation_budget_failure_is_atomic() {
    let plan = recursive_object_plan(StructuredLimits::default());
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"next":"#));
    state.commit_checkpoint().unwrap();
    let before = full_snapshot(&state);
    let slots = state.cursors.slots.clone();
    let generations = state.cursors.generations.clone();
    let epochs = state.cursors.epochs.clone();
    let free_slots = state.cursors.free_slots.clone();
    let free_structured = state.cursors.free_structured.clone();
    let live = state.session_memory.live();
    state.limits.max_session_bytes = state.retained_bytes();
    let error = state.try_push_byte(b'{').unwrap_err();
    assert_eq!(error.kind, LimitKind::SessionBytes);
    assert_eq!(full_snapshot(&state), before);
    assert_eq!(state.session_memory.live(), live);
    assert_eq!(state.cursors.slots, slots);
    assert_eq!(state.cursors.generations, generations);
    assert_eq!(state.cursors.epochs, epochs);
    assert_eq!(state.cursors.free_slots, free_slots);
    assert_eq!(state.cursors.free_structured, free_structured);
    state.limits.max_session_bytes = usize::MAX;
    assert!(state.try_push_byte(b'{').unwrap());
    assert!(state.try_push_byte(b'}').unwrap());
}

#[test]
fn recursive_reference_cursor_capacity_and_session_bytes_stabilize() {
    let plan = recursive_object_plan(StructuredLimits::default());
    let mut state = StructuredState::new(&plan);
    let mark = state.checkpoint();
    let document = recursive_object_document(8);
    let mut samples = Vec::new();
    for _ in 0..64 {
        assert!(feed(&mut state, &document));
        state.rollback(mark);
        samples.push((
            state.retained_bytes(),
            state.cursors.slots.capacity(),
            state.cursors.structured.capacity(),
        ));
    }
    assert_eq!(samples[1], samples[63]);
    assert!(!state.session_memory.is_terminal());
}

#[test]
fn undo_entries_limit_is_enforced_without_partial_mutation() {
    let limits = StructuredLimits {
        max_undo_entries: 2,
        ..StructuredLimits::default()
    };
    let ir = Arc::new(
        crate::frontend::schema_to_ir(r#"{"type":"string"}"#, CompileOptions::default()).unwrap(),
    );
    let plan = StructuredPlan::compile(ir, limits).unwrap();
    let mut state = StructuredState::new(&plan);
    assert!(state.try_push_byte(b'"').unwrap());
    assert!(state.try_push_byte(b'a').unwrap());
    let before = full_snapshot(&state);
    assert!(state.try_push_byte(b'a').is_err());
    assert_eq!(full_snapshot(&state), before);
}

#[test]
fn rejected_byte_leaves_state_unchanged_and_rollback_restores_the_checkpoint() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"a":tru"#));
    let mark = state.checkpoint();
    assert_eq!(state.try_push_byte(b'x'), Ok(false));
    assert!(!state.is_dead());
    assert!(state.try_push_byte(b'e').unwrap());
    state.rollback(mark);
    assert!(state.try_push_byte(b'e').unwrap());
    assert!(feed(&mut state, b"}"));
    assert!(state.is_accepting());
}

#[test]
fn repeated_checkpoint_and_rollback_never_diverges_from_direct_feeding() {
    let plan = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"boolean"}},"required":["a"],"additionalProperties":{"type":"boolean"}}"#,
    );
    let mut probing = StructuredState::new(&plan);
    let mark = probing.checkpoint();
    for candidate in [
        br#"{"a":true,"b":false}"#.as_slice(),
        br#"{"a":1}"#.as_slice(),
    ] {
        probing.rollback(mark);
        let _ = feed(&mut probing, candidate);
    }
    probing.rollback(mark);
    assert!(feed(&mut probing, br#"{"a":true,"b":false}"#));
    assert!(probing.is_accepting());
}

/// For every byte 0..=255 at `prefix`, pushes it, rolls back, and asserts full observable
/// state is unchanged - regardless of whether the byte was accepted or rejected.
fn assert_full_byte_sweep_is_rollback_safe(plan: &StructuredPlan, prefix: &[u8]) {
    let mut state = StructuredState::new(plan);
    assert!(
        feed(&mut state, prefix),
        "prefix must be a valid walk: {prefix:?}"
    );
    let mark = state.checkpoint();
    let before = full_snapshot(&state);
    for byte in 0u16..=255 {
        let b = byte as u8;
        let _ = state.try_push_byte(b);
        state.rollback(mark);
        assert_eq!(
            full_snapshot(&state),
            before,
            "byte {b:#04x} at {prefix:?} broke rollback"
        );
    }
}

#[test]
fn full_byte_sweep_is_rollback_safe_at_every_structural_boundary() {
    let object_schema = build(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"],"additionalProperties":{"type":"integer","minimum":0}}"#,
    );
    let string_schema = build(r#"{"type":"string"}"#);
    let number_schema = build(r#"{"type":"integer","minimum":0,"maximum":999}"#);
    let nested_schema = build(
        r#"{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"boolean"}},"required":["b"],"additionalProperties":false}},"required":["a"],"additionalProperties":false}"#,
    );
    let unique_schema = build(r#"{"type":"array","items":{"type":"string"},"uniqueItems":true}"#);

    let cases: &[(&StructuredPlan, &[u8])] = &[
        (&string_schema, br#"""#),              // inside an open string
        (&string_schema, b"\"tr"),              // mid-literal-like scalar bytes
        (&number_schema, b"12"),                // inside a number
        (&string_schema, b"\"\xe2\x82"),        // inside a raw UTF-8 sequence
        (&string_schema, b"\"\\"),              // right after a backslash
        (&string_schema, b"\"\\u"),             // \u with 0 digits
        (&string_schema, b"\"\\u00"),           // \u with 2 digits
        (&string_schema, b"\"\\uD83D"),         // a completed high surrogate
        (&string_schema, b"\"\\uD83D\\uDE"),    // between surrogate halves
        (&object_schema, br#"{"a"#),            // inside a property name
        (&object_schema, br#"{"a""#),           // right after a completed key
        (&object_schema, br#"{"a":"#),          // right after the colon
        (&object_schema, br#"{"a":tr"#),        // inside a pushed child value
        (&object_schema, br#"{"a":true"#),      // right after child completion
        (&object_schema, br#"{"a":true,"#),     // right after a comma
        (&nested_schema, br#"{"a":{"b":true"#), // before a nested object close
        (&object_schema, br#"{"a":true}"#),     // right after root completion
        (&object_schema, b"{\"a\":true} \t"),   // after trailing whitespace
        (&unique_schema, br#"["a","#),          // right before a duplicate item starts
        (&unique_schema, br#"["a","a"#),        // mid-way through a duplicate item
    ];
    for (plan, prefix) in cases {
        assert_full_byte_sweep_is_rollback_safe(plan, prefix);
    }
}

fn exhaustive_bfs_agrees_with_reference(schema: &str, alphabet: &[u8], max_depth: usize) {
    let ir = Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
    let plan = StructuredPlan::compile(ir.clone(), StructuredLimits::default()).unwrap();
    let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<Vec<u8>> = std::collections::VecDeque::new();
    queue.push_back(Vec::new());
    while let Some(prefix) = queue.pop_front() {
        if !seen.insert(prefix.clone()) {
            continue;
        }
        let mut state = StructuredState::new(&plan);
        let mut walk_ok = true;
        for &b in &prefix {
            if state.try_push_byte(b) != Ok(true) {
                walk_ok = false;
                break;
            }
        }
        let ref_accepts = crate::structured::reference::accepts(&ir, &prefix);
        let ref_continuable = crate::structured::reference::can_continue(&ir, &prefix);
        if walk_ok {
            assert_eq!(
                state.is_accepting(),
                ref_accepts,
                "accepting mismatch at prefix {prefix:?}"
            );
            assert_eq!(
                !state.is_dead(),
                ref_continuable,
                "continuable mismatch at prefix {prefix:?}"
            );
        } else {
            assert!(
                !ref_accepts && !ref_continuable,
                "incremental rejected {prefix:?} but reference still allows it"
            );
        }
        if prefix.len() >= max_depth {
            continue;
        }
        for &b in alphabet {
            let mut next = prefix.clone();
            next.push(b);
            queue.push_back(next);
        }
    }
}

#[test]
fn exhaustive_bfs_open_object_with_additional_schema_agrees_with_reference() {
    exhaustive_bfs_agrees_with_reference(
        r#"{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"],"additionalProperties":{"type":"boolean"}}"#,
        b"{}\"a:,",
        6,
    );
}

#[test]
fn exhaustive_bfs_pattern_properties_agrees_with_reference() {
    exhaustive_bfs_agrees_with_reference(
        r#"{"type":"object","patternProperties":{"^x[0-9]$":{"type":"boolean"}},"additionalProperties":false}"#,
        b"{}\"x1:,",
        6,
    );
}

#[test]
fn exhaustive_bfs_nested_object_agrees_with_reference() {
    exhaustive_bfs_agrees_with_reference(
        r#"{"type":"object","additionalProperties":{"type":"boolean"},"properties":{"a":{"type":"object","additionalProperties":{"type":"boolean"},"properties":{"b":{"type":"boolean"}},"required":["b"]}},"required":["a"]}"#,
        b"{}\"ab:,",
        6,
    );
}

#[test]
fn exhaustive_bfs_full_valid_documents_agree_with_reference() {
    let schema = r#"{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"],"additionalProperties":{"type":"boolean"}}"#;
    let ir = Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
    let plan = StructuredPlan::compile(ir.clone(), StructuredLimits::default()).unwrap();
    for doc in [
        b"{\"a\":true}".as_slice(),
        b"{\"a\":false,\"z\":true}".as_slice(),
    ] {
        for len in 0..=doc.len() {
            let prefix = &doc[..len];
            let mut state = StructuredState::new(&plan);
            let walk_ok = prefix.iter().all(|&b| state.try_push_byte(b) == Ok(true));
            assert!(
                walk_ok,
                "a real document prefix must never be rejected: {prefix:?}"
            );
            assert_eq!(
                state.is_accepting(),
                crate::structured::reference::accepts(&ir, prefix),
                "accepting mismatch at {prefix:?}"
            );
            assert_eq!(
                !state.is_dead(),
                crate::structured::reference::can_continue(&ir, prefix),
                "continuable mismatch at {prefix:?}"
            );
            for bad in [b'\x00', b'x', b'1'] {
                let mut deviated = prefix.to_vec();
                deviated.push(bad);
                let mut trial = StructuredState::new(&plan);
                let trial_ok = deviated.iter().all(|&b| trial.try_push_byte(b) == Ok(true));
                let ref_ok = crate::structured::reference::accepts(&ir, &deviated)
                    || crate::structured::reference::can_continue(&ir, &deviated);
                assert_eq!(
                    trial_ok, ref_ok,
                    "single-byte deviation mismatch at {deviated:?}"
                );
            }
        }
    }
}

/// Per-property allocation profiling. Compiled only under `--features alloc-probe`: it installs
/// a counting global allocator, so it must never share a test binary with a timing assertion.
#[cfg(feature = "alloc-probe")]
mod alloc_probe_tests {
    use super::*;
    use crate::frontend::schema_to_ir;
    use crate::ir::CompileOptions;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingAllocator;
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static TRACK: Cell<bool> = const { Cell::new(false) };
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc(layout);
            if !ptr.is_null() && TRACK.with(Cell::get) {
                COUNT.fetch_add(1, Ordering::Relaxed);
                let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
                PEAK.fetch_max(now, Ordering::Relaxed);
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            if TRACK.with(Cell::get) {
                LIVE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                    Some(live.saturating_sub(layout.size()))
                })
                .ok();
            }
            System.dealloc(ptr, layout);
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAllocator = CountingAllocator;

    /// `cargo test` runs tests on multiple threads; COUNT/LIVE/PEAK are process-global, so every
    /// test in this module must serialize on this guard or they corrupt each other's counts.
    static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn object_document(n: usize, long: bool, unicode: bool) -> String {
        let mut doc = String::from("{");
        for i in 0..n {
            if i > 0 {
                doc.push(',');
            }
            if unicode {
                doc.push_str(&format!("\"caf\\u00e9{i}\":true"));
            } else if long {
                doc.push_str(&format!("\"{}{i}\":true", "k".repeat(64)));
            } else {
                doc.push_str(&format!("\"k{i}\":true"));
            }
        }
        doc.push('}');
        doc
    }

    /// `(allocations, peak_bytes_over_base, session_bytes_charged)` for feeding `doc` end to end.
    fn measure(doc: &str) -> (usize, usize, usize) {
        let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let plan = StructuredPlan::compile(
            ir,
            StructuredLimits {
                max_session_bytes: 64 * 1024 * 1024,
                ..StructuredLimits::default()
            },
        )
        .unwrap();
        COUNT.store(0, Ordering::Relaxed);
        let base = LIVE.load(Ordering::Relaxed);
        PEAK.store(base, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        let mut state = StructuredState::new(&plan);
        for &b in doc.as_bytes() {
            state.try_push_byte(b).unwrap();
        }
        TRACK.with(|track| track.set(false));
        let allocs = COUNT.load(Ordering::Relaxed);
        let peak = PEAK.load(Ordering::Relaxed) - base;
        (allocs, peak, state.session_memory.live())
    }

    #[test]
    fn allocation_profile_across_property_counts() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let cases: &[(&str, usize, bool, bool)] = &[
            ("1 property", 1, false, false),
            ("16 properties", 16, false, false),
            ("1000 short properties", 1000, false, false),
            ("100 long properties", 100, true, false),
            ("100 escaped-unicode-key properties", 100, false, true),
        ];
        for &(label, n, long, unicode) in cases {
            let doc = object_document(n, long, unicode);
            let (allocs, peak, session_bytes) = measure(&doc);
            eprintln!(
                "{label}: {allocs} allocations, {peak} peak bytes, {session_bytes} session bytes charged, {:.2} allocs/property",
                allocs as f64 / n as f64
            );
        }
    }

    #[test]
    fn resource_peak_charge_covers_independent_allocator_peak() {
        let _guard = GUARD
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let limits = crate::frontend::SchemaResourceLimits::default();
        let mut registry = crate::frontend::SchemaRegistry::new(limits);
        registry
            .insert(
                "https://example.test/root",
                r##"{
                    "$defs":{"node":{"$anchor":"node","type":"string"}},
                    "allOf":[{"$ref":"#node"},{"maxLength":32}]
                }"##,
            )
            .unwrap();

        COUNT.store(0, Ordering::Relaxed);
        let base = LIVE.load(Ordering::Relaxed);
        PEAK.store(base, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        let (charged_peak, charged_retained) =
            crate::frontend::discover_memory_for_probe(registry).unwrap();
        TRACK.with(|track| track.set(false));
        let allocator_peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);

        assert!(charged_retained > 0);
        assert!(
            charged_peak >= allocator_peak,
            "charged peak {charged_peak} is below allocator peak {allocator_peak}"
        );
    }

    fn n_obligation_schema(n: usize) -> String {
        let mut patterns = String::new();
        for i in 0..n {
            if i > 0 {
                patterns.push(',');
            }
            let mut regex = String::from("^");
            for j in 0..n {
                regex.push(if j == i { 'a' } else { '.' });
            }
            regex.push('$');
            patterns.push_str(&format!(
                r#""{regex}":{{"type":"integer","minimum":0,"maximum":9}}"#
            ));
        }
        format!(
            r#"{{"type":"object","patternProperties":{{{patterns}}},"additionalProperties":false}}"#
        )
    }

    #[test]
    fn obligation_frame_transitions_allocate_zero_after_construction() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        for n in [1usize, 2, 4, 5, 8] {
            let schema = n_obligation_schema(n);
            let ir = Arc::new(schema_to_ir(&schema, CompileOptions::default()).unwrap());
            let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
            let key = "a".repeat(n);
            let mut state = StructuredState::new(&plan);
            let prefix = format!("{{\"{key}\":");
            for &b in prefix.as_bytes() {
                state.try_push_byte(b).unwrap();
            }
            // One warm-up transition primes any one-time lazy cache in the automaton engines
            // themselves (unrelated to the obligation frame); only steady-state matters here.
            let warm_mark = state.checkpoint();
            state.try_push_byte(b'0').unwrap();
            state.rollback(warm_mark);
            COUNT.store(0, Ordering::Relaxed);
            TRACK.with(|track| track.set(true));
            for i in 0..10_000u32 {
                let digit = b'0' + (i % 10) as u8;
                let mark = state.checkpoint();
                state.try_push_byte(digit).unwrap();
                state.rollback(mark);
            }
            TRACK.with(|track| track.set(false));
            let allocs = COUNT.load(Ordering::Relaxed);
            assert_eq!(
                allocs, 0,
                "n={n} obligations allocated {allocs} times over 10000 transitions"
            );
        }
    }

    fn begin_tracking() {
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
    }

    fn finish_tracking() -> usize {
        TRACK.with(|track| track.set(false));
        COUNT.load(Ordering::Relaxed)
    }

    #[test]
    fn string_enum_transitions_allocate_zero_after_construction() {
        let _guard = GUARD
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let long = "a".repeat(10_000);
        let plan = large_string_enum_plan(vec![long.clone()]);
        let mut state = StructuredState::new(&plan);
        assert!(state.try_push_byte(b'"').unwrap());
        state.reserve_undo(10_001).unwrap();
        begin_tracking();
        for _ in 0..10_000 {
            assert!(state.try_push_byte(b'a').unwrap());
        }
        assert_eq!(finish_tracking(), 0, "ordinary body transitions");

        let shared = "p".repeat(10_000);
        let candidates = (0..=MAX_UNROLLED_ENUM)
            .map(|index| format!("{shared}-{index:04}"))
            .collect();
        let plan = large_string_enum_plan(candidates);
        let mut state = StructuredState::new(&plan);
        assert!(state.try_push_byte(b'"').unwrap());
        state.reserve_undo(10_001).unwrap();
        begin_tracking();
        for _ in 0..10_000 {
            assert!(state.try_push_byte(b'p').unwrap());
        }
        assert_eq!(finish_tracking(), 0, "common-prefix transitions");

        let plan = large_string_enum_plan(vec![long]);
        let escaped = b"\\u0061".repeat(10_000);
        let mut state = StructuredState::new(&plan);
        assert!(state.try_push_byte(b'"').unwrap());
        state.commit_checkpoint().unwrap();
        state.reserve_undo(7).unwrap();
        begin_tracking();
        for spelling in escaped.chunks_exact(6) {
            for &byte in spelling {
                assert!(state.try_push_byte(byte).unwrap());
            }
            state.commit_checkpoint().unwrap();
        }
        assert_eq!(finish_tracking(), 0, "escaped scalar transitions");

        let plan = large_string_enum_plan(vec!["a".to_owned(), "ab".to_owned()]);
        let mut state = StructuredState::new(&plan);
        assert!(state.try_push_byte(b'"').unwrap());
        state.reserve_undo(2).unwrap();
        begin_tracking();
        for _ in 0..10_000 {
            let mark = state.checkpoint();
            assert!(state.try_push_byte(b'a').unwrap());
            state.rollback(mark);
        }
        assert_eq!(finish_tracking(), 0, "speculative rollback transitions");

        assert!(state.try_push_byte(b'a').unwrap());
        state.commit_checkpoint().unwrap();
        state.reserve_undo(2).unwrap();
        begin_tracking();
        assert!(state.try_push_byte(b'"').unwrap());
        assert_eq!(finish_tracking(), 0, "exact closing quote");
    }

    #[test]
    fn string_intersection_transitions_allocate_zero_after_warmup() {
        let _guard = GUARD
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let schema =
            r#"{"allOf":[{"type":"string","pattern":"^a"},{"type":"string","pattern":"z$"}]}"#;
        let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).unwrap());
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        let mut state = StructuredState::new(&plan);
        for &byte in b"\"a" {
            assert!(state.try_push_byte(byte).unwrap());
        }

        let warm_mark = state.checkpoint();
        assert!(state.try_push_byte(b'z').unwrap());
        state.rollback(warm_mark);
        state.reserve_undo(2).unwrap();
        begin_tracking();
        for _ in 0..10_000 {
            let mark = state.checkpoint();
            assert!(state.try_push_byte(b'z').unwrap());
            state.rollback(mark);
        }
        assert_eq!(
            finish_tracking(),
            0,
            "warmed string-intersection transitions must not allocate"
        );
    }

    #[test]
    fn dependent_object_presence_transitions_allocate_zero_after_warmup() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let cases = [
            (
                r#"{"type":"object","dependentSchemas":{"a":{"type":"object","required":["z"],"additionalProperties":true}},"additionalProperties":true}"#,
                br#"{"#.as_slice(),
                b' ',
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"type":"object","required":["z"],"additionalProperties":true}},"additionalProperties":true}"#,
                br#"{"a":1,"#.as_slice(),
                b' ',
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"type":"object","properties":{"bad":{"type":"integer"}},"additionalProperties":true}},"additionalProperties":true}"#,
                br#"{"bad":"x","#.as_slice(),
                b' ',
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"type":"object","required":["a"],"additionalProperties":true},"b":{"type":"object","required":["b"],"additionalProperties":true},"c":{"type":"object","required":["c"],"additionalProperties":true},"d":{"type":"object","required":["d"],"additionalProperties":true},"e":{"type":"object","required":["e"],"additionalProperties":true},"f":{"type":"object","required":["f"],"additionalProperties":true},"g":{"type":"object","required":["g"],"additionalProperties":true},"h":{"type":"object","required":["h"],"additionalProperties":true}},"additionalProperties":true}"#,
                br#"{"#.as_slice(),
                b' ',
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"required":["z"]},"b":{"required":["z"]},"c":{"required":["z"]},"d":{"required":["z"]},"e":{"required":["z"]},"f":{"required":["z"]},"g":{"required":["z"]},"h":{"required":["z"]}},"additionalProperties":true}"#,
                br#"{"#.as_slice(),
                b' ',
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"allOf":[{"not":{"const":0}},{"type":"object"}]}},"additionalProperties":true}"#,
                br#"{"#.as_slice(),
                b' ',
            ),
            (
                r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
                br#"{"a"#.as_slice(),
                b'"',
            ),
            (
                r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
                br#"{"irrelevant"#.as_slice(),
                b'"',
            ),
            (
                r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
                br#"{"a":1,"b":2"#.as_slice(),
                b'}',
            ),
        ];
        for (schema, prefix, byte) in cases {
            let plan = build(schema);
            let mut state = StructuredState::new(&plan);
            assert!(feed(&mut state, prefix));
            let warm = state.checkpoint();
            assert!(state.try_push_byte(byte).unwrap());
            state.rollback(warm);
            COUNT.store(0, Ordering::Relaxed);
            TRACK.with(|track| track.set(true));
            for _ in 0..10_000 {
                let mark = state.checkpoint();
                assert!(state.try_push_byte(byte).unwrap());
                state.rollback(mark);
            }
            TRACK.with(|track| track.set(false));
            assert_eq!(COUNT.load(Ordering::Relaxed), 0, "{schema}");
        }
    }

    #[test]
    fn dependent_required_invalid_close_allocates_zero_after_warmup() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let plan = build(
            r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
        );
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, br#"{"a":1"#));
        let warm = state.checkpoint();
        assert!(!state.try_push_byte(b'}').unwrap());
        state.rollback(warm);
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..10_000 {
            let mark = state.checkpoint();
            assert!(!state.try_push_byte(b'}').unwrap());
            state.rollback(mark);
        }
        TRACK.with(|track| track.set(false));
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn dependent_required_record_and_trie_masks_allocate_zero_after_warmup() {
        use crate::index::{TrieCache, VocabularyHandle};
        use crate::structured::StructuredProgram;
        use crate::vocab::build_vocabulary;
        use rustc_hash::FxHashMap;

        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let schema = Arc::new(
            schema_to_ir(
                r#"{"type":"object","dependentSchemas":{"a":{"type":"object","properties":{"a":true,"alias":true,"z":{"const":1}},"additionalProperties":false},"alias":{"type":"object","properties":{"a":true,"alias":true,"z":{"const":1}},"additionalProperties":false}},"dependentRequired":{"a":["z"],"alias":["z"]},"additionalProperties":true}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let program = StructuredProgram::compile(schema).unwrap();
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [
            (0, b" ".as_slice()),
            (1, b"\"a\""),
            (2, b":"),
            (3, b"1"),
            (4, b","),
            (5, b"\"z\":1}"),
            (6, b"}"),
            (7, b"\"bad\":1"),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let handle = VocabularyHandle::new(Arc::new(build_vocabulary(8, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&handle).unwrap();
        let mut record = program.new_matcher().unwrap();
        let mut walked = program.new_matcher().unwrap();
        assert!(record.advance(b"{").unwrap());
        assert!(walked.advance(b"{").unwrap());
        let mut record_out = [0u8; 4];
        let mut trie_out = [0u8; 4];
        for _ in 0..64 {
            record
                .write_record_mask_le_bytes_into(8, handle.iter_records(), &mut record_out)
                .unwrap();
            walked
                .write_mask_le_bytes_into(&handle, Some(&trie), &mut trie_out)
                .unwrap();
        }
        assert_eq!(record_out, trie_out);

        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..10_000 {
            record
                .write_record_mask_le_bytes_into(8, handle.iter_records(), &mut record_out)
                .unwrap();
        }
        TRACK.with(|track| track.set(false));
        assert_eq!(COUNT.load(Ordering::Relaxed), 0, "record mask allocated");

        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..10_000 {
            walked
                .write_mask_le_bytes_into(&handle, Some(&trie), &mut trie_out)
                .unwrap();
        }
        TRACK.with(|track| track.set(false));
        assert_eq!(COUNT.load(Ordering::Relaxed), 0, "trie mask allocated");
        assert_eq!(record_out, trie_out);
        assert!(record.advance(br#""z":1}"#).unwrap());
        assert!(walked.advance(br#""z":1}"#).unwrap());
        assert!(record.is_accepting() && walked.is_accepting());
    }

    #[test]
    fn recursive_reference_transitions_allocate_zero_after_warmup() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let ir = Arc::new(
            schema_to_ir(
                r##"{
                    "$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"additionalProperties":false}},
                    "$ref":"#/$defs/node"
                }"##,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        let mut state = StructuredState::new(&plan);
        for &byte in br#"{"next":{"next":"# {
            state.try_push_byte(byte).unwrap();
        }
        let warm = state.checkpoint();
        state.try_push_byte(b'{').unwrap();
        state.try_push_byte(b'}').unwrap();
        state.rollback(warm);
        let retained = state.retained_bytes();
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..10_000 {
            let mark = state.checkpoint();
            state.try_push_byte(b'{').unwrap();
            state.try_push_byte(b'}').unwrap();
            state.rollback(mark);
        }
        TRACK.with(|track| track.set(false));
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
        assert_eq!(state.retained_bytes(), retained);
    }

    #[test]
    fn dynamic_reference_resolution_and_rollback_allocate_zero_after_warmup() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let plan = build_resources(
            STRICT_DYNAMIC_TREE,
            "https://example.com/strict-tree",
            &[("https://example.com/tree", DYNAMIC_TREE)],
            StructuredLimits::default(),
        );
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, br#"{"children":["#));

        let warm = state.checkpoint();
        assert!(state.try_push_byte(b'{').unwrap());
        state.rollback(warm);
        let retained = state.retained_bytes();
        begin_tracking();
        for _ in 0..10_000 {
            let mark = state.checkpoint();
            assert!(state.try_push_byte(b'{').unwrap());
            state.rollback(mark);
        }
        assert_eq!(finish_tracking(), 0);
        assert_eq!(state.retained_bytes(), retained);
    }

    #[test]
    fn recursive_reference_trie_masks_allocate_zero_after_warmup() {
        use crate::index::{TrieCache, VocabularyHandle};
        use crate::structured::matcher::{StructuredMatcher, StructuredProgram};
        use crate::vocab::build_vocabulary;
        use rustc_hash::FxHashMap;

        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let schema = Arc::new(
            schema_to_ir(
                r##"{
                    "$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"additionalProperties":false}},
                    "$ref":"#/$defs/node"
                }"##,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let program = StructuredProgram::compile(schema).unwrap();
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [
            (0, b"{".as_slice()),
            (1, b"}"),
            (2, b"\"next\""),
            (3, b":"),
            (4, b"{}"),
            (5, br#"{"next":"#),
            (6, b"null"),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        tokens.insert(b"}".to_vec(), vec![1, 8]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(9, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let mut matcher = StructuredMatcher::new(program).unwrap();
        matcher.advance(br#"{"next":{"next":"#).unwrap();
        let mut output = [0u8; 4];
        for _ in 0..32 {
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                .unwrap();
        }
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..10_000 {
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                .unwrap();
        }
        TRACK.with(|track| track.set(false));
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
        assert!(matcher.advance(b"{}").unwrap());
    }

    #[test]
    fn combinator_repeated_transitions_allocate_zero_after_warmup() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let ir = Arc::new(
            schema_to_ir(
                r#"{"oneOf":[{"type":"string","pattern":"^a+$"},{"type":"string","pattern":"^ab+$"},{"type":"object","additionalProperties":{"type":"integer"}}]}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        let mut state = StructuredState::new(&plan);
        let warm = state.checkpoint();
        assert!(state.try_push_byte(b'"').unwrap());
        state.rollback(warm);
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..10_000 {
            let mark = state.checkpoint();
            assert!(state.try_push_byte(b'"').unwrap());
            state.rollback(mark);
        }
        TRACK.with(|track| track.set(false));
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn negation_transitions_allocate_zero_after_warmup() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        for (schema, prefix, byte) in [
            (r#"{"not":{"const":1}}"#, b"".as_slice(), b'2'),
            (r#"{"not":{"not":{"const":1}}}"#, b"".as_slice(), b'1'),
            (
                r#"{"not":{"type":"object","additionalProperties":{"type":"integer"}}}"#,
                b"".as_slice(),
                b'{',
            ),
            (
                r#"{"if":{"type":"object","required":["kind"],"properties":{"kind":{"const":"a"}},"additionalProperties":true},"then":{"required":["a"]},"else":{"required":["b"]}}"#,
                br#"{"kind":"a","a":1"#,
                b'2',
            ),
        ] {
            let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).unwrap());
            let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
            let mut state = StructuredState::new(&plan);
            assert!(feed(&mut state, prefix), "{schema}");
            let warm = state.checkpoint();
            assert!(state.try_push_byte(byte).unwrap(), "{schema}");
            state.rollback(warm);
            COUNT.store(0, Ordering::Relaxed);
            TRACK.with(|track| track.set(true));
            for _ in 0..10_000 {
                let mark = state.checkpoint();
                assert!(state.try_push_byte(byte).unwrap(), "{schema}");
                state.rollback(mark);
            }
            TRACK.with(|track| track.set(false));
            assert_eq!(COUNT.load(Ordering::Relaxed), 0, "{schema}");
        }
    }

    #[test]
    fn negation_record_and_trie_masks_allocate_zero_after_warmup() {
        use crate::index::{TrieCache, VocabularyHandle};
        use crate::structured::matcher::{StructuredMatcher, StructuredProgram};
        use crate::vocab::build_vocabulary;
        use rustc_hash::FxHashMap;

        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let ir =
            Arc::new(schema_to_ir(r#"{"not":{"const":1}}"#, CompileOptions::default()).unwrap());
        let program = StructuredProgram::compile(ir).unwrap();
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [(0, b"1".as_slice()), (1, b"12"), (2, b"2"), (3, b"[]")] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(4, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let mut matcher = StructuredMatcher::new(program).unwrap();
        let mut output = [0u8; 4];
        for _ in 0..32 {
            matcher
                .write_record_mask_le_bytes_into(
                    vocabulary.mask_vocab_size(),
                    vocabulary.iter_records(),
                    &mut output,
                )
                .unwrap();
        }
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..10_000 {
            matcher
                .write_record_mask_le_bytes_into(
                    vocabulary.mask_vocab_size(),
                    vocabulary.iter_records(),
                    &mut output,
                )
                .unwrap();
        }
        TRACK.with(|track| track.set(false));
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);

        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
            .unwrap();
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..10_000 {
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                .unwrap();
        }
        TRACK.with(|track| track.set(false));
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn combinator_allof_anyof_and_recursive_anyof_transitions_allocate_zero_after_warmup() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let cases: &[(&str, &[u8], u8)] = &[
            (
                r#"{"allOf":[{"type":"object","additionalProperties":{"type":"integer"}},{"type":"object","patternProperties":{"^x":{"type":"integer"}},"additionalProperties":true}]}"#,
                b"",
                b'{',
            ),
            (
                r#"{"anyOf":[{"type":"object","additionalProperties":{"type":"integer"}},{"type":"array","uniqueItems":true},{"type":"string","pattern":"^a+$"},{"type":"string","pattern":"^ab+$"}]}"#,
                br#""a"#,
                b'a',
            ),
            (
                r##"{"$defs":{"node":{"anyOf":[{"type":"null"},{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"required":["next"],"additionalProperties":false}]}},"$ref":"#/$defs/node"}"##,
                br#"{"next":n"#,
                b'u',
            ),
        ];
        for (schema, prefix, byte) in cases {
            let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).unwrap());
            let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
            let mut state = StructuredState::new(&plan);
            assert!(feed(&mut state, prefix), "{schema}");
            state.commit_checkpoint().unwrap();
            let warm = state.checkpoint();
            assert!(state.try_push_byte(*byte).unwrap(), "{schema}");
            state.rollback(warm);
            COUNT.store(0, Ordering::Relaxed);
            TRACK.with(|track| track.set(true));
            for _ in 0..10_000 {
                let mark = state.checkpoint();
                assert!(state.try_push_byte(*byte).unwrap(), "{schema}");
                state.rollback(mark);
            }
            TRACK.with(|track| track.set(false));
            assert_eq!(COUNT.load(Ordering::Relaxed), 0, "{schema}");
        }
    }

    #[test]
    fn combinator_repeated_trie_masks_allocate_zero_after_warmup() {
        use crate::index::{TrieCache, VocabularyHandle};
        use crate::structured::matcher::{StructuredMatcher, StructuredProgram};
        use crate::vocab::build_vocabulary;
        use rustc_hash::FxHashMap;

        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let ir = Arc::new(
            schema_to_ir(
                r#"{"oneOf":[{"type":"string","pattern":"^a+$"},{"type":"string","pattern":"^ab+$"},{"type":"object","additionalProperties":{"type":"integer"}}]}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let program = StructuredProgram::compile(ir).unwrap();
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [
            (0, b"\"a\"".as_slice()),
            (1, b"\"ab\""),
            (2, br#"{"x":1}"#),
            (3, br#""x""#),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(4, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let mut matcher = StructuredMatcher::new(program).unwrap();
        let mut output = [0u8; 4];
        for _ in 0..32 {
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                .unwrap();
        }
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..10_000 {
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                .unwrap();
        }
        TRACK.with(|track| track.set(false));
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn combinator_allof_anyof_and_recursive_anyof_trie_masks_allocate_zero_after_warmup() {
        use crate::index::{TrieCache, VocabularyHandle};
        use crate::structured::matcher::{StructuredMatcher, StructuredProgram};
        use crate::vocab::build_vocabulary;
        use rustc_hash::FxHashMap;

        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let cases: &[(&str, &[u8])] = &[
            (
                r#"{"allOf":[{"type":"object","additionalProperties":{"type":"integer"}},{"type":"object","patternProperties":{"^x":{"type":"integer"}},"additionalProperties":true}]}"#,
                b"",
            ),
            (
                r#"{"anyOf":[{"type":"object","additionalProperties":{"type":"integer"}},{"type":"array","uniqueItems":true},{"type":"string","pattern":"^a+$"},{"type":"string","pattern":"^ab+$"}]}"#,
                br#""a"#,
            ),
            (
                r##"{"$defs":{"node":{"anyOf":[{"type":"null"},{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"required":["next"],"additionalProperties":false}]}},"$ref":"#/$defs/node"}"##,
                br#"{"next":n"#,
            ),
        ];
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [
            (0, b"null".as_slice()),
            (1, b"{"),
            (2, b"}"),
            (3, b"["),
            (4, b"]"),
            (5, br#""a""#),
            (6, b"1"),
            (7, b","),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(8, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        for (schema, prefix) in cases {
            let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).unwrap());
            let program = StructuredProgram::compile(ir).unwrap();
            assert!(program.uses_incremental_backend(), "{schema}");
            let mut matcher = StructuredMatcher::new(program).unwrap();
            if !prefix.is_empty() {
                assert!(matcher.advance(prefix).unwrap(), "{schema}");
            }
            let mut output = [0u8; 4];
            for _ in 0..32 {
                matcher
                    .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                    .unwrap();
            }
            COUNT.store(0, Ordering::Relaxed);
            TRACK.with(|track| track.set(true));
            for _ in 0..10_000 {
                matcher
                    .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                    .unwrap();
            }
            TRACK.with(|track| track.set(false));
            assert_eq!(COUNT.load(Ordering::Relaxed), 0, "{schema}");
        }
    }

    #[test]
    fn a_tiny_plan_budget_rejects_before_allocating_property_plans_and_regex() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        // A schema whose full plan needs many KB: 128 named string properties and their plan arrays.
        let mut props = String::new();
        for i in 0..128 {
            if i > 0 {
                props.push(',');
            }
            props.push_str(&format!(r#""property_number_{i}":{{"type":"string"}}"#));
        }
        let schema = format!(r#"{{"type":"object","properties":{{{props}}}}}"#);
        let ir = Arc::new(schema_to_ir(&schema, CompileOptions::default()).unwrap());

        // A full compile genuinely allocates many KB: property name strings and plan arrays.
        let generous = StructuredLimits {
            max_plan_bytes: 64 * 1024 * 1024,
            ..StructuredLimits::default()
        };
        let base = LIVE.load(Ordering::Relaxed);
        PEAK.store(base, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        assert!(StructuredPlan::compile(ir.clone(), generous).is_ok());
        TRACK.with(|track| track.set(false));
        let full_peak = PEAK.load(Ordering::Relaxed) - base;
        assert!(
            full_peak > 2048,
            "full compile peak {full_peak} too small to be a real test"
        );

        // A one-byte budget must reject WITHOUT ever reaching that peak: admission precedes allocation.
        let limits = StructuredLimits {
            max_plan_bytes: 1,
            ..StructuredLimits::default()
        };
        let base = LIVE.load(Ordering::Relaxed);
        PEAK.store(base, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        let tiny = StructuredPlan::compile(ir, limits);
        TRACK.with(|track| track.set(false));
        let tiny_peak = PEAK.load(Ordering::Relaxed) - base;
        assert!(tiny.is_err(), "tiny plan budget must reject");
        assert!(
            tiny_peak.saturating_mul(8) < full_peak,
            "tiny-budget compile peaked {tiny_peak} vs full {full_peak}: it allocated the large plan before rejecting"
        );
    }

    #[test]
    fn unevaluated_mid_number_value_bytes_allocate_zero_after_warmup() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let schema = r#"{"unevaluatedProperties":{"type":"integer"}}"#;
        let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).unwrap());
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        let mut state = StructuredState::new(&plan);
        for &b in br#"{"a":1234567"# {
            state.try_push_byte(b).unwrap();
        }
        let warm_mark = state.checkpoint();
        state.try_push_byte(b'8').unwrap();
        state.rollback(warm_mark);
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for i in 0..5000u32 {
            let digit = b'0' + (i % 10) as u8;
            let mark = state.checkpoint();
            state.try_push_byte(digit).unwrap();
            state.rollback(mark);
        }
        TRACK.with(|track| track.set(false));
        let allocs = COUNT.load(Ordering::Relaxed);
        assert_eq!(
            allocs, 0,
            "unevaluatedProperties mid-number-value bytes allocated {allocs} times over 5000 transitions"
        );
    }

    #[test]
    fn unevaluated_mid_escaped_string_bytes_allocate_zero_after_warmup() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let schema = r#"{"unevaluatedProperties":{"type":"string"}}"#;
        let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).unwrap());
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        let mut state = StructuredState::new(&plan);
        let prefix = "{\"a\":\"caf\\u00e9".to_string();
        for &b in prefix.as_bytes() {
            state.try_push_byte(b).unwrap();
        }
        let warm_mark = state.checkpoint();
        state.try_push_byte(b'0').unwrap();
        state.rollback(warm_mark);
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for i in 0..5000u32 {
            let hex = b"0123456789abcdef"[(i % 16) as usize];
            let mark = state.checkpoint();
            state.try_push_byte(hex).unwrap();
            state.rollback(mark);
        }
        TRACK.with(|track| track.set(false));
        let allocs = COUNT.load(Ordering::Relaxed);
        assert_eq!(
            allocs, 0,
            "unevaluatedProperties mid-escaped-string bytes allocated {allocs} times over 5000 transitions"
        );
    }

    #[test]
    fn unevaluated_insert_into_an_already_allocated_heap_word_allocates_zero() {
        let _guard = GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let schema = r#"{"unevaluatedProperties":{"type":"integer"}}"#;
        let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).unwrap());
        let limits = StructuredLimits {
            max_properties: 300,
            ..StructuredLimits::default()
        };
        let plan = StructuredPlan::compile(ir, limits).unwrap();
        let mut state = StructuredState::new(&plan);
        assert!(state.try_push_byte(b'{').unwrap());
        // Drive candidates directly to isolate annotation-set allocation behavior.
        // Ordinal 128 grows the heap to four words, covering ordinals through 255.
        for ordinal in 0..129u32 {
            state.start_unevaluated_candidate(0, ordinal).unwrap();
            state.finish_unevaluated_candidate(0, true).unwrap();
        }
        state.commit_checkpoint().unwrap();
        // One warm-up of the exact repeated transition primes any one-time lazy cache in the
        // cursor pool itself; only the annotation word's steady-state behavior matters after.
        let warm_mark = state.checkpoint();
        state.start_unevaluated_candidate(0, 129).unwrap();
        state.finish_unevaluated_candidate(0, true).unwrap();
        state.rollback(warm_mark);
        COUNT.store(0, Ordering::Relaxed);
        TRACK.with(|track| track.set(true));
        for _ in 0..5000u32 {
            let mark = state.checkpoint();
            state.start_unevaluated_candidate(0, 129).unwrap();
            state.finish_unevaluated_candidate(0, true).unwrap();
            state.rollback(mark);
        }
        TRACK.with(|track| track.set(false));
        let allocs = COUNT.load(Ordering::Relaxed);
        assert_eq!(
            allocs, 0,
            "inserting into an already-allocated annotation heap word allocated {allocs} times over 5000 speculative candidates"
        );
    }
}

#[test]
fn a_push_child_rejected_at_the_budget_boundary_never_bricks_the_session() {
    let schema =
        r#"{"type":"array","items":{"type":"object","properties":{"a":{"type":"object"}}}}"#;
    for budget in 200..1400usize {
        let ir =
            Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
        let limits = StructuredLimits {
            max_session_bytes: budget,
            ..StructuredLimits::default()
        };
        let plan = StructuredPlan::compile(ir, limits).unwrap();
        let mut state = StructuredState::new(&plan);
        for &byte in br#"[{"a":{}}]"# {
            if state.try_push_byte(byte).is_err() {
                break;
            }
        }
        assert!(
            !state.session_memory.is_terminal(),
            "budget={budget} bricked the session on a clean rejection"
        );
    }
}

#[test]
fn any_object_free_capacity_matches_the_arena_so_commit_never_allocates() {
    let plan = build(r#"{"type":"array","items":true}"#);
    let mut state = StructuredState::new(&plan);
    for &byte in br#"[{"a":{"b":{"c":{}}}}]"# {
        state.try_push_byte(byte).unwrap();
    }
    assert!(state.any_object_arena.capacity() >= 2);
    assert!(state.any_object_free.capacity() >= state.any_object_arena.capacity());
    assert!(!state.session_memory.is_terminal());
}

#[test]
fn repeated_nested_object_churn_never_leaks_session_bytes_or_goes_terminal() {
    // An exact ledger releases on rollback exactly what it charged, so once capacity is warm the
    // live byte count is flat across identical cycles; a per-cycle leak would grow it monotonically.
    let plan = build(
        r#"{"type":"object","additionalProperties":{"type":"object","additionalProperties":{"type":"boolean"}}}"#,
    );
    let mut state = StructuredState::new(&plan);
    let doc = br#"{"alpha":{"x":true,"y":false},"beta":{"z":true}}"#;
    let mut samples = Vec::new();
    for _ in 0..64 {
        let mark = state.checkpoint();
        for &byte in doc {
            state.try_push_byte(byte).unwrap();
        }
        state.rollback(mark);
        assert!(!state.session_memory.is_terminal());
        samples.push(state.session_memory.live());
    }
    // Cycle 0 warms capacity; every later cycle must land on the identical steady-state charge.
    assert_eq!(samples[1], samples[63], "session bytes leaked across churn");
}

#[test]
fn five_or_more_obligations_do_not_clone_or_leak_on_commit() {
    for count in [5, 8] {
        let schema = overlapping_object_schema(count);
        let plan = build(&schema);
        let mut state = StructuredState::new(&plan);
        assert_exact_session_ledger(&state);
        assert!(feed(&mut state, br#"{"x":"#));
        let resolved = state
            .object_ref(0)
            .resolved_value
            .as_ref()
            .expect("colon resolves the property");
        assert!(
            obligation_owned_bytes(resolved)
                >= (count - MAX_INLINE_OBLIGATIONS) * size_of::<NodeId>()
        );
        assert_exact_session_ledger(&state);
        assert!(feed(&mut state, b"50"));
        assert!(state.object_ref(0).resolved_value.is_none());
        assert!(state.undo.iter().any(|undo| matches!(
            undo,
            Undo::RestoreResolvedValue { old: Some(value), .. }
                if obligation_owned_bytes(value) > 0
        )));
        assert_exact_session_ledger(&state);
        state.commit_checkpoint().unwrap();
        assert!(!state.undo.iter().any(|undo| matches!(
            undo,
            Undo::RestoreResolvedValue { old: Some(value), .. }
                if obligation_owned_bytes(value) > 0
        )));
        assert_exact_session_ledger(&state);
        assert!(feed(&mut state, b"}"));
        state.commit_checkpoint().unwrap();
        assert_exact_session_ledger(&state);
    }
}

#[test]
fn obligation_resolution_limit_failure_is_atomic() {
    let schema = overlapping_object_schema(8);
    let plan = build(&schema);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"x""#));
    state.commit_checkpoint().unwrap();
    assert_exact_session_ledger(&state);
    let before = full_snapshot(&state);
    let live = state.session_memory.live();
    let slots = state.cursors.slots.clone();
    let epochs = state.cursors.epochs.clone();
    state.limits.max_session_bytes = state.retained_bytes();
    let error = state.try_push_byte(b':').unwrap_err();
    assert_eq!(error.kind, LimitKind::SessionBytes);
    assert_eq!(full_snapshot(&state), before);
    assert_eq!(state.session_memory.live(), live);
    assert_eq!(state.cursors.slots, slots);
    assert_eq!(state.cursors.epochs, epochs);
    assert!(!state.session_memory.is_terminal());
    assert_exact_session_ledger(&state);
    state.limits.max_session_bytes = usize::MAX;
    assert!(state.try_push_byte(b':').unwrap());
    assert_exact_session_ledger(&state);
}

#[test]
fn any_object_key_commit_releases_discarded_buffer_charge() {
    let plan = build(r#"{"type":"object","additionalProperties":true}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, br#"{"outer":{"a_very_long_key":1,"b":2"#));
    assert_exact_session_ledger(&state);
    state.commit_checkpoint().unwrap();
    assert_exact_session_ledger(&state);

    let (depth, id) = state
        .frames
        .iter()
        .enumerate()
        .find_map(|(depth, frame)| match frame {
            Frame::Any(AnyFrame::Object(id)) => Some((depth, *id)),
            _ => None,
        })
        .expect("expected active any-object");
    let index = usize::try_from(id.0).unwrap();
    let mut old_key = String::new();
    old_key.try_reserve_exact(3).unwrap();
    let old_capacity = old_key.capacity();
    state
        .session_memory
        .charge(old_capacity, usize::MAX, LimitKind::SessionBytes)
        .unwrap();
    let spare_before = state.any_object_arena[index].spare_key.capacity();
    state.any_object_arena[index]
        .spare_key
        .try_reserve_exact(128)
        .unwrap();
    let spare_capacity = state.any_object_arena[index].spare_key.capacity();
    state
        .session_memory
        .charge(
            spare_capacity - spare_before,
            usize::MAX,
            LimitKind::SessionBytes,
        )
        .unwrap();
    state.reserve_undo(1).unwrap();
    state
        .push_undo(Undo::ResetAnyKey {
            depth,
            old_key,
            old_generation: 0,
            old_decoder: JsonStringDecoder::new(),
        })
        .unwrap();
    state.commit_checkpoint().unwrap();
    assert_eq!(
        state.any_object_arena[index].spare_key.capacity(),
        spare_capacity
    );
    assert_exact_session_ledger(&state);
    assert!(!state.session_memory.is_terminal());

    let mut old_key = String::new();
    old_key.try_reserve_exact(256).unwrap();
    let old_capacity = old_key.capacity();
    state
        .session_memory
        .charge(old_capacity, usize::MAX, LimitKind::SessionBytes)
        .unwrap();
    state.reserve_undo(1).unwrap();
    state
        .push_undo(Undo::ResetAnyKey {
            depth,
            old_key,
            old_generation: 0,
            old_decoder: JsonStringDecoder::new(),
        })
        .unwrap();
    state.commit_checkpoint().unwrap();
    assert_eq!(
        state.any_object_arena[index].spare_key.capacity(),
        old_capacity
    );
    assert_exact_session_ledger(&state);
    assert!(!state.session_memory.is_terminal());
}

#[test]
fn initial_unique_array_charge_uses_real_capacity() {
    let plan = build(r#"{"type":"array","uniqueItems":true}"#);
    let state = StructuredState::new(&plan);
    let Frame::Array(array) = &state.frames[0] else {
        panic!("expected array frame");
    };
    assert_eq!(
        array_owned_charge(array),
        array.canonical.as_ref().unwrap().capacity() * size_of::<CanonicalSet>()
    );
    assert_exact_session_ledger(&state);
}

#[test]
fn a_late_push_child_failure_leaves_logical_state_and_ledger_intact() {
    // At a budget boundary the frame vector can grow (retained) yet the child's own charge fails;
    // that must roll back to an unchanged logical snapshot without bricking the session.
    let schema =
        r#"{"type":"array","items":{"type":"object","properties":{"a":{"type":"object"}}}}"#;
    for budget in 200..1500usize {
        let ir =
            Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
        let limits = StructuredLimits {
            max_session_bytes: budget,
            ..StructuredLimits::default()
        };
        let plan = StructuredPlan::compile(ir, limits).unwrap();
        let mut state = StructuredState::new(&plan);
        for &byte in br#"[{"a":{}}]"# {
            let before = full_snapshot(&state);
            if state.try_push_byte(byte).is_err() {
                assert_eq!(
                    full_snapshot(&state),
                    before,
                    "budget={budget}: byte mutated state"
                );
                assert!(
                    !state.session_memory.is_terminal(),
                    "budget={budget}: bricked"
                );
                let retained = state.retained_bytes();
                let _ = state.try_push_byte(byte);
                assert_eq!(
                    state.retained_bytes(),
                    retained,
                    "budget={budget}: retry grew"
                );
                break;
            }
        }
    }
}

#[test]
fn large_closed_objects_route_to_structured_and_reject_unknown_properties() {
    use crate::ir::CompileOptions;
    // A closed object with more fields than the byte-DFA shuffle can unroll must still be validated
    // incrementally: known properties accepted, unknown ones rejected by additionalProperties:false.
    for n in [8usize, 16, 64] {
        let mut props = String::new();
        for i in 0..n {
            if i > 0 {
                props.push(',');
            }
            props.push_str(&format!(r#""field_{i}":{{"type":"boolean"}}"#));
        }
        let schema =
            format!(r#"{{"type":"object","properties":{{{props}}},"additionalProperties":false}}"#);
        let ir = crate::frontend::schema_to_ir(&schema, CompileOptions::default()).unwrap();
        assert!(
            ir.requires_structured_backend(),
            "n={n} should need the structured backend"
        );
        assert!(
            crate::structured::accepts(&ir, br#"{"field_0":true,"field_1":false}"#),
            "n={n} rejected a valid document"
        );
        assert!(
            !crate::structured::accepts(&ir, br#"{"field_0":true,"unknown_key":true}"#),
            "n={n} accepted an unknown property under additionalProperties:false"
        );
        assert!(
            !crate::structured::accepts(&ir, br#"{"field_0":123}"#),
            "n={n} accepted a wrong value type"
        );
    }
}

#[test]
/// Every boundary candidate, and every ordered pair of them, must leave a committed [0 array
/// item's logical state and canonical ledger byte-for-byte unchanged after speculative rollback.
fn uniqueitems_boundary_candidates_and_pairs_leave_state_unchanged() {
    let plan = build(r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true}"#);
    let candidates: &[&[u8]] = &[b" ", b",", b"]", b" ,", b" ]", b",1", b",1]", b"x"];
    for &cand in candidates {
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, b"[0"));
        let baseline = full_snapshot(&state);
        let mark = state.checkpoint();
        for &b in cand {
            let _ = state.try_push_byte(b);
        }
        state.rollback(mark);
        let after = full_snapshot(&state);
        assert_eq!(
            after, baseline,
            "single candidate {cand:?} corrupted state on rollback"
        );
    }
    for &first in candidates {
        for &second in candidates {
            let mut state = StructuredState::new(&plan);
            assert!(feed(&mut state, b"[0"));
            let baseline = full_snapshot(&state);
            let mark1 = state.checkpoint();
            for &b in first {
                let _ = state.try_push_byte(b);
            }
            state.rollback(mark1);
            let mark2 = state.checkpoint();
            for &b in second {
                let _ = state.try_push_byte(b);
            }
            state.rollback(mark2);
            let after = full_snapshot(&state);
            assert_eq!(
                after, baseline,
                "pair ({first:?}, {second:?}) corrupted state on rollback"
            );
        }
    }
}

#[test]
fn repeated_uniqueitems_speculative_rollback_never_drifts_logical_state_or_ledger() {
    let plan = build(r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true}"#);
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"[0"));
    let baseline = full_snapshot(&state);
    for _ in 0..1000 {
        let mark = state.checkpoint();
        let _ = state.try_push_byte(b' ');
        state.rollback(mark);
        assert_exact_session_ledger(&state);
    }
    assert_eq!(full_snapshot(&state), baseline);
}

#[test]
fn contains_count_survives_speculative_rollback_across_boundary_candidates() {
    let plan = build(
        r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true,"contains":{"type":"integer","minimum":0}}"#,
    );
    let mut state = StructuredState::new(&plan);
    assert!(feed(&mut state, b"[0"));
    let before = state.array_ref(0).contains_count;
    assert_eq!(before, 1);
    for cand in [b" ".as_slice(), b",", b"]", b",1", b",1]", b"x"] {
        let mark = state.checkpoint();
        for &b in cand {
            let _ = state.try_push_byte(b);
        }
        state.rollback(mark);
        assert_eq!(state.array_ref(0).contains_count, before, "{cand:?}");
    }
    assert!(feed(&mut state, b","));
    assert_eq!(state.array_ref(0).contains_count, before);
}

#[test]
fn uniqueitems_survives_speculative_rollback_for_strings_nested_arrays_and_objects() {
    let cases: &[(&str, &[u8])] = &[
        (
            r#"{"type":"array","items":{"type":"string"},"uniqueItems":true}"#,
            br#"["ab""#,
        ),
        (
            r#"{"type":"array","items":{"type":"array"},"uniqueItems":true}"#,
            br#"[[1,2]"#,
        ),
        (
            r#"{"type":"array","items":{"type":"object"},"uniqueItems":true}"#,
            br#"[{"a":1}"#,
        ),
    ];
    for &(schema, prefix) in cases {
        let plan = build(schema);
        let mut state = StructuredState::new(&plan);
        assert!(feed(&mut state, prefix), "{schema}");
        let baseline = full_snapshot(&state);
        for cand in [b" ".as_slice(), b"  ", b"\t]", b",", b"x"] {
            let mark = state.checkpoint();
            for &b in cand {
                let _ = state.try_push_byte(b);
            }
            state.rollback(mark);
            assert_eq!(
                full_snapshot(&state),
                baseline,
                "{schema} candidate {cand:?}"
            );
        }
    }
}

#[test]
fn uniqueitems_budget_exhaustion_mid_candidate_is_atomic() {
    let schema = r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true}"#;
    let ir = Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
    for budget in [400usize, 600, 900, 1400] {
        let limits = StructuredLimits {
            max_session_bytes: budget,
            ..StructuredLimits::default()
        };
        let plan = StructuredPlan::compile(ir.clone(), limits).unwrap();
        let mut state = StructuredState::new(&plan);
        for &byte in b"[0" {
            if state.try_push_byte(byte).is_err() {
                break;
            }
        }
        assert!(!state.session_memory.is_terminal(), "budget={budget}");
        let baseline = full_snapshot(&state);
        let mark = state.checkpoint();
        for &b in b",111111111111111111111111" {
            if state.try_push_byte(b).is_err() {
                break;
            }
        }
        state.rollback(mark);
        assert_eq!(full_snapshot(&state), baseline, "budget={budget}");
        assert!(!state.session_memory.is_terminal(), "budget={budget}");
    }
}
