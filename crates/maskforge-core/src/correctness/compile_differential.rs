//! Compares production and reference compiler output for language equivalence.

use std::{collections::VecDeque, sync::Arc};

use rustc_hash::FxHashSet;

use super::ir_reference::reference_compile;
use crate::automaton::RefEngine;
use crate::compile::compile_ir;
use crate::frontend::schema_to_ir;
use crate::ir::{Builder, CompileOptions, ObjectClosure};

fn equivalent(a: &RefEngine, b: &RefEngine) -> bool {
    let mut seen: FxHashSet<(u32, u32)> = FxHashSet::default();
    let mut queue = VecDeque::new();
    let start = (a.start(), b.start());
    seen.insert((start.0.get(), start.1.get()));
    queue.push_back(start);
    while let Some((sa, sb)) = queue.pop_front() {
        if a.is_accepting(sa) != b.is_accepting(sb)
            || a.is_dead(sa) != b.is_dead(sb)
            || a.can_continue(sa) != b.can_continue(sb)
        {
            return false;
        }
        for byte in 0u8..=255 {
            let na = a.consume_token(sa, &[byte]).unwrap_or(a.dead());
            let nb = b.consume_token(sb, &[byte]).unwrap_or(b.dead());
            if seen.insert((na.get(), nb.get())) {
                queue.push_back((na, nb));
            }
        }
    }
    true
}

fn closed_options() -> CompileOptions {
    CompileOptions {
        object_closure: ObjectClosure::AssumeClosedProfile,
        ..CompileOptions::default()
    }
}

fn assert_paths_agree(schema: &str, options: CompileOptions) {
    let ir = schema_to_ir(schema, options).expect("frontend");
    assert_eq!(
        ir.diagnostics().len(),
        0,
        "schema should be fully supported: {schema}"
    );
    if ir.requires_structured_backend() {
        const DOCUMENTS: &[&[u8]] = &[
            b"null",
            b"true",
            b"false",
            b"-12",
            b"0",
            b"1",
            b"1.0",
            b"10e-1",
            b"12",
            b"123",
            br#""hello""#,
            br#""x""#,
            b"[]",
            b"[true]",
            b"{}",
            br#"{"foo":"bar","baz":"bax"}"#,
        ];
        for document in DOCUMENTS {
            let production = crate::structured::try_accepts(Arc::new(ir.clone()), document)
                .expect("structured production evaluation");
            let oracle = crate::structured::oracle_accepts(&ir, document);
            assert_eq!(
                production, oracle,
                "reference and production disagree for {schema} at {document:?}"
            );
        }
        return;
    }
    let path_a = reference_compile(&ir).expect("reference lowering");
    let path_b = compile_ir(&ir).expect("production compile");
    crate::automaton::assert_exact_equivalence(&path_a, &path_b);
    assert!(
        equivalent(&path_a, &path_b),
        "reference and production disagree for {schema}"
    );
}

const SUPPORTED: &[&str] = &[
    r#"{"type":"null"}"#,
    r#"{"type":"boolean"}"#,
    r#"{"type":"integer"}"#,
    r#"{"type":"integer","minimum":-5,"maximum":42}"#,
    r#"{"type":"integer","exclusiveMinimum":0,"maximum":1000}"#,
    r#"{"type":"integer","minimum":-1000,"maximum":-7}"#,
    r#"{"type":"number"}"#,
    r#"{"type":["number","null"]}"#,
    r#"{"type":"number","minimum":-5,"maximum":42}"#,
    r#"{"type":"number","exclusiveMinimum":0,"maximum":1000}"#,
    r#"{"type":"number","minimum":-1000,"maximum":-7}"#,
    r#"{"type":"number","minimum":-1,"maximum":1}"#,
    r#"{"const":"hello"}"#,
    r#"{"const":7}"#,
    r#"{"const":{"foo":"bar","baz":"bax"}}"#,
    r#"{"const":[{"foo":"bar"}]}"#,
    r#"{"const":{"a":false}}"#,
    r#"{"enum":[1,12,2]}"#,
    r#"{"enum":["a","ab","abc"]}"#,
    r#"{"enum":[true,false,null]}"#,
    r#"{"enum":[{"x":1},{"y":2},[3,4]]}"#,
    r#"{"enum":[6,"foo",[],true,{"foo":12}]}"#,
    r#"{"type":"string"}"#,
    r#"{"type":"string","minLength":2,"maxLength":4}"#,
    r#"{"type":"array","items":{"type":"boolean"}}"#,
    r#"{"type":"array","items":{"type":"integer","minimum":0,"maximum":9},"minItems":1,"maxItems":3}"#,
    r#"{"type":"array","items":{"type":"null"},"maxItems":0}"#,
    r#"{"type":"array","prefixItems":[{"type":"integer"},{"type":"boolean"}],"items":false}"#,
    r#"{"type":"array","prefixItems":[{"type":"boolean"}],"items":{"type":"null"}}"#,
    r#"{"type":"array","prefixItems":[{"type":"integer","minimum":0,"maximum":9},{"type":"null"}],"items":false,"minItems":1}"#,
    r#"{"type":"array","prefixItems":[{"type":"boolean"},{"type":"boolean"}],"items":{"type":"boolean"},"minItems":1,"maxItems":4}"#,
];

const SUPPORTED_CLOSED: &[&str] = &[
    r#"{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"]}"#,
    r#"{"type":"object","properties":{"x":{"type":"integer","minimum":0,"maximum":5},"y":{"type":"null"}}}"#,
    r#"{"type":"object","properties":{"a":{"type":"array","items":{"type":"boolean"}}}}"#,
];

const SUPPORTED_UNIONS: &[&str] = &[
    r#"{"type":["string","null"]}"#,
    r#"{"type":["integer","string","boolean"]}"#,
    r#"{"type":["integer","string"],"enum":[1,"x",true]}"#,
];

const SUPPORTED_ANYOF: &[&str] = &[
    r#"{"anyOf":[{"type":"string"},{"type":"null"}]}"#,
    r#"{"anyOf":[{"type":"boolean"}]}"#,
    r#"{"anyOf":[{"enum":["a","b"]},{"enum":["b","c"]}]}"#,
    r#"{"anyOf":[{"const":"a"},{"const":"b"},{"type":"integer","minimum":0,"maximum":3}]}"#,
    r#"{"anyOf":[{"type":"integer","minimum":0,"maximum":5},{"type":"integer","minimum":3,"maximum":10}]}"#,
];

const SUPPORTED_ALLOF: &[&str] = &[
    r#"{"allOf":[{"type":"null"}]}"#,
    r#"{"allOf":[{"type":"integer","minimum":0,"maximum":10},{"type":"integer","minimum":5,"maximum":20}]}"#,
    r#"{"allOf":[{"enum":["a","b"]},{"enum":["a","b"]}]}"#,
    r#"{"allOf":[{"allOf":[{"type":"boolean"}]}]}"#,
    r#"{"allOf":[{"anyOf":[{"type":"integer","minimum":0,"maximum":10},{"type":"string"}]},{"type":"integer","minimum":5,"maximum":20}]}"#,
    r#"{"allOf":[{"type":"integer","minimum":0,"maximum":5},{"type":"integer","minimum":10,"maximum":20}]}"#,
];

const SUPPORTED_ONEOF: &[&str] = &[
    r#"{"oneOf":[{"type":"boolean"}]}"#,
    r#"{"oneOf":[{"type":"integer","minimum":-100,"maximum":1},{"type":"integer","minimum":2,"maximum":100}]}"#,
    r#"{"oneOf":[{"type":"integer","minimum":0,"maximum":10},{"type":"integer","minimum":5,"maximum":20}]}"#,
    r#"{"oneOf":[{"type":"integer","minimum":0,"maximum":9},{"type":"integer","minimum":0,"maximum":9}]}"#,
    r#"{"oneOf":[{"oneOf":[{"type":"null"}]}]}"#,
];

const SUPPORTED_SIBLING_MERGE: &[&str] = &[
    r#"{"type":"integer","oneOf":[{"type":"integer","minimum":0,"maximum":10},{"type":"integer","minimum":6,"maximum":20}]}"#,
    r#"{"type":"integer","minimum":0,"maximum":100,"allOf":[{"type":"integer","minimum":5,"maximum":50}]}"#,
    r#"{"type":"integer","minimum":0,"maximum":20,"anyOf":[{"type":"integer","minimum":0,"maximum":5},{"type":"integer","minimum":15,"maximum":20}]}"#,
    r#"{"title":"x","description":"y","allOf":[{"type":"boolean"}]}"#,
];

const SUPPORTED_SIBLING_MERGE_CLOSED: &[&str] = &[
    r#"{"type":"object","properties":{"bar":{"type":"integer","minimum":0,"maximum":9}},"required":["bar"],"additionalProperties":false,"allOf":[{"type":"object","properties":{"bar":{"type":"integer","minimum":0,"maximum":9}},"additionalProperties":false}]}"#,
];

#[test]
fn reference_and_production_agree_over_the_supported_dialect() {
    for schema in SUPPORTED {
        assert_paths_agree(schema, CompileOptions::default());
    }
    for schema in SUPPORTED_CLOSED {
        assert_paths_agree(schema, closed_options());
    }
    for schema in SUPPORTED_UNIONS {
        assert_paths_agree(schema, CompileOptions::default());
    }
    for schema in SUPPORTED_ANYOF {
        assert_paths_agree(schema, CompileOptions::default());
    }
    for schema in SUPPORTED_ALLOF {
        assert_paths_agree(schema, CompileOptions::default());
    }
    for schema in SUPPORTED_ONEOF {
        assert_paths_agree(schema, CompileOptions::default());
    }
    for schema in SUPPORTED_SIBLING_MERGE {
        assert_paths_agree(schema, CompileOptions::default());
    }
    for schema in SUPPORTED_SIBLING_MERGE_CLOSED {
        assert_paths_agree(schema, closed_options());
    }
}

#[test]
fn oneof_with_a_base_type_intersects_the_base_and_the_exactly_one() {
    let schema = r#"{"type":"integer","oneOf":[{"type":"integer","minimum":0,"maximum":10},{"type":"integer","minimum":6,"maximum":20}]}"#;
    let ir = schema_to_ir(schema, CompileOptions::default()).unwrap();
    assert_eq!(ir.diagnostics().len(), 0);
    let e = compile_ir(&ir).unwrap();
    assert!(e.accepts(b"5"), "only [0,10] matches");
    assert!(e.accepts(b"18"), "only [6,20] matches");
    assert!(!e.accepts(b"8"), "both branches match: exactly-one fails");
    assert!(
        !e.accepts(b"\"x\""),
        "the base type:integer excludes a string"
    );
}

#[test]
fn number_engine_accepts_the_json_numeric_grammar_and_rejects_malformed() {
    let ir = schema_to_ir(r#"{"type":"number"}"#, CompileOptions::default()).unwrap();
    let e = compile_ir(&ir).unwrap();
    for good in [
        "0", "-0", "42", "-42", "3.14", "-0.5", "1e10", "6.022e23", "1E-9",
    ] {
        assert!(e.accepts(good.as_bytes()), "accept {good}");
    }
    for bad in [
        "", "+1", "01", "1.", ".5", "1e", "1e+", "1..2", "0x1f", "true",
    ] {
        assert!(!e.accepts(bad.as_bytes()), "reject {bad}");
    }
}

#[test]
fn hand_built_ir_equals_the_frontend_ir() {
    let mut b = Builder::new(CompileOptions::default());
    let item = b.boolean().unwrap();
    let arr = b.array(item, 0, None).unwrap();
    let hand = b.finish(arr).unwrap();
    let path_c = compile_ir(&hand).unwrap();

    let ir = schema_to_ir(
        r#"{"type":"array","items":{"type":"boolean"}}"#,
        CompileOptions::default(),
    )
    .unwrap();
    let path_b = compile_ir(&ir).unwrap();
    assert!(equivalent(&path_b, &path_c));
    assert!(ir.same_content_address(&hand));
}

#[test]
fn sample_acceptance_matches_hand_truth() {
    let cases: &[(&str, &[&str], &[&str])] = &[
        (
            r#"{"type":"boolean"}"#,
            &["true", "false"],
            &["tru", "1", "null"],
        ),
        (
            r#"{"type":"integer","minimum":-2,"maximum":10}"#,
            &["-2", "0", "10", "-1"],
            &["-3", "11", "007", "+1", "1.0"],
        ),
        (r#"{"enum":[1,12]}"#, &["1", "12"], &["2", "123", "0"]),
        (
            r#"{"type":"string","pattern":"(cat|car)"}"#,
            &["\"cat\"", "\"car\"", "\"cats\"", "\"xcatx\""],
            &["\"ca\"", "cat", "\"dog\""],
        ),
        (
            r#"{"type":"array","items":{"type":"boolean"},"minItems":1,"maxItems":2}"#,
            &["[true]", "[true,false]"],
            &["[]", "[true,false,true]", "[1]"],
        ),
    ];
    for (schema, valid, invalid) in cases {
        let ir = schema_to_ir(schema, CompileOptions::default()).unwrap();
        let check = |bytes: &[u8]| -> bool {
            if ir.requires_structured_backend() {
                crate::structured::accepts(&ir, bytes)
            } else {
                compile_ir(&ir).unwrap().accepts(bytes)
            }
        };
        for good in *valid {
            assert!(check(good.as_bytes()), "{schema} should accept {good}");
        }
        for bad in *invalid {
            assert!(!check(bad.as_bytes()), "{schema} should reject {bad}");
        }
    }
}

#[test]
fn closed_object_accepts_any_property_order() {
    let schema = r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}},"required":["a"]}"#;
    let ir = schema_to_ir(schema, closed_options()).unwrap();
    let e = compile_ir(&ir).unwrap();
    assert!(e.accepts(br#"{"a":true,"b":null}"#));
    assert!(e.accepts(br#"{"b":null,"a":true}"#));
}

#[test]
fn type_union_branch_order_does_not_change_the_accepted_language() {
    let a = compile_ir(
        &schema_to_ir(r#"{"type":["string","null"]}"#, CompileOptions::default()).unwrap(),
    )
    .unwrap();
    let b = compile_ir(
        &schema_to_ir(r#"{"type":["null","string"]}"#, CompileOptions::default()).unwrap(),
    )
    .unwrap();
    assert!(equivalent(&a, &b));
}

#[test]
fn optional_properties_may_be_omitted_but_required_ones_may_not() {
    let schema = r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"boolean"}},"required":["a"],"additionalProperties":false}"#;
    let ir = schema_to_ir(schema, closed_options()).unwrap();
    let e = compile_ir(&ir).unwrap();
    assert!(e.accepts(br#"{"a":true}"#));
    assert!(e.accepts(br#"{"a":false}"#));
    assert!(e.accepts(br#"{"a":true,"b":true}"#));
    assert!(e.accepts(br#"{"a":true,"b":false}"#));
    assert!(!e.accepts(br#"{"b":true}"#), "required field a is missing");
    assert!(!e.accepts(br#"{}"#), "required field a is missing");
    assert!(
        e.accepts(br#"{"b":true,"a":true}"#),
        "property order is not semantically significant"
    );
}

#[test]
fn optional_property_before_a_required_one_may_be_present_or_absent() {
    let schema = r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"boolean"}},"required":["b"],"additionalProperties":false}"#;
    let ir = schema_to_ir(schema, closed_options()).unwrap();
    let e = compile_ir(&ir).unwrap();
    assert!(e.accepts(br#"{"b":true}"#));
    assert!(e.accepts(br#"{"a":true,"b":true}"#));
    assert!(!e.accepts(br#"{"a":true}"#), "required field b is missing");
}

#[test]
fn three_optional_properties_around_one_required_field_agree_with_the_reference() {
    let schema = r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"boolean"},"c":{"type":"boolean"},"d":{"type":"boolean"}},"required":["b"],"additionalProperties":false}"#;
    assert_paths_agree(schema, closed_options());
    let ir = schema_to_ir(schema, closed_options()).unwrap();
    let e = compile_ir(&ir).unwrap();
    assert!(e.accepts(br#"{"b":true}"#));
    assert!(e.accepts(br#"{"a":true,"b":true,"d":false}"#));
    assert!(e.accepts(br#"{"a":true,"b":true,"c":true,"d":false}"#));
    assert!(!e.accepts(br#"{"a":true,"c":true,"d":false}"#));
}

#[test]
fn all_properties_optional_accepts_every_subset_in_any_order() {
    let schema = r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"boolean"}},"additionalProperties":false}"#;
    assert_paths_agree(schema, closed_options());
    let ir = schema_to_ir(schema, closed_options()).unwrap();
    let e = compile_ir(&ir).unwrap();
    assert!(e.accepts(br#"{}"#));
    assert!(e.accepts(br#"{"a":true}"#));
    assert!(e.accepts(br#"{"b":true}"#));
    assert!(e.accepts(br#"{"a":true,"b":true}"#));
    assert!(
        e.accepts(br#"{"b":true,"a":true}"#),
        "both fields present, reversed order, must also accept"
    );
}

#[test]
fn anti_vacuity_distinct_schemas_are_not_equivalent() {
    let boolean =
        compile_ir(&schema_to_ir(r#"{"type":"boolean"}"#, CompileOptions::default()).unwrap())
            .unwrap();
    let null = compile_ir(&schema_to_ir(r#"{"type":"null"}"#, CompileOptions::default()).unwrap())
        .unwrap();
    assert!(
        !equivalent(&boolean, &null),
        "the differential must be able to fail"
    );
}

#[test]
fn anti_vacuity_a_perturbed_enum_order_would_be_caught() {
    let a = schema_to_ir(r#"{"enum":[1,12]}"#, CompileOptions::default()).unwrap();
    let b = schema_to_ir(r#"{"enum":[1,2]}"#, CompileOptions::default()).unwrap();
    assert!(crate::structured::try_accepts(Arc::new(a), b"12").unwrap());
    assert!(!crate::structured::try_accepts(Arc::new(b), b"12").unwrap());
}

fn independent_integer_membership(bytes: &[u8], lo: Option<i64>, hi: Option<i64>) -> bool {
    let Ok(s) = std::str::from_utf8(bytes) else {
        return false;
    };
    let (negative, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    if digits.len() > 1 && digits.starts_with('0') {
        return false; // a leading zero is not a legal JSON integer token
    }
    if negative && digits == "0" {
        return false; // "-0" is not generated or accepted (see compile::tests)
    }
    let Ok(mut value): Result<i128, _> = digits.parse() else {
        return false;
    };
    if negative {
        value = -value;
    }
    lo.is_none_or(|l| value >= i128::from(l)) && hi.is_none_or(|h| value <= i128::from(h))
}

fn engine_for_integer(lo: Option<i64>, hi: Option<i64>) -> RefEngine {
    let mut b = Builder::new(CompileOptions::default());
    let n = b.integer(lo, hi).unwrap();
    compile_ir(&b.finish(n).unwrap()).unwrap()
}

#[test]
fn compiled_integer_engine_agrees_with_the_independent_membership_oracle() {
    let bounds = [
        (Some(0), Some(0)),
        (Some(0), Some(9)),
        (Some(1), Some(255)),
        (Some(-5), Some(5)),
        (Some(-128), Some(127)),
        (Some(10), Some(1000)),
        (Some(-1000), Some(-7)),
        (Some(-50), Some(50)),
        (Some(99), Some(101)),
        (Some(7), Some(7)),
        (None, Some(3)),
        (Some(-3), None),
        (None, None),
        (Some(i64::MIN), Some(i64::MAX)),
    ];
    for (lo, hi) in bounds {
        let e = engine_for_integer(lo, hi);
        for n in -2000i64..=2000 {
            let text = n.to_string();
            let want = independent_integer_membership(text.as_bytes(), lo, hi);
            assert_eq!(
                e.accepts(text.as_bytes()),
                want,
                "n={n} bounds=({lo:?},{hi:?})"
            );
        }
    }
}

#[test]
fn compiled_integer_engine_rejects_every_malformed_integer_token() {
    let e = engine_for_integer(None, None);
    for bad in ["", "-", "01", "-0", "+1", "1.0", "1e5", "abc", "--1", "1-1"] {
        assert!(
            !independent_integer_membership(bad.as_bytes(), None, None),
            "{bad:?}"
        );
        assert!(!e.accepts(bad.as_bytes()), "{bad:?}");
    }
    for padded in [" 1", "1 "] {
        assert!(
            !independent_integer_membership(padded.as_bytes(), None, None),
            "{padded:?}"
        );
        assert!(e.accepts(padded.as_bytes()), "{padded:?}");
    }
}

#[test]
fn extreme_bounds_agree_with_the_independent_oracle_at_the_boundary() {
    let e = engine_for_integer(Some(i64::MIN), Some(i64::MAX));
    for text in [i64::MIN.to_string(), i64::MAX.to_string(), "0".to_string()] {
        assert!(independent_integer_membership(
            text.as_bytes(),
            Some(i64::MIN),
            Some(i64::MAX)
        ));
        assert!(e.accepts(text.as_bytes()));
    }
}

#[test]
fn enum_triple_shared_prefix_accepts_every_member() {
    let ir = schema_to_ir(r#"{"enum":[1,12,123]}"#, CompileOptions::default()).unwrap();
    let ir = Arc::new(ir);
    assert!(crate::structured::try_accepts(Arc::clone(&ir), b"1").unwrap());
    assert!(crate::structured::try_accepts(Arc::clone(&ir), b"12").unwrap());
    assert!(crate::structured::try_accepts(Arc::clone(&ir), b"123").unwrap());
    assert!(!crate::structured::try_accepts(Arc::clone(&ir), b"1234").unwrap());
    assert!(!crate::structured::try_accepts(ir, b"2").unwrap());
}

fn independent_bool_tuple_membership(
    bytes: &[u8],
    prefix: &[bool],
    tail: Option<bool>,
    min: usize,
    max: Option<usize>,
) -> bool {
    let Ok(s) = std::str::from_utf8(bytes) else {
        return false;
    };
    let Some(inner) = s.strip_prefix('[').and_then(|s| s.strip_suffix(']')) else {
        return false;
    };
    let items: Vec<&str> = if inner.is_empty() {
        Vec::new()
    } else {
        inner.split(',').collect()
    };
    if items.len() < min || max.is_some_and(|m| items.len() > m) {
        return false;
    }
    for (i, item) in items.iter().enumerate() {
        let has_slot = if i < prefix.len() {
            prefix[i]
        } else {
            tail.is_some()
        };
        if !has_slot || !matches!(*item, "true" | "false") {
            return false;
        }
    }
    true
}

#[test]
fn compiled_bool_tuple_agrees_with_the_independent_membership_oracle() {
    let cases: &[(usize, bool, u32, Option<u32>)] = &[
        (2, false, 0, None),
        (3, false, 1, None),
        (1, true, 0, Some(4)),
        (1, true, 2, None),
        (2, true, 0, Some(3)),
    ];
    for &(prefix_len, has_tail, min, max) in cases {
        let mut b = Builder::new(CompileOptions::default());
        let t = b.boolean().unwrap();
        let prefix_ids = vec![t; prefix_len];
        let tail_id = has_tail.then_some(t);
        let node = b.tuple(prefix_ids, tail_id, min, max).unwrap();
        let e = compile_ir(&b.finish(node).unwrap()).unwrap();

        let slot_prefix = vec![true; prefix_len];
        let slot_tail = has_tail.then_some(true);
        for len in 0..=5usize {
            for bits in 0..(1u32 << len) {
                let toks: Vec<&str> = (0..len)
                    .map(|i| {
                        if bits & (1 << i) != 0 {
                            "true"
                        } else {
                            "false"
                        }
                    })
                    .collect();
                let text = format!("[{}]", toks.join(","));
                let want = independent_bool_tuple_membership(
                    text.as_bytes(),
                    &slot_prefix,
                    slot_tail,
                    min as usize,
                    max.map(|m| m as usize),
                );
                assert_eq!(
                    e.accepts(text.as_bytes()),
                    want,
                    "prefix_len={prefix_len} tail={has_tail} min={min} max={max:?} text={text}"
                );
            }
        }
    }
}

#[test]
fn large_enum_set_compiles_and_accepts_every_member() {
    let values: Vec<String> = (0..500).map(|i| i.to_string()).collect();
    let schema = format!(r#"{{"enum":[{}]}}"#, values.join(","));
    let ir = schema_to_ir(&schema, CompileOptions::default()).unwrap();
    let ir = Arc::new(ir);
    for v in [0, 1, 250, 499] {
        assert!(crate::structured::try_accepts(Arc::clone(&ir), v.to_string().as_bytes()).unwrap());
    }
    assert!(!crate::structured::try_accepts(ir, b"500").unwrap());
}
