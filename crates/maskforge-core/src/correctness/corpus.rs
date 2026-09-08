//! Defines correctness cases from official and synthetic sources.

use super::reference_engine::build_reference;
use super::schema_skeleton::{KeyOrder, ScalarLit, SchemaIR};
use crate::automaton::RefEngine;
use crate::error::CompileError;

#[derive(Copy, Clone, Debug)]
pub struct SuiteRef {
    pub file: &'static str,
    pub description: &'static str,
}

#[derive(Copy, Clone, Debug)]
pub struct Sample {
    pub instance: &'static str,
    pub suite_expected: Option<bool>,
}

pub struct CorpusCase {
    pub name: &'static str,
    pub schema_id: &'static str,
    pub json_schema: &'static str,
    pub provenance: Option<SuiteRef>,
    pub engine: RefEngine,
    pub samples: Vec<Sample>,
}

fn syn(instances: &[&'static str]) -> Vec<Sample> {
    instances
        .iter()
        .map(|&instance| Sample {
            instance,
            suite_expected: None,
        })
        .collect()
}

fn suite(pairs: &[(&'static str, bool)]) -> Vec<Sample> {
    pairs
        .iter()
        .map(|&(instance, valid)| Sample {
            instance,
            suite_expected: Some(valid),
        })
        .collect()
}

fn suite_ref(file: &'static str, description: &'static str) -> SuiteRef {
    SuiteRef { file, description }
}

pub fn corpus_cases() -> Result<Vec<CorpusCase>, CompileError> {
    let mut cases = Vec::new();
    let mut push = |name,
                    schema_id,
                    json_schema,
                    provenance,
                    ir: &SchemaIR,
                    samples|
     -> Result<(), CompileError> {
        cases.push(CorpusCase {
            name,
            schema_id,
            json_schema,
            provenance,
            engine: build_reference(ir)?,
            samples,
        });
        Ok(())
    };

    push(
        "suite-type-boolean",
        "type-boolean",
        r#"{"type":"boolean"}"#,
        Some(suite_ref("type.json", "boolean type matches booleans")),
        &SchemaIR::Boolean,
        suite(&[
            ("1", false),
            ("0", false),
            ("1.1", false),
            ("\"foo\"", false),
            ("\"\"", false),
            ("{}", false),
            ("[]", false),
            ("true", true),
            ("false", true),
            ("null", false),
        ]),
    )?;

    push(
        "suite-type-null",
        "type-null",
        r#"{"type":"null"}"#,
        Some(suite_ref(
            "type.json",
            "null type matches only the null object",
        )),
        &SchemaIR::Null,
        suite(&[
            ("1", false),
            ("1.1", false),
            ("0", false),
            ("\"foo\"", false),
            ("\"\"", false),
            ("{}", false),
            ("[]", false),
            ("true", false),
            ("false", false),
            ("null", true),
        ]),
    )?;

    push(
        "suite-const-null",
        "const-null",
        r#"{"const":null}"#,
        Some(suite_ref("const.json", "const with null")),
        &SchemaIR::Null,
        suite(&[("null", true), ("0", false)]),
    )?;

    push(
        "suite-const-false",
        "const-false",
        r#"{"const":false}"#,
        Some(suite_ref("const.json", "const with false does not match 0")),
        &SchemaIR::Enum {
            values: vec![ScalarLit::Bool(false)],
        },
        suite(&[("false", true), ("0", false), ("0.0", false)]),
    )?;

    push(
        "suite-const-true",
        "const-true",
        r#"{"const":true}"#,
        Some(suite_ref("const.json", "const with true does not match 1")),
        &SchemaIR::Enum {
            values: vec![ScalarLit::Bool(true)],
        },
        suite(&[("true", true), ("1", false), ("1.0", false)]),
    )?;

    push(
        "suite-const-nul-char",
        "const-nul-char",
        r#"{"const":"hello\u0000there"}"#,
        Some(suite_ref("const.json", "nul characters in strings")),
        &SchemaIR::StringConst {
            value: "hello\u{0000}there".to_string(),
        },
        suite(&[(r#""hello\u0000there""#, true), (r#""hellothere""#, false)]),
    )?;

    push(
        "suite-enum-simple",
        "enum-simple",
        r#"{"enum":[1,2,3]}"#,
        Some(suite_ref("enum.json", "simple enum validation")),
        &SchemaIR::Enum {
            values: vec![ScalarLit::Int(1), ScalarLit::Int(2), ScalarLit::Int(3)],
        },
        suite(&[("1", true), ("4", false)]),
    )?;

    push(
        "suite-enum-with-null",
        "enum-with-null",
        r#"{"enum":[6,null]}"#,
        Some(suite_ref(
            "enum.json",
            "heterogeneous enum-with-null validation",
        )),
        &SchemaIR::Enum {
            values: vec![ScalarLit::Int(6), ScalarLit::Null],
        },
        suite(&[("null", true), ("6", true), ("\"test\"", false)]),
    )?;

    push(
        "suite-enum-escaped",
        "enum-escaped",
        r#"{"enum":["foo\nbar","foo\rbar"]}"#,
        Some(suite_ref("enum.json", "enum with escaped characters")),
        &SchemaIR::Enum {
            values: vec![
                ScalarLit::Str("foo\nbar".to_string()),
                ScalarLit::Str("foo\rbar".to_string()),
            ],
        },
        suite(&[
            (r#""foo\nbar""#, true),
            (r#""foo\rbar""#, true),
            (r#""abc""#, false),
        ]),
    )?;

    push(
        "syn-string-pattern",
        "syn-string-pattern",
        r#"{"type":"string","pattern":"^(cat|car|carbon)$"}"#,
        None,
        &SchemaIR::StringPattern {
            regex: "(cat|car|carbon)".to_string(),
        },
        syn(&[
            "\"car\"",
            "\"cat\"",
            "\"carbon\"",
            "\"dog\"",
            "\"ca\"",
            "\"carx\"",
            "123",
        ]),
    )?;

    push(
        "syn-enum-prefix",
        "syn-enum-prefix",
        r#"{"enum":[1,12]}"#,
        None,
        &SchemaIR::Enum {
            values: vec![ScalarLit::Int(1), ScalarLit::Int(12)],
        },
        syn(&["1", "12", "2", "123", "\"1\""]),
    )?;

    push(
        "syn-array-bounded",
        "syn-array-bounded",
        r#"{"type":"array","items":{"type":"boolean"},"minItems":1,"maxItems":2}"#,
        None,
        &SchemaIR::Array {
            items: Box::new(SchemaIR::Boolean),
            min: 1,
            max: Some(2),
        },
        syn(&[
            "[true]",
            "[true,false]",
            "[]",
            "[true,false,true]",
            "[1]",
            "true",
        ]),
    )?;

    push(
        "syn-array-unbounded",
        "syn-array-unbounded",
        r#"{"type":"array","items":{"type":"null"},"minItems":0}"#,
        None,
        &SchemaIR::Array {
            items: Box::new(SchemaIR::Null),
            min: 0,
            max: None,
        },
        syn(&["[]", "[null]", "[null,null]", "[true]", "[null,1]"]),
    )?;

    push(
        "syn-object-nested",
        "syn-object-nested",
        r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"array","items":{"type":"null"},"minItems":0}},"required":["a","b"],"additionalProperties":false}"#,
        None,
        &SchemaIR::Object {
            fields: vec![
                ("a".to_string(), SchemaIR::Boolean),
                (
                    "b".to_string(),
                    SchemaIR::Array {
                        items: Box::new(SchemaIR::Null),
                        min: 0,
                        max: None,
                    },
                ),
            ],
            order: KeyOrder::AsDeclared,
        },
        syn(&[
            r#"{"a":true,"b":[null,null]}"#,
            r#"{"a":false,"b":[]}"#,
            r#"{"a":true}"#,
            r#"{"a":1,"b":[]}"#,
            r#"{"a":true,"b":[null],"c":1}"#,
        ]),
    )?;

    Ok(cases)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TYPE_JSON: &str =
        include_str!("../../tools/json-schema-test-suite/draft2020-12/type.json");
    const CONST_JSON: &str =
        include_str!("../../tools/json-schema-test-suite/draft2020-12/const.json");
    const ENUM_JSON: &str =
        include_str!("../../tools/json-schema-test-suite/draft2020-12/enum.json");

    fn vendored_file(name: &str) -> &'static str {
        match name {
            "type.json" => TYPE_JSON,
            "const.json" => CONST_JSON,
            "enum.json" => ENUM_JSON,
            other => panic!("unvendored suite file: {other}"),
        }
    }

    fn vendored_group(file: &str, description: &str) -> serde_json::Value {
        let groups: Vec<serde_json::Value> =
            serde_json::from_str(vendored_file(file)).expect("vendored suite file parses");
        groups
            .into_iter()
            .find(|g| g["description"] == description)
            .unwrap_or_else(|| panic!("group {description:?} not found in {file}"))
    }

    #[test]
    fn embedded_suite_cases_match_the_vendored_files_exactly() {
        for case in corpus_cases().expect("corpus") {
            let Some(prov) = case.provenance else {
                continue;
            };
            let group = vendored_group(prov.file, prov.description);

            let mut want_schema: serde_json::Value = group["schema"].clone();
            want_schema.as_object_mut().unwrap().remove("$schema");
            let got_schema: serde_json::Value =
                serde_json::from_str(case.json_schema).expect("embedded schema parses");
            assert_eq!(got_schema, want_schema, "{}: schema drifted", case.name);

            let want_tests = group["tests"].as_array().expect("tests array");
            assert_eq!(
                case.samples.len(),
                want_tests.len(),
                "{}: sample count drifted from the vendored group",
                case.name
            );
            for (sample, test) in case.samples.iter().zip(want_tests) {
                let got_instance: serde_json::Value =
                    serde_json::from_str(sample.instance).expect("embedded instance parses");
                assert_eq!(
                    &got_instance, &test["data"],
                    "{}: instance drifted",
                    case.name
                );
                assert_eq!(
                    sample.suite_expected,
                    test["valid"].as_bool(),
                    "{}: expected verdict drifted",
                    case.name
                );
            }
        }
    }

    #[test]
    fn corpus_builds_with_suite_and_synthetic_cases() {
        let cases = corpus_cases().expect("corpus builds");
        let suite = cases.iter().filter(|c| c.provenance.is_some()).count();
        let synthetic = cases.iter().filter(|c| c.provenance.is_none()).count();
        assert_eq!(suite, 9);
        assert_eq!(synthetic, 5);
        for c in &cases {
            assert_eq!(c.engine.dead().get(), 0);
            assert!(!c.samples.is_empty());
        }
    }

    #[test]
    fn every_corpus_engine_satisfies_the_three_predicate_law() {
        for case in corpus_cases().expect("corpus") {
            let e = &case.engine;
            assert!(e.is_dead(e.dead()), "{}: DEAD must be dead", case.name);
            assert!(
                !e.is_accepting(e.dead()),
                "{}: DEAD must not accept",
                case.name
            );
            assert!(
                !e.can_continue(e.dead()),
                "{}: DEAD must not continue",
                case.name
            );
            for raw in 0..e.state_count() {
                let s = crate::primitives::StateId::try_from(raw).unwrap();
                assert_eq!(
                    e.is_dead(s),
                    !e.is_accepting(s) && !e.can_continue(s),
                    "{}: state {raw} breaks is_dead == !accept && !continue",
                    case.name
                );
            }
        }
    }

    #[test]
    fn engine_matches_official_suite_ground_truth_offline() {
        for case in corpus_cases().expect("corpus") {
            for s in &case.samples {
                if let Some(expected) = s.suite_expected {
                    assert_eq!(
                        case.engine.accepts(s.instance.as_bytes()),
                        expected,
                        "{} instance {:?} disagreed with suite",
                        case.name,
                        s.instance
                    );
                }
            }
        }
    }
}
