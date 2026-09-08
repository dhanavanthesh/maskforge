//! Builds the reference engine used by correctness checks.

use super::schema_skeleton::{ScalarLit, SchemaIR};
use crate::automaton::{build_from_regex, RefEngine};
use crate::error::CompileError;

pub(crate) fn build_reference(ir: &SchemaIR) -> Result<RefEngine, CompileError> {
    ir.validate()?;
    let pattern = ir_to_regex(ir)?;
    build_from_regex(&pattern)
}

fn ir_to_regex(ir: &SchemaIR) -> Result<String, CompileError> {
    Ok(match ir {
        SchemaIR::Null => "null".to_string(),
        SchemaIR::Boolean => "(?:true|false)".to_string(),
        SchemaIR::StringConst { value } => escape_regex(&json_string(value)),
        SchemaIR::StringPattern { regex } => format!("\"(?:{regex})\""),
        SchemaIR::Enum { values } => {
            let mut seen = rustc_hash::FxHashSet::default();
            let mut alts: Vec<String> = values
                .iter()
                .filter(|v| seen.insert(*v))
                .map(scalar_to_regex)
                .collect();
            alts.sort_by_key(|a| std::cmp::Reverse(a.len()));
            format!("(?:{})", alts.join("|"))
        }
        SchemaIR::Array { items, min, max } => {
            let element = format!("(?:{})", ir_to_regex(items)?);
            format!("\\[{}\\]", array_body(&element, *min, *max))
        }
        SchemaIR::Object { fields, .. } => {
            let mut parts = Vec::with_capacity(fields.len());
            for (key, sub) in fields {
                parts.push(format!(
                    "{}:{}",
                    escape_regex(&json_string(key)),
                    ir_to_regex(sub)?
                ));
            }
            format!("\\{{{}\\}}", parts.join(","))
        }
    })
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn array_body(element: &str, min: u32, max: Option<u32>) -> String {
    match max {
        None => {
            if min == 0 {
                format!("(?:{element}(?:,{element})*)?")
            } else {
                format!("{element}(?:,{element}){{{},}}", min - 1)
            }
        }
        Some(max) => {
            if max == 0 {
                String::new()
            } else if min == 0 {
                format!("(?:{element}(?:,{element}){{0,{}}})?", max - 1)
            } else {
                format!("{element}(?:,{element}){{{},{}}}", min - 1, max - 1)
            }
        }
    }
}

fn scalar_to_regex(lit: &ScalarLit) -> String {
    match lit {
        ScalarLit::Null => "null".to_string(),
        ScalarLit::Bool(true) => "true".to_string(),
        ScalarLit::Bool(false) => "false".to_string(),
        ScalarLit::Int(i) => i.to_string(),
        ScalarLit::Str(s) => escape_regex(&json_string(s)),
    }
}

fn escape_regex(s: &str) -> String {
    const META: &[char] = &[
        '\\', '.', '+', '*', '?', '(', ')', '[', ']', '{', '}', '^', '$', '|',
    ];
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if META.contains(&c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::mask_width;
    use crate::primitives::{StateId, TokenId};
    use crate::vocab::build_vocabulary;
    use rustc_hash::FxHashMap;

    fn engine(ir: &SchemaIR) -> RefEngine {
        build_reference(ir).expect("build")
    }

    #[test]
    fn boolean_accepts_only_true_or_false() {
        let e = engine(&SchemaIR::Boolean);
        assert!(e.accepts(b"true"));
        assert!(e.accepts(b"false"));
        assert!(!e.accepts(b"tru"));
        assert!(!e.accepts(b"null"));
        assert!(!e.accepts(b"truex"));
    }

    #[test]
    fn anchored_rejects_embedded_valid_value() {
        let e = engine(&SchemaIR::StringPattern {
            regex: "(cat|car|carbon)".to_string(),
        });
        assert!(e.accepts(b"\"car\""));
        assert!(!e.accepts(b"x\"car\"x"));
        assert!(!e.accepts(b"\"ca\""));
    }

    #[test]
    fn three_predicates_are_consistent() {
        let e = engine(&SchemaIR::Boolean);
        assert!(!e.is_accepting(e.dead()));
        assert!(!e.can_continue(e.dead()));
        assert!(e.is_dead(e.dead()));
        let mut cur = e.start();
        for &b in b"true" {
            cur = e.consume_token(cur, &[b]).expect("live");
        }
        assert!(e.is_accepting(cur));
        assert!(!e.is_dead(cur));
    }

    #[test]
    fn dead_is_permanent() {
        let e = engine(&SchemaIR::Boolean);
        assert!(e.consume_token(e.start(), b"zzz").is_none());
        assert!(e.consume_token(e.dead(), b"a").is_none());
    }

    #[test]
    fn allowed_tokens_is_dup_safe_and_excludes_eos() {
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        map.insert(b"t".to_vec(), vec![0]);
        map.insert(b"r".to_vec(), vec![1]);
        map.insert(b"z".to_vec(), vec![2, 3]);
        let vocab = build_vocabulary(4, map).expect("vocab");
        let e = engine(&SchemaIR::Boolean);
        let mask = e.allowed_tokens(e.start(), &vocab).expect("mask");
        assert!(mask.get(TokenId(0)));
        assert!(!mask.get(TokenId(2)) && !mask.get(TokenId(3)));
        assert!(!mask.get(TokenId(4)));
    }

    #[test]
    fn incompatible_vocabulary_fails_soft_with_an_empty_mask() {
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        map.insert(b"z".to_vec(), vec![0]);
        let vocab = build_vocabulary(1, map).expect("vocab");
        let e = engine(&SchemaIR::Boolean);
        let mask = e
            .allowed_tokens(e.start(), &vocab)
            .expect("no error, not IncompatibleVocabulary");
        assert_eq!(mask.count_ones(), 0);
    }

    #[test]
    fn sparse_token_ids_do_not_overflow_mask() {
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        map.insert(b"true".to_vec(), vec![50]);
        let vocab = build_vocabulary(100, map).expect("vocab");
        let e = engine(&SchemaIR::Boolean);
        let mask = e
            .allowed_tokens(e.start(), &vocab)
            .expect("mask must not overflow");
        assert!(mask.vocab_size() >= 101);
        assert!(mask.get(TokenId(50)));
        assert!(!mask.get(TokenId(100))); // eos bit present but never set
    }

    #[test]
    fn mask_width_low_and_high_sparse_ids() {
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        map.insert(b"a".to_vec(), vec![1, 2, 3]);
        let vocab = build_vocabulary(999, map).expect("vocab");
        assert_eq!(mask_width(&vocab).unwrap(), 1000); // width from EOS, not the {1,2,3} entries

        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        map.insert(b"a".to_vec(), vec![50, 1000]);
        let vocab = build_vocabulary(0, map).expect("vocab");
        assert_eq!(mask_width(&vocab).unwrap(), 1001); // width from the token ids, not EOS
    }

    #[test]
    fn mask_width_takes_the_larger_of_eos_and_ordinary_ids_either_direction() {
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        map.insert(b"a".to_vec(), vec![5]);
        let eos_larger = build_vocabulary(9000, map.clone()).expect("vocab");
        assert_eq!(mask_width(&eos_larger).unwrap(), 9001);

        let eos_smaller = build_vocabulary(1, map).expect("vocab");
        assert_eq!(mask_width(&eos_smaller).unwrap(), 6);
    }

    #[test]
    fn token_id_u32_max_produces_a_valid_mask_no_truncation_no_panic() {
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        map.insert(b"true".to_vec(), vec![u32::MAX]);
        let vocab = build_vocabulary(0, map).expect("vocab");
        let width = mask_width(&vocab).unwrap();
        assert_eq!(width, usize::try_from(u32::MAX).unwrap() + 1);
        let e = engine(&SchemaIR::Boolean);
        let mask = e
            .allowed_tokens(e.start(), &vocab)
            .expect("no error, not IncompatibleVocabulary");
        assert!(mask.get(TokenId(u32::MAX)));
        assert!(!mask.get(TokenId(0)));
    }

    #[test]
    fn json_string_const_escapes_quote_and_backslash() {
        let e = engine(&SchemaIR::StringConst {
            value: "a\"b\\c".to_string(),
        });
        assert!(e.accepts(b"\"a\\\"b\\\\c\""));
        assert!(!e.accepts(b"\"a\"b\\c\""));
    }

    #[test]
    fn json_string_escapes_every_control_category() {
        let value = "a\n\r\t\u{0}\u{8}\u{c}(b)\u{1f600}".to_string();
        let e = engine(&SchemaIR::StringConst {
            value: value.clone(),
        });
        let literal = json_string(&value);
        assert!(literal.contains("\\n") && literal.contains("\\u0000"));
        assert!(e.accepts(literal.as_bytes()));
        assert!(!e.accepts(format!("\"{value}\"").as_bytes()));
    }

    #[test]
    fn enum_prefix_overlapping_scalars_accept_both_values() {
        let e = engine(&SchemaIR::Enum {
            values: vec![ScalarLit::Int(1), ScalarLit::Int(12)],
        });
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"12"));
        assert!(!e.accepts(b"2"));
        assert!(!e.accepts(b"123"));
        let after_one = e.consume_token(e.start(), b"1").expect("live after \"1\"");
        assert!(e.is_accepting(after_one) && e.can_continue(after_one));
    }

    #[test]
    fn enum_duplicate_literals_are_set_collapsed() {
        let e = engine(&SchemaIR::Enum {
            values: vec![ScalarLit::Int(1), ScalarLit::Int(1), ScalarLit::Int(2)],
        });
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"2"));
        assert!(!e.accepts(b"3"));
    }

    #[test]
    fn array_max_zero_accepts_only_empty() {
        let e = engine(&SchemaIR::Array {
            items: Box::new(SchemaIR::Boolean),
            min: 0,
            max: Some(0),
        });
        assert!(e.accepts(b"[]"));
        assert!(!e.accepts(b"[true]"));
    }

    #[test]
    fn forged_state_id_is_dead_equivalent_no_panic() {
        let e = engine(&SchemaIR::Boolean);
        let forged = StateId(u32::MAX);
        assert!(e.is_dead(forged));
        assert!(!e.is_accepting(forged));
        assert!(!e.can_continue(forged));
        assert!(e.consume_token(forged, b"x").is_none());
    }

    #[test]
    fn invalid_regex_syntax_is_a_structured_error() {
        let ir = SchemaIR::StringPattern {
            regex: "(unclosed".to_string(),
        };
        let code = build_reference(&ir).unwrap_err().code;
        assert!(matches!(
            code,
            crate::error::ErrorCode::Malformed | crate::error::ErrorCode::Unsupported
        ));
    }
}
