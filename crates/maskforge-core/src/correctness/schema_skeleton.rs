//! Defines the crate-private schema representation for correctness checks.

use rustc_hash::FxHashSet;

use crate::error::{CompileError, ErrorCode, LimitKind, Stage};
use crate::frontend::regex::{analyze_pattern, PatternAnalysis};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ScalarLit {
    Null,
    Bool(bool),
    Str(String),
    Int(i64),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum KeyOrder {
    AsDeclared,
}

#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum SchemaIR {
    Null,
    Boolean,
    StringConst {
        value: String,
    },
    StringPattern {
        regex: String,
    },
    Enum {
        values: Vec<ScalarLit>,
    },
    Array {
        items: Box<SchemaIR>,
        min: u32,
        max: Option<u32>,
    },
    Object {
        fields: Vec<(String, SchemaIR)>,
        order: KeyOrder,
    },
}

impl SchemaIR {
    pub(crate) fn validate(&self) -> Result<(), CompileError> {
        self.validate_at("")
    }

    fn validate_at(&self, ptr: &str) -> Result<(), CompileError> {
        limit_seam(LimitKind::NodeCount)?;
        match self {
            SchemaIR::Null | SchemaIR::Boolean | SchemaIR::StringConst { .. } => Ok(()),
            SchemaIR::StringPattern { regex } => validate_pattern(regex, ptr),
            SchemaIR::Enum { values } => {
                if values.is_empty() {
                    return Err(malformed(
                        ptr,
                        "enum",
                        "an enum must have at least one value",
                    ));
                }
                Ok(())
            }
            SchemaIR::Array { items, min, max } => {
                if let Some(max) = max {
                    if max < min {
                        return Err(malformed(ptr, "maxItems", "array max is less than min"));
                    }
                }
                limit_seam(LimitKind::ArrayLength)?;
                items.validate_at(&format!("{ptr}/items"))
            }
            SchemaIR::Object { fields, .. } => {
                let mut seen: FxHashSet<&str> = FxHashSet::default();
                seen.reserve(fields.len());
                for (key, sub) in fields {
                    let child = format!("{ptr}/{}", ptr_escape(key));
                    if !seen.insert(key.as_str()) {
                        return Err(malformed(&child, "properties", "duplicate object key"));
                    }
                    sub.validate_at(&child)?;
                }
                Ok(())
            }
        }
    }
}

fn limit_seam(_kind: LimitKind) -> Result<(), CompileError> {
    Ok(())
}

fn ptr_escape(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

fn validate_pattern(regex: &str, ptr: &str) -> Result<(), CompileError> {
    if crate::ir::has_anchor_outside_brackets(regex.as_bytes()) {
        return Err(unsupported(
            ptr,
            "pattern",
            "explicit anchors are not allowed",
        ));
    }
    match analyze_pattern(regex) {
        PatternAnalysis::Ordinary => {}
        PatternAnalysis::Leading(_) => {
            return Err(unsupported(
                ptr,
                "pattern",
                "look-ahead is not supported by the byte skeleton",
            ));
        }
        PatternAnalysis::Unsupported(what) => {
            return Err(
                unsupported(ptr, "pattern", "unsupported regex construct").with_observed(what)
            );
        }
        PatternAnalysis::Malformed(what) => {
            return Err(malformed(ptr, "pattern", what));
        }
    }
    let bytes = regex.as_bytes();
    if contains_unicode_property_escape(bytes) {
        return Err(unsupported(
            ptr,
            "pattern",
            "Unicode property classes are not supported by the byte skeleton",
        ));
    }
    if bracket_expression_has_non_ascii(bytes) {
        return Err(unsupported(
            ptr,
            "pattern",
            "non-ASCII bytes inside [...] are not supported by this byte-level engine",
        ));
    }
    Ok(())
}

fn contains_unicode_property_escape(bytes: &[u8]) -> bool {
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'\\' {
            index += 1;
            continue;
        }
        let Some(&escaped) = bytes.get(index + 1) else {
            return false;
        };
        if matches!(escaped, b'p' | b'P') && bytes.get(index + 2) == Some(&b'{') {
            return true;
        }
        index += 2;
    }
    false
}

fn bracket_expression_has_non_ascii(bytes: &[u8]) -> bool {
    let mut in_bracket = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'[' if !in_bracket => in_bracket = true,
            b']' if in_bracket => in_bracket = false,
            b if in_bracket && b >= 0x80 => return true,
            _ => {}
        }
        i += 1;
    }
    false
}

fn malformed(ptr: &str, keyword: &str, message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L3, message)
        .with_pointer(ptr.to_string())
        .with_keyword(keyword.to_string())
}

fn unsupported(ptr: &str, keyword: &str, message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Unsupported, Stage::L3, message)
        .with_pointer(ptr.to_string())
        .with_keyword(keyword.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_must_be_non_empty() {
        assert_eq!(
            SchemaIR::Enum { values: vec![] }
                .validate()
                .unwrap_err()
                .code,
            ErrorCode::Malformed
        );
        assert!(SchemaIR::Enum {
            values: vec![ScalarLit::Int(1)]
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn array_max_below_min_is_malformed() {
        let ir = SchemaIR::Array {
            items: Box::new(SchemaIR::Boolean),
            min: 3,
            max: Some(2),
        };
        assert_eq!(ir.validate().unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn unbounded_array_is_allowed() {
        let ir = SchemaIR::Array {
            items: Box::new(SchemaIR::Boolean),
            min: 0,
            max: None,
        };
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn duplicate_object_keys_rejected() {
        let ir = SchemaIR::Object {
            fields: vec![
                ("a".into(), SchemaIR::Null),
                ("a".into(), SchemaIR::Boolean),
            ],
            order: KeyOrder::AsDeclared,
        };
        let e = ir.validate().unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
        assert_eq!(e.json_pointer_path.as_deref(), Some("/a"));
    }

    #[test]
    fn forbidden_regex_constructs_are_unsupported() {
        for pat in ["^ab", "ab$", "a(?=b)", r"a\p{L}", r"(a)\1", r"\k<n>"] {
            let ir = SchemaIR::StringPattern {
                regex: pat.to_string(),
            };
            assert_eq!(
                ir.validate().unwrap_err().code,
                ErrorCode::Unsupported,
                "pattern {pat:?} should be unsupported"
            );
        }
    }

    #[test]
    fn non_ascii_bracket_expression_is_unsupported_not_a_late_dfa_error() {
        for pat in ["[😈-😍]", "[😈😇]", "a[b😀]c"] {
            let ir = SchemaIR::StringPattern {
                regex: pat.to_string(),
            };
            assert_eq!(
                ir.validate().unwrap_err().code,
                ErrorCode::Unsupported,
                "pattern {pat:?} should be caught here, not at DFA build"
            );
        }
        assert!(SchemaIR::StringPattern {
            regex: "😇|😈".to_string(),
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn ordinary_pattern_is_accepted() {
        let ir = SchemaIR::StringPattern {
            regex: "(cat|car|carbon)".to_string(),
        };
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn deeply_nested_array_validates_without_a_configured_limit() {
        let mut ir = SchemaIR::Boolean;
        for _ in 0..200 {
            ir = SchemaIR::Array {
                items: Box::new(ir),
                min: 0,
                max: None,
            };
        }
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn large_flat_object_field_count_validates_without_a_configured_limit() {
        let fields = (0..5000)
            .map(|i| (format!("field{i}"), SchemaIR::Boolean))
            .collect();
        let ir = SchemaIR::Object {
            fields,
            order: KeyOrder::AsDeclared,
        };
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn long_pattern_validates_without_a_configured_limit() {
        let regex = "a".repeat(10_000);
        let ir = SchemaIR::StringPattern { regex };
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn json_pointer_escapes_slash_and_tilde_in_keys() {
        let ir = SchemaIR::Object {
            fields: vec![
                ("a/b".into(), SchemaIR::Null),
                ("a/b".into(), SchemaIR::Boolean),
            ],
            order: KeyOrder::AsDeclared,
        };
        assert_eq!(
            ir.validate().unwrap_err().json_pointer_path.as_deref(),
            Some("/a~1b")
        );
        assert_eq!(ptr_escape("a~b"), "a~0b");
        assert_eq!(ptr_escape("a~/b"), "a~0~1b");
    }
}
