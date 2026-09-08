//! Lowers regular expressions into JSON string schema IR.

use crate::error::CompileError;
use crate::ir::{Builder, Charset, CompileOptions, SchemaIR};

const MAX_LEADING_ASSERTIONS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AssertionKind {
    Positive,
    Negative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LeadingAssertion<'a> {
    pub(crate) kind: AssertionKind,
    pub(crate) operand: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LeadingAssertions<'a> {
    assertions: [Option<AssertionSpan>; MAX_LEADING_ASSERTIONS],
    len: usize,
    pattern: &'a str,
    pub(crate) body: &'a str,
    pub(crate) end_anchored: bool,
}

impl<'a> LeadingAssertions<'a> {
    pub(crate) fn iter(self) -> impl Iterator<Item = LeadingAssertion<'a>> {
        self.assertions
            .into_iter()
            .take(self.len)
            .flatten()
            .filter_map(move |span| {
                let start = usize::try_from(span.start).ok()?;
                let end = usize::try_from(span.end).ok()?;
                Some(LeadingAssertion {
                    kind: span.kind,
                    operand: self.pattern.get(start..end)?,
                })
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AssertionSpan {
    kind: AssertionKind,
    start: u32,
    end: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PatternAnalysis<'a> {
    Ordinary,
    Leading(LeadingAssertions<'a>),
    Unsupported(&'static str),
    Malformed(&'static str),
}

#[derive(Default)]
struct Scan {
    lookaheads: usize,
    lookbehinds: usize,
    numeric_backreference: bool,
    named_backreference: bool,
    incompatible_group: bool,
}

pub(crate) fn analyze_pattern(pattern: &str) -> PatternAnalysis<'_> {
    analyze_pattern_impl(pattern, false)
}

pub(crate) fn analyze_compiled_pattern(pattern: &str) -> PatternAnalysis<'_> {
    analyze_pattern_impl(pattern, true)
}

fn analyze_pattern_impl(pattern: &str, allow_internal_flags: bool) -> PatternAnalysis<'_> {
    let scan = match scan_pattern(pattern, allow_internal_flags) {
        Ok(scan) => scan,
        Err(message) => return PatternAnalysis::Malformed(message),
    };
    if scan.numeric_backreference {
        return PatternAnalysis::Unsupported("numeric backreference");
    }
    if scan.named_backreference {
        return PatternAnalysis::Unsupported("named backreference");
    }
    if scan.lookbehinds != 0 {
        return PatternAnalysis::Unsupported("look-behind");
    }
    if scan.incompatible_group {
        return PatternAnalysis::Unsupported("unsupported ECMA group construct");
    }
    if scan.lookaheads == 0 {
        return PatternAnalysis::Ordinary;
    }
    match parse_leading_assertions(pattern, scan.lookaheads) {
        Some(parsed) => PatternAnalysis::Leading(parsed),
        None => PatternAnalysis::Unsupported("non-leading or nested look-ahead"),
    }
}

fn scan_pattern(pattern: &str, allow_internal_flags: bool) -> Result<Scan, &'static str> {
    let bytes = pattern.as_bytes();
    let mut scan = Scan::default();
    let mut in_class = false;
    let mut depth = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => {
                let Some(&escaped) = bytes.get(index + 1) else {
                    return Err("trailing regex escape");
                };
                if !in_class && escaped.is_ascii_digit() && escaped != b'0' {
                    scan.numeric_backreference = true;
                } else if !in_class && escaped == b'k' && bytes.get(index + 2) == Some(&b'<') {
                    scan.named_backreference = true;
                }
                index += 2;
                continue;
            }
            b'[' if !in_class => in_class = true,
            b']' if in_class => in_class = false,
            b'(' if !in_class => {
                depth = depth.checked_add(1).ok_or("regex nesting overflow")?;
                classify_group(bytes, index, allow_internal_flags, &mut scan);
            }
            b')' if !in_class => {
                depth = depth
                    .checked_sub(1)
                    .ok_or("unmatched closing parenthesis")?;
            }
            _ => {}
        }
        index += 1;
    }
    if in_class {
        Err("unclosed character class")
    } else if depth != 0 {
        Err("unclosed parenthesis")
    } else {
        Ok(scan)
    }
}

fn classify_group(bytes: &[u8], index: usize, allow_internal_flags: bool, scan: &mut Scan) {
    let tail = &bytes[index..];
    if tail.starts_with(b"(?<=") || tail.starts_with(b"(?<!") {
        scan.lookbehinds += 1;
    } else if tail.starts_with(b"(?=") || tail.starts_with(b"(?!") {
        scan.lookaheads += 1;
    } else if tail.starts_with(b"(?:")
        || (allow_internal_flags && tail.starts_with(b"(?s:"))
        || !tail.starts_with(b"(?")
    {
    } else {
        scan.incompatible_group = true;
    }
}

fn parse_leading_assertions(pattern: &str, expected: usize) -> Option<LeadingAssertions<'_>> {
    let bytes = pattern.as_bytes();
    if bytes.first() != Some(&b'^') {
        return None;
    }
    let mut parsed = LeadingAssertions {
        assertions: [None; MAX_LEADING_ASSERTIONS],
        len: 0,
        pattern,
        body: "",
        end_anchored: false,
    };
    let mut cursor = 1usize;
    while bytes
        .get(cursor..)
        .is_some_and(|tail| tail.starts_with(b"(?=") || tail.starts_with(b"(?!"))
    {
        if parsed.len == MAX_LEADING_ASSERTIONS {
            return None;
        }
        let negative = bytes.get(cursor + 2) == Some(&b'!');
        let close = matching_group_end(bytes, cursor)?;
        parsed.assertions[parsed.len] = Some(AssertionSpan {
            kind: if negative {
                AssertionKind::Negative
            } else {
                AssertionKind::Positive
            },
            start: u32::try_from(cursor.checked_add(3)?).ok()?,
            end: u32::try_from(close).ok()?,
        });
        parsed.len += 1;
        cursor = close + 1;
    }
    if parsed.len != expected {
        return None;
    }
    let remainder = pattern.get(cursor..)?;
    parsed.end_anchored = trailing_dollar_is_anchor(remainder);
    parsed.body = if parsed.end_anchored {
        remainder.get(..remainder.len().checked_sub(1)?)?
    } else {
        remainder
    };
    parsed.body = parsed.body.strip_prefix('^').unwrap_or(parsed.body);
    Some(parsed)
}

fn matching_group_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_class = false;
    let mut index = start;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index = index.checked_add(1)?,
            b'[' if !in_class => in_class = true,
            b']' if in_class => in_class = false,
            b'(' if !in_class => depth = depth.checked_add(1)?,
            b')' if !in_class => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        index = index.checked_add(1)?;
    }
    None
}

fn trailing_dollar_is_anchor(pattern: &str) -> bool {
    pattern.strip_suffix('$').is_some_and(|prefix| {
        prefix
            .as_bytes()
            .iter()
            .rev()
            .take_while(|&&byte| byte == b'\\')
            .count()
            % 2
            == 0
    })
}

pub fn regex_to_ir(regex: &str, options: CompileOptions) -> Result<SchemaIR, CompileError> {
    let mut builder = Builder::new(options);
    let node = match analyze_pattern(regex) {
        PatternAnalysis::Leading(assertions) => lower_leading_assertions(&mut builder, assertions)?,
        PatternAnalysis::Ordinary => builder.string_pattern(regex, None, None, Charset::Utf8Any)?,
        PatternAnalysis::Unsupported(what) => return Err(regex_construct(what)),
        PatternAnalysis::Malformed(what) => {
            return Err(CompileError::new(
                crate::error::ErrorCode::Malformed,
                crate::error::Stage::L2,
                "malformed regex syntax",
            )
            .with_observed(what));
        }
    };
    builder.finish(node)
}

pub(crate) fn lower_leading_assertions(
    builder: &mut Builder,
    assertions: LeadingAssertions<'_>,
) -> Result<crate::primitives::NodeId, CompileError> {
    let body =
        crate::structured::leading_fragment_pattern(assertions.body, assertions.end_anchored)?;
    let body = builder.string_pattern(&body, None, None, Charset::Utf8Any)?;
    let mut branches = Vec::new();
    branches
        .try_reserve_exact(MAX_LEADING_ASSERTIONS + 1)
        .map_err(|_| regex_limit())?;
    branches.push(body);
    for assertion in assertions.iter() {
        let operand = crate::structured::leading_fragment_pattern(assertion.operand, false)?;
        let operand = builder.string_pattern(&operand, None, None, Charset::Utf8Any)?;
        branches.push(match assertion.kind {
            AssertionKind::Positive => operand,
            AssertionKind::Negative => builder.not(operand)?,
        });
    }
    builder.intersection_of(branches)
}

fn regex_limit() -> CompileError {
    CompileError::new(
        crate::error::ErrorCode::InternalLimitExceeded,
        crate::error::Stage::L2,
        "regex assertion branch allocation",
    )
}

fn regex_construct(observed: &'static str) -> CompileError {
    CompileError::new(
        crate::error::ErrorCode::Unsupported,
        crate::error::Stage::L2,
        "unsupported regex construct",
    )
    .with_observed(observed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_regex_lowers_and_matches_as_a_quoted_string() {
        let ir = regex_to_ir("(cat|car|carbon)", CompileOptions::default()).unwrap();
        assert!(ir.requires_structured_backend());
        assert!(crate::structured::accepts(&ir, b"\"cat\""));
        assert!(crate::structured::accepts(&ir, b"\"carbon\""));
        assert!(!crate::structured::accepts(&ir, b"cat"));
        assert!(!crate::structured::accepts(&ir, b"\"dog\""));
    }

    #[test]
    fn an_anchored_regex_matches_only_the_complete_content() {
        let ir = regex_to_ir("^cat$", CompileOptions::default()).unwrap();
        assert!(
            crate::structured::try_accepts(std::sync::Arc::new(ir.clone()), b"\"cat\"").unwrap()
        );
        assert!(
            !crate::structured::try_accepts(std::sync::Arc::new(ir.clone()), b"\"xcat\"").unwrap()
        );
        assert!(!crate::structured::try_accepts(std::sync::Arc::new(ir), b"\"catx\"").unwrap());
    }

    #[test]
    fn scanner_ignores_escaped_and_character_class_operator_text() {
        for pattern in [r"\(\?=literal", r"[(?=!)]", r"[\\1]", r"\\1", r"\0"] {
            assert_eq!(analyze_pattern(pattern), PatternAnalysis::Ordinary);
        }
    }

    #[test]
    fn scanner_classifies_backreferences_lookbehind_and_malformed_syntax() {
        assert_eq!(
            analyze_pattern(r"(a)\1"),
            PatternAnalysis::Unsupported("numeric backreference")
        );
        assert_eq!(
            analyze_pattern(r"\k<name>"),
            PatternAnalysis::Unsupported("named backreference")
        );
        assert_eq!(
            analyze_pattern(r"(?<=a)b"),
            PatternAnalysis::Unsupported("look-behind")
        );
        assert!(matches!(
            analyze_pattern("[abc"),
            PatternAnalysis::Malformed(_)
        ));
        assert_eq!(
            analyze_pattern(r"(?s:.)"),
            PatternAnalysis::Unsupported("unsupported ECMA group construct")
        );
        assert_eq!(
            analyze_compiled_pattern(r"(?s:.)"),
            PatternAnalysis::Ordinary
        );
    }

    #[test]
    fn leading_assertion_parser_preserves_operands_body_and_end_anchor() {
        let PatternAnalysis::Leading(parsed) = analyze_pattern(r"^(?=.{1,8}$)(?!bad)\w+$") else {
            panic!("expected supported leading assertions");
        };
        let assertions: Vec<_> = parsed.iter().collect();
        assert_eq!(assertions.len(), 2);
        assert_eq!(assertions[0].kind, AssertionKind::Positive);
        assert_eq!(assertions[0].operand, ".{1,8}$");
        assert_eq!(assertions[1].kind, AssertionKind::Negative);
        assert_eq!(assertions[1].operand, "bad");
        assert_eq!(parsed.body, r"\w+");
        assert!(parsed.end_anchored);
        assert_eq!(
            analyze_pattern(r"a(?=b)"),
            PatternAnalysis::Unsupported("non-leading or nested look-ahead")
        );
    }

    #[test]
    fn leading_positive_and_negative_assertions_match_decoded_text() {
        let ir = regex_to_ir(r"^(?=.{2,4}$)(?!bad$)[a-z]+$", CompileOptions::default())
            .expect("leading assertions");
        let ir = std::sync::Arc::new(ir);
        for (json, expected) in [
            (br#""ok""#.as_slice(), true),
            (br#""good""#.as_slice(), true),
            (br#""bad""#.as_slice(), false),
            (br#""x""#.as_slice(), false),
            (br#""toolong""#.as_slice(), false),
            (br#""\u006fk""#.as_slice(), true),
        ] {
            assert_eq!(
                crate::structured::try_accepts(ir.clone(), json).unwrap(),
                expected,
                "json={}",
                String::from_utf8_lossy(json)
            );
        }
    }
}
