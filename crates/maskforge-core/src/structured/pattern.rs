//! Production JSON Schema string-pattern helpers.

use crate::automaton::{build_from_unicode_regex, Combinator, ProductAutomaton, RefEngine};
use crate::error::{CompileError, ErrorCode, Stage};
use crate::frontend::regex::{analyze_pattern, AssertionKind, LeadingAssertions, PatternAnalysis};

/// Strips a leading `^` and an unescaped trailing `$`.
pub(crate) fn split_anchors(src: &str) -> (bool, &str, bool) {
    let mut middle = src;
    let anchor_start = middle.starts_with('^');
    if anchor_start {
        middle = &middle[1..];
    }
    let anchor_end = trailing_dollar_is_anchor(middle);
    if anchor_end {
        middle = &middle[..middle.len() - 1];
    }
    (anchor_start, middle, anchor_end)
}

fn trailing_dollar_is_anchor(src: &str) -> bool {
    src.strip_suffix('$').is_some_and(|prefix| {
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

const DECODED_SCALAR: &str = r"(?s:.)";
// Source bytes protect parsing only. Automaton construction remains independently bounded by the
// Thompson-NFA, determinization, DFA-state/cell, and product limits in `automaton::byte_dfa`.
const MAX_PATTERN_INPUT_BYTES: usize = 1 << 20;
const MAX_PATTERN_HIR_NODES: usize = 1 << 18;
const MAX_PATTERN_COMPILE_WORK: usize = 1 << 24;

/// Turns a regex matched at decoded-text offset zero into a complete-string language.
pub(crate) fn leading_fragment_pattern(
    fragment: &str,
    end_anchored: bool,
) -> Result<String, CompileError> {
    let (_, middle, fragment_end) = split_anchors(fragment);
    let middle = normalize_ecma_classes(middle)?;
    let exact_end = end_anchored || fragment_end;
    let suffix = if exact_end { "" } else { DECODED_SCALAR };
    let suffix_quantifier = if exact_end { "" } else { "*" };
    let capacity = middle
        .len()
        .checked_add(suffix.len())
        .and_then(|bytes| bytes.checked_add(suffix_quantifier.len()))
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or_else(pattern_limit)?;
    let mut pattern = String::new();
    pattern
        .try_reserve_exact(capacity)
        .map_err(|_| pattern_limit())?;
    pattern.push_str("(?:");
    pattern.push_str(&middle);
    pattern.push(')');
    pattern.push_str(suffix);
    pattern.push_str(suffix_quantifier);
    Ok(pattern)
}

const ECMA_SPACE_CONTENT: &str =
    r"\x09-\x0D\x20\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}";
const ECMA_DOT: &str = r"[^\n\r\x{2028}\x{2029}]";

fn normalize_ecma_classes(source: &str) -> Result<String, CompileError> {
    let bytes = source.as_bytes();
    let mut output = String::new();
    let mut in_class = false;
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index..].starts_with(br"[\s\S]") {
            push_checked(&mut output, "(?s:.)")?;
            index += br"[\s\S]".len();
            continue;
        }
        match bytes[index] {
            b'[' => in_class = true,
            b']' => in_class = false,
            b'.' if !in_class => {
                push_checked(&mut output, ECMA_DOT)?;
                index += 1;
                continue;
            }
            b'\\' => {
                let Some(&escaped) = bytes.get(index + 1) else {
                    return Err(pattern_subset("trailing regex escape"));
                };
                if !escaped.is_ascii() {
                    return Err(pattern_subset("non-ASCII identity escape"));
                }
                if normalize_class_escape(&mut output, escaped, in_class)? {
                    index += 2;
                    continue;
                }
                push_checked(&mut output, "\\")?;
                push_char_checked(&mut output, char::from(escaped))?;
                index += 2;
                continue;
            }
            _ => {}
        }
        let ch = source[index..].chars().next().ok_or_else(pattern_limit)?;
        push_char_checked(&mut output, ch)?;
        index += ch.len_utf8();
    }
    Ok(output)
}

fn normalize_class_escape(
    output: &mut String,
    escaped: u8,
    in_class: bool,
) -> Result<bool, CompileError> {
    let replacement = match (escaped, in_class) {
        (b'w', true) => "0-9A-Z_a-z",
        (b'w', false) => "[0-9A-Z_a-z]",
        (b'W', false) => "[^0-9A-Z_a-z]",
        (b'd', true) => "0-9",
        (b'd', false) => "[0-9]",
        (b'D', false) => "[^0-9]",
        (b's', true) => ECMA_SPACE_CONTENT,
        (b's', false) => return push_wrapped_class(output, false),
        (b'S', false) => return push_wrapped_class(output, true),
        (b'W' | b'D' | b'S', true) => {
            return Err(pattern_subset("negated shorthand inside a class"));
        }
        (b'b', false) => return Err(pattern_subset("word-boundary assertion")),
        (_, false)
            if !matches!(
                escaped,
                b'^' | b'$'
                    | b'\\'
                    | b'.'
                    | b'*'
                    | b'+'
                    | b'?'
                    | b'('
                    | b')'
                    | b'['
                    | b']'
                    | b'{'
                    | b'}'
                    | b'|'
                    | b'/'
                    | b'f'
                    | b'n'
                    | b'r'
                    | b't'
                    | b'v'
                    | b'x'
                    | b'u'
                    | b'p'
                    | b'P'
            ) =>
        {
            return Err(pattern_subset(
                "identity escape under Unicode pattern semantics",
            ));
        }
        _ => return Ok(false),
    };
    push_checked(output, replacement)?;
    Ok(true)
}

fn push_checked(output: &mut String, value: &str) -> Result<(), CompileError> {
    output
        .try_reserve(value.len())
        .map_err(|_| pattern_limit())?;
    output.push_str(value);
    Ok(())
}

fn push_char_checked(output: &mut String, value: char) -> Result<(), CompileError> {
    output
        .try_reserve(value.len_utf8())
        .map_err(|_| pattern_limit())?;
    output.push(value);
    Ok(())
}

fn push_wrapped_class(output: &mut String, negated: bool) -> Result<bool, CompileError> {
    output
        .try_reserve(ECMA_SPACE_CONTENT.len() + 3)
        .map_err(|_| pattern_limit())?;
    output.push('[');
    if negated {
        output.push('^');
    }
    output.push_str(ECMA_SPACE_CONTENT);
    output.push(']');
    Ok(true)
}

fn pattern_subset(observed: &'static str) -> CompileError {
    CompileError::new(
        ErrorCode::Unsupported,
        Stage::L2,
        "unsupported ECMA regex subset",
    )
    .with_observed(observed)
}

fn pattern_limit() -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L2,
        "regex assertion lowering allocation",
    )
}

fn search_pattern_limit(
    kind: crate::error::LimitKind,
    observed: usize,
    limit: usize,
) -> CompileError {
    let mut error = CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L2,
        "ECMA regex pattern exceeds a layered compilation budget",
    );
    error.limit = Some((kind, observed, limit));
    error
}

fn check_pattern_budgets(regex: &str) -> Result<(), CompileError> {
    if regex.len() > MAX_PATTERN_INPUT_BYTES {
        return Err(search_pattern_limit(
            crate::error::LimitKind::PatternInputBytes,
            regex.len(),
            MAX_PATTERN_INPUT_BYTES,
        ));
    }
    let hir_nodes = regex
        .chars()
        .try_fold(1usize, |nodes, ch| {
            nodes.checked_add(
                usize::from(matches!(
                    ch,
                    '|' | '(' | ')' | '[' | ']' | '{' | '}' | '*' | '+' | '?'
                )) + 1,
            )
        })
        .unwrap_or(usize::MAX);
    if hir_nodes > MAX_PATTERN_HIR_NODES {
        return Err(search_pattern_limit(
            crate::error::LimitKind::PatternHirNodes,
            hir_nodes,
            MAX_PATTERN_HIR_NODES,
        ));
    }
    let work = hir_nodes.checked_mul(64).unwrap_or(usize::MAX);
    if work > MAX_PATTERN_COMPILE_WORK {
        return Err(search_pattern_limit(
            crate::error::LimitKind::PatternCompileWork,
            work,
            MAX_PATTERN_COMPILE_WORK,
        ));
    }
    Ok(())
}

/// Compiles ECMA search semantics while preserving per-alternative anchors.
pub(crate) fn build_search_pattern(regex: &str) -> Result<String, CompileError> {
    check_pattern_budgets(regex)?;
    let normalized = normalize_ecma_classes(regex)?;
    build_search_pattern_with_item(&normalized).ok_or_else(pattern_limit)
}

/// Compiles search semantics for an already-decoded property name.
pub(crate) fn build_property_search_pattern(regex: &str) -> Result<String, CompileError> {
    check_pattern_budgets(regex)?;
    let (anchor_start, _, anchor_end) = split_anchors(regex);
    if anchor_start {
        leading_fragment_pattern(regex, anchor_end)
    } else {
        build_search_pattern(regex)
    }
}

/// Builds a decoded-property-name search engine, including the bounded leading-assertion subset.
pub(crate) fn build_property_search_engine(regex: &str) -> Result<RefEngine, CompileError> {
    match analyze_pattern(regex) {
        PatternAnalysis::Ordinary => {
            let pattern = build_property_search_pattern(regex)?;
            build_from_unicode_regex(&pattern)
        }
        PatternAnalysis::Leading(assertions) => compile_leading_assertions(assertions),
        PatternAnalysis::Unsupported(what) | PatternAnalysis::Malformed(what) => {
            Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L2,
                "unsupported regex construct",
            )
            .with_observed(what))
        }
    }
}

fn compile_leading_assertions(
    assertions: LeadingAssertions<'_>,
) -> Result<RefEngine, CompileError> {
    let body = leading_fragment_pattern(assertions.body, assertions.end_anchored)?;
    let mut positive = Vec::new();
    positive.try_reserve_exact(9).map_err(|_| pattern_limit())?;
    positive.push(build_from_unicode_regex(&body)?);
    let mut negative = Vec::new();
    negative.try_reserve_exact(8).map_err(|_| pattern_limit())?;
    for assertion in assertions.iter() {
        let operand = leading_fragment_pattern(assertion.operand, false)?;
        let engine = build_from_unicode_regex(&operand)?;
        match assertion.kind {
            AssertionKind::Positive => positive.push(engine),
            AssertionKind::Negative => negative.push(engine),
        }
    }
    let accepted = intersect_engines(positive)?;
    if negative.is_empty() {
        return Ok(accepted);
    }
    let mut refs = Vec::new();
    let ref_count = negative.len().checked_add(1).ok_or_else(pattern_limit)?;
    refs.try_reserve_exact(ref_count)
        .map_err(|_| pattern_limit())?;
    refs.push(&accepted);
    refs.extend(negative.iter());
    ProductAutomaton::build_reachable(&refs, Combinator::Difference)?.into_engine()
}

fn intersect_engines(mut engines: Vec<RefEngine>) -> Result<RefEngine, CompileError> {
    if engines.is_empty() {
        return Err(pattern_limit());
    }
    if engines.len() == 1 {
        return engines.pop().ok_or_else(pattern_limit);
    }
    let mut refs = Vec::new();
    refs.try_reserve_exact(engines.len())
        .map_err(|_| pattern_limit())?;
    refs.extend(engines.iter());
    ProductAutomaton::build_reachable(&refs, Combinator::All)?.into_engine()
}

fn build_search_pattern_with_item(regex: &str) -> Option<String> {
    Some(format!("{DECODED_SCALAR}*?(?:{regex}){DECODED_SCALAR}*"))
}

/// Tests one compiled property search with one DFA transition per byte.
pub(crate) fn property_pattern_matches(engine: &RefEngine, bytes: &[u8]) -> bool {
    engine
        .consume_token(engine.start(), bytes)
        .is_some_and(|state| engine.is_accepting(state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_patterns_preserve_top_level_branch_anchors() {
        let compiled = build_search_pattern("^cat$|dog").expect("supported search pattern");
        assert!(compiled.contains("cat"));
        assert!(compiled.contains("dog"));
        assert!(compiled.contains("(?:^cat$|dog)"));
    }

    #[test]
    fn source_length_is_not_an_automaton_cost_proxy() {
        for length in [511usize, 512, 513, 1_300, 8_192] {
            let literal = "a".repeat(length);
            let compiled = build_search_pattern(&literal).expect("long literal remains budgeted");
            assert!(compiled.contains(&literal));
        }
        let oversized = "a".repeat(MAX_PATTERN_INPUT_BYTES + 1);
        let error = build_search_pattern(&oversized).unwrap_err();
        assert_eq!(
            error.limit,
            Some((
                crate::error::LimitKind::PatternInputBytes,
                MAX_PATTERN_INPUT_BYTES + 1,
                MAX_PATTERN_INPUT_BYTES,
            ))
        );
    }

    #[test]
    fn trailing_anchor_uses_backslash_parity() {
        for count in 0..=5 {
            let pattern = format!("x{}$", "\\".repeat(count));
            assert_eq!(split_anchors(&pattern).2, count % 2 == 0, "{pattern:?}");
        }
        let compiled = build_property_search_pattern(r"(^cat$)|(dog\\$)|tail\\$")
            .expect("supported property search");
        assert!(compiled.contains("cat"));
        assert!(compiled.contains("dog"));
        assert!(compiled.contains("tail"));
    }

    #[test]
    fn leading_fragments_use_ecma_shorthand_classes() {
        let word = build_from_unicode_regex(
            &leading_fragment_pattern(r"\w+", true).expect("word fragment"),
        )
        .expect("word engine");
        assert!(property_pattern_matches(&word, b"Az_09"));
        assert!(!property_pattern_matches(&word, "λ".as_bytes()));

        let digit = build_from_unicode_regex(
            &leading_fragment_pattern(r"\d", true).expect("digit fragment"),
        )
        .expect("digit engine");
        assert!(property_pattern_matches(&digit, b"7"));
        assert!(!property_pattern_matches(&digit, "٧".as_bytes()));

        let space = build_from_unicode_regex(
            &leading_fragment_pattern(r"\s", true).expect("space fragment"),
        )
        .expect("space engine");
        assert!(property_pattern_matches(&space, "\u{a0}".as_bytes()));
        assert!(property_pattern_matches(&space, "\u{feff}".as_bytes()));

        let any = build_from_unicode_regex(
            &leading_fragment_pattern(r"[\s\S]", true).expect("any fragment"),
        )
        .expect("any engine");
        assert!(property_pattern_matches(&any, b"\n"));

        let dot =
            build_from_unicode_regex(&leading_fragment_pattern(".", true).expect("dot fragment"))
                .expect("dot engine");
        assert!(property_pattern_matches(&dot, "λ".as_bytes()));
        for terminator in ["\n", "\r", "\u{2028}", "\u{2029}"] {
            assert!(!property_pattern_matches(&dot, terminator.as_bytes()));
        }
    }

    #[test]
    fn unanchored_property_patterns_use_the_canonical_ecma_classes() {
        let word = build_property_search_engine(r"\w+").expect("word property pattern");
        assert!(property_pattern_matches(&word, b"key_09"));
        assert!(!property_pattern_matches(&word, "λ".as_bytes()));

        let space = build_property_search_engine(r"\s+").expect("space property pattern");
        assert!(property_pattern_matches(&space, "x\u{a0}y".as_bytes()));
    }
}
