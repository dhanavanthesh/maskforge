//! Builds automata for JSON numbers constrained by `multipleOf`.

#[cfg(test)]
use super::elimination::{to_regex_via_state_elimination, EliminationState, MAX_ELIMINATION_BYTES};
use super::graph::ByteState;
use super::{byte_string_set, concatenate, union, AutomatonGraph, RefEngine};
use crate::error::{CompileError, ErrorCode, Stage};
use crate::mem_gate::acquire_automaton_build_bytes;

const MAX_MULTIPLE_OF_MODULUS: u64 = 1 << 20;
const MAX_SUFFIX_ENDINGS: u64 = 1 << 16;
#[cfg(test)]
const MAX_DIRECT_REMAINDER_MODULUS: u64 = 8;
const MAX_ENGINE_REMAINDER_MODULUS: u64 = 1 << 12;

#[cfg(test)]
pub(crate) fn multiple_of_regex(modulus: u64) -> Result<String, CompileError> {
    if modulus == 0 {
        return Err(malformed("multipleOf must be a positive integer"));
    }
    if modulus > MAX_MULTIPLE_OF_MODULUS {
        return Err(limit("multipleOf modulus"));
    }
    if let Some((suffix_len, ending_count)) = power_of_ten_factor_suffix(modulus) {
        return Ok(suffix_regex(suffix_len, modulus, ending_count));
    }
    if modulus <= MAX_DIRECT_REMAINDER_MODULUS {
        return remainder_dfa_regex(modulus);
    }
    Err(limit("multipleOf modulus needs a factor of 2 or 5"))
}

pub(crate) fn multiple_of_engine(modulus: u64) -> Result<RefEngine, CompileError> {
    if modulus == 0 {
        return Err(malformed("multipleOf must be a positive integer"));
    }
    if modulus > MAX_MULTIPLE_OF_MODULUS {
        return Err(limit("multipleOf modulus"));
    }
    if modulus == 1 {
        return signed_digit_star_engine();
    }
    if modulus <= MAX_ENGINE_REMAINDER_MODULUS {
        return remainder_dfa_engine(modulus);
    }
    if let Some((suffix_len, ending_count)) = power_of_ten_factor_suffix(modulus) {
        return suffix_engine(suffix_len, modulus, ending_count);
    }
    Err(limit("multipleOf modulus needs a factor of 2 or 5"))
}

fn power_of_ten_factor_suffix(modulus: u64) -> Option<(u32, u64)> {
    let mut rest = modulus;
    let mut a = 0u32;
    while rest % 2 == 0 {
        rest /= 2;
        a += 1;
    }
    let mut b = 0u32;
    while rest % 5 == 0 {
        rest /= 5;
        b += 1;
    }
    if rest != 1 {
        return None;
    }
    let suffix_len = a.max(b);
    let ten_pow_l = 10u64.checked_pow(suffix_len)?;
    let ending_count = ten_pow_l / modulus;
    if ending_count > MAX_SUFFIX_ENDINGS {
        return None;
    }
    Some((suffix_len, ending_count))
}

#[cfg(test)]
fn suffix_regex(suffix_len: u32, modulus: u64, ending_count: u64) -> String {
    if suffix_len == 0 {
        return r"-?[0-9]*".to_string();
    }
    let len = suffix_len as usize;
    let threshold = 10u64.pow(suffix_len - 1);
    let padded: Vec<String> = (0..ending_count)
        .map(|index| format!("{:0len$}", index * modulus))
        .collect();
    let short: Vec<String> = (0..ending_count)
        .map(|index| index * modulus)
        .take_while(|ending| *ending < threshold)
        .map(|ending| ending.to_string())
        .collect();
    format!("-?(?:{}|[0-9]*(?:{}))", short.join("|"), padded.join("|"))
}

fn suffix_engine(
    suffix_len: u32,
    modulus: u64,
    ending_count: u64,
) -> Result<RefEngine, CompileError> {
    if suffix_len == 0 {
        return signed_digit_star_engine();
    }
    let width = usize::try_from(suffix_len).map_err(|_| limit("multipleOf suffix width"))?;
    let count = usize::try_from(ending_count).map_err(|_| limit("multipleOf ending count"))?;
    let per_ending = std::mem::size_of::<Vec<u8>>()
        .checked_mul(2)
        .and_then(|bytes| {
            width
                .checked_mul(2)
                .and_then(|width| bytes.checked_add(width))
        })
        .ok_or_else(|| limit("multipleOf suffix memory"))?;
    let retained = count
        .checked_mul(per_ending)
        .ok_or_else(|| limit("multipleOf suffix memory"))?;
    let _permit = acquire_automaton_build_bytes(retained)?;

    let mut padded = Vec::new();
    padded
        .try_reserve_exact(count)
        .map_err(|_| limit("multipleOf padded ending allocation"))?;
    let mut short = Vec::new();
    short
        .try_reserve_exact(count)
        .map_err(|_| limit("multipleOf short ending allocation"))?;
    let threshold = if suffix_len == 0 {
        0
    } else {
        10u64
            .checked_pow(suffix_len - 1)
            .ok_or_else(|| limit("multipleOf suffix width"))?
    };
    for index in 0..ending_count {
        let ending = index
            .checked_mul(modulus)
            .ok_or_else(|| limit("multipleOf ending value"))?;
        padded.push(decimal_bytes(ending, width)?);
        if ending < threshold {
            short.push(decimal_bytes(ending, 0)?);
        }
    }
    suffix_parts_engine(&short, &padded)
}

fn suffix_parts_engine(short: &[Vec<u8>], padded: &[Vec<u8>]) -> Result<RefEngine, CompileError> {
    let short_engine = RefEngine::from_graph(&byte_string_set(short)?)?;
    let padded_engine = RefEngine::from_graph(&byte_string_set(padded)?)?;
    let digit_star = digit_star_engine()?;
    let long = RefEngine::from_graph(&concatenate(&[&digit_star, &padded_engine])?)?;
    let body = RefEngine::from_graph(&union(&[&short_engine, &long])?)?;
    let sign = optional_sign_engine()?;
    RefEngine::from_graph(&concatenate(&[&sign, &body])?)
}

fn signed_digit_star_engine() -> Result<RefEngine, CompileError> {
    let sign = optional_sign_engine()?;
    let digits = digit_star_engine()?;
    RefEngine::from_graph(&concatenate(&[&sign, &digits])?)
}

fn optional_sign_engine() -> Result<RefEngine, CompileError> {
    RefEngine::from_graph(&AutomatonGraph::from_byte_states(
        &[(true, vec![(b'-', 1)]), (true, Vec::new())],
        0,
    )?)
}

fn digit_star_engine() -> Result<RefEngine, CompileError> {
    RefEngine::from_graph(&AutomatonGraph::from_byte_states(
        &[(true, (b'0'..=b'9').map(|byte| (byte, 0)).collect())],
        0,
    )?)
}

fn decimal_bytes(value: u64, width: usize) -> Result<Vec<u8>, CompileError> {
    let natural = if value == 0 {
        1
    } else {
        usize::try_from(value.ilog10() + 1).map_err(|_| limit("multipleOf digit count"))?
    };
    let len = width.max(natural);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| limit("multipleOf ending allocation"))?;
    bytes.resize(len, b'0');
    let mut remaining = value;
    for index in (0..len).rev() {
        let digit = u8::try_from(remaining % 10).map_err(|_| limit("multipleOf digit"))?;
        bytes[index] = b'0' + digit;
        remaining /= 10;
    }
    Ok(bytes)
}

#[cfg(test)]
fn remainder_dfa_regex(modulus: u64) -> Result<String, CompileError> {
    let states = remainder_dfa_states(modulus)?;
    let elimination: Vec<EliminationState> = states
        .into_iter()
        .map(|(accepting, edges)| EliminationState { accepting, edges })
        .collect();
    to_regex_via_state_elimination(&elimination, 0, MAX_ELIMINATION_BYTES)
}

fn remainder_dfa_engine(modulus: u64) -> Result<RefEngine, CompileError> {
    let graph = AutomatonGraph::from_byte_states(&remainder_dfa_states(modulus)?, 0)?;
    RefEngine::from_graph(&graph)
}

fn remainder_dfa_states(modulus: u64) -> Result<Vec<ByteState>, CompileError> {
    let start = 0usize;
    let sign = 1usize;
    let accept = remainder_index(0)?;
    let state_count = usize::try_from(modulus)
        .map_err(|_| limit("multipleOf state count"))?
        .checked_add(2)
        .ok_or_else(|| limit("multipleOf state count"))?;

    let mut states = Vec::new();
    states
        .try_reserve_exact(state_count)
        .map_err(|_| limit("multipleOf state allocation"))?;
    for i in 0..state_count {
        let mut edges = Vec::new();
        edges
            .try_reserve_exact(11)
            .map_err(|_| limit("multipleOf edge allocation"))?;
        if i == start || i == sign {
            if i == start {
                edges.push((
                    b'-',
                    u32::try_from(sign).map_err(|_| limit("sign state id"))?,
                ));
            }
            for d in 0u8..=9 {
                let target = remainder_index(u64::from(d) % modulus)?;
                edges.push((
                    b'0' + d,
                    u32::try_from(target).map_err(|_| limit("multipleOf state id"))?,
                ));
            }
        } else {
            let r = u64::try_from(i - 2).map_err(|_| limit("multipleOf remainder"))?;
            for d in 0u8..=9 {
                let target = remainder_index((r * 10 + u64::from(d)) % modulus)?;
                edges.push((
                    b'0' + d,
                    u32::try_from(target).map_err(|_| limit("multipleOf state id"))?,
                ));
            }
        }
        states.push((i == accept, edges));
    }
    Ok(states)
}

fn remainder_index(remainder: u64) -> Result<usize, CompileError> {
    usize::try_from(remainder)
        .map_err(|_| limit("multipleOf remainder"))?
        .checked_add(2)
        .ok_or_else(|| limit("multipleOf state id"))
}

#[cfg(test)]
const MAX_DECIMAL_MULTIPLE_OF_EXP: u32 = 12;

#[cfg(test)]
pub(crate) fn decimal_multiple_of_regex(coef: u64, exp: u32) -> Result<String, CompileError> {
    if coef == 0 {
        return Err(malformed("multipleOf must be positive"));
    }
    if exp > MAX_DECIMAL_MULTIPLE_OF_EXP {
        return Err(limit("multipleOf decimal scale"));
    }
    if coef == 1 {
        return Ok(at_most_places_regex(exp));
    }
    if exp == 0 {
        return Ok(format!("(?:{})(?:\\.0+)?", multiple_of_regex(coef)?));
    }
    if let Some(regex) = decimal_power_of_ten_suffix(coef, exp) {
        return Ok(regex);
    }
    decimal_remainder_dfa_regex(coef, exp)
}

#[cfg(test)]
fn factor_two_five(coef: u64) -> Option<(u32, u32)> {
    let (mut rest, mut a, mut b) = (coef, 0u32, 0u32);
    while rest % 2 == 0 {
        rest /= 2;
        a += 1;
    }
    while rest % 5 == 0 {
        rest /= 5;
        b += 1;
    }
    (rest == 1).then_some((a, b))
}

#[cfg(test)]
fn decimal_power_of_ten_suffix(coef: u64, exp: u32) -> Option<String> {
    let (a, b) = factor_two_five(coef)?;
    if exp < a.max(b) {
        return None;
    }
    let ten_exp = 10u64.checked_pow(exp)?;
    let ending_count = ten_exp / coef;
    if ending_count > MAX_SUFFIX_ENDINGS {
        return None;
    }
    let width = exp as usize;
    let mut sigs: Vec<String> = Vec::new();
    for k in 1..ending_count {
        let padded = format!("{:0width$}", k * coef);
        sigs.push(padded.trim_end_matches('0').to_string());
    }
    Some(format!(r"-?[0-9]+(?:\.(?:{})0*|\.0+)?", sigs.join("|")))
}

#[cfg(test)]
fn at_most_places_regex(exp: u32) -> String {
    if exp == 0 {
        return r"-?[0-9]+(?:\.0+)?".to_string();
    }
    format!(r"-?[0-9]+(?:\.[0-9]{{1,{exp}}}0*)?")
}

#[cfg(test)]
fn decimal_remainder_dfa_regex(coef: u64, exp: u32) -> Result<String, CompileError> {
    let m = usize::try_from(coef).map_err(|_| limit("multipleOf modulus"))?;
    if coef > MAX_MULTIPLE_OF_MODULUS {
        return Err(limit("multipleOf modulus"));
    }
    let scale = exp as usize;
    let pow10 = |c: usize| -> u64 { (0..c).fold(1u64, |acc, _| (acc * 10) % coef) };
    let modn = |v: u64| v % coef;
    let int_state = |r: u64| 2 + r as usize;
    let frac_state = |r: u64, c: usize| 2 + m + c * m + r as usize;
    let dead = 2 + m + (scale + 1) * m;
    let total = dead + 1;

    let mut states: Vec<EliminationState> = Vec::with_capacity(total);
    for _ in 0..total {
        states.push(EliminationState {
            accepting: false,
            edges: Vec::new(),
        });
    }
    let digit = |d: u8| b'0' + d;
    states[0].edges.push((b'-', 1u32));
    for s in [0usize, 1usize] {
        for d in 0u8..=9 {
            states[s]
                .edges
                .push((digit(d), int_state(modn(u64::from(d))) as u32));
        }
        states[s].edges.push((b'.', frac_state(0, scale) as u32));
    }
    for r in 0..coef {
        let st = int_state(r);
        for d in 0u8..=9 {
            let nr = modn(r * 10 + u64::from(d));
            states[st].edges.push((digit(d), int_state(nr) as u32));
        }
        states[st].edges.push((b'.', frac_state(r, scale) as u32));
        states[st].accepting = modn(r * pow10(scale)) == 0;
    }
    for c in 0..=scale {
        for r in 0..coef {
            let st = frac_state(r, c);
            if c > 0 {
                for d in 0u8..=9 {
                    let nr = modn(r * 10 + u64::from(d));
                    states[st]
                        .edges
                        .push((digit(d), frac_state(nr, c - 1) as u32));
                }
            } else {
                states[st].edges.push((b'0', frac_state(r, 0) as u32));
            }
            states[st].accepting = modn(r * pow10(c)) == 0;
        }
    }
    to_regex_via_state_elimination(&states, 0, MAX_ELIMINATION_BYTES)
}

fn malformed(message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L2, message)
}

fn limit(what: &'static str) -> CompileError {
    let mut e = CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L2,
        "multipleOf cap",
    );
    e.observed = Some(what.to_string());
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::build_from_regex;
    use crate::automaton::byte_dfa::assert_exact_equivalence;
    use crate::automaton::ProductAutomaton;

    fn engine(modulus: u64) -> crate::automaton::RefEngine {
        multiple_of_engine(modulus).unwrap()
    }

    #[test]
    fn direct_remainder_graph_is_exactly_equivalent_to_elimination_oracle() {
        for modulus in [3, 6, 7, 8] {
            let old = build_from_regex(&remainder_dfa_regex(modulus).unwrap()).unwrap();
            let new = remainder_dfa_engine(modulus).unwrap();
            assert_exact_equivalence(&old, &new);
        }
    }

    #[test]
    fn direct_power_suffix_language_is_exactly_equivalent_to_regex_oracle() {
        let integer = build_from_regex(r"(?:0|-?[1-9][0-9]*)").unwrap();
        for modulus in [1, 2, 5, 10] {
            let old_multiple = build_from_regex(&multiple_of_regex(modulus).unwrap()).unwrap();
            let new_multiple = multiple_of_engine(modulus).unwrap();
            let old = ProductAutomaton::build(
                &[&integer, &old_multiple],
                crate::automaton::Combinator::All,
            )
            .unwrap()
            .into_engine()
            .unwrap();
            let new = ProductAutomaton::build(
                &[&integer, &new_multiple],
                crate::automaton::Combinator::All,
            )
            .unwrap()
            .into_engine()
            .unwrap();
            assert_exact_equivalence(&old, &new);
        }
    }

    #[test]
    fn direct_suffix_path_above_remainder_cap_matches_integer_arithmetic() {
        for modulus in [10_000u64, 15_625] {
            let engine = multiple_of_engine(modulus).unwrap();
            for value in [0i64, 1, -1, 10_000, -10_000, 15_625, -15_625, 31_250] {
                assert_eq!(
                    engine.accepts(value.to_string().as_bytes()),
                    value.unsigned_abs() % modulus == 0,
                    "value={value} modulus={modulus}"
                );
            }
        }
    }

    fn assert_matches_brute_force(modulus: u64) {
        let e = engine(modulus);
        for n in -2000i64..=2000 {
            let want = n % (modulus as i64) == 0;
            let got = e.accepts(n.to_string().as_bytes());
            assert_eq!(got, want, "n={n} modulus={modulus}");
        }
    }

    fn decimal_is_multiple(s: &str, coef: u64, exp: u32) -> bool {
        let body = s.strip_prefix('-').unwrap_or(s);
        let (int_part, frac_part) = match body.split_once('.') {
            Some((i, f)) => (i, f),
            None => (body, ""),
        };
        let digits: String = format!("{int_part}{frac_part}");
        let p: i128 = digits.parse().unwrap();
        let fdigits = frac_part.len() as u32;
        let lhs = p * 10i128.pow(exp);
        let rhs = coef as i128 * 10i128.pow(fdigits);
        lhs % rhs == 0
    }

    fn assert_decimal_matches_oracle(coef: u64, exp: u32, candidates: &[&str]) {
        let e = build_from_regex(&decimal_multiple_of_regex(coef, exp).unwrap()).unwrap();
        for s in candidates {
            let want = decimal_is_multiple(s, coef, exp);
            assert_eq!(e.accepts(s.as_bytes()), want, "s={s} coef={coef} exp={exp}");
        }
    }

    #[test]
    fn decimal_multiple_of_agrees_with_exact_oracle_on_corpus_divisors() {
        let candidates = [
            "0", "1", "2", "3", "16", "-16", "256", "12", "-1", "10", "100", "-100", "1.0", "2.0",
            "1.5", "-1.5", "0.5", "0.25", "0.50", "0.75", "1.25", "0.1", "0.01", "0.10", "0.100",
            "0.3", "0.03", "1.23", "1.230", "1.234", "12.34", "3.14", "-0.25", "16.0", "16.5",
            "0.001", "2.5", "-2.5", "100.00",
        ];
        for (coef, exp) in [
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (16, 0),
            (3, 0),
            (25, 2),
            (5, 1),
        ] {
            assert_decimal_matches_oracle(coef, exp, &candidates);
        }
    }

    #[test]
    fn power_of_two_and_five_moduli_agree_with_brute_force() {
        for modulus in [
            1u64, 2, 4, 5, 8, 10, 16, 20, 25, 50, 100, 125, 200, 250, 500, 1000,
        ] {
            assert_matches_brute_force(modulus);
        }
    }

    #[test]
    fn small_non_power_of_ten_moduli_agree_with_brute_force_via_the_direct_dfa() {
        for modulus in [3u64, 6, 7, 8] {
            assert_matches_brute_force(modulus);
        }
    }

    #[test]
    fn a_non_power_of_ten_modulus_past_the_measured_safe_cap_is_a_limit_error_not_a_blowup() {
        let err = multiple_of_regex(9).unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn a_pure_power_with_too_many_suffix_endings_and_no_direct_dfa_fallback_is_a_limit_error() {
        let err = multiple_of_regex(128).unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn a_pure_power_just_under_the_suffix_ending_cap_builds_quickly_and_correctly() {
        assert_matches_brute_force(64); // 2^6: 5^6=15625 endings, well under the cap
    }

    #[test]
    fn zero_is_divisible_by_every_modulus() {
        for modulus in [2u64, 10, 100, 7, 1000] {
            assert!(engine(modulus).accepts(b"0"));
        }
    }

    #[test]
    fn zero_modulus_is_rejected() {
        let err = multiple_of_regex(0).unwrap_err();
        assert_eq!(err.code, ErrorCode::Malformed);
    }

    #[test]
    fn modulus_over_the_absolute_cap_is_a_structured_limit_error() {
        let err = multiple_of_regex(MAX_MULTIPLE_OF_MODULUS + 1).unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn a_large_non_power_of_ten_modulus_is_a_structured_limit_error_not_a_multi_megabyte_regex() {
        let err = multiple_of_regex(999_983).unwrap_err(); // a large prime
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn negative_multiples_are_accepted_for_both_construction_paths() {
        let e = engine(4); // power-of-two path
        assert!(e.accepts(b"-8"));
        assert!(!e.accepts(b"-6"));
        let e = engine(3); // direct remainder-DFA path
        assert!(e.accepts(b"-9"));
        assert!(!e.accepts(b"-8"));
    }

    #[test]
    fn suffix_path_rejects_a_short_number_that_is_not_zero_or_a_true_multiple() {
        let e = engine(1000);
        assert!(!e.accepts(b"5"));
        assert!(e.accepts(b"1000"));
        assert!(!e.accepts(b"1001"));
    }

    #[test]
    fn intersecting_multiple_of_with_a_range_via_the_shared_product_automaton() {
        use crate::automaton::{Combinator, ProductAutomaton};
        let range = build_from_regex(&crate::compile::integer_regex(Some(0), Some(100))).unwrap();
        let mult = engine(10);
        let product = ProductAutomaton::build(&[&range, &mult], Combinator::All).unwrap();
        let e = build_from_regex(&product.to_regex().unwrap()).unwrap();
        assert!(e.accepts(b"0"));
        assert!(e.accepts(b"10"));
        assert!(e.accepts(b"100"));
        assert!(!e.accepts(b"110"), "out of the [0,100] range");
        assert!(!e.accepts(b"15"), "not a multiple of 10");
        assert!(!e.accepts(b"-10"), "out of the [0,100] range");
    }
}
