//! Incrementally validates one unconstrained JSON value without recursion.

/// JSON number syntax, tracked without buffering the digits: only the shape needed to know
/// which bytes may follow and whether the number is complete right now.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NumberPhase {
    IntFirst,
    IntRest { leading_zero: bool },
    FracFirst,
    FracRest,
    ExpSign,
    ExpFirst,
    ExpRest,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ScalarPhase {
    BeforeValue,
    True(u8),
    False(u8),
    Null(u8),
    Number(NumberPhase),
    /// A literal's own final matching byte is consumed into this phase; the NEXT byte is what
    /// decides `Done` (mirrors how a number's last digit still just `Continue`s).
    LiteralDone,
}

/// One step of a scalar (non-container) transition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ScalarStep {
    Continue(ScalarPhase),
    /// The scalar is a string: the caller switches to string-body decoding.
    StartString,
    /// The scalar is a container: the caller pushes an array/object frame.
    StartArray,
    StartObject,
    /// The scalar is already complete; the byte belongs to the parent, not this value.
    Done,
    Invalid,
}

/// Advances `phase` by one byte of a scalar (or dispatches into a string/array/object start).
pub(crate) fn step_scalar(phase: ScalarPhase, byte: u8) -> ScalarStep {
    match phase {
        ScalarPhase::BeforeValue => match byte {
            b'"' => ScalarStep::StartString,
            b'[' => ScalarStep::StartArray,
            b'{' => ScalarStep::StartObject,
            b't' => ScalarStep::Continue(ScalarPhase::True(1)),
            b'f' => ScalarStep::Continue(ScalarPhase::False(1)),
            b'n' => ScalarStep::Continue(ScalarPhase::Null(1)),
            b'-' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::IntFirst)),
            b'0' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::IntRest {
                leading_zero: true,
            })),
            b'1'..=b'9' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::IntRest {
                leading_zero: false,
            })),
            _ => ScalarStep::Invalid,
        },
        ScalarPhase::True(n) => step_literal(b"true", n, byte),
        ScalarPhase::False(n) => step_literal(b"false", n, byte),
        ScalarPhase::Null(n) => step_literal(b"null", n, byte),
        ScalarPhase::Number(n) => step_number(n, byte),
        ScalarPhase::LiteralDone => ScalarStep::Done,
    }
}

fn step_literal(word: &'static [u8], matched: u8, byte: u8) -> ScalarStep {
    if (matched as usize) >= word.len() || byte != word[matched as usize] {
        return ScalarStep::Invalid;
    }
    let matched = matched + 1;
    if matched as usize == word.len() {
        ScalarStep::Continue(ScalarPhase::LiteralDone)
    } else {
        ScalarStep::Continue(match word[0] {
            b't' => ScalarPhase::True(matched),
            b'f' => ScalarPhase::False(matched),
            _ => ScalarPhase::Null(matched),
        })
    }
}

fn step_number(phase: NumberPhase, byte: u8) -> ScalarStep {
    match phase {
        NumberPhase::IntFirst => match byte {
            b'0' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::IntRest {
                leading_zero: true,
            })),
            b'1'..=b'9' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::IntRest {
                leading_zero: false,
            })),
            _ => ScalarStep::Invalid,
        },
        NumberPhase::IntRest { leading_zero } => match byte {
            b'0'..=b'9' if !leading_zero => {
                ScalarStep::Continue(ScalarPhase::Number(NumberPhase::IntRest { leading_zero }))
            }
            b'.' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::FracFirst)),
            b'e' | b'E' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::ExpSign)),
            _ => ScalarStep::Done,
        },
        NumberPhase::FracFirst => match byte {
            b'0'..=b'9' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::FracRest)),
            _ => ScalarStep::Invalid,
        },
        NumberPhase::FracRest => match byte {
            b'0'..=b'9' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::FracRest)),
            b'e' | b'E' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::ExpSign)),
            _ => ScalarStep::Done,
        },
        NumberPhase::ExpSign => match byte {
            b'+' | b'-' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::ExpFirst)),
            b'0'..=b'9' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::ExpRest)),
            _ => ScalarStep::Invalid,
        },
        NumberPhase::ExpFirst => match byte {
            b'0'..=b'9' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::ExpRest)),
            _ => ScalarStep::Invalid,
        },
        NumberPhase::ExpRest => match byte {
            b'0'..=b'9' => ScalarStep::Continue(ScalarPhase::Number(NumberPhase::ExpRest)),
            _ => ScalarStep::Done,
        },
    }
}

/// Whether `phase` is already a syntactically complete value (so an unrecognized byte belongs
/// to the parent, not a rejection).
pub(crate) fn scalar_is_complete(phase: ScalarPhase) -> bool {
    matches!(
        phase,
        ScalarPhase::LiteralDone
            | ScalarPhase::Number(
                NumberPhase::IntRest { .. } | NumberPhase::FracRest | NumberPhase::ExpRest
            )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(bytes: &[u8]) -> Option<ScalarPhase> {
        let mut phase = ScalarPhase::BeforeValue;
        for &b in bytes {
            match step_scalar(phase, b) {
                ScalarStep::Continue(next) => phase = next,
                ScalarStep::Done | ScalarStep::Invalid => return None,
                ScalarStep::StartString | ScalarStep::StartArray | ScalarStep::StartObject => {
                    return None
                }
            }
        }
        Some(phase)
    }

    #[test]
    fn true_false_null_match_exactly() {
        assert!(matches!(run(b"tru"), Some(ScalarPhase::True(3))));
        assert!(matches!(
            step_scalar(ScalarPhase::True(3), b'e'),
            ScalarStep::Continue(ScalarPhase::LiteralDone)
        ));
        assert!(matches!(
            step_scalar(ScalarPhase::LiteralDone, b','),
            ScalarStep::Done
        ));
        assert!(matches!(run(b"fals"), Some(ScalarPhase::False(4))));
        assert!(matches!(run(b"nul"), Some(ScalarPhase::Null(3))));
    }

    #[test]
    fn a_leading_zero_forbids_more_int_digits() {
        let phase = run(b"0").unwrap();
        assert!(matches!(step_scalar(phase, b'1'), ScalarStep::Done));
    }

    #[test]
    fn negative_and_fractional_and_exponent_numbers_are_syntactically_valid() {
        for text in ["-12", "0.5", "1e10", "1E-10", "-0.25e+3"] {
            let phase = run(text.as_bytes());
            assert!(phase.is_some(), "{text}");
            assert!(scalar_is_complete(phase.unwrap()), "{text}");
        }
    }

    #[test]
    fn a_bare_leading_dot_or_trailing_dot_is_invalid() {
        assert!(matches!(
            step_scalar(ScalarPhase::BeforeValue, b'.'),
            ScalarStep::Invalid
        ));
        let after_int = run(b"1").unwrap();
        assert!(matches!(
            step_scalar(after_int, b'.'),
            ScalarStep::Continue(ScalarPhase::Number(NumberPhase::FracFirst))
        ));
        assert!(matches!(
            step_scalar(ScalarPhase::Number(NumberPhase::FracFirst), b'e'),
            ScalarStep::Invalid
        ));
    }
}
