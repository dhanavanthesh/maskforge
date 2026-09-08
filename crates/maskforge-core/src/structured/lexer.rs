//! Strict incremental JSON string decoding: byte-by-byte UTF-8 structural validation and escape
//! handling, so pattern/length constraints apply to canonical decoded scalars, not raw spelling.

/// State of the current JSON backslash-escape, if any.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EscapeState {
    None,
    Started,
    Unicode { digits: u8, value: u16 },
}

/// State of a raw (non-escaped) multi-byte UTF-8 sequence in progress.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Utf8State {
    Start,
    /// `first_min`/`first_max` bound only the next continuation byte when `is_first` - the one
    /// byte whose range depends on the lead byte to exclude overlong/surrogate/over-max encodings.
    Continuation {
        remaining: u8,
        cp: u32,
        first_min: u8,
        first_max: u8,
        is_first: bool,
    },
}

/// The effect of feeding one raw JSON-string-body byte into a [`JsonStringDecoder`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DecodeStep {
    /// The byte was consumed; no scalar value completed yet.
    Continue,
    /// The byte completed one decoded Unicode scalar value.
    Scalar(char),
    /// The byte is illegal here; this string body can never become valid JSON from this point.
    Invalid,
}

/// Incremental strict JSON string body decoder, fed one byte at a time between the quotes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct JsonStringDecoder {
    escape: EscapeState,
    utf8: Utf8State,
    pending_high_surrogate: Option<u16>,
    decoded_codepoints: u64,
}

/// `(continuation_count, first_continuation_min, first_continuation_max, initial_cp_bits)`, or
/// `None` if `lead` can never start a valid UTF-8 sequence.
fn utf8_lead_shape(lead: u8) -> Option<(u8, u8, u8, u32)> {
    match lead {
        0xc2..=0xdf => Some((1, 0x80, 0xbf, u32::from(lead & 0x1f))),
        0xe0 => Some((2, 0xa0, 0xbf, u32::from(lead & 0x0f))),
        0xe1..=0xec => Some((2, 0x80, 0xbf, u32::from(lead & 0x0f))),
        0xed => Some((2, 0x80, 0x9f, u32::from(lead & 0x0f))),
        0xee..=0xef => Some((2, 0x80, 0xbf, u32::from(lead & 0x0f))),
        0xf0 => Some((3, 0x90, 0xbf, u32::from(lead & 0x07))),
        0xf1..=0xf3 => Some((3, 0x80, 0xbf, u32::from(lead & 0x07))),
        0xf4 => Some((3, 0x80, 0x8f, u32::from(lead & 0x07))),
        _ => None,
    }
}

impl JsonStringDecoder {
    pub(crate) fn new() -> Self {
        Self {
            escape: EscapeState::None,
            utf8: Utf8State::Start,
            pending_high_surrogate: None,
            decoded_codepoints: 0,
        }
    }

    /// Scalars decoded so far, for a caller that wants length/count semantics fused into this
    /// same pass instead of a second full re-decode.
    #[cfg(any(test, feature = "bench-internals"))]
    pub(crate) fn decoded_codepoints(&self) -> u64 {
        self.decoded_codepoints
    }

    /// Whether the string could legally close right now (no incomplete escape, UTF-8 sequence, or
    /// unpaired high surrogate).
    pub(crate) fn at_boundary(&self) -> bool {
        self.escape == EscapeState::None
            && self.utf8 == Utf8State::Start
            && self.pending_high_surrogate.is_none()
    }

    /// Tests the scalar forced by a pending UTF-8 or `\uXXXX` sequence.
    /// `None` means no scalar is forced within `max_candidates`.
    #[cfg(test)]
    pub(crate) fn any_pending_scalar_satisfies(
        &self,
        max_candidates: usize,
        predicate: impl FnMut(char) -> bool,
    ) -> Option<bool> {
        self.any_pending_scalar_satisfies_with_supplementary(
            max_candidates,
            predicate,
            || None,
            |_, _| None,
        )
    }

    /// Extends [`Self::any_pending_scalar_satisfies`] with a symbolic query for a possible high
    /// surrogate. The callback receives the exact inclusive supplementary scalar interval.
    pub(crate) fn any_pending_scalar_satisfies_with_supplementary(
        &self,
        max_candidates: usize,
        mut predicate: impl FnMut(char) -> bool,
        mut any_scalar: impl FnMut() -> Option<bool>,
        mut supplementary_range: impl FnMut(u32, u32) -> Option<bool>,
    ) -> Option<bool> {
        fn supplementary(high: u16, low: u16) -> Option<char> {
            let cp = 0x10000
                + ((u32::from(high).checked_sub(0xd800)?) << 10)
                + u32::from(low).checked_sub(0xdc00)?;
            char::from_u32(cp)
        }

        fn any_low_surrogate(high: u16, predicate: &mut impl FnMut(char) -> bool) -> bool {
            (0xdc00..=0xdfff).any(|low| supplementary(high, low).is_some_and(&mut *predicate))
        }

        fn visit_utf8(
            decoder: JsonStringDecoder,
            predicate: &mut impl FnMut(char) -> bool,
        ) -> bool {
            let Utf8State::Continuation {
                first_min,
                first_max,
                is_first,
                ..
            } = decoder.utf8
            else {
                return false;
            };
            let (lo, hi) = if is_first {
                (first_min, first_max)
            } else {
                (0x80, 0xbf)
            };
            (lo..=hi).any(|byte| {
                let mut next = decoder;
                match next.push(byte) {
                    DecodeStep::Scalar(value) => predicate(value),
                    DecodeStep::Continue => visit_utf8(next, predicate),
                    DecodeStep::Invalid => false,
                }
            })
        }

        if let Utf8State::Continuation {
            remaining,
            first_min,
            first_max,
            is_first,
            ..
        } = self.utf8
        {
            let first = if is_first {
                usize::from(first_max - first_min) + 1
            } else {
                64
            };
            let tail = 64usize.checked_pow(u32::from(remaining.saturating_sub(1)))?;
            if first.checked_mul(tail)? > max_candidates {
                return None;
            }
            return Some(visit_utf8(*self, &mut predicate));
        }

        match (self.escape, self.pending_high_surrogate) {
            (EscapeState::Unicode { digits, value }, pending_high) => {
                let remaining = u32::from(4u8.checked_sub(digits)?);
                let count = 16usize.checked_pow(remaining)?;
                let shift = remaining.checked_mul(4)?;
                let start = u32::from(value).checked_shl(shift)?;
                let end = start.checked_add(u32::try_from(count).ok()?.checked_sub(1)?)?;
                if let Some(high) = pending_high {
                    let lo = start.max(0xdc00);
                    let hi = end.min(0xdfff);
                    return Some(
                        lo <= hi
                            && (lo..=hi).any(|low| {
                                supplementary(high, low as u16).is_some_and(&mut predicate)
                            }),
                    );
                }
                if start <= 0xdbff && end >= 0xd800 {
                    let high_lo = start.max(0xd800);
                    let high_hi = end.min(0xdbff);
                    let scalar_lo = 0x1_0000 + ((high_lo - 0xd800) << 10);
                    let scalar_hi = 0x1_0000 + ((high_hi - 0xd800) << 10) + 0x3ff;

                    let lower_bmp_hi = end.min(0xd7ff);
                    let lower_count = lower_bmp_hi
                        .checked_sub(start)
                        .and_then(|width| width.checked_add(1))
                        .unwrap_or(0);
                    let upper_bmp_lo = start.max(0xe000);
                    let upper_count = end
                        .checked_sub(upper_bmp_lo)
                        .and_then(|width| width.checked_add(1))
                        .unwrap_or(0);
                    let bmp_count = usize::try_from(lower_count.checked_add(upper_count)?).ok()?;
                    if bmp_count > max_candidates {
                        return None;
                    }
                    let bmp_viable = (start..=lower_bmp_hi)
                        .chain(upper_bmp_lo..=end)
                        .any(|cp| char::from_u32(cp).is_some_and(&mut predicate));
                    if bmp_viable {
                        return Some(true);
                    }
                    return supplementary_range(scalar_lo, scalar_hi);
                }
                if count > max_candidates {
                    return None;
                }
                Some((start..=end).any(|cp| {
                    !(0xd800..=0xdfff).contains(&cp)
                        && char::from_u32(cp).is_some_and(&mut predicate)
                }))
            }
            (EscapeState::None | EscapeState::Started, Some(high)) => {
                (max_candidates >= 1024).then(|| any_low_surrogate(high, &mut predicate))
            }
            (EscapeState::Started, None) => any_scalar(),
            (EscapeState::None, None) => None,
        }
    }

    pub(crate) fn push(&mut self, byte: u8) -> DecodeStep {
        if self.pending_high_surrogate.is_some() {
            let expects_u_escape = match self.escape {
                EscapeState::None => byte == b'\\',
                EscapeState::Started => byte == b'u',
                EscapeState::Unicode { .. } => true,
            };
            if !expects_u_escape {
                return DecodeStep::Invalid;
            }
        }
        match self.escape {
            EscapeState::None => self.push_body_byte(byte),
            EscapeState::Started => self.push_escape_selector(byte),
            EscapeState::Unicode { digits, value } => self.push_unicode_digit(byte, digits, value),
        }
    }

    fn push_body_byte(&mut self, byte: u8) -> DecodeStep {
        if let Utf8State::Continuation {
            remaining,
            cp,
            first_min,
            first_max,
            is_first,
        } = self.utf8
        {
            let (lo, hi) = if is_first {
                (first_min, first_max)
            } else {
                (0x80, 0xbf)
            };
            if byte < lo || byte > hi {
                return DecodeStep::Invalid;
            }
            let cp = (cp << 6) | u32::from(byte & 0x3f);
            if remaining == 1 {
                self.utf8 = Utf8State::Start;
                return self.emit_scalar_codepoint(cp);
            }
            self.utf8 = Utf8State::Continuation {
                remaining: remaining - 1,
                cp,
                first_min,
                first_max,
                is_first: false,
            };
            return DecodeStep::Continue;
        }
        match byte {
            0x00..=0x1f | 0x22 => DecodeStep::Invalid,
            0x5c => {
                self.escape = EscapeState::Started;
                DecodeStep::Continue
            }
            0x20..=0x7f => self.emit_scalar_codepoint(u32::from(byte)),
            _ => match utf8_lead_shape(byte) {
                Some((remaining, first_min, first_max, cp)) => {
                    self.utf8 = Utf8State::Continuation {
                        remaining,
                        cp,
                        first_min,
                        first_max,
                        is_first: true,
                    };
                    DecodeStep::Continue
                }
                None => DecodeStep::Invalid,
            },
        }
    }

    fn push_escape_selector(&mut self, byte: u8) -> DecodeStep {
        self.escape = EscapeState::None;
        match byte {
            b'"' => self.emit_scalar_codepoint(u32::from(b'"')),
            b'\\' => self.emit_scalar_codepoint(u32::from(b'\\')),
            b'/' => self.emit_scalar_codepoint(u32::from(b'/')),
            b'b' => self.emit_scalar_codepoint(0x08),
            b'f' => self.emit_scalar_codepoint(0x0c),
            b'n' => self.emit_scalar_codepoint(u32::from(b'\n')),
            b'r' => self.emit_scalar_codepoint(u32::from(b'\r')),
            b't' => self.emit_scalar_codepoint(u32::from(b'\t')),
            b'u' => {
                self.escape = EscapeState::Unicode {
                    digits: 0,
                    value: 0,
                };
                DecodeStep::Continue
            }
            _ => DecodeStep::Invalid,
        }
    }

    fn push_unicode_digit(&mut self, byte: u8, digits: u8, value: u16) -> DecodeStep {
        let Some(nibble) = (byte as char).to_digit(16) else {
            return DecodeStep::Invalid;
        };
        let value = (value << 4) | nibble as u16;
        if digits + 1 < 4 {
            self.escape = EscapeState::Unicode {
                digits: digits + 1,
                value,
            };
            return DecodeStep::Continue;
        }
        self.escape = EscapeState::None;
        if (0xd800..=0xdbff).contains(&value) {
            if self.pending_high_surrogate.is_some() {
                return DecodeStep::Invalid;
            }
            self.pending_high_surrogate = Some(value);
            return DecodeStep::Continue;
        }
        if (0xdc00..=0xdfff).contains(&value) {
            return match self.pending_high_surrogate.take() {
                Some(high) => {
                    let cp =
                        0x10000 + ((u32::from(high) - 0xd800) << 10) + (u32::from(value) - 0xdc00);
                    self.emit_scalar_codepoint(cp)
                }
                None => DecodeStep::Invalid,
            };
        }
        if self.pending_high_surrogate.take().is_some() {
            return DecodeStep::Invalid;
        }
        self.emit_scalar_codepoint(u32::from(value))
    }

    fn emit_scalar_codepoint(&mut self, cp: u32) -> DecodeStep {
        match char::from_u32(cp) {
            Some(c) => {
                self.decoded_codepoints += 1;
                DecodeStep::Scalar(c)
            }
            None => DecodeStep::Invalid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(bytes: &[u8]) -> Result<String, usize> {
        let mut dec = JsonStringDecoder::new();
        let mut out = String::new();
        for (i, &b) in bytes.iter().enumerate() {
            match dec.push(b) {
                DecodeStep::Continue => {}
                DecodeStep::Scalar(c) => out.push(c),
                DecodeStep::Invalid => return Err(i),
            }
        }
        if dec.at_boundary() {
            Ok(out)
        } else {
            Err(bytes.len())
        }
    }

    #[test]
    fn pending_scalar_search_is_bounded_and_exact_for_escape_and_utf8_states() {
        let mut unicode = JsonStringDecoder::new();
        for byte in br"\u2" {
            assert_ne!(unicode.push(*byte), DecodeStep::Invalid);
        }
        assert_eq!(
            unicode.any_pending_scalar_satisfies(4096, |value| value == '\u{2345}'),
            Some(true)
        );
        assert_eq!(
            unicode.any_pending_scalar_satisfies(4096, |value| value.is_ascii_lowercase()),
            Some(false)
        );

        let mut broad = JsonStringDecoder::new();
        for byte in br"\u" {
            assert_ne!(broad.push(*byte), DecodeStep::Invalid);
        }
        assert_eq!(
            broad.any_pending_scalar_satisfies(4096, |_| false),
            None,
            "the full BMP stays conservative rather than doing unbounded work"
        );

        let mut utf8 = JsonStringDecoder::new();
        assert_eq!(utf8.push(0xc3), DecodeStep::Continue);
        assert_eq!(
            utf8.any_pending_scalar_satisfies(64, |value| value == 'é'),
            Some(true)
        );
        assert_eq!(
            utf8.any_pending_scalar_satisfies(63, |_| false),
            None,
            "candidate cap is enforced"
        );
    }

    #[test]
    fn rejects_raw_ff() {
        assert!(decode(b"\xff").is_err());
    }

    #[test]
    fn rejects_c2_28() {
        assert!(decode(b"\xc2\x28").is_err());
    }

    #[test]
    fn rejects_overlong_c0_af() {
        assert!(decode(b"\xc0\xaf").is_err());
    }

    #[test]
    fn rejects_ed_a0_80_surrogate() {
        assert!(decode(b"\xed\xa0\x80").is_err());
    }

    #[test]
    fn rejects_f4_90_80_80_over_max() {
        assert!(decode(b"\xf4\x90\x80\x80").is_err());
    }

    #[test]
    fn rejects_isolated_low_surrogate() {
        assert!(decode(b"\\uDC00").is_err());
    }

    #[test]
    fn rejects_high_surrogate_without_low() {
        assert!(decode(b"\\uD800").is_err());
        assert!(decode(b"\\uD800x").is_err());
        assert!(decode(b"\\uD800\\n").is_err());
    }

    #[test]
    fn rejects_a_second_high_surrogate_before_the_first_is_paired() {
        assert!(decode(b"\\uD800\\uD801\\uDC00").is_err());
    }

    #[test]
    fn accepts_a_valid_surrogate_pair() {
        assert_eq!(decode(b"\\uD83D\\uDE00").unwrap(), "\u{1F600}");
    }

    #[test]
    fn decoded_codepoints_counts_scalars_not_bytes() {
        let mut dec = JsonStringDecoder::new();
        for b in *b"a\xe2\x82\xac\\uD83D\\uDE00" {
            dec.push(b);
        }
        assert_eq!(dec.decoded_codepoints(), 3);
    }

    #[test]
    fn raw_and_escaped_spellings_decode_to_the_same_scalar() {
        assert_eq!(decode(b"a").unwrap(), decode(b"\\u0061").unwrap());
    }

    #[test]
    fn split_valid_utf8_across_pushes_stays_viable_until_the_final_byte() {
        let mut dec = JsonStringDecoder::new();
        assert_eq!(dec.push(0xe2), DecodeStep::Continue);
        assert!(!dec.at_boundary());
        assert_eq!(dec.push(0x82), DecodeStep::Continue);
        assert_eq!(dec.push(0xac), DecodeStep::Scalar('\u{20ac}'));
        assert!(dec.at_boundary());
    }

    #[test]
    fn split_invalid_utf8_dies_at_the_first_impossible_byte() {
        let mut dec = JsonStringDecoder::new();
        assert_eq!(dec.push(0xe0), DecodeStep::Continue);
        assert_eq!(dec.push(0x80), DecodeStep::Invalid);
    }
}
