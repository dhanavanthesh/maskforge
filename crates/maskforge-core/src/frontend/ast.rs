//! Parses JSON into a source-preserving syntax tree.

use std::mem::size_of;

use crate::error::{CompileError, ErrorCode, LimitKind, Stage};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Ast {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Ast>),
    Obj(Vec<(String, Ast)>),
}

const MAX_PARSE_DEPTH: u32 = 1024;

pub fn parse(text: &str) -> Result<Ast, CompileError> {
    parse_bounded(text, usize::MAX, usize::MAX).map(|(ast, _)| ast)
}

pub(crate) fn parse_bounded(
    text: &str,
    max_nodes: usize,
    max_bytes: usize,
) -> Result<(Ast, AstMemory), CompileError> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        pos: 0,
        depth: 0,
        memory: AstMemory::new(max_nodes, max_bytes),
    };
    p.skip_ws();
    let value = p.value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(p.malformed("trailing bytes after the JSON document"));
    }
    Ok((value, p.memory))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AstMemory {
    pub(crate) nodes: usize,
    pub(crate) bytes: usize,
    max_nodes: usize,
    max_bytes: usize,
}

impl AstMemory {
    fn new(max_nodes: usize, max_bytes: usize) -> Self {
        Self {
            nodes: 0,
            bytes: 0,
            max_nodes,
            max_bytes,
        }
    }

    fn add_node(&mut self) -> Result<(), CompileError> {
        self.nodes = self.nodes.checked_add(1).ok_or_else(ast_limit_overflow)?;
        self.check(LimitKind::AstNodeCount, self.nodes, self.max_nodes)
    }

    fn add_bytes(&mut self, bytes: usize) -> Result<(), CompileError> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(ast_limit_overflow)?;
        self.check(LimitKind::AstBytes, self.bytes, self.max_bytes)
    }

    fn ensure_bytes(&self, bytes: usize) -> Result<(), CompileError> {
        let observed = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(ast_limit_overflow)?;
        self.check(LimitKind::AstBytes, observed, self.max_bytes)
    }

    fn check(&self, kind: LimitKind, observed: usize, cap: usize) -> Result<(), CompileError> {
        if observed <= cap {
            return Ok(());
        }
        let mut error = CompileError::new(
            ErrorCode::InternalLimitExceeded,
            Stage::L1,
            "schema AST resource limit exceeded",
        );
        error.limit = Some((kind, observed, cap));
        Err(error)
    }
}

fn ast_limit_overflow() -> CompileError {
    let mut error = CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L1,
        "schema AST size overflow",
    );
    error.limit = Some((LimitKind::AstBytes, usize::MAX, usize::MAX));
    error
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: u32,
    memory: AstMemory,
}

impl Parser<'_> {
    fn malformed(&self, message: &'static str) -> CompileError {
        CompileError::new(ErrorCode::Malformed, Stage::L1, message)
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn value(&mut self) -> Result<Ast, CompileError> {
        self.memory.add_node()?;
        match self.peek() {
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => Ok(Ast::Str(self.string()?)),
            Some(b't') => self.literal("true", Ast::Bool(true)),
            Some(b'f') => self.literal("false", Ast::Bool(false)),
            Some(b'n') => self.literal("null", Ast::Null),
            Some(b) if b == b'-' || b.is_ascii_digit() => self.number(),
            _ => Err(self.malformed("expected a JSON value")),
        }
    }

    fn nested(
        &mut self,
        f: fn(&mut Self) -> Result<Ast, CompileError>,
    ) -> Result<Ast, CompileError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(CompileError::new(
                ErrorCode::InternalLimitExceeded,
                Stage::L1,
                "JSON nesting is too deep",
            ));
        }
        let result = f(self);
        self.depth -= 1;
        result
    }

    fn literal(&mut self, word: &str, ast: Ast) -> Result<Ast, CompileError> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(ast)
        } else {
            Err(self.malformed("invalid literal"))
        }
    }

    fn object(&mut self) -> Result<Ast, CompileError> {
        self.pos += 1; // '{'
        let mut fields = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Ast::Obj(fields));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.malformed("expected an object key"));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.malformed("expected ':' after an object key"));
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.value()?;
            self.reserve_vec(&mut fields, 1)?;
            fields.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Ast::Obj(fields));
                }
                _ => return Err(self.malformed("expected ',' or '}' in an object")),
            }
        }
    }

    fn array(&mut self) -> Result<Ast, CompileError> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Ast::Arr(items));
        }
        loop {
            self.skip_ws();
            let value = self.value()?;
            self.reserve_vec(&mut items, 1)?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Ast::Arr(items));
                }
                _ => return Err(self.malformed("expected ',' or ']' in an array")),
            }
        }
    }

    fn digits(&mut self) -> Result<(), CompileError> {
        let start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(self.malformed("expected a digit"));
        }
        Ok(())
    }

    fn number(&mut self) -> Result<Ast, CompileError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => self.digits()?,
            _ => return Err(self.malformed("expected a digit")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            self.digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            self.digits()?;
        }
        let token = &self.bytes[start..self.pos];
        let token = std::str::from_utf8(token).map_err(|_| self.malformed("invalid number"))?;
        self.memory.ensure_bytes(token.len())?;
        let mut s = String::new();
        s.try_reserve_exact(token.len())
            .map_err(|_| ast_limit_overflow())?;
        self.memory.add_bytes(s.capacity())?;
        s.push_str(token);
        Ok(Ast::Num(s))
    }

    fn string(&mut self) -> Result<String, CompileError> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err(self.malformed("unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    self.escape(&mut out)?;
                }
                Some(b) if b < 0x20 => return Err(self.malformed("control byte in a string")),
                Some(_) => {
                    let ch = self.utf8_char()?;
                    self.reserve_string(&mut out, ch.len_utf8())?;
                    out.push(ch);
                }
            }
        }
    }

    fn escape(&mut self, out: &mut String) -> Result<(), CompileError> {
        self.reserve_string(out, 1)?;
        match self.peek() {
            Some(b'"') => out.push('"'),
            Some(b'\\') => out.push('\\'),
            Some(b'/') => out.push('/'),
            Some(b'b') => out.push('\u{08}'),
            Some(b'f') => out.push('\u{0c}'),
            Some(b'n') => out.push('\n'),
            Some(b'r') => out.push('\r'),
            Some(b't') => out.push('\t'),
            Some(b'u') => return self.unicode_escape(out),
            _ => return Err(self.malformed("invalid string escape")),
        }
        self.pos += 1;
        Ok(())
    }

    fn unicode_escape(&mut self, out: &mut String) -> Result<(), CompileError> {
        self.pos += 1; // 'u'
        let hi = self.hex4()?;
        let code = if (0xD800..=0xDBFF).contains(&hi) {
            if self.bytes.get(self.pos) != Some(&b'\\')
                || self.bytes.get(self.pos + 1) != Some(&b'u')
            {
                return Err(self.malformed("unpaired high surrogate"));
            }
            self.pos += 2;
            let lo = self.hex4()?;
            if !(0xDC00..=0xDFFF).contains(&lo) {
                return Err(self.malformed("invalid low surrogate"));
            }
            0x1_0000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00)
        } else if (0xDC00..=0xDFFF).contains(&hi) {
            return Err(self.malformed("unexpected low surrogate"));
        } else {
            u32::from(hi)
        };
        let scalar = char::from_u32(code).ok_or_else(|| self.malformed("invalid code point"))?;
        self.reserve_string(out, scalar.len_utf8())?;
        out.push(scalar);
        Ok(())
    }

    fn reserve_vec<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), CompileError> {
        if values
            .len()
            .checked_add(additional)
            .ok_or_else(ast_limit_overflow)?
            <= values.capacity()
        {
            return Ok(());
        }
        self.memory.ensure_bytes(
            additional
                .checked_mul(size_of::<T>())
                .ok_or_else(ast_limit_overflow)?,
        )?;
        let before = values.capacity();
        values
            .try_reserve_exact(additional)
            .map_err(|_| ast_limit_overflow())?;
        self.memory.add_bytes(
            values
                .capacity()
                .checked_sub(before)
                .and_then(|growth| growth.checked_mul(size_of::<T>()))
                .ok_or_else(ast_limit_overflow)?,
        )
    }

    fn reserve_string(
        &mut self,
        value: &mut String,
        additional: usize,
    ) -> Result<(), CompileError> {
        if value
            .len()
            .checked_add(additional)
            .ok_or_else(ast_limit_overflow)?
            <= value.capacity()
        {
            return Ok(());
        }
        self.memory.ensure_bytes(additional)?;
        let before = value.capacity();
        value
            .try_reserve_exact(additional)
            .map_err(|_| ast_limit_overflow())?;
        self.memory.add_bytes(
            value
                .capacity()
                .checked_sub(before)
                .ok_or_else(ast_limit_overflow)?,
        )
    }

    fn hex4(&mut self) -> Result<u16, CompileError> {
        let slice = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.malformed("truncated unicode escape"))?;
        let mut value: u16 = 0;
        for &b in slice {
            let digit = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(self.malformed("invalid hex digit")),
            };
            value = value * 16 + u16::from(digit);
        }
        self.pos += 4;
        Ok(value)
    }

    fn utf8_char(&mut self) -> Result<char, CompileError> {
        let rest = &self.bytes[self.pos..];
        let len = match rest[0] {
            0x00..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => return Err(self.malformed("invalid UTF-8 lead byte")),
        };
        let chunk = rest
            .get(..len)
            .ok_or_else(|| self.malformed("truncated UTF-8 sequence"))?;
        let s = std::str::from_utf8(chunk).map_err(|_| self.malformed("invalid UTF-8"))?;
        let ch = s
            .chars()
            .next()
            .ok_or_else(|| self.malformed("invalid UTF-8"))?;
        self.pos += len;
        Ok(ch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars_and_nesting_in_source_order() {
        let ast = parse(r#"{"b": 1, "a": [true, null, "x"]}"#).unwrap();
        let Ast::Obj(fields) = ast else { panic!() };
        assert_eq!(fields[0].0, "b");
        assert_eq!(fields[1].0, "a");
        assert_eq!(fields[0].1, Ast::Num("1".into()));
    }

    #[test]
    fn preserves_duplicate_object_keys() {
        let Ast::Obj(fields) = parse(r#"{"a":1,"a":2}"#).unwrap() else {
            panic!()
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].1, Ast::Num("1".into()));
        assert_eq!(fields[1].1, Ast::Num("2".into()));
    }

    #[test]
    fn keeps_the_raw_number_token() {
        assert_eq!(parse("-12").unwrap(), Ast::Num("-12".into()));
        assert_eq!(parse("3.5").unwrap(), Ast::Num("3.5".into()));
        assert_eq!(parse("1e9").unwrap(), Ast::Num("1e9".into()));
        assert_eq!(parse("0").unwrap(), Ast::Num("0".into()));
        assert_eq!(parse("0.5").unwrap(), Ast::Num("0.5".into()));
        assert_eq!(parse("1E+9").unwrap(), Ast::Num("1E+9".into()));
        assert_eq!(parse("-0").unwrap(), Ast::Num("-0".into()));
    }

    #[test]
    fn rejects_malformed_number_tokens() {
        for bad in [
            "-", "01", "-01", "1.", "1e", "1e+", "-1.", "00", "1..2", "1.2.3", "1e1e1", "+1", ".5",
        ] {
            let e = parse(bad).unwrap_err();
            assert_eq!(e.code, ErrorCode::Malformed, "{bad:?} should be Malformed");
        }
    }

    #[test]
    fn decodes_string_escapes_and_surrogate_pairs() {
        assert_eq!(parse(r#""a\"b""#).unwrap(), Ast::Str("a\"b".into()));
        assert_eq!(parse(r#""\n""#).unwrap(), Ast::Str("\n".into()));
        assert_eq!(parse(r#""😀""#).unwrap(), Ast::Str("\u{1F600}".into()));
    }

    #[test]
    fn rejects_malformed_documents() {
        for bad in [
            "{",
            "[1,]",
            r#"{"a" 1}"#,
            "nul",
            "1 2",
            r#""\uZZZZ""#,
            r#""\uD83D""#,
            "",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} should be malformed");
        }
    }

    #[test]
    fn rejects_control_byte_in_string() {
        assert!(parse("\"a\nb\"").is_err());
    }

    #[test]
    fn enforces_ast_node_limit_at_the_exact_boundary() {
        let (_, memory) = parse_bounded("[null,true]", 3, usize::MAX).unwrap();
        assert_eq!(memory.nodes, 3);

        let error = parse_bounded("[null,true]", 2, usize::MAX).unwrap_err();
        assert_eq!(error.limit, Some((LimitKind::AstNodeCount, 3, 2)));
    }

    #[test]
    fn enforces_ast_bytes_from_actual_container_capacity() {
        let source = r#"{"long-key":["abcdefgh",12]}"#;
        let (_, measured) = parse_bounded(source, usize::MAX, usize::MAX).unwrap();
        assert!(measured.bytes > 0);
        parse_bounded(source, usize::MAX, measured.bytes).unwrap();

        let cap = measured.bytes - 1;
        let error = parse_bounded(source, usize::MAX, cap).unwrap_err();
        let Some((LimitKind::AstBytes, observed, configured)) = error.limit else {
            panic!("expected AST byte limit error, got {error:?}");
        };
        assert!(observed > configured);
        assert_eq!(configured, cap);
    }
}
