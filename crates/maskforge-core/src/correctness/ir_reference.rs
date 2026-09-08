//! Provides a reference lowering from schema IR to byte automata.

use crate::automaton::{build_from_regex, Combinator, ProductAutomaton, RefEngine};
use crate::compile::integer_regex;
use crate::error::{CompileError, ErrorCode, Stage};
use crate::ir::{Charset, ItemsPolicy, Node, ScalarLit, SchemaIR};
use crate::primitives::NodeId;

const MAX_DEPTH: u32 = 20_000;

const WS: &str = r"[\x20\x09\x0A\x0D]*";

pub(crate) fn reference_compile(ir: &SchemaIR) -> Result<RefEngine, CompileError> {
    if let Some(d) = ir.diagnostics().next() {
        let mut e = CompileError::new(ErrorCode::Unsupported, Stage::L2, d.reason.advisory());
        if let Some(p) = ir.str_at(d.json_pointer) {
            e = e.with_pointer(p.to_string());
        }
        return Err(e);
    }
    let pattern = lower(ir, ir.root(), 0)?;
    build_from_regex(&format!("{WS}(?:{pattern}){WS}"))
}

fn lower(ir: &SchemaIR, id: NodeId, depth: u32) -> Result<String, CompileError> {
    if depth > MAX_DEPTH {
        return Err(oops("reference lowering is too deep"));
    }
    let node = ir.node(id).ok_or_else(|| oops("node id out of range"))?;
    Ok(match node {
        Node::Null => "null".to_owned(),
        Node::Boolean => "(?:true|false)".to_owned(),
        Node::Never => r"[^\x00-\xff]".to_owned(),
        Node::StringConst { value } => {
            let raw = ir.str_at(*value).ok_or_else(|| oops("string ref"))?;
            quote(&encode(raw))
        }
        Node::StringPattern {
            regex,
            min_len,
            max_len,
            charset,
        } => {
            let base = ir.str_at(*regex).ok_or_else(|| oops("pattern ref"))?;
            let inner = match charset {
                Charset::AsciiPrintableNoQuoteBackslash | Charset::Utf8CountedCodepoints => {
                    format!("(?:{base}){}", rep(*min_len, *max_len))
                }
                Charset::Utf8Any => format!("(?:{base})"),
            };
            format!("\"{inner}\"")
        }
        Node::Integer {
            minimum,
            maximum,
            multiple_of,
        } => {
            let range = integer_regex(*minimum, *maximum);
            match multiple_of {
                None => range,
                Some(m) => {
                    let a = build_from_regex(&range)?;
                    let b = build_from_regex(&crate::automaton::multiple_of_regex(*m)?)?;
                    ProductAutomaton::build(&[&a, &b], Combinator::All)?.to_regex()?
                }
            }
        }
        Node::Number {
            integer_only,
            minimum,
            maximum,
            multiple_of,
        } => {
            if *integer_only {
                return Err(oops("mathematical integer requires structured evaluation"));
            }
            let narrow = |bound: Option<(i128, u32, bool)>| {
                bound
                    .map(|(coefficient, scale, exclusive)| {
                        i64::try_from(coefficient)
                            .map(|coefficient| (coefficient, scale, exclusive))
                    })
                    .transpose()
                    .map_err(|_| oops("wide decimal bound requires structured evaluation"))
            };
            crate::compile::number_node_regex(narrow(*minimum)?, narrow(*maximum)?, *multiple_of)
                .map_err(|_| oops("number node regex"))?
        }
        Node::LexicalNumber { regex } => ir
            .str_at(*regex)
            .ok_or_else(|| oops("lexical number pattern"))?
            .to_owned(),
        Node::Enum { values } => {
            let lits = ir.lits_at(*values).ok_or_else(|| oops("enum ref"))?;
            let mut alts: Vec<String> = lits.iter().map(lit).collect();
            alts.sort_by_key(|a| std::cmp::Reverse(a.len()));
            format!("(?:{})", alts.join("|"))
        }
        Node::Array {
            unique_items: true, ..
        }
        | Node::Array {
            items: ItemsPolicy::AllowAny,
            ..
        }
        | Node::Array {
            contains: Some(_), ..
        } => {
            return Err(oops(
                "uniqueItems/open items/contains is not a regular language",
            ))
        }
        Node::Array {
            items: ItemsPolicy::Schema(id),
            min_items,
            max_items,
            ..
        } => {
            let e = lower(ir, *id, depth + 1)?;
            format!("\\[{WS}{}{WS}\\]", repeat_csv(&e, *min_items, *max_items))
        }
        Node::Tuple {
            unique_items: true, ..
        }
        | Node::Tuple {
            contains: Some(_), ..
        } => return Err(oops("uniqueItems/contains is not a regular language")),
        Node::Tuple {
            prefix,
            tail,
            min_items,
            max_items,
            ..
        } => {
            let ids = ir.refs_at(*prefix).ok_or_else(|| oops("tuple ref"))?;
            let mut elems = Vec::with_capacity(ids.len());
            for id in ids {
                elems.push(lower(ir, *id, depth + 1)?);
            }
            let tail = match tail {
                Some(t) => Some(lower(ir, *t, depth + 1)?),
                None => None,
            };
            let body = tuple_csv(&elems, tail.as_deref(), *min_items, *max_items)?;
            format!("\\[{WS}{body}{WS}\\]")
        }
        Node::Object { dependent: dep, .. } if ir.props_at(*dep).is_some_and(|d| !d.is_empty()) => {
            return Err(oops("dependentSchemas is not a regular language"));
        }
        Node::Object {
            dependent_required, ..
        } if ir
            .dependent_required_at(*dependent_required)
            .is_some_and(|pairs| !pairs.is_empty()) =>
        {
            return Err(oops("dependentRequired is not a regular language"));
        }
        Node::Object {
            fields, required, ..
        } => {
            let props = ir.props_at(*fields).ok_or_else(|| oops("object ref"))?;
            if props.len() > 8 {
                return Err(oops(
                    "reference object lowering: too many fields to permute",
                ));
            }
            let mut frags = Vec::with_capacity(props.len());
            let mut req = Vec::with_capacity(props.len());
            for (i, (name, sub)) in props.iter().enumerate() {
                let key = ir.str_at(*name).ok_or_else(|| oops("field name"))?;
                let f = format!(
                    "{}{WS}:{WS}{}",
                    quote(&encode(key)),
                    lower(ir, *sub, depth + 1)?
                );
                frags.push(f);
                let idx = u32::try_from(i).map_err(|_| oops("field index overflow"))?;
                req.push(ir.is_required(*required, idx));
            }
            let n = frags.len();
            let required_idx: Vec<usize> = (0..n).filter(|&i| req[i]).collect();
            let optional: Vec<usize> = (0..n).filter(|&i| !req[i]).collect();
            let opt_count = u32::try_from(optional.len()).map_err(|_| oops("too many optional"))?;
            let mut alts = Vec::new();
            for mask in 0..(1u32 << opt_count) {
                let mut present = required_idx.clone();
                for (bit, &idx) in optional.iter().enumerate() {
                    if mask & (1 << bit) != 0 {
                        present.push(idx);
                    }
                }
                for perm in permutations(&present) {
                    let members: Vec<&str> = perm.iter().map(|&i| frags[i].as_str()).collect();
                    alts.push(members.join(&format!("{WS},{WS}")));
                }
            }
            format!("\\{{{WS}(?:{}){WS}\\}}", alts.join("|"))
        }
        Node::OpenObject { .. } => return Err(oops("open object is not a regular language")),
        Node::Union { branches } => {
            let ids = ir.refs_at(*branches).ok_or_else(|| oops("union ref"))?;
            let mut alts = Vec::with_capacity(ids.len());
            for id in ids {
                alts.push(lower(ir, *id, depth + 1)?);
            }
            format!("(?:{})", alts.join("|"))
        }
        Node::Intersection { branches } => {
            let ids = ir
                .refs_at(*branches)
                .ok_or_else(|| oops("intersection ref"))?;
            let mut fragments = Vec::with_capacity(ids.len());
            for id in ids {
                fragments.push(lower(ir, *id, depth + 1)?);
            }
            product_regex(&fragments, Combinator::All)?
        }
        Node::ExactlyOne { branches } => {
            let ids = ir
                .refs_at(*branches)
                .ok_or_else(|| oops("exactly-one ref"))?;
            let mut fragments = Vec::with_capacity(ids.len());
            for id in ids {
                fragments.push(lower(ir, *id, depth + 1)?);
            }
            product_regex(&fragments, Combinator::ExactlyOne)?
        }
        Node::Not { .. } => return Err(oops("not is not a regular language")),
        Node::Ref { .. } => return Err(oops("recursive ref is not a regular language")),
        Node::DynamicRef { .. } => return Err(oops("dynamic ref is not a regular language")),
        Node::Unevaluated { .. } => return Err(oops("unevaluated is not a regular language")),
        Node::Unsupported { .. } => return Err(oops("unsupported node in reference lowering")),
    })
}

fn product_regex(fragments: &[String], combinator: Combinator) -> Result<String, CompileError> {
    let mut engines = Vec::with_capacity(fragments.len());
    for f in fragments {
        engines.push(build_from_regex(f)?);
    }
    let refs: Vec<&RefEngine> = engines.iter().collect();
    let product = ProductAutomaton::build(&refs, combinator)?;
    product.to_regex()
}

fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    let Some((&first, rest)) = items.split_first() else {
        return vec![Vec::new()];
    };
    let mut out = Vec::new();
    for p in permutations(rest) {
        for pos in 0..=p.len() {
            let mut with_first = p.clone();
            with_first.insert(pos, first);
            out.push(with_first);
        }
    }
    out
}

fn lit(l: &ScalarLit) -> String {
    match l {
        ScalarLit::Null => "null".to_owned(),
        ScalarLit::Bool(true) => "true".to_owned(),
        ScalarLit::Bool(false) => "false".to_owned(),
        ScalarLit::Int(n) => n.to_string(),
        ScalarLit::Str(s) => decoded_string_regex(s),
        ScalarLit::Json(s) => raw_literal(s),
    }
}

fn decoded_string_regex(value: &str) -> String {
    let mut out = String::from("\"");
    for scalar in value.chars() {
        let spellings = scalar_spellings(scalar);
        if spellings.len() == 1 {
            out.push_str(&spellings[0]);
        } else {
            out.push_str("(?:");
            out.push_str(&spellings.join("|"));
            out.push(')');
        }
    }
    out.push('"');
    out
}

fn scalar_spellings(scalar: char) -> Vec<String> {
    let mut spellings = Vec::new();
    if scalar >= '\u{20}' && scalar != '"' && scalar != '\\' {
        let mut raw = String::new();
        escape_into(&mut raw, scalar);
        spellings.push(raw);
    }
    let short = match scalar {
        '"' => Some(r#"\\\""#),
        '\\' => Some(r"\\\\"),
        '/' => Some(r"\\/"),
        '\u{08}' => Some(r"\\b"),
        '\u{0c}' => Some(r"\\f"),
        '\n' => Some(r"\\n"),
        '\r' => Some(r"\\r"),
        '\t' => Some(r"\\t"),
        _ => None,
    };
    if let Some(short) = short {
        spellings.push(short.to_owned());
    }
    spellings.push(unicode_scalar_spelling(scalar));
    spellings
}

fn unicode_scalar_spelling(scalar: char) -> String {
    let code = u32::from(scalar);
    if code <= 0xffff {
        return unicode_code_unit(code);
    }
    let adjusted = code - 0x1_0000;
    let high = 0xd800 + (adjusted >> 10);
    let low = 0xdc00 + (adjusted & 0x3ff);
    format!("{}{}", unicode_code_unit(high), unicode_code_unit(low))
}

fn unicode_code_unit(code: u32) -> String {
    let mut out = String::from(r"\\u");
    for digit in format!("{code:04x}").chars() {
        if digit.is_ascii_alphabetic() {
            out.push('[');
            out.push(digit);
            out.push(digit.to_ascii_uppercase());
            out.push(']');
        } else {
            out.push(digit);
        }
    }
    out
}

fn raw_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut prev_was_backslash = false;
    for c in s.chars() {
        if in_string {
            escape_into(&mut out, c);
            if prev_was_backslash {
                prev_was_backslash = false;
            } else if c == '\\' {
                prev_was_backslash = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            escape_into(&mut out, c);
        } else if matches!(c, '{' | '}' | '[' | ']' | ':' | ',') {
            out.push_str(WS);
            escape_into(&mut out, c);
            out.push_str(WS);
        } else {
            escape_into(&mut out, c);
        }
    }
    out
}

fn escape_into(out: &mut String, c: char) {
    if matches!(
        c,
        '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|'
    ) {
        out.push('\\');
    }
    out.push(c);
}

fn encode(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn quote(body: &str) -> String {
    let mut out = String::from("\"");
    for c in body.chars() {
        escape_into(&mut out, c);
    }
    out.push('"');
    out
}

fn rep(min: Option<u32>, max: Option<u32>) -> String {
    match (min, max) {
        (Some(a), Some(b)) => format!("{{{a},{b}}}"),
        (Some(a), None) => format!("{{{a},}}"),
        (None, Some(b)) => format!("{{0,{b}}}"),
        (None, None) => "*".to_owned(),
    }
}

fn repeat_csv(elem: &str, min: u32, max: Option<u32>) -> String {
    let e = format!("(?:{elem})");
    if max == Some(0) {
        return String::new();
    }
    let tail = match max {
        None => format!("(?:{WS},{WS}{e})*"),
        Some(m) => format!("(?:{WS},{WS}{e}){{0,{}}}", m - 1),
    };
    if min == 0 {
        format!("(?:{e}{tail})?")
    } else {
        let required_tail = format!("(?:{WS},{WS}{e}){{{},}}", min - 1);
        match max {
            None => format!("{e}{required_tail}"),
            Some(m) => format!("{e}(?:{WS},{WS}{e}){{{},{}}}", min - 1, m - 1),
        }
    }
}

fn tuple_csv(
    elems: &[String],
    tail: Option<&str>,
    min: u32,
    max: Option<u32>,
) -> Result<String, CompileError> {
    let n = u32::try_from(elems.len())
        .map_err(|_| oops("tuple prefix length exceeds the reference limit"))?;
    let pick = |i: u32| -> String {
        let piece = if i < n {
            elems[i as usize].as_str()
        } else {
            tail.unwrap_or("")
        };
        format!("(?:{piece})")
    };
    let sequence = |count: u32| -> String {
        let items: Vec<String> = (0..count).map(pick).collect();
        items.join(&format!("{WS},{WS}"))
    };
    let mut branches: Vec<String> = Vec::new();
    match (tail, max) {
        (None, _) => {
            let hi = max.map_or(n, |m| m.min(n));
            for count in min..=hi {
                branches.push(sequence(count));
            }
        }
        (Some(_), Some(hi)) => {
            for count in min..=hi {
                branches.push(sequence(count));
            }
        }
        (Some(t), None) => {
            for count in min..n {
                branches.push(sequence(count));
            }
            let head = sequence(n);
            let extra = min.saturating_sub(n);
            let more = if extra == 0 {
                format!("(?:{WS},{WS}(?:{t}))*")
            } else {
                format!("(?:{WS},{WS}(?:{t})){{{extra},}}")
            };
            branches.push(format!("{head}{more}"));
        }
    }
    branches.sort_by_key(|b| std::cmp::Reverse(b.len()));
    Ok(format!("(?:{})", branches.join("|")))
}

fn oops(what: &'static str) -> CompileError {
    CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L2, what)
}
