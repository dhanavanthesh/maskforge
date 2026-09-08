//! The provenance sidecar. A `Diagnostic` records where and why an occurrence was rejected;
//! it is keyed by occurrence and never enters the semantic hash.

use bincode::{Decode, Encode};

use crate::ir::StrRef;
use crate::primitives::NodeId;

/// Why a keyword or shape is not accepted by the regular-subset engine. The discriminant is the
/// stable machine code (folded into the content hash and pinned by a wire test); variants are
/// append-only. `advisory` is display-only.
#[non_exhaustive]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub enum UnsupportedReason {
    /// A keyword with no supported arm in this dialect.
    UnsupportedKeyword = 0,
    /// A `type` union or array of types (never narrowed to one).
    UnionType = 1,
    /// An object whose additional properties are open (absent or a non-`false` schema).
    OpenAdditionalProperties = 2,
    /// `additionalProperties` given as a subschema rather than `false`.
    AdditionalPropertiesSchema = 3,
    /// A cycle in the `$ref` graph (needs a pushdown grammar).
    RecursiveReference = 4,
    /// A `$ref` to an external document or the network.
    NonRegularRef = 5,
    /// A regex construct the anchored byte engine cannot accept.
    InvalidPattern = 6,
    /// `type:number`, `multipleOf`, or a numeric bound outside the supported integer subset.
    NumericRangeUnsupported = 7,
    /// `uniqueItems` (set distinctness is context-sensitive).
    SetDistinctness = 8,
    /// `oneOf` (exactly-one is context-sensitive; never downgraded to a union).
    ExactlyOne = 9,
    /// `not` (complement).
    Complement = 10,
    /// `if`/`then`/`else` or `dependentSchemas` (conditional applicator).
    ConditionalApplicator = 11,
    /// `dependentRequired` (a deferred obligation).
    DeferredObligation = 12,
    /// A keyword whose meaning depends on collected annotations.
    AnnotationDependent = 13,
    /// `$dynamicRef`/`$dynamicAnchor` (dynamic scope).
    DynamicScope = 14,
    /// A non-scalar `const`/`enum` member.
    NonScalarLiteral = 15,
    /// A length-bounded string that also carries a pattern or a non-ASCII body.
    LengthWithPattern = 16,
    /// A `format` this dialect does not enforce even under `format_assertion`.
    FormatUnsupported = 17,
    /// A `format` combined with `pattern`, `minLength`, or `maxLength` on the same string.
    FormatWithPattern = 18,
    /// A `const`/`enum` numeric member whose lexical form is not a plain JSON integer token
    /// (a fraction, exponent, or magnitude outside `i64`), e.g. `1.0`, `1e0`.
    NonIntegerNumericLiteral = 19,
    /// `allOf` (language intersection needs a product-automaton construction this dialect does not
    /// build; never approximated as a union or a regex concatenation).
    AllOfIntersectionUnsupported = 20,
    /// A `minimum`/`maximum` bound on `type:number`. The engine does not build a decimal-comparison
    /// automaton for a bounded real range under the current profile, so only unbounded `type:number`
    /// is supported. This is a current-implementation limit, not a proof of non-representability.
    NumberBoundUnsupported = 21,
    /// Unconstrained array items (bare `prefixItems`, `items:true`, or no item schema): they admit
    /// arbitrary-depth JSON, which is not regular. A regular-item array or `items:false` is supported.
    OpenArrayTail = 22,
    /// An `OpenObject` reached the byte-DFA compiler; it needs the structured backend instead.
    StructuredBackendRequired = 23,
    /// A look-around assertion is outside the supported anchored regular subset.
    RegexAssertionUnsupported = 24,
    /// A numeric or named regex backreference is not compiled as an automaton.
    RegexBackreferenceUnsupported = 25,
}

impl UnsupportedReason {
    /// A short, display-only description. Not the machine contract (the discriminant is).
    #[must_use]
    pub fn advisory(self) -> &'static str {
        match self {
            Self::UnsupportedKeyword => "keyword is not supported in this dialect",
            Self::UnionType => "a type union is never narrowed; declare one type",
            Self::OpenAdditionalProperties => {
                "object has open additionalProperties; a closed-object profile is required"
            }
            Self::AdditionalPropertiesSchema => {
                "additionalProperties as a subschema is not supported"
            }
            Self::RecursiveReference => "a cyclic reference needs a pushdown grammar",
            Self::NonRegularRef => "external or network references are not supported",
            Self::InvalidPattern => "the regex uses a construct the byte engine cannot accept",
            Self::NumericRangeUnsupported => {
                "only signed 64-bit integers with integer bounds are supported"
            }
            Self::SetDistinctness => "uniqueItems is context-sensitive",
            Self::ExactlyOne => "oneOf is context-sensitive and is not downgraded to a union",
            Self::Complement => "not is context-sensitive",
            Self::ConditionalApplicator => "conditional applicators are not supported",
            Self::DeferredObligation => "dependentRequired is not supported",
            Self::AnnotationDependent => "the keyword depends on collected annotations",
            Self::DynamicScope => "dynamic scope resolution is not supported",
            Self::NonScalarLiteral => "a const/enum member must be a scalar",
            Self::LengthWithPattern => {
                "a length-bounded string with a pattern or non-ASCII body is not supported"
            }
            Self::FormatUnsupported => "this format is not enforced",
            Self::FormatWithPattern => {
                "format combined with pattern or a length bound is not supported"
            }
            Self::NonIntegerNumericLiteral => "not a plain JSON integer token (fraction/exponent)",
            Self::AllOfIntersectionUnsupported => {
                "allOf is a language intersection, which needs a product-automaton construction \
                 this dialect does not build; never approximated as a union or concatenation"
            }
            Self::NumberBoundUnsupported => {
                "a bounded real-number range is not implemented; only unbounded type:number \
                 is supported"
            }
            Self::OpenArrayTail => {
                "unconstrained trailing items admit arbitrary-depth JSON (not regular); use \
                 items:false or a schema-valued items to close the array"
            }
            Self::StructuredBackendRequired => {
                "this schema needs the structured backend, not the byte-DFA compiler: check \
                 SchemaIR::requires_structured_backend() before calling compile_ir, or use a \
                 caller that already dispatches automatically (Python: ConstraintSession.from_json_schema)"
            }
            Self::RegexAssertionUnsupported => {
                "the regex assertion is outside the supported anchored regular subset"
            }
            Self::RegexBackreferenceUnsupported => {
                "regex backreferences are not supported by the automaton compiler"
            }
        }
    }
}

/// A byte span into the original source text, when the frontend tracks one.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub struct Span {
    /// Byte offset of the span start.
    pub start: u32,
    /// Byte offset one past the span end.
    pub end: u32,
}

/// One rejected occurrence: the node it produced, the keyword, its RFC 6901 pointer, and the
/// machine reason. Keyed by occurrence; never hashed into the content address.
#[derive(Clone, PartialEq, Eq, Debug, Encode, Decode)]
pub struct Diagnostic {
    /// The `Unsupported` node this occurrence lowered to.
    pub node: NodeId,
    /// The offending keyword, as an interned string reference.
    pub keyword: StrRef,
    /// The RFC 6901 pointer to the occurrence, as an interned string reference.
    pub json_pointer: StrRef,
    /// The source span, when the frontend tracked one.
    pub source_span: Option<Span>,
    /// The machine-stable reason.
    pub reason: UnsupportedReason,
}
