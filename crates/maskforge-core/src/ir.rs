//! The public arena schema IR: a flat, immutable, hash-consed DAG built once through [`Builder`]
//! and consumed directly by the compiler. All fields are private; the only ways in are
//! [`Builder::finish`] and [`SchemaIR::from_wire`], and the only way out is a read-only accessor.

use std::sync::OnceLock;

use bincode::{Decode, Encode};
use fluent_uri::Uri;
use rustc_hash::FxHashMap;

use crate::diagnostics::{Diagnostic, UnsupportedReason};
use crate::error::{CompileError, ErrorCode, Stage};
use crate::primitives::NodeId;
use crate::wire::SchemaIRWire;

/// The IR schema version. Checked exactly on every wire crossing and folded into the content hash.
pub const IR_VERSION: u16 = 6;

// Arena caps, enforced at BOTH construction paths: incrementally by the builder and on the decoded
// value by the wire layer. Chosen so a within-cap arena always encodes below the envelope byte
// limit, which keeps encoding infallible and keeps decode able to accept anything encode produces.
pub(crate) const MAX_NODES: usize = 1 << 18;
pub(crate) const MAX_STRINGS_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_LITERALS: usize = 1 << 18;
/// Aggregate byte cap over all `ScalarLit::Str` payloads, which the count cap alone does not bound.
pub(crate) const MAX_LITERAL_STR_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_PROPS: usize = 1 << 18;
pub(crate) const MAX_DEPENDENT_REQUIRED_PAIRS: usize = 1 << 18;
pub(crate) const MAX_REQUIRED_WORDS: usize = 1 << 18;
pub(crate) const MAX_DIAGNOSTICS: usize = 1 << 16;
pub(crate) const MAX_NODE_REFS: usize = 1 << 18;
pub(crate) const MAX_RESOURCES: usize = 1 << 16;
pub(crate) const MAX_ANCHORS: usize = 1 << 18;

/// Total bytes of owned variable-length literal content (`Str` and composite `Json`) in a slice.
/// Both are unbounded String payloads, so both count toward the aggregate byte cap.
pub(crate) fn literal_str_bytes(lits: &[ScalarLit]) -> usize {
    lits.iter()
        .map(|l| match l {
            ScalarLit::Str(s) | ScalarLit::Json(s) => s.len(),
            _ => 0,
        })
        .sum()
}

/// A contiguous run in a side blob: `off` is the start index, `len` the element/field count.
macro_rules! slice_ref {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Encode, Decode)]
        pub struct $name {
            /// Start index into the owning blob.
            pub off: u32,
            /// Element or field count.
            pub len: u32,
        }
    };
}

slice_ref!(
    /// A run of bytes in the `strings` blob (a UTF-8 substring).
    StrRef
);
slice_ref!(
    /// A run of `ScalarLit` values in the `literals` blob.
    LitSlice
);
slice_ref!(
    /// A run of `(name, child)` pairs in the `props` blob.
    PropSlice
);
slice_ref!(
    /// A run of `(trigger, required)` property-name pairs.
    DependentRequiredSlice
);
slice_ref!(
    /// A run of required-flag words in the `required_bits` blob: `off` is the first word, `len`
    /// the field count.
    BitSlice
);
slice_ref!(
    /// A run of `NodeId` branches in the `node_refs` blob, in declaration order.
    RefSlice
);
slice_ref!(
    /// A run of anchor records in the resource metadata table.
    AnchorSlice
);

/// Dense frozen schema-resource identifier.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Encode, Decode)]
pub struct ResourceId(pub(crate) u32);

impl ResourceId {
    /// Returns the underlying dense index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Dense frozen anchor-record identifier.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Encode, Decode)]
pub struct AnchorId(pub(crate) u32);

impl AnchorId {
    /// Returns the underlying dense index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Frozen metadata for one Draft 2020-12 schema resource.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub struct ResourceRecord {
    /// Canonical absolute URI, absent only for an anonymous resource.
    pub canonical_uri: Option<StrRef>,
    /// Root node of this schema resource.
    pub root: NodeId,
    /// Static `$anchor` records owned by this resource.
    pub static_anchors: AnchorSlice,
    /// `$dynamicAnchor` records owned by this resource.
    pub dynamic_anchors: AnchorSlice,
}

/// Frozen static or dynamic anchor declaration.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub struct AnchorRecord {
    /// Resource declaring this anchor.
    pub resource: ResourceId,
    /// Plain-name anchor value.
    pub name: StrRef,
    /// Schema node identified by the anchor.
    pub target: NodeId,
}

/// Builder input for one already-resolved resource. Anchor lists must be name-sorted.
pub(crate) struct ResourceInput {
    pub(crate) canonical_uri: Option<String>,
    pub(crate) root: NodeId,
    pub(crate) static_anchors: Vec<(String, NodeId)>,
    pub(crate) dynamic_anchors: Vec<(String, NodeId)>,
}

/// A literal usable as a `const` value or an `enum` member.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Encode, Decode)]
pub enum ScalarLit {
    /// JSON `null`.
    Null,
    /// JSON `true`/`false`.
    Bool(bool),
    /// A signed 64-bit integer.
    Int(i64),
    /// A UTF-8 string.
    Str(String),
    /// A composite value held as its canonical JSON serialization (minimal whitespace, declared key
    /// order); only that one spelling is accepted (see PROVENANCE.md).
    Json(String),
}

/// The byte alphabet a bounded string is drawn from. The discriminant is the stable wire/hash
/// code; variants are append-only (a reorder is a wire break and bumps the envelope format).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub enum Charset {
    /// Any UTF-8 body (no length enforcement in this profile).
    Utf8Any = 0,
    /// Printable ASCII excluding `"` and `\`, where one code point equals one byte.
    AsciiPrintableNoQuoteBackslash = 1,
    /// Full JSON-string alphabet where `regex` denotes one code point, so `{min,max}` counts them.
    Utf8CountedCodepoints = 2,
}

/// How an object treats properties not named in `fields`. Discriminant is the stable code.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub enum ObjectClosure {
    /// An open object (absent or non-`false` additional properties); rejected at compile.
    RejectOpenObjects = 0,
    /// The caller opted in to generating as a closed object.
    AssumeClosedProfile = 1,
    /// `additionalProperties: false` stated by the schema itself.
    Forbidden = 2,
    /// Default: absent `additionalProperties` compiles as truly open (spec-correct `additionalProperties:true`), not rejected.
    AllowOpenProfile = 3,
}

/// How an open object treats keys past `known`/`patterns`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub enum AdditionalPolicy {
    /// No further keys are allowed.
    Forbid,
    /// Any further key with any JSON value is allowed.
    AllowAny,
    /// Any further key is allowed if its value matches this schema.
    Schema(NodeId),
    /// Allows an unevaluated key to flow to the unevaluated check.
    /// It validates like `AllowAny` but differs in annotation collection.
    Open,
}

/// Which instance type a `Node::Unevaluated` obligation targets.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub enum UnevaluatedKind {
    /// `unevaluatedProperties`: obligation applies to object members.
    Properties,
    /// `unevaluatedItems`: obligation applies to array elements.
    Items,
}

/// What an array's elements must satisfy. Unlike `AdditionalPolicy`, there is no `Forbid`: an array
/// always accepts elements, only their type may be unconstrained.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub enum ItemsPolicy {
    /// Every element must match this schema.
    Schema(NodeId),
    /// Any JSON value, arbitrary depth - not a regular language, so this always routes through
    /// `crate::structured`.
    AllowAny,
}

/// What `contains` requires an element to be, independent of `items`/`prefixItems`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub enum ContainsPolicy {
    /// `contains: false` - no element can ever satisfy it.
    Never,
    /// `contains: true` - every element satisfies it.
    Always,
    /// An element satisfies it by matching this schema.
    Schema(NodeId),
}

/// A `contains` assertion: `min..=max` elements (independent of `items`/`prefixItems`) must
/// satisfy `policy`. Not a regular-language property, so this always needs `crate::structured`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub struct ContainsConstraint {
    /// What counts as a match.
    pub policy: ContainsPolicy,
    /// Inclusive minimum number of matching elements.
    pub min: u32,
    /// Inclusive maximum number of matching elements, when bounded.
    pub max: Option<u32>,
}

/// What the compiler does when an occurrence is `Unsupported`. Discriminant is the stable code.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub enum UnsupportedPolicy {
    /// Any `Unsupported` occurrence fails the compile with its diagnostics.
    RejectAtCompile = 0,
}

/// Options that govern lowering. Language-affecting choices (`object_closure`, `format_assertion`)
/// are resolved into node fields at build time; `max_diagnostics` bounds the diagnostics buffer.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub struct CompileOptions {
    /// The default closure applied to an object with absent additional properties.
    pub object_closure: ObjectClosure,
    /// Whether `format` keywords are enforced as lexical DFAs.
    pub format_assertion: bool,
    /// The policy for `Unsupported` occurrences.
    pub unsupported_policy: UnsupportedPolicy,
    /// The diagnostics buffer cap; must be at least one.
    pub max_diagnostics: u32,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            object_closure: ObjectClosure::AllowOpenProfile,
            format_assertion: false,
            unsupported_policy: UnsupportedPolicy::RejectAtCompile,
            max_diagnostics: 8192,
        }
    }
}

/// Above this many fields, a closed object's shuffle-into-one-regex can take unbounded time to even
/// DISCOVER it exceeds `automaton::elimination`'s byte cap (subset explosion over optional fields),
/// so both `requires_structured_backend` and `structured::MatchCache` treat it as a pre-check, never
/// a "try it and see" attempt.
pub(crate) const MAX_SAFE_SHUFFLE_FIELDS: usize = 4;
/// Above this, a counted-codepoint string routes to `crate::structured`'s O(1)-per-char counter
/// instead of an unrolled DFA whose state count is O(max_len).
pub(crate) const MAX_UNROLLED_CODEPOINT_LEN: u32 = 256;
/// Above this `maxItems`, an array routes to `crate::structured`'s native element counter instead of
/// unrolling the element DFA `max_items` times.
pub(crate) const MAX_UNROLLED_ARRAY_ITEMS: u32 = 64;
/// Above this member count, an all-string enum routes to `crate::structured`'s O(log N) binary
/// search instead of a giant alternation DFA (which state-explodes and crashes past ~10k).
pub(crate) const MAX_UNROLLED_ENUM: usize = 128;
/// One arena node. Children are `NodeId` indices; scalars live in side blobs. `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub enum Node {
    /// Matches literal `null`.
    Null,
    /// Matches `true` or `false`.
    Boolean,
    /// Matches no value at all (the bare `false` schema). A regular language like any other leaf:
    /// an impossible byte class, dead from the very first byte, never needs the structured backend.
    Never,
    /// A fixed string literal.
    StringConst {
        /// The literal value (a JSON string body, before JSON quoting).
        value: StrRef,
    },
    /// A regular pattern over a chosen byte alphabet, optionally length-bounded.
    StringPattern {
        /// The byte-level regex body.
        regex: StrRef,
        /// Inclusive minimum length in code points, when bounded.
        min_len: Option<u32>,
        /// Inclusive maximum length in code points, when bounded.
        max_len: Option<u32>,
        /// The alphabet the body is drawn from.
        charset: Charset,
    },
    /// A signed 64-bit integer with optional inclusive bounds and an optional exact divisor.
    Integer {
        /// Inclusive lower bound.
        minimum: Option<i64>,
        /// Inclusive upper bound.
        maximum: Option<i64>,
        /// `multipleOf`: the value must be exactly divisible by this positive integer.
        multiple_of: Option<u64>,
    },
    /// Any JSON number with sign, integer part, and optional fraction.
    /// Bounded forms omit exponents; genuinely fractional bounds remain unsupported.
    Number {
        /// Whether the mathematical value, rather than its spelling, must be integral.
        integer_only: bool,
        /// Lower bound as `(scaled_value, scale, exclusive)`: the value is `scaled_value * 10^-scale`
        /// (`scale = 0` is an integer bound), `exclusive = true` for a strict bound.
        minimum: Option<(i128, u32, bool)>,
        /// Upper bound, same `(scaled_value, scale, exclusive)` encoding.
        maximum: Option<(i128, u32, bool)>,
        /// `multipleOf` as `coef * 10^-exp` (exact decimal), when present.
        multiple_of: Option<(u64, u32)>,
    },
    /// A JSON number constrained by lexical digit counts.
    LexicalNumber {
        /// Anchored byte regex for the complete number token.
        regex: StrRef,
    },
    /// A closed set of scalar literals (non-empty, canonically sorted, duplicate-free).
    Enum {
        /// The member values.
        values: LitSlice,
    },
    /// A homogeneous array with a cardinality bound.
    Array {
        /// What every element must satisfy.
        items: ItemsPolicy,
        /// Inclusive minimum item count.
        min_items: u32,
        /// Inclusive maximum item count, when bounded.
        max_items: Option<u32>,
        /// `uniqueItems`: no two items may be the same JSON value. Not a regular-language property
        /// (needs unbounded memory of prior items), so this always routes through `crate::structured`.
        unique_items: bool,
        /// `contains`/`minContains`/`maxContains`, when present.
        contains: Option<ContainsConstraint>,
        /// Whether `items` was textually present (`true` or a schema), for `unevaluatedItems`: an
        /// absent `items` validates identically but annotates no index (proven by the official suite).
        items_annotates: bool,
    },
    /// A positional (tuple) array: `prefix` positions validate in order, `tail` validates the rest
    /// (absent = closed); length runs from `min_items` to `max_items` (and the prefix, if closed).
    Tuple {
        /// The positional element schemas, in array order.
        prefix: RefSlice,
        /// The schema for items past the prefix, or `None` for a closed tuple.
        tail: Option<NodeId>,
        /// Inclusive minimum item count.
        min_items: u32,
        /// Inclusive maximum item count, when bounded.
        max_items: Option<u32>,
        /// `uniqueItems`, same semantics and structured-backend requirement as `Node::Array`'s.
        unique_items: bool,
        /// Whether `items` was textually present past `prefixItems`, mirroring `items_annotates`.
        tail_annotates: bool,
        /// `contains`/`minContains`/`maxContains`, when present.
        contains: Option<ContainsConstraint>,
    },
    /// An object with source-ordered fields, a positional required bitset, and a closure policy.
    Object {
        /// The `(name, child)` fields, in source order.
        fields: PropSlice,
        /// The required-field bitset (indexed by field position).
        required: BitSlice,
        /// How additional properties are treated.
        closure: ObjectClosure,
        /// `dependentSchemas`: when `key` is present, the whole instance must also satisfy
        /// `schema`. Instance-dependent, so a non-empty run always needs `crate::structured`.
        dependent: PropSlice,
        /// `dependentRequired` flattened as decoded trigger/required name pairs.
        dependent_required: DependentRequiredSlice,
    },
    /// An order-independent object: known properties, `patternProperties`, and an additional-key
    /// policy, none of which can be represented as one finite byte-level regex.
    OpenObject {
        /// The exact `(name, child)` properties, matched independent of textual order.
        known: PropSlice,
        /// The required-field bitset, aligned to `known` by position.
        known_required: BitSlice,
        /// The `(pattern, child)` `patternProperties` entries; a key may match more than one.
        patterns: PropSlice,
        /// How a key matching neither `known` nor `patterns` is treated.
        additional: AdditionalPolicy,
        /// An optional schema every key (known, pattern-matched, or additional) must satisfy.
        property_names: Option<NodeId>,
        /// Inclusive minimum property count.
        min_properties: Option<u32>,
        /// Inclusive maximum property count, when bounded.
        max_properties: Option<u32>,
        /// `dependentSchemas`, same semantics as `Node::Object`'s.
        dependent: PropSlice,
        /// `dependentRequired` flattened as decoded trigger/required name pairs.
        dependent_required: DependentRequiredSlice,
    },
    /// A `type` union: matches any of its branches, in declaration order (deduplicated by type).
    Union {
        /// The branch nodes.
        branches: RefSlice,
    },
    /// An `allOf`: matches iff every branch matches (language intersection).
    Intersection {
        /// The branch nodes.
        branches: RefSlice,
    },
    /// A `oneOf`: matches iff exactly one branch matches.
    ExactlyOne {
        /// The branch nodes.
        branches: RefSlice,
    },
    /// Matches complete JSON values that `inner` does not match.
    /// Full JSON-value complement requires `crate::structured`.
    Not {
        /// The negated schema.
        inner: NodeId,
    },
    /// A recursion cut point: a leaf whose body is `def_targets[def]`, an edge free of the back-edge
    /// rule so a `$ref` cycle needs no cyclic child graph. Followed only by `crate::structured`.
    Ref {
        /// Index into the `SchemaIR::def_targets` table naming the definition body.
        def: u32,
    },
    /// A `$dynamicRef` whose initially resolved fragment names a `$dynamicAnchor`.
    DynamicRef {
        /// Statically resolved target used when dynamic scope does not rebind the name.
        initial_target: NodeId,
        /// Initial dynamic-anchor declaration, which supplies the plain-name identity.
        anchor: AnchorId,
    },
    /// Requires unannotated properties or items in `scope` to match `unevaluated`.
    /// Annotation tracking requires `crate::structured`.
    Unevaluated {
        /// Which instance type this obligation targets.
        kind: UnevaluatedKind,
        /// The schema minus the unevaluated keyword (local assertions and in-place applicators).
        scope: NodeId,
        /// The schema each unevaluated member must satisfy (`Never` for `false`).
        unevaluated: NodeId,
    },
    /// An occurrence the dialect does not support. Carries no pointer; provenance is the sidecar.
    Unsupported {
        /// The offending keyword.
        keyword: StrRef,
        /// The machine-stable reason.
        reason: UnsupportedReason,
    },
}

/// The arena IR. Built once and frozen; all fields private.
#[derive(Clone)]
pub struct SchemaIR {
    ir_version: u16,
    root: NodeId,
    nodes: Vec<Node>,
    strings: Vec<u8>,
    literals: Vec<ScalarLit>,
    props: Vec<(StrRef, NodeId)>,
    dependent_required_pairs: Vec<DependentRequiredPair>,
    required_bits: Vec<u64>,
    node_refs: Vec<NodeId>,
    def_targets: Vec<NodeId>,
    retrieval_uri: Option<StrRef>,
    resources: Vec<ResourceRecord>,
    anchors: Vec<AnchorRecord>,
    node_resources: Vec<ResourceId>,
    options: CompileOptions,
    diagnostics: Vec<Diagnostic>,
    cached_hash: OnceLock<[u8; 32]>,
    route_table: OnceLock<crate::routing::RouteTable>,
}

/// One decoded-name presence implication used by `dependentRequired`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Encode, Decode)]
pub struct DependentRequiredPair {
    /// The decoded property name whose presence activates the rule.
    pub trigger: StrRef,
    /// A decoded property name that must be present when the trigger is present.
    pub required: StrRef,
}

impl PartialEq for SchemaIR {
    /// Exact frozen representation equality: every field (arena order, blob offsets, options, and
    /// diagnostics) except the memoized hash. Two IRs with the same schema but a different arena
    /// layout or different rejected sites compare unequal; for language identity use
    /// [`SchemaIR::same_content_address`] or compare [`SchemaIR::canonical_hash`].
    fn eq(&self, other: &Self) -> bool {
        self.ir_version == other.ir_version
            && self.root == other.root
            && self.nodes == other.nodes
            && self.strings == other.strings
            && self.literals == other.literals
            && self.props == other.props
            && self.dependent_required_pairs == other.dependent_required_pairs
            && self.required_bits == other.required_bits
            && self.node_refs == other.node_refs
            && self.def_targets == other.def_targets
            && self.retrieval_uri == other.retrieval_uri
            && self.resources == other.resources
            && self.anchors == other.anchors
            && self.node_resources == other.node_resources
            && self.options == other.options
            && self.diagnostics == other.diagnostics
    }
}

impl Eq for SchemaIR {}

impl std::fmt::Debug for SchemaIR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaIR")
            .field("ir_version", &self.ir_version)
            .field("root", &self.root)
            .field("nodes", &self.nodes)
            .field("resources", &self.resources)
            .field("diagnostics", &self.diagnostics.len())
            .finish_non_exhaustive()
    }
}

impl SchemaIR {
    /// The IR schema version.
    #[must_use]
    pub fn ir_version(&self) -> u16 {
        self.ir_version
    }

    /// The root node id.
    #[must_use]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// The node at `id`, or `None` if the id is out of range.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.get() as usize)
    }

    /// All nodes in arena order.
    pub fn nodes(&self) -> impl ExactSizeIterator<Item = &Node> {
        self.nodes.iter()
    }

    /// The number of nodes in the arena.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Optional root retrieval URI retained in the compact string blob.
    #[must_use]
    pub fn retrieval_uri(&self) -> Option<&str> {
        self.retrieval_uri.and_then(|uri| self.str_at(uri))
    }

    /// Frozen schema resources in canonical order.
    #[must_use]
    pub fn resources(&self) -> &[ResourceRecord] {
        &self.resources
    }

    /// Frozen static and dynamic anchor records.
    #[must_use]
    pub fn anchors(&self) -> &[AnchorRecord] {
        &self.anchors
    }

    /// Resource that owns `node`, or `None` for an invalid node ID.
    #[must_use]
    pub fn node_resource(&self, node: NodeId) -> Option<ResourceId> {
        self.node_resources.get(node.get() as usize).copied()
    }

    /// Every diagnostic, one per rejected occurrence.
    pub fn diagnostics(&self) -> impl ExactSizeIterator<Item = &Diagnostic> {
        self.diagnostics.iter()
    }

    /// A counted-codepoint string whose bound is past `MAX_UNROLLED_CODEPOINT_LEN`, and whose regex
    /// is exactly `JSON_STRING_CODEPOINT` (the only text the frontend ever pairs with this charset).
    pub(crate) fn is_unrolled_codepoint_string(&self, node: &Node) -> bool {
        let Node::StringPattern {
            regex,
            min_len,
            max_len,
            charset: Charset::Utf8CountedCodepoints,
        } = node
        else {
            return false;
        };
        let long = max_len.is_some_and(|n| n > MAX_UNROLLED_CODEPOINT_LEN)
            || min_len.is_some_and(|n| n > MAX_UNROLLED_CODEPOINT_LEN);
        long && self.str_at(*regex) == Some(crate::frontend::JSON_STRING_CODEPOINT)
    }

    /// A `pattern`/`format` string, as opposed to the unconstrained-body sentinel: its regex must
    /// run over decoded Unicode scalar bytes (`crate::structured`), never raw JSON escape spelling.
    pub(crate) fn is_user_pattern(&self, node: &Node) -> bool {
        let Node::StringPattern {
            regex,
            charset: Charset::Utf8Any,
            ..
        } = node
        else {
            return false;
        };
        self.str_at(*regex) != Some(crate::frontend::JSON_STRING_BODY)
    }

    /// An enum past `MAX_UNROLLED_ENUM` whose members are all strings: compiled as a decoded-value
    /// trie when regular, or matched by the structured backend inside non-regular composition.
    pub(crate) fn is_large_string_enum(&self, node: &Node) -> bool {
        let Node::Enum { values } = node else {
            return false;
        };
        let Some(lits) = self.lits_at(*values) else {
            return false;
        };
        lits.len() > MAX_UNROLLED_ENUM && lits.iter().all(|l| matches!(l, ScalarLit::Str(_)))
    }

    /// Whether any arena node needs `crate::structured` instead of `compile::compile_ir`: an
    /// `OpenObject`, an oversized object shuffle, `uniqueItems`/`contains`, `dependentSchemas`, or an
    /// unrolled-codepoint-count string too long to build as one DFA cheaply.
    #[must_use]
    pub fn requires_structured_backend(&self) -> bool {
        self.route_table()
            .get(self.root)
            .is_none_or(|route| route.kind != crate::routing::ExecutionKind::Regular)
    }

    /// The shared immutable per-node route table, computed once for this frozen IR.
    #[must_use]
    pub fn route_table(&self) -> &crate::routing::RouteTable {
        self.route_table
            .get_or_init(|| crate::routing::analyze(self))
    }

    /// The compile options this IR was built with.
    #[must_use]
    pub fn options(&self) -> &CompileOptions {
        &self.options
    }

    /// Resolves a string reference to its UTF-8 slice, or `None` if the run is out of range.
    #[must_use]
    pub fn str_at(&self, r: StrRef) -> Option<&str> {
        let end = (r.off as usize).checked_add(r.len as usize)?;
        let bytes = self.strings.get(r.off as usize..end)?;
        std::str::from_utf8(bytes).ok()
    }

    /// Resolves a literal run, or `None` if out of range.
    #[must_use]
    pub fn lits_at(&self, r: LitSlice) -> Option<&[ScalarLit]> {
        let end = (r.off as usize).checked_add(r.len as usize)?;
        self.literals.get(r.off as usize..end)
    }

    /// Resolves a property run, or `None` if out of range.
    #[must_use]
    pub fn props_at(&self, r: PropSlice) -> Option<&[(StrRef, NodeId)]> {
        let end = (r.off as usize).checked_add(r.len as usize)?;
        self.props.get(r.off as usize..end)
    }

    /// Resolves a `dependentRequired` pair run, or `None` if out of range.
    #[must_use]
    pub fn dependent_required_at(
        &self,
        r: DependentRequiredSlice,
    ) -> Option<&[DependentRequiredPair]> {
        let end = (r.off as usize).checked_add(r.len as usize)?;
        self.dependent_required_pairs.get(r.off as usize..end)
    }

    /// The definition body a `Node::Ref { def }` names, or `None` if the slot is out of range.
    #[must_use]
    pub fn def_target(&self, def: u32) -> Option<NodeId> {
        self.def_targets.get(def as usize).copied()
    }

    /// Resolves a branch run, or `None` if out of range.
    #[must_use]
    pub fn refs_at(&self, r: RefSlice) -> Option<&[NodeId]> {
        let end = (r.off as usize).checked_add(r.len as usize)?;
        self.node_refs.get(r.off as usize..end)
    }

    /// Whether field `idx` of an object with required set `r` is required.
    #[must_use]
    pub fn is_required(&self, r: BitSlice, idx: u32) -> bool {
        if idx >= r.len {
            return false;
        }
        let word = r.off as usize + (idx / 64) as usize;
        self.required_bits
            .get(word)
            .is_some_and(|w| (w >> (idx % 64)) & 1 == 1)
    }

    /// The memoized blake3 Merkle content address over node semantics (tags, scalars, child digests,
    /// closure, required bits) only; pointers, layout, diagnostics, and `CompileOptions` are excluded.
    #[must_use]
    pub fn canonical_hash(&self) -> [u8; 32] {
        *self.cached_hash.get_or_init(|| self.compute_hash())
    }

    /// Whether two IRs share the same content address. This is cryptographic identity, not a
    /// structural proof: equal addresses mean the schemas are the same up to a BLAKE3 collision,
    /// which is the same guarantee any content-addressed store relies on.
    #[must_use]
    pub fn same_content_address(&self, other: &Self) -> bool {
        self.canonical_hash() == other.canonical_hash()
    }

    /// Serializes the IR into the versioned MFIR envelope.
    #[must_use]
    pub fn to_wire(&self) -> Vec<u8> {
        crate::wire::encode(self)
    }

    /// Deserializes and validates an MFIR envelope into a frozen IR.
    pub fn from_wire(bytes: &[u8]) -> Result<Self, CompileError> {
        crate::wire::decode(bytes)
    }

    /// Builds the wire payload struct (a clone of the semantic fields, without the memo).
    pub(crate) fn to_wire_struct(&self) -> SchemaIRWire {
        SchemaIRWire {
            ir_version: self.ir_version,
            root: self.root,
            nodes: self.nodes.clone(),
            strings: self.strings.clone(),
            literals: self.literals.clone(),
            props: self.props.clone(),
            dependent_required_pairs: self.dependent_required_pairs.clone(),
            required_bits: self.required_bits.clone(),
            node_refs: self.node_refs.clone(),
            def_targets: self.def_targets.clone(),
            retrieval_uri: self.retrieval_uri,
            resources: self.resources.clone(),
            anchors: self.anchors.clone(),
            node_resources: self.node_resources.clone(),
            options: self.options.clone(),
            diagnostics: self.diagnostics.clone(),
        }
    }

    /// The single construction choke point: validate the parts, then freeze with an empty memo.
    /// Panic-free on adversarial input (used by both the builder and the wire decoder).
    pub(crate) fn assemble(wire: SchemaIRWire) -> Result<Self, CompileError> {
        let ir = Self {
            ir_version: wire.ir_version,
            root: wire.root,
            nodes: wire.nodes,
            strings: wire.strings,
            literals: wire.literals,
            props: wire.props,
            dependent_required_pairs: wire.dependent_required_pairs,
            required_bits: wire.required_bits,
            node_refs: wire.node_refs,
            def_targets: wire.def_targets,
            retrieval_uri: wire.retrieval_uri,
            resources: wire.resources,
            anchors: wire.anchors,
            node_resources: wire.node_resources,
            options: wire.options,
            diagnostics: wire.diagnostics,
            cached_hash: OnceLock::new(),
            route_table: OnceLock::new(),
        };
        ir.validate()?;
        Ok(ir)
    }

    /// Structural validation. Total and panic-free: every index is checked, every child is a
    /// strict back-edge, every set is canonical. Rejects a forged arena with a structured error.
    fn validate(&self) -> Result<(), CompileError> {
        if self.ir_version != IR_VERSION {
            return Err(err(
                ErrorCode::ArtifactMismatch,
                "ir_version does not match",
            ));
        }
        if self.options.max_diagnostics == 0 {
            return Err(malformed("max_diagnostics must be at least one"));
        }
        if self.root.get() as usize >= self.nodes.len() {
            return Err(malformed("root node id out of range"));
        }
        if literal_str_bytes(&self.literals) > MAX_LITERAL_STR_BYTES {
            return Err(limit("literal string bytes"));
        }
        if self.dependent_required_pairs.len() > MAX_DEPENDENT_REQUIRED_PAIRS {
            return Err(malformed("dependentRequired pair count exceeds cap"));
        }
        self.validate_resources()?;
        for (i, node) in self.nodes.iter().enumerate() {
            let parent = u32::try_from(i).map_err(|_| limit("node index"))?;
            self.validate_node(node, parent)?;
        }
        for target in &self.def_targets {
            if target.get() as usize >= self.nodes.len() {
                return Err(malformed("def target node id out of range"));
            }
        }
        self.validate_diagnostics()?;
        Ok(())
    }

    /// Re-runs structural validation for benchmark phase attribution.
    #[doc(hidden)]
    #[cfg(feature = "bench-internals")]
    pub fn validate_for_bench(&self) -> Result<(), CompileError> {
        self.validate()
    }

    fn validate_diagnostics(&self) -> Result<(), CompileError> {
        if self.diagnostics.len() as u64 > u64::from(self.options.max_diagnostics) {
            return Err(CompileError::new(
                ErrorCode::TooManyDiagnostics,
                Stage::L2,
                "more diagnostics than the configured cap",
            ));
        }
        for d in &self.diagnostics {
            let node = self
                .nodes
                .get(d.node.get() as usize)
                .ok_or_else(|| malformed("diagnostic references a node out of range"))?;
            let Node::Unsupported { keyword, reason } = node else {
                return Err(malformed("a diagnostic must point at an Unsupported node"));
            };
            if *keyword != d.keyword {
                return Err(malformed("diagnostic keyword disagrees with its node"));
            }
            if *reason != d.reason {
                return Err(malformed("diagnostic reason disagrees with its node"));
            }
            if self.str_at(d.keyword).is_none() || self.str_at(d.json_pointer).is_none() {
                return Err(malformed("diagnostic string reference out of range"));
            }
            if let Some(span) = d.source_span {
                if span.start > span.end {
                    return Err(malformed("diagnostic span start is past its end"));
                }
            }
        }
        Ok(())
    }

    fn validate_node(&self, node: &Node, parent: u32) -> Result<(), CompileError> {
        let child_ok = |c: NodeId| c.get() < parent;
        match node {
            Node::Null | Node::Boolean | Node::Never => Ok(()),
            Node::StringConst { value } => {
                self.str_at(*value)
                    .ok_or_else(|| malformed("string const reference out of range"))?;
                Ok(())
            }
            Node::StringPattern {
                regex,
                min_len,
                max_len,
                charset,
            } => {
                let re = self
                    .str_at(*regex)
                    .ok_or_else(|| malformed("pattern reference out of range"))?;
                if *charset == Charset::Utf8Any && re != crate::frontend::JSON_STRING_BODY {
                    validate_search_pattern(re)?;
                } else {
                    validate_pattern(re)?;
                }
                if let (Some(lo), Some(hi)) = (min_len, max_len) {
                    if hi < lo {
                        return Err(malformed("string maxLength is less than minLength"));
                    }
                }
                Ok(())
            }
            Node::Integer {
                minimum,
                maximum,
                multiple_of,
            } => {
                if let (Some(lo), Some(hi)) = (minimum, maximum) {
                    if hi < lo {
                        return Err(malformed("integer maximum is less than minimum"));
                    }
                }
                if *multiple_of == Some(0) {
                    return Err(malformed("multipleOf must be a positive integer"));
                }
                Ok(())
            }
            Node::Number {
                integer_only: _,
                minimum,
                maximum,
                multiple_of,
            } => {
                if let (Some((lo, ls, lo_excl)), Some((hi, hs, hi_excl))) = (minimum, maximum) {
                    let ord = decimal_cmp((*lo, *ls), (*hi, *hs));
                    if ord == std::cmp::Ordering::Greater
                        || (ord == std::cmp::Ordering::Equal && (*lo_excl || *hi_excl))
                    {
                        return Err(malformed("number maximum is not above minimum"));
                    }
                }
                if let Some((coef, _)) = multiple_of {
                    if *coef == 0 {
                        return Err(malformed("multipleOf must be positive"));
                    }
                }
                Ok(())
            }
            Node::LexicalNumber { regex } => {
                let regex = self
                    .str_at(*regex)
                    .ok_or_else(|| malformed("lexical number pattern out of range"))?;
                validate_pattern(regex)
            }
            Node::Enum { values } => {
                let lits = self
                    .lits_at(*values)
                    .ok_or_else(|| malformed("enum literal run out of range"))?;
                if lits.is_empty() {
                    return Err(malformed("an enum must have at least one value"));
                }
                for pair in lits.windows(2) {
                    if pair[0] >= pair[1] {
                        return Err(malformed(
                            "enum values must be canonically sorted and unique",
                        ));
                    }
                }
                Ok(())
            }
            Node::Array {
                items,
                min_items,
                max_items,
                ..
            } => {
                if let ItemsPolicy::Schema(id) = items {
                    if !child_ok(*id) {
                        return Err(malformed("array item is not a back-edge"));
                    }
                }
                if let Some(max) = max_items {
                    if max < min_items {
                        return Err(malformed("array maxItems is less than minItems"));
                    }
                }
                Ok(())
            }
            Node::Tuple {
                prefix,
                tail,
                min_items,
                max_items,
                ..
            } => {
                let ids = self
                    .refs_at(*prefix)
                    .ok_or_else(|| malformed("tuple prefix run out of range"))?;
                if ids.is_empty() {
                    return Err(malformed("a tuple must have at least one prefix position"));
                }
                for id in ids {
                    if !child_ok(*id) {
                        return Err(malformed("tuple prefix element is not a back-edge"));
                    }
                }
                if let Some(t) = tail {
                    if !child_ok(*t) {
                        return Err(malformed("tuple tail is not a back-edge"));
                    }
                }
                if let Some(max) = max_items {
                    if max < min_items {
                        return Err(malformed("tuple maxItems is less than minItems"));
                    }
                }
                // A closed tuple (no tail) cannot hold more items than the prefix supplies, so a
                // minItems past the prefix length is an empty language, rejected at build.
                if tail.is_none() && *min_items > prefix.len {
                    return Err(malformed("closed tuple minItems exceeds the prefix length"));
                }
                Ok(())
            }
            Node::Object {
                fields,
                required,
                dependent,
                dependent_required,
                ..
            } => {
                let props = self
                    .props_at(*fields)
                    .ok_or_else(|| malformed("object field run out of range"))?;
                let mut seen: FxHashMap<&str, ()> = FxHashMap::default();
                seen.reserve(props.len());
                for (name, child) in props {
                    let key = self
                        .str_at(*name)
                        .ok_or_else(|| malformed("object field name out of range"))?;
                    if seen.insert(key, ()).is_some() {
                        return Err(malformed("duplicate object key"));
                    }
                    if !child_ok(*child) {
                        return Err(malformed("object field value is not a back-edge"));
                    }
                }
                self.validate_required(*required, props.len())?;
                self.validate_dependent_props(*dependent, child_ok)?;
                self.validate_dependent_required(*dependent_required)
            }
            Node::OpenObject {
                known,
                known_required,
                patterns,
                additional,
                property_names,
                min_properties,
                max_properties,
                dependent,
                dependent_required,
            } => {
                let known_props = self
                    .props_at(*known)
                    .ok_or_else(|| malformed("open object known props out of range"))?;
                let mut seen: FxHashMap<&str, ()> = FxHashMap::default();
                seen.reserve(known_props.len());
                for (name, child) in known_props {
                    let key = self
                        .str_at(*name)
                        .ok_or_else(|| malformed("open object field name out of range"))?;
                    if seen.insert(key, ()).is_some() {
                        return Err(malformed("duplicate object key"));
                    }
                    if !child_ok(*child) {
                        return Err(malformed("open object field value is not a back-edge"));
                    }
                }
                self.validate_required(*known_required, known_props.len())?;
                let pattern_props = self
                    .props_at(*patterns)
                    .ok_or_else(|| malformed("open object pattern props out of range"))?;
                for (regex, child) in pattern_props {
                    let re = self
                        .str_at(*regex)
                        .ok_or_else(|| malformed("open object pattern regex out of range"))?;
                    validate_search_pattern(re)?;
                    if !child_ok(*child) {
                        return Err(malformed("open object pattern value is not a back-edge"));
                    }
                }
                if let AdditionalPolicy::Schema(id) = additional {
                    if !child_ok(*id) {
                        return Err(malformed(
                            "open object additional schema is not a back-edge",
                        ));
                    }
                }
                if let Some(pn) = property_names {
                    if !child_ok(*pn) {
                        return Err(malformed(
                            "open object propertyNames schema is not a back-edge",
                        ));
                    }
                }
                if let (Some(lo), Some(hi)) = (min_properties, max_properties) {
                    if hi < lo {
                        return Err(malformed(
                            "open object maxProperties is less than minProperties",
                        ));
                    }
                }
                self.validate_dependent_props(*dependent, child_ok)?;
                self.validate_dependent_required(*dependent_required)
            }
            Node::Union { branches } => {
                let ids = self
                    .refs_at(*branches)
                    .ok_or_else(|| malformed("union branch run out of range"))?;
                if ids.is_empty() {
                    return Err(malformed("a type union must have at least one branch"));
                }
                for id in ids {
                    if !child_ok(*id) {
                        return Err(malformed("union branch is not a back-edge"));
                    }
                }
                Ok(())
            }
            Node::Intersection { branches } => {
                let ids = self
                    .refs_at(*branches)
                    .ok_or_else(|| malformed("intersection branch run out of range"))?;
                if ids.is_empty() {
                    return Err(malformed("an intersection must have at least one branch"));
                }
                for id in ids {
                    if !child_ok(*id) {
                        return Err(malformed("intersection branch is not a back-edge"));
                    }
                }
                Ok(())
            }
            Node::ExactlyOne { branches } => {
                let ids = self
                    .refs_at(*branches)
                    .ok_or_else(|| malformed("exactly-one branch run out of range"))?;
                if ids.is_empty() {
                    return Err(malformed("an exactly-one must have at least one branch"));
                }
                for id in ids {
                    if !child_ok(*id) {
                        return Err(malformed("exactly-one branch is not a back-edge"));
                    }
                }
                Ok(())
            }
            Node::Not { inner } => {
                if !child_ok(*inner) {
                    return Err(malformed("not target is not a back-edge"));
                }
                Ok(())
            }
            Node::Ref { def } => {
                if (*def as usize) >= self.def_targets.len() {
                    return Err(malformed("ref names a def slot out of range"));
                }
                Ok(())
            }
            Node::DynamicRef {
                initial_target,
                anchor,
            } => {
                if initial_target.get() as usize >= self.nodes.len() {
                    return Err(malformed("dynamic reference target is out of range"));
                }
                let record = self
                    .anchors
                    .get(anchor.get() as usize)
                    .ok_or_else(|| malformed("dynamic reference anchor is out of range"))?;
                let resource = self
                    .resources
                    .get(record.resource.get() as usize)
                    .ok_or_else(|| {
                        malformed("dynamic reference anchor resource is out of range")
                    })?;
                let dynamic_start = resource.dynamic_anchors.off as usize;
                let dynamic_end = dynamic_start
                    .checked_add(resource.dynamic_anchors.len as usize)
                    .ok_or_else(|| malformed("dynamic anchor run overflows"))?;
                let anchor_index = anchor.get() as usize;
                if !(dynamic_start..dynamic_end).contains(&anchor_index) {
                    return Err(malformed(
                        "dynamic reference names an anchor that is not dynamic",
                    ));
                }
                let consistent = record.target == *initial_target
                    || matches!(
                        self.node(*initial_target),
                        Some(Node::Ref { def }) if self.def_target(*def) == Some(record.target)
                    );
                if !consistent {
                    return Err(malformed(
                        "dynamic reference target disagrees with its initial anchor",
                    ));
                }
                Ok(())
            }
            Node::Unevaluated {
                scope, unevaluated, ..
            } => {
                if !child_ok(*scope) || !child_ok(*unevaluated) {
                    return Err(malformed("unevaluated child is not a back-edge"));
                }
                Ok(())
            }
            Node::Unsupported { keyword, .. } => {
                self.str_at(*keyword)
                    .ok_or_else(|| malformed("unsupported keyword reference out of range"))?;
                Ok(())
            }
        }
    }

    fn validate_resources(&self) -> Result<(), CompileError> {
        if self.resources.is_empty() || self.resources.len() > MAX_RESOURCES {
            return Err(malformed("resource table is empty or exceeds its cap"));
        }
        if self.anchors.len() > MAX_ANCHORS {
            return Err(malformed("anchor table exceeds its cap"));
        }
        if self.node_resources.len() != self.nodes.len() {
            return Err(malformed("node-to-resource table length mismatch"));
        }
        if let Some(uri) = self.retrieval_uri {
            let uri = self
                .str_at(uri)
                .ok_or_else(|| malformed("retrieval URI string run is invalid"))?;
            validate_canonical_uri(uri)?;
        }
        for resource in &self.node_resources {
            if resource.get() as usize >= self.resources.len() {
                return Err(malformed("node is assigned to an invalid resource"));
            }
        }
        let mut previous_uri: Option<&str> = None;
        let mut anonymous_seen = false;
        let mut anchor_cursor = 0usize;
        for (index, resource) in self.resources.iter().enumerate() {
            if resource.root.get() as usize >= self.nodes.len() {
                return Err(malformed("resource root is out of range"));
            }
            let resource_id =
                ResourceId(u32::try_from(index).map_err(|_| limit("resource index"))?);
            if self.node_resource(resource.root) != Some(resource_id) {
                return Err(malformed("resource root has an inconsistent owner"));
            }
            match resource.canonical_uri {
                None => {
                    if anonymous_seen || previous_uri.is_some() {
                        return Err(malformed("anonymous resource ordering is not canonical"));
                    }
                    anonymous_seen = true;
                }
                Some(uri) => {
                    let uri = self
                        .str_at(uri)
                        .ok_or_else(|| malformed("resource URI string run is invalid"))?;
                    validate_canonical_uri(uri)?;
                    if previous_uri.is_some_and(|previous| previous >= uri) {
                        return Err(malformed("resource URIs are duplicate or not canonical"));
                    }
                    previous_uri = Some(uri);
                }
            }
            let mut names = rustc_hash::FxHashSet::default();
            for run in [resource.static_anchors, resource.dynamic_anchors] {
                if usize::try_from(run.off).map_err(|_| malformed("anchor run offset"))?
                    != anchor_cursor
                {
                    return Err(malformed("anchor runs are not in canonical order"));
                }
                let anchors = self.validate_anchor_run(run, resource_id)?;
                anchor_cursor = anchor_cursor
                    .checked_add(anchors.len())
                    .ok_or_else(|| malformed("anchor run overflows"))?;
                names
                    .try_reserve(anchors.len())
                    .map_err(|_| limit("anchor validation set"))?;
                for anchor in anchors {
                    let name = self
                        .str_at(anchor.name)
                        .ok_or_else(|| malformed("anchor name string run is invalid"))?;
                    if !names.insert(name) {
                        return Err(malformed(
                            "static and dynamic anchors collide inside one resource",
                        ));
                    }
                }
            }
        }
        if anchor_cursor != self.anchors.len() {
            return Err(malformed("anchor table has unclaimed records"));
        }
        Ok(())
    }

    fn validate_anchor_run(
        &self,
        run: AnchorSlice,
        resource: ResourceId,
    ) -> Result<&[AnchorRecord], CompileError> {
        let start = usize::try_from(run.off).map_err(|_| malformed("anchor run offset"))?;
        let end = start
            .checked_add(usize::try_from(run.len).map_err(|_| malformed("anchor run length"))?)
            .ok_or_else(|| malformed("anchor run overflows"))?;
        let anchors = self
            .anchors
            .get(start..end)
            .ok_or_else(|| malformed("anchor run is out of range"))?;
        let mut previous_name: Option<&str> = None;
        for anchor in anchors {
            if anchor.resource != resource {
                return Err(malformed("anchor run contains another resource"));
            }
            if anchor.target.get() as usize >= self.nodes.len()
                || self.node_resource(anchor.target) != Some(resource)
            {
                return Err(malformed("anchor target or resource is invalid"));
            }
            let name = self
                .str_at(anchor.name)
                .ok_or_else(|| malformed("anchor name string run is invalid"))?;
            if previous_name.is_some_and(|previous| previous >= name) {
                return Err(malformed("anchor names are duplicate or not canonical"));
            }
            previous_name = Some(name);
        }
        Ok(anchors)
    }

    fn validate_required(
        &self,
        required: BitSlice,
        field_count: usize,
    ) -> Result<(), CompileError> {
        if required.len as usize != field_count {
            return Err(malformed(
                "required bitset width does not match field count",
            ));
        }
        let words = field_count.div_ceil(64);
        let end = (required.off as usize)
            .checked_add(words)
            .ok_or_else(|| malformed("required bitset overflows"))?;
        let slice = self
            .required_bits
            .get(required.off as usize..end)
            .ok_or_else(|| malformed("required bitset out of range"))?;
        // The tail bits past the field count must be zero so the words hash canonically.
        if field_count % 64 != 0 {
            if let Some(last) = slice.last() {
                let used = field_count % 64;
                if last >> used != 0 {
                    return Err(malformed("required bitset has non-zero padding bits"));
                }
            }
        }
        Ok(())
    }

    /// Validates a `dependentSchemas` run: distinct keys, every value a strict back-edge.
    fn validate_dependent_props(
        &self,
        dependent: PropSlice,
        child_ok: impl Fn(NodeId) -> bool,
    ) -> Result<(), CompileError> {
        let props = self
            .props_at(dependent)
            .ok_or_else(|| malformed("dependentSchemas prop run out of range"))?;
        let mut seen: FxHashMap<&str, ()> = FxHashMap::default();
        seen.reserve(props.len());
        for (name, child) in props {
            let key = self
                .str_at(*name)
                .ok_or_else(|| malformed("dependentSchemas key out of range"))?;
            if seen.insert(key, ()).is_some() {
                return Err(malformed("duplicate dependentSchemas key"));
            }
            if !child_ok(*child) {
                return Err(malformed("dependentSchemas value is not a back-edge"));
            }
        }
        Ok(())
    }

    fn validate_dependent_required(
        &self,
        dependent: DependentRequiredSlice,
    ) -> Result<(), CompileError> {
        let pairs = self
            .dependent_required_at(dependent)
            .ok_or_else(|| malformed("dependentRequired pair run out of range"))?;
        if pairs.len() > MAX_DEPENDENT_REQUIRED_PAIRS {
            return Err(malformed("dependentRequired pair count exceeds cap"));
        }
        let mut seen = rustc_hash::FxHashSet::default();
        seen.try_reserve(pairs.len())
            .map_err(|_| limit("dependentRequired validation set"))?;
        for pair in pairs {
            let trigger = self
                .str_at(pair.trigger)
                .ok_or_else(|| malformed("dependentRequired trigger out of range"))?;
            let required = self
                .str_at(pair.required)
                .ok_or_else(|| malformed("dependentRequired required name out of range"))?;
            if !seen.insert((trigger, required)) {
                return Err(malformed("duplicate dependentRequired pair"));
            }
        }
        Ok(())
    }

    fn compute_hash(&self) -> [u8; 32] {
        let mut digests: Vec<[u8; 32]> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            digests.push(self.node_digest(node, &digests));
        }
        let mut top = blake3::Hasher::new();
        top.update(b"MFIR-ir-v");
        top.update(&self.ir_version.to_le_bytes());
        top.update(&digests[self.root.get() as usize]);
        // A def body is reachable only through the def table, so its content is folded in by slot;
        // without this two recursive schemas sharing ref structure but differing bodies would collide.
        top.update(&(self.def_targets.len() as u64).to_le_bytes());
        for target in &self.def_targets {
            top.update(&digests[target.get() as usize]);
        }
        top.update(&[self.options.object_closure as u8]);
        top.update(&[u8::from(self.options.format_assertion)]);
        top.update(&[self.options.unsupported_policy as u8]);
        top.update(&self.options.max_diagnostics.to_le_bytes());
        match self.retrieval_uri() {
            None => {
                top.update(&[0u8]);
            }
            Some(uri) => {
                top.update(&[1u8]);
                feed_bytes(&mut top, uri.as_bytes())
            }
        };
        top.update(&(self.resources.len() as u64).to_le_bytes());
        for (index, resource) in self.resources.iter().enumerate() {
            match resource.canonical_uri.and_then(|uri| self.str_at(uri)) {
                None => {
                    top.update(&[0u8]);
                }
                Some(uri) => {
                    top.update(&[1u8]);
                    feed_bytes(&mut top, uri.as_bytes())
                }
            };
            top.update(&resource.root.get().to_le_bytes());
            let resource_id = ResourceId(u32::try_from(index).unwrap_or(u32::MAX));
            for (dynamic, run) in [
                (false, resource.static_anchors),
                (true, resource.dynamic_anchors),
            ] {
                top.update(&[u8::from(dynamic)]);
                let start = run.off as usize;
                let end = start.saturating_add(run.len as usize);
                let anchors = self.anchors.get(start..end).unwrap_or(&[]);
                top.update(&(anchors.len() as u64).to_le_bytes());
                for anchor in anchors {
                    top.update(&resource_id.get().to_le_bytes());
                    feed_bytes(&mut top, self.str_at(anchor.name).unwrap_or("").as_bytes());
                    top.update(&anchor.target.get().to_le_bytes());
                }
            }
        }
        top.update(&(self.node_resources.len() as u64).to_le_bytes());
        for resource in &self.node_resources {
            top.update(&resource.get().to_le_bytes());
        }
        *top.finalize().as_bytes()
    }

    fn node_digest(&self, node: &Node, digests: &[[u8; 32]]) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        match node {
            Node::Null => h.update(&[0u8]),
            Node::Boolean => h.update(&[1u8]),
            Node::Never => h.update(&[16u8]),
            Node::StringConst { value } => {
                h.update(&[2u8]);
                feed_bytes(&mut h, self.str_at(*value).unwrap_or("").as_bytes());
                &mut h
            }
            Node::StringPattern {
                regex,
                min_len,
                max_len,
                charset,
            } => {
                h.update(&[3u8]);
                feed_bytes(&mut h, self.str_at(*regex).unwrap_or("").as_bytes());
                feed_opt_u32(&mut h, *min_len);
                feed_opt_u32(&mut h, *max_len);
                h.update(&[charset_tag(*charset)]);
                &mut h
            }
            Node::Integer {
                minimum,
                maximum,
                multiple_of,
            } => {
                h.update(&[4u8]);
                feed_opt_i64(&mut h, *minimum);
                feed_opt_i64(&mut h, *maximum);
                feed_opt_u64(&mut h, *multiple_of);
                &mut h
            }
            Node::Number {
                integer_only,
                minimum,
                maximum,
                multiple_of,
            } => {
                h.update(&[12u8]);
                h.update(&[u8::from(*integer_only)]);
                feed_opt_decimal_bound(&mut h, *minimum);
                feed_opt_decimal_bound(&mut h, *maximum);
                match multiple_of {
                    None => h.update(&[0u8]),
                    Some((coef, exp)) => h
                        .update(&[1u8])
                        .update(&coef.to_le_bytes())
                        .update(&exp.to_le_bytes()),
                };
                &mut h
            }
            Node::LexicalNumber { regex } => {
                h.update(&[17u8]);
                feed_bytes(&mut h, self.str_at(*regex).unwrap_or("").as_bytes());
                &mut h
            }
            Node::Enum { values } => {
                h.update(&[5u8]);
                let lits = self.lits_at(*values).unwrap_or(&[]);
                h.update(&(lits.len() as u64).to_le_bytes());
                for lit in lits {
                    feed_bytes(&mut h, &lit_bytes(lit));
                }
                &mut h
            }
            Node::Array {
                items,
                min_items,
                max_items,
                unique_items,
                contains,
                items_annotates,
            } => {
                h.update(&[6u8]);
                match items {
                    ItemsPolicy::Schema(id) => {
                        h.update(&[0u8]);
                        h.update(&digests[id.get() as usize]);
                    }
                    ItemsPolicy::AllowAny => {
                        h.update(&[1u8]);
                    }
                }
                h.update(&min_items.to_le_bytes());
                feed_opt_u32(&mut h, *max_items);
                h.update(&[u8::from(*unique_items)]);
                feed_contains(&mut h, contains, digests);
                h.update(&[u8::from(*items_annotates)]);
                &mut h
            }
            Node::Tuple {
                prefix,
                tail,
                min_items,
                max_items,
                unique_items,
                tail_annotates,
                contains,
            } => {
                h.update(&[13u8]);
                let ids = self.refs_at(*prefix).unwrap_or(&[]);
                h.update(&(ids.len() as u64).to_le_bytes());
                for id in ids {
                    h.update(&digests[id.get() as usize]);
                }
                match tail {
                    None => h.update(&[0u8]),
                    Some(t) => h.update(&[1u8]).update(&digests[t.get() as usize]),
                };
                h.update(&min_items.to_le_bytes());
                feed_opt_u32(&mut h, *max_items);
                h.update(&[u8::from(*unique_items)]);
                feed_contains(&mut h, contains, digests);
                h.update(&[u8::from(*tail_annotates)]);
                &mut h
            }
            Node::Object {
                fields,
                required,
                closure,
                dependent,
                dependent_required,
            } => {
                h.update(&[7u8]);
                let props = self.props_at(*fields).unwrap_or(&[]);
                h.update(&(props.len() as u64).to_le_bytes());
                // Field count is a u32 by construction, so the u32 index never overflows here.
                for ((name, child), idx) in props.iter().zip(0u32..) {
                    feed_bytes(&mut h, self.str_at(*name).unwrap_or("").as_bytes());
                    h.update(&digests[child.get() as usize]);
                    h.update(&[u8::from(self.is_required(*required, idx))]);
                }
                h.update(&[closure_tag(*closure)]);
                self.feed_dependent(&mut h, *dependent, digests);
                self.feed_dependent_required(&mut h, *dependent_required);
                &mut h
            }
            Node::OpenObject {
                known,
                known_required,
                patterns,
                additional,
                property_names,
                min_properties,
                max_properties,
                dependent,
                dependent_required,
            } => {
                h.update(&[14u8]);
                let kp = self.props_at(*known).unwrap_or(&[]);
                h.update(&(kp.len() as u64).to_le_bytes());
                for ((name, child), idx) in kp.iter().zip(0u32..) {
                    feed_bytes(&mut h, self.str_at(*name).unwrap_or("").as_bytes());
                    h.update(&digests[child.get() as usize]);
                    h.update(&[u8::from(self.is_required(*known_required, idx))]);
                }
                let pp = self.props_at(*patterns).unwrap_or(&[]);
                h.update(&(pp.len() as u64).to_le_bytes());
                for (regex, child) in pp {
                    feed_bytes(&mut h, self.str_at(*regex).unwrap_or("").as_bytes());
                    h.update(&digests[child.get() as usize]);
                }
                match additional {
                    AdditionalPolicy::Forbid => h.update(&[0u8]),
                    AdditionalPolicy::AllowAny => h.update(&[1u8]),
                    AdditionalPolicy::Schema(id) => {
                        h.update(&[2u8]).update(&digests[id.get() as usize])
                    }
                    AdditionalPolicy::Open => h.update(&[3u8]),
                };
                match property_names {
                    None => h.update(&[0u8]),
                    Some(id) => h.update(&[1u8]).update(&digests[id.get() as usize]),
                };
                feed_opt_u32(&mut h, *min_properties);
                feed_opt_u32(&mut h, *max_properties);
                self.feed_dependent(&mut h, *dependent, digests);
                self.feed_dependent_required(&mut h, *dependent_required);
                &mut h
            }
            Node::Union { branches } => {
                h.update(&[9u8]);
                let ids = self.refs_at(*branches).unwrap_or(&[]);
                h.update(&(ids.len() as u64).to_le_bytes());
                for id in ids {
                    h.update(&digests[id.get() as usize]);
                }
                &mut h
            }
            Node::Intersection { branches } => {
                h.update(&[10u8]);
                let ids = self.refs_at(*branches).unwrap_or(&[]);
                h.update(&(ids.len() as u64).to_le_bytes());
                for id in ids {
                    h.update(&digests[id.get() as usize]);
                }
                &mut h
            }
            Node::ExactlyOne { branches } => {
                h.update(&[11u8]);
                let ids = self.refs_at(*branches).unwrap_or(&[]);
                h.update(&(ids.len() as u64).to_le_bytes());
                for id in ids {
                    h.update(&digests[id.get() as usize]);
                }
                &mut h
            }
            Node::Not { inner } => {
                h.update(&[15u8]);
                h.update(&digests[inner.get() as usize]);
                &mut h
            }
            Node::Ref { def } => {
                h.update(&[17u8]);
                h.update(&def.to_le_bytes());
                &mut h
            }
            Node::DynamicRef {
                initial_target,
                anchor,
            } => {
                h.update(&[19u8]);
                h.update(&initial_target.get().to_le_bytes());
                h.update(&anchor.get().to_le_bytes());
                &mut h
            }
            Node::Unevaluated {
                kind,
                scope,
                unevaluated,
            } => {
                h.update(&[18u8]);
                h.update(&[*kind as u8]);
                h.update(&digests[scope.get() as usize]);
                h.update(&digests[unevaluated.get() as usize]);
                &mut h
            }
            Node::Unsupported { keyword, reason } => {
                h.update(&[8u8]);
                feed_bytes(&mut h, self.str_at(*keyword).unwrap_or("").as_bytes());
                h.update(&(*reason as u32).to_le_bytes());
                &mut h
            }
        };
        *h.finalize().as_bytes()
    }

    /// Feeds a `dependentSchemas` run into a node digest, keyed by name so equal sets hash equal
    /// regardless of arena insertion order.
    fn feed_dependent(&self, h: &mut blake3::Hasher, dependent: PropSlice, digests: &[[u8; 32]]) {
        let props = self.props_at(dependent).unwrap_or(&[]);
        h.update(&(props.len() as u64).to_le_bytes());
        for (name, child) in props {
            feed_bytes(h, self.str_at(*name).unwrap_or("").as_bytes());
            h.update(&digests[child.get() as usize]);
        }
    }

    fn feed_dependent_required(&self, h: &mut blake3::Hasher, dependent: DependentRequiredSlice) {
        let mut pairs: Vec<(&[u8], &[u8])> = self
            .dependent_required_at(dependent)
            .unwrap_or(&[])
            .iter()
            .map(|pair| {
                (
                    self.str_at(pair.trigger).unwrap_or("").as_bytes(),
                    self.str_at(pair.required).unwrap_or("").as_bytes(),
                )
            })
            .collect();
        pairs.sort_unstable();
        h.update(&(pairs.len() as u64).to_le_bytes());
        for (trigger, required) in pairs {
            feed_bytes(h, trigger);
            feed_bytes(h, required);
        }
    }
}

fn feed_bytes(h: &mut blake3::Hasher, bytes: &[u8]) {
    h.update(&(bytes.len() as u64).to_le_bytes());
    h.update(bytes);
}

fn feed_opt_u32(h: &mut blake3::Hasher, v: Option<u32>) {
    match v {
        None => h.update(&[0u8]),
        Some(x) => h.update(&[1u8]).update(&x.to_le_bytes()),
    };
}

fn feed_opt_i64(h: &mut blake3::Hasher, v: Option<i64>) {
    match v {
        None => h.update(&[0u8]),
        Some(x) => h.update(&[1u8]).update(&x.to_le_bytes()),
    };
}

fn feed_opt_decimal_bound(h: &mut blake3::Hasher, v: Option<(i128, u32, bool)>) {
    match v {
        None => h.update(&[0u8]),
        Some((x, scale, excl)) => h
            .update(&[1u8])
            .update(&x.to_le_bytes())
            .update(&scale.to_le_bytes())
            .update(&[u8::from(excl)]),
    };
}

/// Compares `(coefficient, scale)` decimals without scale-sized work or overflow.
pub(crate) fn decimal_cmp(a: (i128, u32), b: (i128, u32)) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if a.0 == 0 || b.0 == 0 || a.0.is_negative() != b.0.is_negative() {
        return a.0.signum().cmp(&b.0.signum());
    }
    let (a_digits, a_len) = decimal_digits(a.0.unsigned_abs());
    let (b_digits, b_len) = decimal_digits(b.0.unsigned_abs());
    let a_exponent = i64::try_from(a_len).unwrap_or(i64::MAX) - i64::from(a.1);
    let b_exponent = i64::try_from(b_len).unwrap_or(i64::MAX) - i64::from(b.1);
    let mut magnitude = a_exponent.cmp(&b_exponent);
    if magnitude == Ordering::Equal {
        for index in 0..a_len.max(b_len) {
            magnitude = a_digits[index].cmp(&b_digits[index]);
            if magnitude != Ordering::Equal {
                break;
            }
        }
    }
    if a.0.is_negative() {
        magnitude.reverse()
    } else {
        magnitude
    }
}

fn decimal_digits(mut value: u128) -> ([u8; 39], usize) {
    let mut reverse = [0u8; 39];
    let mut length = 0;
    while value > 0 {
        reverse[length] = u8::try_from(value % 10).unwrap_or(0);
        value /= 10;
        length += 1;
    }
    let mut digits = [0u8; 39];
    for index in 0..length {
        digits[index] = reverse[length - index - 1];
    }
    (digits, length)
}

fn feed_contains(
    h: &mut blake3::Hasher,
    contains: &Option<ContainsConstraint>,
    digests: &[[u8; 32]],
) {
    match contains {
        None => {
            h.update(&[0u8]);
        }
        Some(c) => {
            h.update(&[1u8]);
            match c.policy {
                ContainsPolicy::Never => h.update(&[0u8]),
                ContainsPolicy::Always => h.update(&[1u8]),
                ContainsPolicy::Schema(id) => h.update(&[2u8]).update(&digests[id.get() as usize]),
            };
            h.update(&c.min.to_le_bytes());
            feed_opt_u32(h, c.max);
        }
    }
}

fn feed_opt_u64(h: &mut blake3::Hasher, v: Option<u64>) {
    match v {
        None => h.update(&[0u8]),
        Some(x) => h.update(&[1u8]).update(&x.to_le_bytes()),
    };
}

fn charset_tag(c: Charset) -> u8 {
    match c {
        Charset::Utf8Any => 0,
        Charset::AsciiPrintableNoQuoteBackslash => 1,
        Charset::Utf8CountedCodepoints => 2,
    }
}

fn closure_tag(c: ObjectClosure) -> u8 {
    match c {
        ObjectClosure::RejectOpenObjects => 0,
        ObjectClosure::AssumeClosedProfile => 1,
        ObjectClosure::Forbidden => 2,
        ObjectClosure::AllowOpenProfile => 3,
    }
}

fn lit_bytes(lit: &ScalarLit) -> Vec<u8> {
    match lit {
        ScalarLit::Null => vec![0u8],
        ScalarLit::Bool(b) => vec![1u8, u8::from(*b)],
        ScalarLit::Int(i) => {
            let mut v = vec![2u8];
            v.extend_from_slice(&i.to_le_bytes());
            v
        }
        ScalarLit::Str(s) => {
            let mut v = vec![3u8];
            v.extend_from_slice(s.as_bytes());
            v
        }
        ScalarLit::Json(s) => {
            let mut v = vec![4u8];
            v.extend_from_slice(s.as_bytes());
            v
        }
    }
}

/// Rejects patterns the byte-level engine cannot compile.
fn validate_pattern(regex: &str) -> Result<(), CompileError> {
    if has_anchor_outside_brackets(regex.as_bytes()) {
        return Err(unsupported("explicit anchors are not allowed"));
    }
    validate_pattern_body(regex, false)
}

/// Like `validate_pattern`, but permits `^` and `$` for structured search.
/// Used by `patternProperties` and `propertyNames`.
pub(crate) fn validate_search_pattern(regex: &str) -> Result<(), CompileError> {
    validate_pattern_body(regex, true)
}

fn validate_pattern_body(regex: &str, allow_leading_assertions: bool) -> Result<(), CompileError> {
    use crate::frontend::regex::{analyze_compiled_pattern, PatternAnalysis};

    match analyze_compiled_pattern(regex) {
        PatternAnalysis::Ordinary => Ok(()),
        PatternAnalysis::Leading(_) if allow_leading_assertions => Ok(()),
        PatternAnalysis::Leading(_) => {
            Err(unsupported("unsupported regex construct").with_observed("look-ahead"))
        }
        PatternAnalysis::Unsupported(what) => {
            Err(unsupported("unsupported regex construct").with_observed(what))
        }
        PatternAnalysis::Malformed(what) => {
            Err(malformed("malformed regex syntax").with_observed(what))
        }
    }
}

/// True if `^` or `$` appears as an anchor (outside a bracket expression, unescaped). Inside
/// `[...]` those bytes are a negation or a literal, which the byte engine accepts.
pub(crate) fn has_anchor_outside_brackets(bytes: &[u8]) -> bool {
    let mut in_bracket = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'[' if !in_bracket => in_bracket = true,
            b']' if in_bracket => in_bracket = false,
            b'^' | b'$' if !in_bracket => return true,
            _ => {}
        }
        i += 1;
    }
    false
}

fn err(code: ErrorCode, message: &'static str) -> CompileError {
    CompileError::new(code, Stage::L2, message)
}

fn malformed(message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L2, message)
}

fn validate_canonical_uri(value: &str) -> Result<(), CompileError> {
    let uri = Uri::parse(value).map_err(|_| malformed("canonical resource URI is invalid"))?;
    if uri.fragment().is_some() || uri.normalize().as_str() != value {
        return Err(malformed(
            "resource URI is not fragmentless canonical RFC 3986 form",
        ));
    }
    Ok(())
}

fn unsupported(message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Unsupported, Stage::L2, message)
}

/// Hash-consing arena builder. Structurally equal subtrees collapse to one `NodeId` because every
/// side blob is content-interned, so equal children yield equal `Node` values.
pub struct Builder {
    nodes: Vec<Node>,
    strings: Vec<u8>,
    literals: Vec<ScalarLit>,
    props: Vec<(StrRef, NodeId)>,
    dependent_required_pairs: Vec<DependentRequiredPair>,
    required_bits: Vec<u64>,
    node_refs: Vec<NodeId>,
    def_targets: Vec<Option<NodeId>>,
    diagnostics: Vec<Diagnostic>,
    literal_str_bytes: usize,
    node_intern: FxHashMap<(ResourceId, Node), NodeId>,
    node_resources: Vec<ResourceId>,
    current_resource: ResourceId,
    str_intern: FxHashMap<String, StrRef>,
    lit_intern: FxHashMap<Vec<ScalarLit>, LitSlice>,
    prop_intern: FxHashMap<Vec<(StrRef, NodeId)>, PropSlice>,
    dependent_required_intern: FxHashMap<Vec<DependentRequiredPair>, DependentRequiredSlice>,
    bit_intern: FxHashMap<Vec<u64>, BitSlice>,
    ref_intern: FxHashMap<Vec<NodeId>, RefSlice>,
    options: CompileOptions,
}

impl Builder {
    /// A new builder with the given options.
    #[must_use]
    pub fn new(options: CompileOptions) -> Self {
        Self {
            nodes: Vec::new(),
            strings: Vec::new(),
            literals: Vec::new(),
            props: Vec::new(),
            dependent_required_pairs: Vec::new(),
            required_bits: Vec::new(),
            node_refs: Vec::new(),
            def_targets: Vec::new(),
            diagnostics: Vec::new(),
            literal_str_bytes: 0,
            node_intern: FxHashMap::default(),
            node_resources: Vec::new(),
            current_resource: ResourceId(0),
            str_intern: FxHashMap::default(),
            lit_intern: FxHashMap::default(),
            prop_intern: FxHashMap::default(),
            dependent_required_intern: FxHashMap::default(),
            bit_intern: FxHashMap::default(),
            ref_intern: FxHashMap::default(),
            options,
        }
    }

    /// Interns a UTF-8 string, returning a content-addressed reference.
    fn intern_str(&mut self, s: &str) -> Result<StrRef, CompileError> {
        if let Some(r) = self.str_intern.get(s) {
            return Ok(*r);
        }
        if self
            .strings
            .len()
            .checked_add(s.len())
            .is_none_or(|n| n > MAX_STRINGS_BYTES)
        {
            return Err(limit("strings blob cap"));
        }
        let off = u32::try_from(self.strings.len()).map_err(|_| limit("strings blob"))?;
        let len = u32::try_from(s.len()).map_err(|_| limit("string length"))?;
        self.strings.extend_from_slice(s.as_bytes());
        let r = StrRef { off, len };
        self.str_intern.insert(s.to_owned(), r);
        Ok(r)
    }

    fn intern_lits(&mut self, lits: Vec<ScalarLit>) -> Result<LitSlice, CompileError> {
        if let Some(r) = self.lit_intern.get(&lits) {
            return Ok(*r);
        }
        if self
            .literals
            .len()
            .checked_add(lits.len())
            .is_none_or(|n| n > MAX_LITERALS)
        {
            return Err(limit("literals cap"));
        }
        if self
            .literal_str_bytes
            .checked_add(literal_str_bytes(&lits))
            .is_none_or(|n| n > MAX_LITERAL_STR_BYTES)
        {
            return Err(limit("literal string bytes cap"));
        }
        self.literal_str_bytes += literal_str_bytes(&lits);
        let off = u32::try_from(self.literals.len()).map_err(|_| limit("literals blob"))?;
        let len = u32::try_from(lits.len()).map_err(|_| limit("literal run"))?;
        self.literals.extend_from_slice(&lits);
        let r = LitSlice { off, len };
        self.lit_intern.insert(lits, r);
        Ok(r)
    }

    fn intern_props(&mut self, pairs: Vec<(StrRef, NodeId)>) -> Result<PropSlice, CompileError> {
        if let Some(r) = self.prop_intern.get(&pairs) {
            return Ok(*r);
        }
        if self
            .props
            .len()
            .checked_add(pairs.len())
            .is_none_or(|n| n > MAX_PROPS)
        {
            return Err(limit("props cap"));
        }
        let off = u32::try_from(self.props.len()).map_err(|_| limit("props blob"))?;
        let len = u32::try_from(pairs.len()).map_err(|_| limit("prop run"))?;
        self.props.extend_from_slice(&pairs);
        let r = PropSlice { off, len };
        self.prop_intern.insert(pairs, r);
        Ok(r)
    }

    /// `groups` is `(trigger, required names)`: each trigger and each required name is owned
    /// exactly once by the caller. Interns the trigger once per group (never once per edge),
    /// reserves the flattened pair vector fallibly against the checked total edge count, and
    /// keeps the existing sorted/deduplicated/shared-slice arena representation.
    fn intern_dependent_required(
        &mut self,
        mut groups: Vec<(String, Vec<String>)>,
    ) -> Result<DependentRequiredSlice, CompileError> {
        let total_edges = groups
            .iter()
            .try_fold(0usize, |total, (_, required)| {
                total.checked_add(required.len())
            })
            .ok_or_else(|| limit("dependentRequired pair count"))?;
        if total_edges > MAX_DEPENDENT_REQUIRED_PAIRS {
            return Err(limit("dependentRequired pair count"));
        }
        groups.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        if groups.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(malformed("duplicate dependentRequired trigger"));
        }
        for (_, required) in &mut groups {
            required.sort_unstable();
            if required.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(malformed("duplicate dependentRequired pair"));
            }
        }
        let mut refs = Vec::new();
        refs.try_reserve_exact(total_edges)
            .map_err(|_| limit("dependentRequired pair arena"))?;
        for (trigger, required) in &groups {
            let trigger_ref = self.intern_str(trigger)?;
            for name in required {
                refs.push(DependentRequiredPair {
                    trigger: trigger_ref,
                    required: self.intern_str(name)?,
                });
            }
        }
        if let Some(r) = self.dependent_required_intern.get(&refs) {
            return Ok(*r);
        }
        if self
            .dependent_required_pairs
            .len()
            .checked_add(refs.len())
            .is_none_or(|n| n > MAX_DEPENDENT_REQUIRED_PAIRS)
        {
            return Err(limit("dependentRequired pair arena"));
        }
        let off = u32::try_from(self.dependent_required_pairs.len())
            .map_err(|_| limit("dependentRequired pair arena"))?;
        let len = u32::try_from(refs.len()).map_err(|_| limit("dependentRequired pair run"))?;
        self.dependent_required_pairs
            .try_reserve_exact(refs.len())
            .map_err(|_| limit("dependentRequired pair arena"))?;
        self.dependent_required_pairs.extend_from_slice(&refs);
        let slice = DependentRequiredSlice { off, len };
        self.dependent_required_intern.insert(refs, slice);
        Ok(slice)
    }

    fn intern_bits(&mut self, words: Vec<u64>, field_count: u32) -> Result<BitSlice, CompileError> {
        if let Some(r) = self.bit_intern.get(&words) {
            return Ok(BitSlice {
                off: r.off,
                len: field_count,
            });
        }
        if self
            .required_bits
            .len()
            .checked_add(words.len())
            .is_none_or(|n| n > MAX_REQUIRED_WORDS)
        {
            return Err(limit("required words cap"));
        }
        let off = u32::try_from(self.required_bits.len()).map_err(|_| limit("required blob"))?;
        self.required_bits.extend_from_slice(&words);
        let r = BitSlice {
            off,
            len: field_count,
        };
        self.bit_intern.insert(words, r);
        Ok(r)
    }

    fn intern_refs(&mut self, refs: Vec<NodeId>) -> Result<RefSlice, CompileError> {
        if let Some(r) = self.ref_intern.get(&refs) {
            return Ok(*r);
        }
        if self
            .node_refs
            .len()
            .checked_add(refs.len())
            .is_none_or(|n| n > MAX_NODE_REFS)
        {
            return Err(limit("node refs cap"));
        }
        let off = u32::try_from(self.node_refs.len()).map_err(|_| limit("node refs blob"))?;
        let len = u32::try_from(refs.len()).map_err(|_| limit("node ref run"))?;
        self.node_refs.extend_from_slice(&refs);
        let r = RefSlice { off, len };
        self.ref_intern.insert(refs, r);
        Ok(r)
    }

    fn intern_node(&mut self, node: Node) -> Result<NodeId, CompileError> {
        if let Some(id) = self.node_intern.get(&(self.current_resource, node.clone())) {
            return Ok(*id);
        }
        if self.nodes.len() >= MAX_NODES {
            return Err(limit("node cap"));
        }
        let id = NodeId::try_from(self.nodes.len()).map_err(|_| limit("node count"))?;
        self.nodes.push(node.clone());
        self.node_resources.push(self.current_resource);
        self.node_intern.insert((self.current_resource, node), id);
        Ok(id)
    }

    pub(crate) fn set_current_resource(&mut self, resource: ResourceId) -> ResourceId {
        std::mem::replace(&mut self.current_resource, resource)
    }

    /// Interns a `null` node.
    pub fn null(&mut self) -> Result<NodeId, CompileError> {
        self.intern_node(Node::Null)
    }

    /// Interns a `boolean` node.
    pub fn boolean(&mut self) -> Result<NodeId, CompileError> {
        self.intern_node(Node::Boolean)
    }

    /// Interns a node matching no value at all (the bare `false` schema).
    pub fn never(&mut self) -> Result<NodeId, CompileError> {
        self.intern_node(Node::Never)
    }

    /// Interns a string-const node.
    pub fn string_const(&mut self, value: &str) -> Result<NodeId, CompileError> {
        let value = self.intern_str(value)?;
        self.intern_node(Node::StringConst { value })
    }

    /// Interns a string-pattern node.
    pub fn string_pattern(
        &mut self,
        regex: &str,
        min_len: Option<u32>,
        max_len: Option<u32>,
        charset: Charset,
    ) -> Result<NodeId, CompileError> {
        let regex = self.intern_str(regex)?;
        self.intern_node(Node::StringPattern {
            regex,
            min_len,
            max_len,
            charset,
        })
    }

    /// Interns an integer node.
    pub fn integer(
        &mut self,
        minimum: Option<i64>,
        maximum: Option<i64>,
    ) -> Result<NodeId, CompileError> {
        self.integer_multiple_of(minimum, maximum, None)
    }

    /// Interns an integer node with an exact `multipleOf` divisor.
    pub fn integer_multiple_of(
        &mut self,
        minimum: Option<i64>,
        maximum: Option<i64>,
        multiple_of: Option<u64>,
    ) -> Result<NodeId, CompileError> {
        if multiple_of == Some(0) {
            return Err(malformed("multipleOf must be a positive integer"));
        }
        self.intern_node(Node::Integer {
            minimum,
            maximum,
            multiple_of,
        })
    }

    /// Interns an unbounded `number` node (the full JSON numeric grammar).
    pub fn number(&mut self) -> Result<NodeId, CompileError> {
        self.number_range(None, None, None)
    }

    /// Interns a `number` node with an integer-valued inclusive/exclusive bound and/or a decimal
    /// `multipleOf` (`coef * 10^-exp`).
    pub fn number_range(
        &mut self,
        minimum: Option<(i128, u32, bool)>,
        maximum: Option<(i128, u32, bool)>,
        multiple_of: Option<(u64, u32)>,
    ) -> Result<NodeId, CompileError> {
        self.intern_node(Node::Number {
            integer_only: false,
            minimum,
            maximum,
            multiple_of,
        })
    }

    /// Interns exact decimal constraints whose accepted values must be mathematically integral.
    pub fn integer_number_range(
        &mut self,
        minimum: Option<(i128, u32, bool)>,
        maximum: Option<(i128, u32, bool)>,
        multiple_of: Option<(u64, u32)>,
    ) -> Result<NodeId, CompileError> {
        self.intern_node(Node::Number {
            integer_only: true,
            minimum,
            maximum,
            multiple_of,
        })
    }

    /// Interns a lexical JSON-number constraint.
    pub fn lexical_number(&mut self, regex: &str) -> Result<NodeId, CompileError> {
        validate_pattern(regex)?;
        let regex = self.intern_str(regex)?;
        self.intern_node(Node::LexicalNumber { regex })
    }

    /// Interns an enum node. Values are canonically sorted; a duplicate value is rejected.
    pub fn enum_values(&mut self, mut values: Vec<ScalarLit>) -> Result<NodeId, CompileError> {
        if values.is_empty() {
            return Err(malformed("an enum must have at least one value"));
        }
        values.sort();
        if values.windows(2).any(|w| w[0] == w[1]) {
            return Err(malformed("duplicate enum value"));
        }
        let values = self.intern_lits(values)?;
        self.intern_node(Node::Enum { values })
    }

    /// Interns an array node.
    pub fn array(
        &mut self,
        items: NodeId,
        min_items: u32,
        max_items: Option<u32>,
    ) -> Result<NodeId, CompileError> {
        self.array_unique(items, min_items, max_items, false)
    }

    /// Interns an array node with an explicit `uniqueItems` flag.
    pub fn array_unique(
        &mut self,
        items: NodeId,
        min_items: u32,
        max_items: Option<u32>,
        unique_items: bool,
    ) -> Result<NodeId, CompileError> {
        self.array_full(
            ItemsPolicy::Schema(items),
            min_items,
            max_items,
            unique_items,
            None,
            true,
        )
    }

    /// Interns an array node with unconstrained (`items: true` or absent) elements.
    pub fn array_open(
        &mut self,
        min_items: u32,
        max_items: Option<u32>,
        unique_items: bool,
    ) -> Result<NodeId, CompileError> {
        self.array_full(
            ItemsPolicy::AllowAny,
            min_items,
            max_items,
            unique_items,
            None,
            true,
        )
    }

    /// Interns an array node with every keyword explicit, including `contains` and whether `items`
    /// was textually present (`items_annotates`, for `unevaluatedItems`).
    #[allow(clippy::too_many_arguments)]
    pub fn array_full(
        &mut self,
        items: ItemsPolicy,
        min_items: u32,
        max_items: Option<u32>,
        unique_items: bool,
        contains: Option<ContainsConstraint>,
        items_annotates: bool,
    ) -> Result<NodeId, CompileError> {
        self.intern_node(Node::Array {
            items,
            min_items,
            max_items,
            unique_items,
            contains,
            items_annotates,
        })
    }

    /// Interns a tuple array with positional `prefix` schemas and an optional `tail` schema.
    /// Closed tuples whose minimum exceeds the prefix length are rejected.
    pub fn tuple(
        &mut self,
        prefix: Vec<NodeId>,
        tail: Option<NodeId>,
        min_items: u32,
        max_items: Option<u32>,
    ) -> Result<NodeId, CompileError> {
        self.tuple_unique(prefix, tail, min_items, max_items, false)
    }

    /// Interns a positional (tuple) array node with an explicit `uniqueItems` flag.
    pub fn tuple_unique(
        &mut self,
        prefix: Vec<NodeId>,
        tail: Option<NodeId>,
        min_items: u32,
        max_items: Option<u32>,
        unique_items: bool,
    ) -> Result<NodeId, CompileError> {
        self.tuple_full(prefix, tail, min_items, max_items, unique_items, None, true)
    }

    /// Interns a positional (tuple) array node with every keyword explicit, including `contains` and
    /// whether the tail was textually present (`tail_annotates`, for `unevaluatedItems`).
    #[allow(clippy::too_many_arguments)]
    pub fn tuple_full(
        &mut self,
        prefix: Vec<NodeId>,
        tail: Option<NodeId>,
        min_items: u32,
        max_items: Option<u32>,
        unique_items: bool,
        contains: Option<ContainsConstraint>,
        tail_annotates: bool,
    ) -> Result<NodeId, CompileError> {
        if prefix.is_empty() {
            return Err(malformed("a tuple must have at least one prefix position"));
        }
        let prefix = self.intern_refs(prefix)?;
        self.intern_node(Node::Tuple {
            prefix,
            tail,
            min_items,
            max_items,
            unique_items,
            tail_annotates,
            contains,
        })
    }

    /// Interns a type-union node. `branches` are child nodes in declaration order; a single branch
    /// collapses to that branch directly rather than a one-element union.
    pub fn union_of(&mut self, branches: Vec<NodeId>) -> Result<NodeId, CompileError> {
        if branches.is_empty() {
            return Err(malformed("a type union must have at least one branch"));
        }
        if branches.len() == 1 {
            return Ok(branches[0]);
        }
        let branches = self.intern_refs(branches)?;
        self.intern_node(Node::Union { branches })
    }

    /// Reserves a def slot for a recursive definition and returns its index. The slot is filled by
    /// [`Builder::set_def_target`] once the body is lowered; a `Node::Ref` may point at it meanwhile.
    pub fn alloc_def_slot(&mut self) -> Result<u32, CompileError> {
        let slot = u32::try_from(self.def_targets.len()).map_err(|_| limit("def slot count"))?;
        if self.def_targets.len() >= MAX_NODES {
            return Err(limit("def slot cap"));
        }
        self.def_targets.push(None);
        Ok(slot)
    }

    /// Points a reserved def slot at its lowered body root. Fails on an out-of-range slot or one
    /// already assigned, rather than silently overwriting or indexing out of bounds.
    pub fn set_def_target(&mut self, slot: u32, body: NodeId) -> Result<(), CompileError> {
        let entry = self
            .def_targets
            .get_mut(slot as usize)
            .ok_or_else(|| malformed("def slot index out of range"))?;
        if entry.replace(body).is_some() {
            return Err(malformed("def slot already assigned"));
        }
        Ok(())
    }

    /// Interns a `Node::Ref` naming a reserved def slot (a recursion cut point).
    pub fn ref_node(&mut self, def: u32) -> Result<NodeId, CompileError> {
        self.intern_node(Node::Ref { def })
    }

    /// Interns an already-statically-resolved dynamic reference.
    pub fn dynamic_ref(
        &mut self,
        initial_target: NodeId,
        anchor: AnchorId,
    ) -> Result<NodeId, CompileError> {
        self.intern_node(Node::DynamicRef {
            initial_target,
            anchor,
        })
    }

    /// Interns an `unevaluatedProperties`/`unevaluatedItems` obligation over `scope`, with
    /// `unevaluated` applied to each member `scope` did not annotate.
    pub fn unevaluated(
        &mut self,
        kind: UnevaluatedKind,
        scope: NodeId,
        unevaluated: NodeId,
    ) -> Result<NodeId, CompileError> {
        self.intern_node(Node::Unevaluated {
            kind,
            scope,
            unevaluated,
        })
    }

    /// Interns an `allOf` intersection in declaration order.
    /// A single branch collapses to itself.
    pub fn intersection_of(&mut self, branches: Vec<NodeId>) -> Result<NodeId, CompileError> {
        if branches.is_empty() {
            return Err(malformed("an intersection must have at least one branch"));
        }
        if branches.len() == 1 {
            return Ok(branches[0]);
        }
        let branches = self.intern_refs(branches)?;
        self.intern_node(Node::Intersection { branches })
    }

    /// Interns a `oneOf` exactly-one node in declaration order.
    /// A single branch collapses to itself because exactly-one preserves its language.
    pub fn exactly_one_of(&mut self, branches: Vec<NodeId>) -> Result<NodeId, CompileError> {
        if branches.is_empty() {
            return Err(malformed("an exactly-one must have at least one branch"));
        }
        if branches.len() == 1 {
            return Ok(branches[0]);
        }
        let branches = self.intern_refs(branches)?;
        self.intern_node(Node::ExactlyOne { branches })
    }

    /// Interns a `not`: matches any complete JSON value `inner` does not.
    pub fn not(&mut self, inner: NodeId) -> Result<NodeId, CompileError> {
        self.intern_node(Node::Not { inner })
    }

    /// Interns `(name, child)` pairs plus their aligned required bitset; shared by `object` and
    /// `open_object`. A duplicate field name is rejected.
    fn intern_named_fields(
        &mut self,
        fields: Vec<(String, NodeId)>,
        required: &[bool],
    ) -> Result<(PropSlice, BitSlice), CompileError> {
        if required.len() != fields.len() {
            return Err(malformed("required flags do not align with fields"));
        }
        let mut names: FxHashMap<&str, ()> = FxHashMap::default();
        names.reserve(fields.len());
        for (name, _) in &fields {
            if names.insert(name.as_str(), ()).is_some() {
                return Err(malformed("duplicate object key"));
            }
        }
        let mut pairs = Vec::with_capacity(fields.len());
        for (name, child) in &fields {
            let name = self.intern_str(name)?;
            pairs.push((name, *child));
        }
        let field_count = u32::try_from(fields.len()).map_err(|_| limit("object field count"))?;
        let words = fields.len().div_ceil(64);
        let mut bits = vec![0u64; words];
        for (i, &req) in required.iter().enumerate() {
            if req {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        let fields = self.intern_props(pairs)?;
        let required = self.intern_bits(bits, field_count)?;
        Ok((fields, required))
    }

    /// Interns an object node. `fields` are `(name, child)` in source order; `required` is aligned
    /// to `fields` by position. A duplicate field name is rejected.
    pub fn object(
        &mut self,
        fields: Vec<(String, NodeId)>,
        required: &[bool],
        closure: ObjectClosure,
    ) -> Result<NodeId, CompileError> {
        self.object_full(fields, required, closure, Vec::new(), Vec::new())
    }

    /// Interns an object with `dependentSchemas`: `(key, schema)` pairs where `key`'s presence in
    /// the instance additionally requires the whole instance to satisfy `schema`.
    pub fn object_full(
        &mut self,
        fields: Vec<(String, NodeId)>,
        required: &[bool],
        closure: ObjectClosure,
        dependent: Vec<(String, NodeId)>,
        dependent_required: Vec<(String, Vec<String>)>,
    ) -> Result<NodeId, CompileError> {
        let (fields, required) = self.intern_named_fields(fields, required)?;
        let dependent = self.intern_dependent(dependent)?;
        let dependent_required = self.intern_dependent_required(dependent_required)?;
        self.intern_node(Node::Object {
            fields,
            required,
            closure,
            dependent,
            dependent_required,
        })
    }

    fn intern_dependent(
        &mut self,
        dependent: Vec<(String, NodeId)>,
    ) -> Result<PropSlice, CompileError> {
        let mut pairs = Vec::with_capacity(dependent.len());
        for (name, child) in dependent {
            let name_ref = self.intern_str(&name)?;
            pairs.push((name_ref, child));
        }
        self.intern_props(pairs)
    }

    /// Interns an order-independent open object. `known`/`known_required` behave like `object`'s
    /// `fields`/`required`; `patterns` are `(regex, child)` `patternProperties` entries.
    #[allow(clippy::too_many_arguments)]
    pub fn open_object(
        &mut self,
        known: Vec<(String, NodeId)>,
        known_required: &[bool],
        patterns: Vec<(String, NodeId)>,
        additional: AdditionalPolicy,
        property_names: Option<NodeId>,
        min_properties: Option<u32>,
        max_properties: Option<u32>,
    ) -> Result<NodeId, CompileError> {
        self.open_object_full(
            known,
            known_required,
            patterns,
            additional,
            property_names,
            min_properties,
            max_properties,
            Vec::new(),
            Vec::new(),
        )
    }

    /// Interns an order-independent open object with `dependentSchemas`, same semantics as
    /// `object_full`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn open_object_full(
        &mut self,
        known: Vec<(String, NodeId)>,
        known_required: &[bool],
        patterns: Vec<(String, NodeId)>,
        additional: AdditionalPolicy,
        property_names: Option<NodeId>,
        min_properties: Option<u32>,
        max_properties: Option<u32>,
        dependent: Vec<(String, NodeId)>,
        dependent_required: Vec<(String, Vec<String>)>,
    ) -> Result<NodeId, CompileError> {
        if let (Some(lo), Some(hi)) = (min_properties, max_properties) {
            if hi < lo {
                return Err(malformed("maxProperties is less than minProperties"));
            }
        }
        let (known, known_required) = self.intern_named_fields(known, known_required)?;
        let mut pattern_pairs = Vec::with_capacity(patterns.len());
        for (regex, child) in &patterns {
            let regex_ref = self.intern_str(regex)?;
            pattern_pairs.push((regex_ref, *child));
        }
        let patterns = self.intern_props(pattern_pairs)?;
        let dependent = self.intern_dependent(dependent)?;
        let dependent_required = self.intern_dependent_required(dependent_required)?;
        self.intern_node(Node::OpenObject {
            known,
            known_required,
            patterns,
            additional,
            property_names,
            min_properties,
            max_properties,
            dependent,
            dependent_required,
        })
    }

    /// Interns an `Unsupported` node and records a diagnostic for this occurrence. Returns the
    /// node id, or `TooManyDiagnostics` once the bounded buffer is full.
    pub fn unsupported(
        &mut self,
        keyword: &str,
        reason: UnsupportedReason,
        json_pointer: &str,
    ) -> Result<NodeId, CompileError> {
        if self.diagnostics.len() >= MAX_DIAGNOSTICS
            || self.diagnostics.len() as u64 >= u64::from(self.options.max_diagnostics)
        {
            return Err(CompileError::new(
                ErrorCode::TooManyDiagnostics,
                Stage::L2,
                "diagnostics buffer is full",
            ));
        }
        let keyword_ref = self.intern_str(keyword)?;
        let pointer_ref = self.intern_str(json_pointer)?;
        let node = self.intern_node(Node::Unsupported {
            keyword: keyword_ref,
            reason,
        })?;
        self.diagnostics.push(Diagnostic {
            node,
            keyword: keyword_ref,
            json_pointer: pointer_ref,
            source_span: None,
            reason,
        });
        Ok(node)
    }

    /// Freezes the arena into an immutable IR rooted at `root`, running full validation.
    pub fn finish(self, root: NodeId) -> Result<SchemaIR, CompileError> {
        self.finish_with_resources(
            root,
            None,
            vec![ResourceInput {
                canonical_uri: None,
                root,
                static_anchors: Vec::new(),
                dynamic_anchors: Vec::new(),
            }],
        )
    }

    pub(crate) fn finish_with_resources(
        mut self,
        root: NodeId,
        retrieval_uri: Option<String>,
        resource_inputs: Vec<ResourceInput>,
    ) -> Result<SchemaIR, CompileError> {
        let retrieval_uri = retrieval_uri
            .as_deref()
            .map(|uri| self.intern_str(uri))
            .transpose()?;
        let mut resources = Vec::new();
        resources
            .try_reserve_exact(resource_inputs.len())
            .map_err(|_| limit("resource allocation"))?;
        let mut anchors = Vec::new();
        for (index, input) in resource_inputs.into_iter().enumerate() {
            let resource = ResourceId(u32::try_from(index).map_err(|_| limit("resource index"))?);
            let canonical_uri = input
                .canonical_uri
                .as_deref()
                .map(|uri| self.intern_str(uri))
                .transpose()?;
            let static_off = u32::try_from(anchors.len()).map_err(|_| limit("anchor offset"))?;
            for (name, target) in input.static_anchors {
                let name = self.intern_str(&name)?;
                anchors.push(AnchorRecord {
                    resource,
                    name,
                    target,
                });
            }
            let static_len = u32::try_from(anchors.len())
                .map_err(|_| limit("anchor count"))?
                .checked_sub(static_off)
                .ok_or_else(|| limit("anchor run"))?;
            let dynamic_off = u32::try_from(anchors.len()).map_err(|_| limit("anchor offset"))?;
            for (name, target) in input.dynamic_anchors {
                let name = self.intern_str(&name)?;
                anchors.push(AnchorRecord {
                    resource,
                    name,
                    target,
                });
            }
            let dynamic_len = u32::try_from(anchors.len())
                .map_err(|_| limit("anchor count"))?
                .checked_sub(dynamic_off)
                .ok_or_else(|| limit("anchor run"))?;
            resources.push(ResourceRecord {
                canonical_uri,
                root: input.root,
                static_anchors: AnchorSlice {
                    off: static_off,
                    len: static_len,
                },
                dynamic_anchors: AnchorSlice {
                    off: dynamic_off,
                    len: dynamic_len,
                },
            });
        }
        let def_targets = self
            .def_targets
            .into_iter()
            .map(|t| t.ok_or_else(|| malformed("def slot never assigned a target")))
            .collect::<Result<Vec<NodeId>, CompileError>>()?;
        SchemaIR::assemble(SchemaIRWire {
            ir_version: IR_VERSION,
            root,
            nodes: self.nodes,
            strings: self.strings,
            literals: self.literals,
            props: self.props,
            dependent_required_pairs: self.dependent_required_pairs,
            required_bits: self.required_bits,
            node_refs: self.node_refs,
            def_targets,
            retrieval_uri,
            resources,
            anchors,
            node_resources: self.node_resources,
            options: self.options,
            diagnostics: self.diagnostics,
        })
    }
}

fn limit(what: &'static str) -> CompileError {
    CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L2, what)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> CompileOptions {
        CompileOptions::default()
    }

    #[test]
    fn identical_subschemas_share_one_node_id() {
        let mut b = Builder::new(opts());
        let a = b.boolean().unwrap();
        let c = b.boolean().unwrap();
        assert_eq!(a, c, "boolean is interned once");
        let obj = b
            .object(
                vec![("x".into(), a), ("y".into(), c)],
                &[false, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir = b.finish(obj).unwrap();
        // two occurrences of boolean, one distinct boolean node (plus the object).
        assert_eq!(ir.node_count(), 2);
    }

    #[test]
    fn finish_rejects_an_unfilled_def_slot() {
        let mut b = Builder::new(opts());
        let slot = b.alloc_def_slot().unwrap();
        let root = b.ref_node(slot).unwrap();
        assert!(b.finish(root).is_err());
    }

    #[test]
    fn set_def_target_rejects_a_second_assignment_to_the_same_slot() {
        let mut b = Builder::new(opts());
        let slot = b.alloc_def_slot().unwrap();
        let a = b.boolean().unwrap();
        let c = b.boolean().unwrap();
        b.set_def_target(slot, a).unwrap();
        assert!(b.set_def_target(slot, c).is_err());
    }

    #[test]
    fn set_def_target_rejects_an_out_of_range_slot() {
        let mut b = Builder::new(opts());
        let a = b.boolean().unwrap();
        assert!(b.set_def_target(0, a).is_err());
    }

    #[test]
    fn object_with_few_fields_but_heavy_bounded_strings_needs_structured_backend() {
        let mut b = Builder::new(opts());
        let a = b
            .string_pattern("(?:.)", None, Some(100), Charset::Utf8CountedCodepoints)
            .unwrap();
        let c = b
            .string_pattern("(?:.)", None, Some(2048), Charset::Utf8CountedCodepoints)
            .unwrap();
        let obj = b
            .object(
                vec![("a".into(), a), ("c".into(), c)],
                &[false, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir = b.finish(obj).unwrap();
        assert!(
            ir.requires_structured_backend(),
            "2 fields, well under MAX_SAFE_SHUFFLE_FIELDS, but weight 100+2048 exceeds the cap"
        );
    }

    #[test]
    fn three_fields_with_a_255_scalar_string_exceed_the_shuffle_edge_bound() {
        let mut b = Builder::new(opts());
        let label = b
            .string_pattern("(?:.)", Some(2), Some(255), Charset::Utf8CountedCodepoints)
            .unwrap();
        let boolean = b.boolean().unwrap();
        let obj = b
            .object(
                vec![
                    ("category".into(), label),
                    ("generate_entities".into(), boolean),
                    ("label".into(), label),
                ],
                &[false, false, true],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir = b.finish(obj).unwrap();
        assert!(ir.requires_structured_backend());
    }

    #[test]
    fn large_minimum_array_needs_structured_backend_even_without_a_maximum() {
        let mut b = Builder::new(opts());
        let item = b.null().unwrap();
        let array = b.array(item, MAX_UNROLLED_ARRAY_ITEMS + 1, None).unwrap();
        let ir = b.finish(array).unwrap();
        assert!(ir.requires_structured_backend());
    }

    #[test]
    fn object_with_small_bounded_strings_stays_on_the_fast_dfa_path() {
        let mut b = Builder::new(opts());
        let a = b
            .string_pattern("(?:.)", None, Some(50), Charset::Utf8CountedCodepoints)
            .unwrap();
        let c = b
            .string_pattern("(?:.)", None, Some(50), Charset::Utf8CountedCodepoints)
            .unwrap();
        let obj = b
            .object(
                vec![("a".into(), a), ("c".into(), c)],
                &[false, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir = b.finish(obj).unwrap();
        assert!(!ir.requires_structured_backend());
    }

    #[test]
    fn single_optional_heavy_bounded_string_routes_before_shuffle_construction() {
        let mut b = Builder::new(opts());
        let a = b
            .string_pattern("(?:.)", None, Some(4096), Charset::Utf8CountedCodepoints)
            .unwrap();
        let obj = b
            .object(vec![("a".into(), a)], &[false], ObjectClosure::Forbidden)
            .unwrap();
        let ir = b.finish(obj).unwrap();
        assert!(ir.requires_structured_backend());
    }

    #[test]
    fn heavy_bounded_string_wrapped_in_a_nullable_union_still_counts_its_weight() {
        let mut b = Builder::new(opts());
        let null = b.null().unwrap();
        let big = b
            .string_pattern("(?:.)", None, Some(2048), Charset::Utf8CountedCodepoints)
            .unwrap();
        let nullable_big = b.union_of(vec![null, big]).unwrap();
        let small = b
            .string_pattern("(?:.)", None, Some(100), Charset::Utf8CountedCodepoints)
            .unwrap();
        let obj = b
            .object(
                vec![("a".into(), nullable_big), ("b".into(), small)],
                &[false, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir = b.finish(obj).unwrap();
        assert!(
            ir.requires_structured_backend(),
            "type:[X,null] must not hide a field's real weight from the shuffle pre-check"
        );
    }

    #[test]
    fn distinct_node_count_never_exceeds_occurrence_count() {
        let mut b = Builder::new(opts());
        let inner = b.boolean().unwrap();
        let arr = b.array(inner, 0, None).unwrap();
        let ir = b.finish(arr).unwrap();
        assert!(ir.node_count() <= 2);
    }

    #[test]
    fn two_builders_different_insertion_order_hash_equal() {
        let mut b1 = Builder::new(opts());
        let x = b1.boolean().unwrap();
        let y = b1.null().unwrap();
        let o1 = b1
            .object(
                vec![("a".into(), x), ("b".into(), y)],
                &[true, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir1 = b1.finish(o1).unwrap();

        let mut b2 = Builder::new(opts());
        let y = b2.null().unwrap();
        let x = b2.boolean().unwrap();
        let o2 = b2
            .object(
                vec![("a".into(), x), ("b".into(), y)],
                &[true, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir2 = b2.finish(o2).unwrap();

        assert_eq!(ir1.canonical_hash(), ir2.canonical_hash());
    }

    #[test]
    fn different_property_order_changes_the_hash() {
        let mut b1 = Builder::new(opts());
        let (x, y) = (b1.boolean().unwrap(), b1.null().unwrap());
        let o1 = b1
            .object(
                vec![("a".into(), x), ("b".into(), y)],
                &[false, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir1 = b1.finish(o1).unwrap();

        let mut b2 = Builder::new(opts());
        let (x, y) = (b2.boolean().unwrap(), b2.null().unwrap());
        let o2 = b2
            .object(
                vec![("b".into(), y), ("a".into(), x)],
                &[false, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir2 = b2.finish(o2).unwrap();

        assert_ne!(ir1.canonical_hash(), ir2.canonical_hash());
    }

    #[test]
    fn enum_order_does_not_change_the_hash() {
        let mut b1 = Builder::new(opts());
        let e1 = b1
            .enum_values(vec![ScalarLit::Int(1), ScalarLit::Int(2)])
            .unwrap();
        let ir1 = b1.finish(e1).unwrap();
        let mut b2 = Builder::new(opts());
        let e2 = b2
            .enum_values(vec![ScalarLit::Int(2), ScalarLit::Int(1)])
            .unwrap();
        let ir2 = b2.finish(e2).unwrap();
        assert_eq!(ir1.canonical_hash(), ir2.canonical_hash());
    }

    #[test]
    fn required_flag_change_changes_the_hash() {
        let mk = |req: &[bool]| {
            let mut b = Builder::new(opts());
            let x = b.boolean().unwrap();
            let o = b
                .object(vec![("a".into(), x)], req, ObjectClosure::Forbidden)
                .unwrap();
            b.finish(o).unwrap().canonical_hash()
        };
        assert_ne!(mk(&[true]), mk(&[false]));
    }

    #[test]
    fn length_prefix_prevents_concatenation_collision() {
        let mut b1 = Builder::new(opts());
        let e1 = b1
            .enum_values(vec![
                ScalarLit::Str("ab".into()),
                ScalarLit::Str("c".into()),
            ])
            .unwrap();
        let ir1 = b1.finish(e1).unwrap();
        let mut b2 = Builder::new(opts());
        let e2 = b2
            .enum_values(vec![
                ScalarLit::Str("a".into()),
                ScalarLit::Str("bc".into()),
            ])
            .unwrap();
        let ir2 = b2.finish(e2).unwrap();
        assert_ne!(ir1.canonical_hash(), ir2.canonical_hash());
    }

    #[test]
    fn enum_rejects_duplicate_value() {
        let mut b = Builder::new(opts());
        let e = b.enum_values(vec![ScalarLit::Int(1), ScalarLit::Int(1)]);
        assert_eq!(e.unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn object_rejects_duplicate_key() {
        let mut b = Builder::new(opts());
        let x = b.boolean().unwrap();
        let y = b.null().unwrap();
        let e = b.object(
            vec![("a".into(), x), ("a".into(), y)],
            &[false, false],
            ObjectClosure::Forbidden,
        );
        assert_eq!(e.unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn array_max_below_min_is_rejected_at_finish() {
        let mut b = Builder::new(opts());
        let inner = b.boolean().unwrap();
        let arr = b.array(inner, 3, Some(2)).unwrap();
        assert_eq!(b.finish(arr).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn unsupported_records_a_diagnostic_per_occurrence() {
        let mut b = Builder::new(opts());
        let u1 = b
            .unsupported("anyOf", UnsupportedReason::UnionType, "/anyOf")
            .unwrap();
        let u2 = b
            .unsupported("anyOf", UnsupportedReason::UnionType, "/properties/x/anyOf")
            .unwrap();
        assert_eq!(u1, u2, "same keyword+reason interns to one node");
        let ir = b.finish(u1).unwrap();
        assert_eq!(
            ir.diagnostics().len(),
            2,
            "each occurrence keeps its own pointer"
        );
        let pointers: Vec<&str> = ir
            .diagnostics()
            .map(|d| ir.str_at(d.json_pointer).unwrap())
            .collect();
        assert_eq!(pointers, ["/anyOf", "/properties/x/anyOf"]);
    }

    #[test]
    fn diagnostics_are_bounded() {
        let mut options = opts();
        options.max_diagnostics = 2;
        let mut b = Builder::new(options);
        b.unsupported("not", UnsupportedReason::Complement, "/a")
            .unwrap();
        b.unsupported("not", UnsupportedReason::Complement, "/b")
            .unwrap();
        let e = b.unsupported("not", UnsupportedReason::Complement, "/c");
        assert_eq!(e.unwrap_err().code, ErrorCode::TooManyDiagnostics);
    }

    #[test]
    fn default_max_diagnostics_absorbs_a_real_world_scale_schema_without_hard_failing() {
        // The measured jsonschemabench worst case was ~4500 occurrences; the default must clear it.
        let mut b = Builder::new(CompileOptions::default());
        for i in 0..5000u32 {
            b.unsupported("not", UnsupportedReason::Complement, &format!("/{i}"))
                .unwrap();
        }
        let last = b
            .unsupported("not", UnsupportedReason::Complement, "/last")
            .unwrap();
        let ir = b.finish(last).unwrap();
        assert_eq!(ir.diagnostics().len(), 5001);
    }

    #[test]
    fn canonical_hash_ignores_diagnostic_pointer() {
        let mk = |ptr: &str| {
            let mut b = Builder::new(opts());
            let u = b
                .unsupported("not", UnsupportedReason::Complement, ptr)
                .unwrap();
            b.finish(u).unwrap().canonical_hash()
        };
        assert_eq!(mk("/a"), mk("/properties/deeply/nested/b"));
    }

    #[test]
    fn integer_bounds_reject_empty_range_at_finish() {
        let mut b = Builder::new(opts());
        let n = b.integer(Some(5), Some(1)).unwrap();
        assert_eq!(b.finish(n).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn object_closure_change_changes_the_hash() {
        let mk = |closure| {
            let mut b = Builder::new(opts());
            let x = b.boolean().unwrap();
            let o = b.object(vec![("a".into(), x)], &[false], closure).unwrap();
            b.finish(o).unwrap().canonical_hash()
        };
        assert_ne!(
            mk(ObjectClosure::Forbidden),
            mk(ObjectClosure::AssumeClosedProfile)
        );
    }

    #[test]
    fn compile_options_are_part_of_the_program4_hash() {
        let mk = |cap| {
            let mut o = opts();
            o.max_diagnostics = cap;
            let mut b = Builder::new(o);
            let n = b.boolean().unwrap();
            b.finish(n).unwrap().canonical_hash()
        };
        assert_ne!(mk(8), mk(64));
    }

    #[test]
    fn all_compile_options_change_the_program4_hash() {
        let mk = |f: &dyn Fn(&mut CompileOptions)| {
            let mut o = opts();
            f(&mut o);
            let mut b = Builder::new(o);
            let n = b.boolean().unwrap();
            b.finish(n).unwrap().canonical_hash()
        };
        let base = mk(&|_| {});
        assert_ne!(base, mk(&|o| o.max_diagnostics = 8));
        assert_ne!(base, mk(&|o| o.format_assertion = true));
    }

    #[test]
    fn same_content_address_holds_across_build_order() {
        let mut b1 = Builder::new(opts());
        let x = b1.boolean().unwrap();
        let y = b1.null().unwrap();
        let o1 = b1
            .object(
                vec![("a".into(), x), ("b".into(), y)],
                &[true, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir1 = b1.finish(o1).unwrap();

        let mut b2 = Builder::new(opts());
        let y = b2.null().unwrap();
        let x = b2.boolean().unwrap();
        let o2 = b2
            .object(
                vec![("a".into(), x), ("b".into(), y)],
                &[true, false],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir2 = b2.finish(o2).unwrap();

        assert!(ir1.same_content_address(&ir2));
    }

    #[test]
    fn object_field_counts_around_word_boundaries_build_and_read_back() {
        for n in [0usize, 1, 63, 64, 65, 130] {
            let mut b = Builder::new(opts());
            let mut fields = Vec::with_capacity(n);
            let mut required = Vec::with_capacity(n);
            for i in 0..n {
                let child = b.boolean().unwrap();
                fields.push((format!("f{i}"), child));
                required.push(i % 2 == 0);
            }
            let o = b
                .object(fields, &required, ObjectClosure::Forbidden)
                .unwrap();
            let ir = b.finish(o).unwrap();
            let Node::Object { required: bits, .. } = ir.node(o).unwrap() else {
                panic!("object");
            };
            for i in 0..n {
                let idx = u32::try_from(i).unwrap();
                assert_eq!(ir.is_required(*bits, idx), i % 2 == 0, "field {i} of {n}");
            }
        }
    }

    #[test]
    fn integer_and_enum_carry_i64_extremes() {
        let mut b = Builder::new(opts());
        let n = b.integer(Some(i64::MIN), Some(i64::MAX)).unwrap();
        let ir = b.finish(n).unwrap();
        assert_eq!(SchemaIR::from_wire(&ir.to_wire()).unwrap(), ir);

        let mut b = Builder::new(opts());
        let e = b
            .enum_values(vec![ScalarLit::Int(i64::MIN), ScalarLit::Int(i64::MAX)])
            .unwrap();
        let ir = b.finish(e).unwrap();
        assert_eq!(SchemaIR::from_wire(&ir.to_wire()).unwrap(), ir);
    }

    #[test]
    fn literal_string_bytes_at_the_cap_build_and_round_trip() {
        let mut b = Builder::new(opts());
        let big = "a".repeat(MAX_LITERAL_STR_BYTES);
        let e = b.enum_values(vec![ScalarLit::Str(big)]).unwrap();
        let ir = b.finish(e).unwrap();
        assert_eq!(SchemaIR::from_wire(&ir.to_wire()).unwrap(), ir);
    }

    #[test]
    fn a_single_over_cap_literal_string_is_rejected_at_build() {
        let mut b = Builder::new(opts());
        let too_big = "a".repeat(MAX_LITERAL_STR_BYTES + 1);
        let e = b.enum_values(vec![ScalarLit::Str(too_big)]);
        assert_eq!(e.unwrap_err().code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn repeated_large_literals_cannot_bypass_the_aggregate_cap() {
        let half = MAX_LITERAL_STR_BYTES / 2;
        let mut b = Builder::new(opts());
        // Two distinct enums, each just under the cap, together exceed it.
        b.enum_values(vec![ScalarLit::Str("a".repeat(half))])
            .unwrap();
        let second = b.enum_values(vec![ScalarLit::Str("b".repeat(half + 1))]);
        assert_eq!(second.unwrap_err().code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn tuple_node_round_trips_and_rejects_a_closed_min_past_the_prefix() {
        let mut b = Builder::new(opts());
        let x = b.boolean().unwrap();
        let y = b.null().unwrap();
        let node = b.tuple(vec![x, y], Some(x), 1, Some(4)).unwrap();
        let ir = b.finish(node).unwrap();
        assert_eq!(SchemaIR::from_wire(&ir.to_wire()).unwrap(), ir);

        // A closed tuple (no tail) cannot hold more items than its prefix, so minItems past the
        // prefix length is an empty language, rejected at finish.
        let mut b = Builder::new(opts());
        let x = b.boolean().unwrap();
        let node = b.tuple(vec![x], None, 3, None).unwrap();
        assert_eq!(b.finish(node).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn empty_tuple_prefix_is_rejected_at_build() {
        let mut b = Builder::new(opts());
        assert_eq!(
            b.tuple(vec![], None, 0, None).unwrap_err().code,
            ErrorCode::Malformed
        );
    }

    #[test]
    fn tuple_and_array_hash_differently_even_with_one_prefix_element() {
        let mk_tuple = || {
            let mut b = Builder::new(opts());
            let x = b.boolean().unwrap();
            let n = b.tuple(vec![x], Some(x), 0, None).unwrap();
            b.finish(n).unwrap().canonical_hash()
        };
        let mk_array = || {
            let mut b = Builder::new(opts());
            let x = b.boolean().unwrap();
            let n = b.array(x, 0, None).unwrap();
            b.finish(n).unwrap().canonical_hash()
        };
        assert_ne!(mk_tuple(), mk_array());
    }

    #[test]
    fn deeply_nested_and_repeated_subtrees_build_via_arena_iteration() {
        let mut b = Builder::new(opts());
        let mut node = b.boolean().unwrap();
        for _ in 0..5_000 {
            node = b.array(node, 0, None).unwrap();
        }
        let ir = b.finish(node).unwrap();
        assert_eq!(ir.node_count(), 5_001);
        let _ = ir.canonical_hash();
    }

    #[test]
    fn open_object_rejects_duplicate_known_keys() {
        let mut b = Builder::new(opts());
        let x = b.boolean().unwrap();
        let err = b
            .open_object(
                vec![("a".into(), x), ("a".into(), x)],
                &[false, false],
                Vec::new(),
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
            )
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::Malformed);
    }

    #[test]
    fn open_object_rejects_max_properties_below_min_properties() {
        let mut b = Builder::new(opts());
        let err = b
            .open_object(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                Some(5),
                Some(2),
            )
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::Malformed);
    }

    #[test]
    fn open_object_additional_policy_changes_the_hash() {
        let mk = |policy: AdditionalPolicy| {
            let mut b = Builder::new(opts());
            let n = b
                .open_object(Vec::new(), &[], Vec::new(), policy, None, None, None)
                .unwrap();
            b.finish(n).unwrap().canonical_hash()
        };
        let forbid = mk(AdditionalPolicy::Forbid);
        let allow_any = mk(AdditionalPolicy::AllowAny);
        assert_ne!(forbid, allow_any);

        let mut b = Builder::new(opts());
        let s = b.boolean().unwrap();
        let schema_hash = {
            let n = b
                .open_object(
                    Vec::new(),
                    &[],
                    Vec::new(),
                    AdditionalPolicy::Schema(s),
                    None,
                    None,
                    None,
                )
                .unwrap();
            b.finish(n).unwrap().canonical_hash()
        };
        assert_ne!(schema_hash, forbid);
        assert_ne!(schema_hash, allow_any);
    }

    #[test]
    fn open_object_and_closed_object_hash_differently_for_the_same_fields() {
        let mk_open = || {
            let mut b = Builder::new(opts());
            let x = b.boolean().unwrap();
            let n = b
                .open_object(
                    vec![("a".into(), x)],
                    &[true],
                    Vec::new(),
                    AdditionalPolicy::Forbid,
                    None,
                    None,
                    None,
                )
                .unwrap();
            b.finish(n).unwrap().canonical_hash()
        };
        let mk_closed = || {
            let mut b = Builder::new(opts());
            let x = b.boolean().unwrap();
            let n = b
                .object(vec![("a".into(), x)], &[true], ObjectClosure::Forbidden)
                .unwrap();
            b.finish(n).unwrap().canonical_hash()
        };
        assert_ne!(mk_open(), mk_closed());
    }

    #[test]
    fn open_object_pattern_and_property_names_round_trip_through_wire() {
        let mut b = Builder::new(opts());
        let known_val = b.boolean().unwrap();
        let pattern_val = b.integer(None, None).unwrap();
        let names_schema = b
            .string_pattern("[a-z]+", None, None, Charset::Utf8Any)
            .unwrap();
        let n = b
            .open_object(
                vec![("a".into(), known_val)],
                &[true],
                vec![("^x_".into(), pattern_val)],
                AdditionalPolicy::AllowAny,
                Some(names_schema),
                Some(1),
                Some(10),
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        let bytes = ir.to_wire();
        let decoded = SchemaIR::from_wire(&bytes).unwrap();
        assert_eq!(ir.canonical_hash(), decoded.canonical_hash());
        let Node::OpenObject {
            min_properties,
            max_properties,
            property_names,
            ..
        } = decoded.node(decoded.root()).unwrap()
        else {
            panic!("expected OpenObject");
        };
        assert_eq!(*min_properties, Some(1));
        assert_eq!(*max_properties, Some(10));
        assert!(property_names.is_some());
    }

    #[test]
    fn dependent_required_canonical_hash_is_unordered_and_wire_stable() {
        let options = CompileOptions::default();
        let first = crate::frontend::schema_to_ir(
            r#"{"dependentRequired":{"a":["b","c"],"x":["y"]}}"#,
            options.clone(),
        )
        .unwrap();
        let second = crate::frontend::schema_to_ir(
            r#"{"dependentRequired":{"x":["y"],"a":["c","b"]}}"#,
            options,
        )
        .unwrap();
        assert_eq!(IR_VERSION, 6);
        assert_eq!(first.canonical_hash(), second.canonical_hash());
        let encoded = first.to_wire();
        let decoded = SchemaIR::from_wire(&encoded).unwrap();
        assert_eq!(decoded, first);
        assert_eq!(decoded.canonical_hash(), first.canonical_hash());
        let mut v2 = encoded;
        v2[6..8].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            SchemaIR::from_wire(&v2).unwrap_err().code,
            ErrorCode::Malformed
        );
    }

    #[test]
    fn decimal_comparison_is_exact_at_coefficient_and_scale_limits() {
        use std::cmp::Ordering;

        assert_eq!(decimal_cmp((1, u32::MAX), (2, u32::MAX)), Ordering::Less);
        assert_eq!(
            decimal_cmp((-1, u32::MAX), (-2, u32::MAX)),
            Ordering::Greater
        );
        assert_eq!(decimal_cmp((10, 2), (1, 1)), Ordering::Equal);
        assert_eq!(decimal_cmp((i128::MAX, 38), (1, 0)), Ordering::Greater);
        assert_eq!(decimal_cmp((i128::MIN, 38), (-1, 0)), Ordering::Less);
    }

    #[test]
    fn forged_dependent_required_arena_is_rejected_without_panicking() {
        let valid = crate::frontend::schema_to_ir(
            r#"{"type":"object","additionalProperties":true,"dependentRequired":{"a":["b"]}}"#,
            CompileOptions::default(),
        )
        .unwrap();
        let root = valid.root().get() as usize;
        let mut bad_slice = valid.to_wire_struct();
        match &mut bad_slice.nodes[root] {
            Node::OpenObject {
                dependent_required, ..
            }
            | Node::Object {
                dependent_required, ..
            } => dependent_required.off = u32::MAX,
            _ => panic!("expected object branch"),
        }
        assert_eq!(
            SchemaIR::assemble(bad_slice).unwrap_err().code,
            ErrorCode::Malformed
        );

        for (off, len) in [(0, u32::MAX), (u32::MAX, u32::MAX)] {
            let mut bad_range = valid.to_wire_struct();
            match &mut bad_range.nodes[root] {
                Node::OpenObject {
                    dependent_required, ..
                }
                | Node::Object {
                    dependent_required, ..
                } => {
                    dependent_required.off = off;
                    dependent_required.len = len;
                }
                _ => unreachable!(),
            }
            assert_eq!(
                SchemaIR::assemble(bad_range).unwrap_err().code,
                ErrorCode::Malformed
            );
        }

        for trigger in [true, false] {
            let mut bad_name = valid.to_wire_struct();
            let bad = StrRef {
                off: u32::MAX,
                len: 1,
            };
            if trigger {
                bad_name.dependent_required_pairs[0].trigger = bad;
            } else {
                bad_name.dependent_required_pairs[0].required = bad;
            }
            assert_eq!(
                SchemaIR::assemble(bad_name).unwrap_err().code,
                ErrorCode::Malformed
            );
        }

        let mut duplicate = valid.to_wire_struct();
        duplicate
            .dependent_required_pairs
            .push(duplicate.dependent_required_pairs[0]);
        match &mut duplicate.nodes[root] {
            Node::OpenObject {
                dependent_required, ..
            }
            | Node::Object {
                dependent_required, ..
            } => dependent_required.len = 2,
            _ => unreachable!(),
        }
        assert_eq!(
            SchemaIR::assemble(duplicate).unwrap_err().code,
            ErrorCode::Malformed
        );

        let mut over_cap = valid.to_wire_struct();
        over_cap.dependent_required_pairs =
            vec![over_cap.dependent_required_pairs[0]; MAX_DEPENDENT_REQUIRED_PAIRS + 1];
        assert_eq!(
            SchemaIR::assemble(over_cap).unwrap_err().code,
            ErrorCode::Malformed
        );
    }
}
