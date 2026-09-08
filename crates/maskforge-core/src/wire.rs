//! MFIR wire envelope: a framed, checksummed, version-tagged IR blob for FFI.
//! Decoding fails closed through framing, allocation, and semantic validation checks.

use bincode::{Decode, Encode};

use crate::diagnostics::Diagnostic;
use crate::error::{CompileError, ErrorCode, Stage};
use crate::ir::{
    literal_str_bytes, AnchorRecord, CompileOptions, DependentRequiredPair, Node, ResourceId,
    ResourceRecord, ScalarLit, SchemaIR, StrRef, IR_VERSION, MAX_ANCHORS,
    MAX_DEPENDENT_REQUIRED_PAIRS, MAX_DIAGNOSTICS, MAX_LITERALS, MAX_LITERAL_STR_BYTES, MAX_NODES,
    MAX_NODE_REFS, MAX_PROPS, MAX_REQUIRED_WORDS, MAX_RESOURCES, MAX_STRINGS_BYTES,
};
use crate::primitives::NodeId;

const MAGIC: [u8; 4] = *b"MFIR";
const FORMAT_VERSION: u16 = 1;
/// Bytes before the payload: magic(4) + format_version(2) + ir_version(2) + payload_len(4).
const HEADER_LEN: usize = 12;
const CHECKSUM_LEN: usize = 32;
/// Minimum envelope size (empty payload): header + checksum.
const MIN_ENVELOPE: usize = HEADER_LEN + CHECKSUM_LEN;

/// The decoder's allocation bound. Set above the largest arena the builder caps admit, so any
/// value the encoder can produce still decodes, while a forged length prefix is refused.
const MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;

/// The wire payload: exactly the semantic fields, never the memoized hash. `cached_hash` is
/// excluded structurally (it is not a field here), not by a serde attribute.
#[derive(Encode, Decode)]
pub(crate) struct SchemaIRWire {
    pub(crate) ir_version: u16,
    pub(crate) root: NodeId,
    pub(crate) nodes: Vec<Node>,
    pub(crate) strings: Vec<u8>,
    pub(crate) literals: Vec<ScalarLit>,
    pub(crate) props: Vec<(StrRef, NodeId)>,
    pub(crate) dependent_required_pairs: Vec<DependentRequiredPair>,
    pub(crate) required_bits: Vec<u64>,
    pub(crate) node_refs: Vec<NodeId>,
    pub(crate) def_targets: Vec<NodeId>,
    pub(crate) retrieval_uri: Option<StrRef>,
    pub(crate) resources: Vec<ResourceRecord>,
    pub(crate) anchors: Vec<AnchorRecord>,
    pub(crate) node_resources: Vec<ResourceId>,
    pub(crate) options: CompileOptions,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

/// The base byte format shared by both directions: little-endian, fixed-width ints.
fn encode_config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
}

/// The decode config: the same byte format plus a hard allocation limit for untrusted input.
fn decode_config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
        .with_limit::<MAX_ENCODED_BYTES>()
}

/// Encodes an IR into a framed, checksummed envelope.
pub(crate) fn encode(ir: &SchemaIR) -> Vec<u8> {
    frame(ir.ir_version(), &encode_payload(&ir.to_wire_struct()))
}

fn encode_payload(wire: &SchemaIRWire) -> Vec<u8> {
    // INVARIANT: encoding a builder-capped IR into a growable Vec cannot fail; the arena is bounded
    // well below the byte limit, and there is no io or length constraint on an in-memory writer.
    bincode::encode_to_vec(wire, encode_config()).expect("encoding an in-memory IR cannot fail")
}

/// Wraps an already-encoded payload in the magic/version/length/checksum frame.
fn frame(ir_version: u16, payload: &[u8]) -> Vec<u8> {
    let payload_len =
        u32::try_from(payload.len()).expect("payload length fits u32 (bounded by construction)");
    let mut out = Vec::with_capacity(MIN_ENVELOPE + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&ir_version.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(payload);

    let mut hasher = blake3::Hasher::new();
    hasher.update(&out[..HEADER_LEN]);
    hasher.update(payload);
    out.extend_from_slice(hasher.finalize().as_bytes());
    out
}

/// Decodes and validates an envelope into a frozen IR. Never panics on adversarial input.
pub(crate) fn decode(bytes: &[u8]) -> Result<SchemaIR, CompileError> {
    if bytes.len() < MIN_ENVELOPE {
        return Err(err("envelope shorter than the minimum header"));
    }
    if bytes[..4] != MAGIC {
        return Err(err("bad envelope magic"));
    }
    let format_version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if format_version != FORMAT_VERSION {
        return Err(err("unsupported envelope format version"));
    }
    let ir_version = u16::from_le_bytes([bytes[6], bytes[7]]);
    if ir_version != IR_VERSION {
        return Err(err("ir_version mismatch"));
    }
    let payload_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if payload_len > MAX_ENCODED_BYTES {
        return Err(err("declared payload exceeds the size cap"));
    }
    let total = HEADER_LEN
        .checked_add(payload_len)
        .and_then(|n| n.checked_add(CHECKSUM_LEN))
        .ok_or_else(|| err("payload length overflows"))?;
    if total != bytes.len() {
        return Err(err("envelope length does not match the declared payload"));
    }
    let payload = &bytes[HEADER_LEN..HEADER_LEN + payload_len];
    let checksum = &bytes[HEADER_LEN + payload_len..];

    let mut hasher = blake3::Hasher::new();
    hasher.update(&bytes[..HEADER_LEN]);
    hasher.update(payload);
    if !constant_time_eq(checksum, hasher.finalize().as_bytes()) {
        return Err(CompileError::new(
            ErrorCode::ArtifactMismatch,
            Stage::L2,
            "envelope checksum mismatch",
        ));
    }

    let (wire, read): (SchemaIRWire, usize) =
        bincode::decode_from_slice(payload, decode_config()).map_err(map_decode_err)?;
    if read != payload_len {
        return Err(err("payload has trailing bytes after decode"));
    }

    check_caps(&wire)?;
    SchemaIR::assemble(wire)
}

fn check_caps(wire: &SchemaIRWire) -> Result<(), CompileError> {
    let over = |kind: &'static str| Err(limit(kind));
    if wire.ir_version != IR_VERSION {
        return Err(err("decoded ir_version mismatch"));
    }
    if wire.nodes.len() > MAX_NODES {
        return over("node count");
    }
    if wire.strings.len() > MAX_STRINGS_BYTES {
        return over("strings size");
    }
    if wire.literals.len() > MAX_LITERALS {
        return over("literal count");
    }
    if wire.props.len() > MAX_PROPS {
        return over("prop count");
    }
    if wire.dependent_required_pairs.len() > MAX_DEPENDENT_REQUIRED_PAIRS {
        return Err(err("dependentRequired pair count exceeds cap"));
    }
    if wire.required_bits.len() > MAX_REQUIRED_WORDS {
        return over("required words");
    }
    if wire.node_refs.len() > MAX_NODE_REFS {
        return over("node ref count");
    }
    if wire.def_targets.len() > MAX_NODES {
        return over("def target count");
    }
    if wire.resources.len() > MAX_RESOURCES {
        return over("resource count");
    }
    if wire.anchors.len() > MAX_ANCHORS {
        return over("anchor count");
    }
    if wire.node_resources.len() > MAX_NODES {
        return over("node resource count");
    }
    if wire.diagnostics.len() > MAX_DIAGNOSTICS {
        return over("diagnostic count");
    }
    if literal_str_bytes(&wire.literals) > MAX_LITERAL_STR_BYTES {
        return over("literal string bytes");
    }
    Ok(())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc = 0u8;
    for (x, y) in a.iter().zip(b) {
        acc |= x ^ y;
    }
    acc == 0
}

/// A hostile length prefix that would over-allocate trips the byte limit; other decode failures
/// are malformed bytes.
fn map_decode_err(e: bincode::error::DecodeError) -> CompileError {
    match e {
        bincode::error::DecodeError::LimitExceeded => limit("decode allocation"),
        _ => err("payload failed to decode"),
    }
}

fn err(message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L2, message)
}

fn limit(what: &'static str) -> CompileError {
    let mut e = CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L2,
        "decoded arena exceeds a cap",
    );
    e.observed = Some(what.to_string());
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Builder, CompileOptions, ItemsPolicy, ScalarLit};

    fn sample() -> SchemaIR {
        let mut b = Builder::new(CompileOptions::default());
        let item = b
            .enum_values(vec![ScalarLit::Int(1), ScalarLit::Int(12)])
            .unwrap();
        let arr = b.array(item, 0, Some(3)).unwrap();
        b.finish(arr).unwrap()
    }

    fn minimal_wire(root: NodeId, nodes: Vec<Node>) -> SchemaIRWire {
        let node_resources = vec![ResourceId(0); nodes.len()];
        SchemaIRWire {
            ir_version: IR_VERSION,
            root,
            nodes,
            strings: Vec::new(),
            literals: Vec::new(),
            props: Vec::new(),
            dependent_required_pairs: Vec::new(),
            required_bits: Vec::new(),
            node_refs: Vec::new(),
            def_targets: Vec::new(),
            retrieval_uri: None,
            resources: vec![ResourceRecord {
                canonical_uri: None,
                root,
                static_anchors: crate::ir::AnchorSlice::default(),
                dynamic_anchors: crate::ir::AnchorSlice::default(),
            }],
            anchors: Vec::new(),
            node_resources,
            options: CompileOptions::default(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn roundtrip_preserves_value_and_hash() {
        let ir = sample();
        let bytes = ir.to_wire();
        let back = SchemaIR::from_wire(&bytes).expect("decode");
        assert_eq!(ir, back);
        assert_eq!(ir.canonical_hash(), back.canonical_hash());
    }

    #[test]
    fn intersection_and_exactly_one_round_trip() {
        let mut b = Builder::new(CompileOptions::default());
        let a = b.integer(Some(0), Some(10)).unwrap();
        let c = b.integer(Some(5), Some(20)).unwrap();
        let inter = b.intersection_of(vec![a, c]).unwrap();
        let ir = b.finish(inter).unwrap();
        let back = SchemaIR::from_wire(&ir.to_wire()).expect("decode");
        assert_eq!(ir, back);
        assert_eq!(ir.canonical_hash(), back.canonical_hash());

        let mut b = Builder::new(CompileOptions::default());
        let a = b.integer(Some(0), Some(10)).unwrap();
        let c = b.integer(Some(5), Some(20)).unwrap();
        let one = b.exactly_one_of(vec![a, c]).unwrap();
        let ir = b.finish(one).unwrap();
        let back = SchemaIR::from_wire(&ir.to_wire()).expect("decode");
        assert_eq!(ir, back);
        assert_eq!(ir.canonical_hash(), back.canonical_hash());
    }

    #[test]
    fn number_and_composite_literal_round_trip() {
        let mut b = Builder::new(CompileOptions::default());
        let n = b.number().unwrap();
        let ir = b.finish(n).unwrap();
        let back = SchemaIR::from_wire(&ir.to_wire()).expect("decode");
        assert_eq!(ir, back);
        assert_eq!(ir.canonical_hash(), back.canonical_hash());

        let mut b = Builder::new(CompileOptions::default());
        let e = b
            .enum_values(vec![
                ScalarLit::Json(r#"{"x":1}"#.to_string()),
                ScalarLit::Json("[2,3]".to_string()),
            ])
            .unwrap();
        let ir = b.finish(e).unwrap();
        let back = SchemaIR::from_wire(&ir.to_wire()).expect("decode");
        assert_eq!(ir, back);
        assert_eq!(ir.canonical_hash(), back.canonical_hash());
    }

    #[test]
    fn payload_is_identical_whether_or_not_the_hash_was_computed() {
        let a = sample();
        let _ = a.canonical_hash(); // populate the memo
        let b = sample(); // memo empty
        assert_eq!(a.to_wire(), b.to_wire());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample().to_wire();
        bytes[0] = b'X';
        assert_eq!(decode(&bytes).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn rejects_bad_format_version() {
        let mut bytes = sample().to_wire();
        bytes[4] = 0xFF;
        assert_eq!(decode(&bytes).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn rejects_ir_version_mismatch_both_directions() {
        for v in [0u16, IR_VERSION + 1] {
            let mut bytes = sample().to_wire();
            bytes[6..8].copy_from_slice(&v.to_le_bytes());
            // corrupting the header also breaks the checksum; either way it must fail closed.
            assert!(decode(&bytes).is_err());
        }
    }

    #[test]
    fn rejects_truncated_header_and_trailing_bytes() {
        let bytes = sample().to_wire();
        assert_eq!(decode(&bytes[..8]).unwrap_err().code, ErrorCode::Malformed);
        let mut extra = bytes.clone();
        extra.push(0);
        assert_eq!(decode(&extra).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn rejects_bad_checksum() {
        let mut bytes = sample().to_wire();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert_eq!(
            decode(&bytes).unwrap_err().code,
            ErrorCode::ArtifactMismatch
        );
    }

    #[test]
    fn rejects_forged_root_out_of_range() {
        let wire = minimal_wire(NodeId(9), vec![Node::Boolean]);
        let bytes = frame(IR_VERSION, &encode_payload(&wire));
        assert_eq!(decode(&bytes).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn rejects_forged_back_edge_child() {
        // Array item points forward (or to itself): not a strict back-edge -> a forged cycle.
        let wire = minimal_wire(
            NodeId(0),
            vec![Node::Array {
                items: ItemsPolicy::Schema(NodeId(0)),
                min_items: 0,
                max_items: None,
                unique_items: false,
                contains: None,
                items_annotates: true,
            }],
        );
        let bytes = frame(IR_VERSION, &encode_payload(&wire));
        assert_eq!(decode(&bytes).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn rejects_out_of_range_string_reference() {
        let mut wire = minimal_wire(
            NodeId(0),
            vec![Node::StringConst {
                value: StrRef { off: 0, len: 5 },
            }],
        );
        wire.strings = vec![b'a'];
        let bytes = frame(IR_VERSION, &encode_payload(&wire));
        assert_eq!(decode(&bytes).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn hostile_vector_length_trips_the_byte_limit_before_allocating() {
        // A small payload whose `nodes` length prefix declares a billion elements. The bincode
        // byte limit must reject it as LimitExceeded, not attempt the allocation.
        let mut payload = encode_payload(&sample().to_wire_struct());
        // Layout (fixed-int, little-endian): ir_version(2) + root(4) + nodes_len(8) + ...
        payload[6..14].copy_from_slice(&1_000_000_000u64.to_le_bytes());
        let bytes = frame(IR_VERSION, &payload);
        assert_eq!(
            decode(&bytes).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
    }

    #[test]
    fn every_fieldless_enum_wire_tag_is_stable() {
        use crate::diagnostics::UnsupportedReason as R;
        use crate::ir::{Charset, ObjectClosure, UnsupportedPolicy};
        fn tag<T: Encode>(v: T) -> u32 {
            let b = bincode::encode_to_vec(v, encode_config()).unwrap();
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }
        assert_eq!(tag(Charset::Utf8Any), 0);
        assert_eq!(tag(Charset::AsciiPrintableNoQuoteBackslash), 1);
        assert_eq!(tag(Charset::Utf8CountedCodepoints), 2);
        assert_eq!(tag(ObjectClosure::RejectOpenObjects), 0);
        assert_eq!(tag(ObjectClosure::AssumeClosedProfile), 1);
        assert_eq!(tag(ObjectClosure::Forbidden), 2);
        assert_eq!(tag(ObjectClosure::AllowOpenProfile), 3);
        assert_eq!(tag(UnsupportedPolicy::RejectAtCompile), 0);
        for (i, r) in [
            R::UnsupportedKeyword,
            R::UnionType,
            R::OpenAdditionalProperties,
            R::AdditionalPropertiesSchema,
            R::RecursiveReference,
            R::NonRegularRef,
            R::InvalidPattern,
            R::NumericRangeUnsupported,
            R::SetDistinctness,
            R::ExactlyOne,
            R::Complement,
            R::ConditionalApplicator,
            R::DeferredObligation,
            R::AnnotationDependent,
            R::DynamicScope,
            R::NonScalarLiteral,
            R::LengthWithPattern,
            R::FormatUnsupported,
            R::FormatWithPattern,
            R::NonIntegerNumericLiteral,
            R::AllOfIntersectionUnsupported,
            R::NumberBoundUnsupported,
            R::OpenArrayTail,
            R::StructuredBackendRequired,
            R::RegexAssertionUnsupported,
            R::RegexBackreferenceUnsupported,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(tag(r), u32::try_from(i).unwrap(), "reason {r:?}");
        }
    }

    #[test]
    fn unknown_enum_wire_tag_is_rejected() {
        let payload = 99u32.to_le_bytes();
        let decoded: Result<(crate::diagnostics::UnsupportedReason, usize), _> =
            bincode::decode_from_slice(&payload, decode_config());
        assert!(decoded.is_err());
    }

    #[test]
    fn from_wire_recomputes_the_same_hash_from_an_empty_memo() {
        let ir = sample();
        let expected = ir.canonical_hash();
        let back = SchemaIR::from_wire(&ir.to_wire()).unwrap();
        assert_eq!(back.canonical_hash(), expected);
    }

    #[test]
    fn decode_rejects_an_over_cap_literal_string_payload() {
        // A forged arena the builder would never produce: one enum literal past the byte cap.
        let big = "a".repeat(MAX_LITERAL_STR_BYTES + 1);
        let mut wire = minimal_wire(
            NodeId(0),
            vec![Node::Enum {
                values: crate::ir::LitSlice { off: 0, len: 1 },
            }],
        );
        wire.literals = vec![ScalarLit::Str(big)];
        let bytes = frame(IR_VERSION, &encode_payload(&wire));
        assert_eq!(
            decode(&bytes).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
    }

    #[test]
    fn an_over_cap_collection_is_rejected() {
        use crate::diagnostics::{Diagnostic, UnsupportedReason};
        let dummy = Diagnostic {
            node: NodeId(0),
            keyword: StrRef { off: 0, len: 0 },
            json_pointer: StrRef { off: 0, len: 0 },
            source_span: None,
            reason: UnsupportedReason::Complement,
        };
        let mut wire = minimal_wire(NodeId(0), vec![Node::Boolean]);
        wire.diagnostics = vec![dummy; MAX_DIAGNOSTICS + 1];
        assert_eq!(
            check_caps(&wire).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
    }

    #[test]
    fn rejects_non_zero_required_padding_bits() {
        // One field, but the required word has a bit set past the field count.
        let mut wire = minimal_wire(
            NodeId(1),
            vec![
                Node::Boolean,
                Node::Object {
                    fields: crate::ir::PropSlice { off: 0, len: 1 },
                    required: crate::ir::BitSlice { off: 0, len: 1 },
                    closure: crate::ir::ObjectClosure::Forbidden,
                    dependent: crate::ir::PropSlice { off: 0, len: 0 },
                    dependent_required: crate::ir::DependentRequiredSlice { off: 0, len: 0 },
                },
            ],
        );
        wire.strings = vec![b'a'];
        wire.props = vec![(StrRef { off: 0, len: 1 }, NodeId(0))];
        wire.required_bits = vec![0b10]; // bit 1 set, but only field 0 exists
        let bytes = frame(IR_VERSION, &encode_payload(&wire));
        assert_eq!(decode(&bytes).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn rejects_diagnostic_pointing_at_a_non_unsupported_node() {
        use crate::diagnostics::{Diagnostic, UnsupportedReason};
        let mut wire = minimal_wire(NodeId(0), vec![Node::Boolean]);
        wire.strings = b"not/x".to_vec();
        wire.diagnostics = vec![Diagnostic {
            node: NodeId(0), // a Boolean, not an Unsupported
            keyword: StrRef { off: 0, len: 3 },
            json_pointer: StrRef { off: 3, len: 2 },
            source_span: None,
            reason: UnsupportedReason::Complement,
        }];
        let bytes = frame(IR_VERSION, &encode_payload(&wire));
        assert_eq!(decode(&bytes).unwrap_err().code, ErrorCode::Malformed);
    }

    fn assert_forged_malformed(wire: SchemaIRWire) {
        let bytes = frame(IR_VERSION, &encode_payload(&wire));
        assert_eq!(decode(&bytes).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn rejects_invalid_resource_and_node_ownership_metadata() {
        let mut bad_owner = minimal_wire(NodeId(0), vec![Node::Boolean]);
        bad_owner.node_resources[0] = ResourceId(1);
        assert_forged_malformed(bad_owner);

        let mut bad_root = minimal_wire(NodeId(0), vec![Node::Boolean, Node::Boolean]);
        bad_root.resources[0].root = NodeId(1);
        bad_root.node_resources[1] = ResourceId(1);
        assert_forged_malformed(bad_root);

        let mut missing_owner = minimal_wire(NodeId(0), vec![Node::Boolean]);
        missing_owner.node_resources.clear();
        assert_forged_malformed(missing_owner);
    }

    #[test]
    fn rejects_duplicate_and_noncanonical_resource_uris() {
        let mut duplicate = minimal_wire(NodeId(0), vec![Node::Boolean, Node::Boolean]);
        duplicate.strings = b"https://e.test/x".to_vec();
        let uri = StrRef {
            off: 0,
            len: u32::try_from(duplicate.strings.len()).unwrap(),
        };
        duplicate.resources = vec![
            ResourceRecord {
                canonical_uri: Some(uri),
                root: NodeId(0),
                static_anchors: crate::ir::AnchorSlice::default(),
                dynamic_anchors: crate::ir::AnchorSlice::default(),
            },
            ResourceRecord {
                canonical_uri: Some(uri),
                root: NodeId(1),
                static_anchors: crate::ir::AnchorSlice::default(),
                dynamic_anchors: crate::ir::AnchorSlice::default(),
            },
        ];
        duplicate.node_resources = vec![ResourceId(0), ResourceId(1)];
        assert_forged_malformed(duplicate);

        let mut invalid_run = minimal_wire(NodeId(0), vec![Node::Boolean]);
        invalid_run.resources[0].canonical_uri = Some(StrRef { off: 9, len: 1 });
        assert_forged_malformed(invalid_run);
    }

    #[test]
    fn rejects_invalid_anchor_runs_duplicates_and_dynamic_metadata() {
        use crate::ir::{AnchorId, AnchorSlice};

        let mut duplicate = minimal_wire(NodeId(1), vec![Node::Boolean, Node::Boolean]);
        duplicate.strings = b"same".to_vec();
        let name = StrRef { off: 0, len: 4 };
        duplicate.anchors = vec![
            AnchorRecord {
                resource: ResourceId(0),
                name,
                target: NodeId(0),
            },
            AnchorRecord {
                resource: ResourceId(0),
                name,
                target: NodeId(0),
            },
        ];
        duplicate.resources[0].static_anchors = AnchorSlice { off: 0, len: 1 };
        duplicate.resources[0].dynamic_anchors = AnchorSlice { off: 1, len: 1 };
        assert_forged_malformed(duplicate);

        let mut bad_run = minimal_wire(NodeId(0), vec![Node::Boolean]);
        bad_run.resources[0].dynamic_anchors = AnchorSlice {
            off: u32::MAX,
            len: 1,
        };
        assert_forged_malformed(bad_run);

        let mut bad_anchor_id = minimal_wire(
            NodeId(1),
            vec![
                Node::Boolean,
                Node::DynamicRef {
                    initial_target: NodeId(0),
                    anchor: AnchorId(1),
                },
            ],
        );
        bad_anchor_id.resources[0].root = NodeId(1);
        assert_forged_malformed(bad_anchor_id);

        let mut static_anchor = minimal_wire(
            NodeId(1),
            vec![
                Node::Boolean,
                Node::DynamicRef {
                    initial_target: NodeId(0),
                    anchor: AnchorId(0),
                },
            ],
        );
        static_anchor.strings = b"a".to_vec();
        static_anchor.anchors = vec![AnchorRecord {
            resource: ResourceId(0),
            name: StrRef { off: 0, len: 1 },
            target: NodeId(0),
        }];
        static_anchor.resources[0].root = NodeId(1);
        static_anchor.resources[0].static_anchors = AnchorSlice { off: 0, len: 1 };
        static_anchor.resources[0].dynamic_anchors = AnchorSlice { off: 1, len: 0 };
        assert_forged_malformed(static_anchor);

        let mut inconsistent = minimal_wire(
            NodeId(2),
            vec![
                Node::Boolean,
                Node::Boolean,
                Node::DynamicRef {
                    initial_target: NodeId(1),
                    anchor: AnchorId(0),
                },
            ],
        );
        inconsistent.strings = b"a".to_vec();
        inconsistent.anchors = vec![AnchorRecord {
            resource: ResourceId(0),
            name: StrRef { off: 0, len: 1 },
            target: NodeId(0),
        }];
        inconsistent.resources[0].root = NodeId(2);
        inconsistent.resources[0].dynamic_anchors = AnchorSlice { off: 0, len: 1 };
        assert_forged_malformed(inconsistent);
    }

    #[test]
    fn resource_and_anchor_wire_caps_are_exact() {
        let base = ResourceRecord {
            canonical_uri: None,
            root: NodeId(0),
            static_anchors: crate::ir::AnchorSlice::default(),
            dynamic_anchors: crate::ir::AnchorSlice::default(),
        };
        let mut exact_resources = minimal_wire(NodeId(0), vec![Node::Boolean]);
        exact_resources.resources = vec![base.clone(); MAX_RESOURCES];
        assert!(check_caps(&exact_resources).is_ok());
        exact_resources.resources.push(base);
        assert_eq!(
            check_caps(&exact_resources).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );

        let anchor = AnchorRecord {
            resource: ResourceId(0),
            name: StrRef::default(),
            target: NodeId(0),
        };
        let mut exact_anchors = minimal_wire(NodeId(0), vec![Node::Boolean]);
        exact_anchors.anchors = vec![anchor; MAX_ANCHORS];
        assert!(check_caps(&exact_anchors).is_ok());
        exact_anchors.anchors.push(AnchorRecord {
            resource: ResourceId(0),
            name: StrRef::default(),
            target: NodeId(0),
        });
        assert_eq!(
            check_caps(&exact_anchors).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
    }

    #[test]
    fn multiple_resources_static_dynamic_and_productive_cycles_round_trip() {
        let limits = crate::frontend::SchemaResourceLimits::default();
        let mut registry = crate::frontend::SchemaRegistry::new(limits);
        registry
            .insert(
                "https://e.test/other",
                r##"{"$dynamicAnchor":"node","type":"object","properties":{"next":{"$ref":"https://e.test/root"}}}"##,
            )
            .expect("resource");
        let ir = crate::frontend::schema_to_ir_with_resources(
            r##"{"$dynamicAnchor":"node","type":"object","properties":{"child":{"$dynamicRef":"https://e.test/other#node"}}}"##,
            Some("https://e.test/root"),
            CompileOptions::default(),
            limits,
            registry,
        )
        .expect("resource graph IR");
        assert_eq!(ir.resources().len(), 2);
        assert_eq!(ir.anchors().len(), 2);
        assert!(ir
            .nodes()
            .any(|node| matches!(node, Node::DynamicRef { .. })));
        let decoded = SchemaIR::from_wire(&ir.to_wire()).expect("wire round trip");
        assert_eq!(decoded, ir);
        assert_eq!(decoded.canonical_hash(), ir.canonical_hash());
    }
}
