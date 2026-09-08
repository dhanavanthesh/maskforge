//! Lowers JSON Schema syntax into validated schema IR.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::ast::{parse, Ast};
use super::resources::{
    DocumentId, ReferenceKind, ResourceGraph, SchemaLocationId, SchemaRegistry,
    SchemaResourceLimits,
};
use crate::diagnostics::UnsupportedReason;
use crate::error::{CompileError, ErrorCode, Stage};
use crate::ir::{
    AdditionalPolicy, AnchorId, Builder, Charset, CompileOptions, ContainsConstraint,
    ContainsPolicy, ObjectClosure, ResourceId, ResourceInput, ScalarLit, SchemaIR, UnevaluatedKind,
};
use crate::primitives::NodeId;

pub(crate) const JSON_STRING_BODY: &str = concat!(
    r#"(?:[\x20-\x21\x23-\x5b\x5d-\x7f]|"#,
    r"[\xc2-\xdf][\x80-\xbf]",
    r"|\xe0[\xa0-\xbf][\x80-\xbf]|[\xe1-\xec][\x80-\xbf][\x80-\xbf]",
    r"|\xed[\x80-\x9f][\x80-\xbf]|[\xee-\xef][\x80-\xbf][\x80-\xbf]",
    r"|\xf0[\x90-\xbf][\x80-\xbf][\x80-\xbf]|[\xf1-\xf3][\x80-\xbf][\x80-\xbf][\x80-\xbf]",
    r"|\xf4[\x80-\x8f][\x80-\xbf][\x80-\xbf]",
    r#"|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"#
);
pub(crate) const JSON_STRING_CODEPOINT: &str = concat!(
    r#"(?:[\x20\x21\x23-\x5b\x5d-\x7f]|\\["\\/bfnrt]"#,
    r"|\\u(?:[0-9a-ceA-CE][0-9a-fA-F]{3}|[fF][0-9a-fA-F]{3}|[dD][0-7][0-9a-fA-F]{2})",
    r"|\\u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}",
    r"|[\xc2-\xdf][\x80-\xbf]",
    r"|\xe0[\xa0-\xbf][\x80-\xbf]|[\xe1-\xec][\x80-\xbf][\x80-\xbf]",
    r"|\xed[\x80-\x9f][\x80-\xbf]|[\xee-\xef][\x80-\xbf][\x80-\xbf]",
    r"|\xf0[\x90-\xbf][\x80-\xbf][\x80-\xbf]|[\xf1-\xf3][\x80-\xbf][\x80-\xbf][\x80-\xbf]",
    r"|\xf4[\x80-\x8f][\x80-\xbf][\x80-\xbf])"
);

const DATE_RE: &str = concat!(
    r"(?:(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)-02-29)",
    r"|(?:[0-9]{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])",
    r"|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)",
    r"|02-(?:0[1-9]|1[0-9]|2[0-8])))"
);
const TIME_RE: &str = r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])";
const DATE_TIME_RE: &str = concat!(
    r"(?:(?:(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00)-02-29)",
    r"|(?:[0-9]{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])",
    r"|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)",
    r"|02-(?:0[1-9]|1[0-9]|2[0-8]))))T",
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
);
const UUID_RE: &str =
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}";
const IPV4_RE: &str = concat!(
    r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])",
    r"(?:\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}"
);

const MAX_DEPTH: u32 = 64;
const MAX_EXPANDED_NODES: u32 = 100_000;

const VALID_TYPE_NAMES: &[&str] = &[
    "null", "boolean", "integer", "number", "string", "array", "object",
];

const UNSUPPORTED_KEYWORDS: &[(&str, UnsupportedReason)] = &[];

const OBJECT_ONLY_KEYWORDS: &[&str] = &[
    "properties",
    "patternProperties",
    "additionalProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
    "required",
    "dependentSchemas",
    "dependentRequired",
    "dependencies",
];

pub fn schema_to_ir(text: &str, options: CompileOptions) -> Result<SchemaIR, CompileError> {
    schema_to_ir_timed(text, options).map(|(ir, ..)| ir)
}

pub fn schema_to_ir_timed(
    text: &str,
    options: CompileOptions,
) -> Result<(SchemaIR, u64, u64), CompileError> {
    let t0 = std::time::Instant::now();
    let ast = parse(text)?;
    let parse_ns = t0.elapsed().as_nanos() as u64;
    let t1 = std::time::Instant::now();
    let ir = if preflight_unsupported(&ast)?.is_some() {
        ir_from_ast(&ast, options)?
    } else if needs_resource_graph(&ast)? {
        let limits = SchemaResourceLimits::default();
        schema_set_to_ir(text, None, options, limits, SchemaRegistry::new(limits))?
    } else {
        ir_from_ast(&ast, options)?
    };
    let lower_ns = t1.elapsed().as_nanos() as u64;
    Ok((ir, parse_ns, lower_ns))
}

pub(crate) fn ir_from_ast(ast: &Ast, options: CompileOptions) -> Result<SchemaIR, CompileError> {
    if let Some((ptr, keyword, reason)) = preflight_unsupported(ast)? {
        let mut builder = Builder::new(options);
        let root = builder.unsupported(keyword, reason, &ptr)?;
        return builder.finish(root);
    }
    let mut index = DocIndex::default();
    index_document(ast, "", &mut index)?;
    let DocIndex {
        anchors, pointers, ..
    } = index;
    let object_closure = options.object_closure;
    let format_assertion = options.format_assertion;
    let mut lowerer = Lowerer {
        builder: Builder::new(options),
        root: ast,
        anchors,
        pointers,
        def_slots: FxHashMap::default(),
        lowered_refs: FxHashMap::default(),
        recursion_memo: FxHashMap::default(),
        ref_adjacency: FxHashMap::default(),
        open_under_unevaluated_properties: false,
        open_under_unevaluated_items: false,
        budget: MAX_EXPANDED_NODES,
        object_closure,
        format_assertion,
        validation_enabled: true,
        dialect_quirks_root: declared_dialect(ast)
            .map(dialect_quirks)
            .unwrap_or_default(),
        dialect_quirks_by_resource: FxHashMap::default(),
        current_resource: None,
        graph: None,
        current_document: DocumentId(0),
        location_nodes: FxHashMap::default(),
        location_slots: FxHashMap::default(),
        pending_locations: Vec::new(),
        assigned_slots: FxHashSet::default(),
        dynamic_anchor_ids: FxHashMap::default(),
    };
    let root = lowerer.lower(ast, "", 0)?;
    lowerer.builder.finish(root)
}

pub(crate) fn schema_set_to_ir(
    root_text: &str,
    retrieval_uri: Option<&str>,
    options: CompileOptions,
    limits: SchemaResourceLimits,
    mut registry: SchemaRegistry,
) -> Result<SchemaIR, CompileError> {
    match retrieval_uri {
        Some(uri) => {
            registry.insert(uri, Arc::<str>::from(root_text))?;
        }
        None => {
            registry.insert_anonymous(Arc::<str>::from(root_text))?;
        }
    }
    let graph = registry.discover()?;
    lower_resource_graph(&graph, root_text, retrieval_uri, options, limits)
}

#[cfg(feature = "bench-internals")]
pub(crate) fn schema_set_to_ir_profiled(
    root_text: &str,
    retrieval_uri: Option<&str>,
    options: CompileOptions,
    limits: SchemaResourceLimits,
    mut registry: SchemaRegistry,
) -> Result<(SchemaIR, super::SchemaSetProfile), CompileError> {
    let started = std::time::Instant::now();
    match retrieval_uri {
        Some(uri) => registry.insert(uri, Arc::<str>::from(root_text))?,
        None => registry.insert_anonymous(Arc::<str>::from(root_text))?,
    };
    let registry_insert_ns = elapsed_ns(started);
    let (graph, resource) = registry.discover_profiled()?;
    let retained_graph_bytes = graph.retained_bytes;
    let peak_resource_build_bytes = graph.peak_build_bytes;
    let started = std::time::Instant::now();
    let ir = lower_resource_graph(&graph, root_text, retrieval_uri, options, limits)?;
    let ir_lowering_and_construction_validation_ns = elapsed_ns(started);
    let started = std::time::Instant::now();
    ir.validate_for_bench()?;
    let ir_revalidation_ns = elapsed_ns(started);
    Ok((
        ir,
        super::SchemaSetProfile {
            registry_insert_ns,
            document_parse_ns: resource.document_parse_ns,
            resource_uri_anchor_index_ns: resource.resource_uri_anchor_index_ns,
            resource_canonicalize_ns: resource.canonicalize_ns,
            reference_resolution_ns: resource.reference_resolution_ns,
            graph_freeze_ns: resource.graph_freeze_ns,
            ir_lowering_and_construction_validation_ns,
            ir_revalidation_ns,
            retained_graph_bytes,
            peak_resource_build_bytes,
        },
    ))
}

#[cfg(feature = "bench-internals")]
fn elapsed_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn lower_resource_graph(
    graph: &ResourceGraph,
    root_text: &str,
    retrieval_uri: Option<&str>,
    options: CompileOptions,
    limits: SchemaResourceLimits,
) -> Result<SchemaIR, CompileError> {
    if graph.retained_bytes > limits.max_retained_graph_bytes {
        return Err(limit("retained schema-resource graph"));
    }
    if graph.peak_build_bytes > limits.max_peak_build_bytes {
        return Err(limit("peak schema-resource graph construction"));
    }
    let root_location = graph.root_location(retrieval_uri, root_text)?;
    let root_record = graph
        .locations
        .get(usize::try_from(root_location.get()).map_err(|_| limit("root location"))?)
        .ok_or_else(|| limit("root location"))?;
    let root_document = root_record.document;
    let root_ast = graph.location_ast(root_location)?;
    let validation_enabled = validation_vocabulary_enabled(graph, root_ast)?;
    let dialect_quirks_root = declared_dialect(root_ast)
        .map(dialect_quirks)
        .unwrap_or_default();
    let dialect_quirks_by_resource = resource_dialect_quirks(graph, dialect_quirks_root)?;
    let object_closure = options.object_closure;
    let format_assertion = options.format_assertion;
    let dynamic_anchor_ids = canonical_anchor_ids(graph)?;
    let mut lowerer = Lowerer {
        builder: Builder::new(options),
        root: root_ast,
        anchors: FxHashMap::default(),
        pointers: FxHashMap::default(),
        def_slots: FxHashMap::default(),
        lowered_refs: FxHashMap::default(),
        recursion_memo: FxHashMap::default(),
        ref_adjacency: FxHashMap::default(),
        open_under_unevaluated_properties: false,
        open_under_unevaluated_items: false,
        budget: MAX_EXPANDED_NODES,
        object_closure,
        format_assertion,
        validation_enabled,
        dialect_quirks_root,
        dialect_quirks_by_resource,
        current_resource: None,
        graph: Some(graph),
        current_document: root_document,
        location_nodes: FxHashMap::default(),
        location_slots: FxHashMap::default(),
        pending_locations: Vec::new(),
        assigned_slots: FxHashSet::default(),
        dynamic_anchor_ids,
    };
    let root = lowerer.lower_location(root_location)?;
    let resource_roots: Vec<(SchemaLocationId, DocumentId)> = graph
        .resources
        .iter()
        .map(|resource| (resource.root, resource.document))
        .collect();
    for (location, document) in resource_roots {
        let record = graph
            .locations
            .get(usize::try_from(location.get()).map_err(|_| limit("resource root"))?)
            .ok_or_else(|| limit("resource root"))?;
        if record.document != document {
            return Err(limit("resource root document"));
        }
        lowerer.lower_location(location)?;
    }
    lowerer.finish_graph(root, retrieval_uri.map(str::to_owned))
}

fn resource_dialect_quirks(
    graph: &ResourceGraph,
    root_quirks: DialectQuirks,
) -> Result<FxHashMap<ResourceId, DialectQuirks>, CompileError> {
    let mut map = FxHashMap::default();
    map.try_reserve(graph.resources.len())
        .map_err(|_| limit("resource dialect mapping"))?;
    for (index, resource) in graph.resources.iter().enumerate() {
        let id = ResourceId(u32::try_from(index).map_err(|_| limit("resource ID"))?);
        let quirks = declared_dialect(graph.location_ast(resource.root)?)
            .map_or(root_quirks, dialect_quirks);
        map.insert(id, quirks);
    }
    Ok(map)
}

fn canonical_anchor_ids(
    graph: &ResourceGraph,
) -> Result<FxHashMap<AnchorId, AnchorId>, CompileError> {
    let mut mapping = FxHashMap::default();
    mapping
        .try_reserve(graph.anchors.len())
        .map_err(|_| limit("anchor mapping"))?;
    let mut next = 0u32;
    for resource in &graph.resources {
        for run in [&resource.static_anchors, &resource.dynamic_anchors] {
            let mut ids = run.to_vec();
            ids.sort_unstable_by(|left, right| {
                let left_name = graph
                    .anchors
                    .get(left.get() as usize)
                    .map(|anchor| anchor.name.as_ref())
                    .unwrap_or("");
                let right_name = graph
                    .anchors
                    .get(right.get() as usize)
                    .map(|anchor| anchor.name.as_ref())
                    .unwrap_or("");
                left_name.cmp(right_name)
            });
            for id in ids {
                mapping.insert(id, AnchorId(next));
                next = next.checked_add(1).ok_or_else(|| limit("anchor ID"))?;
            }
        }
    }
    Ok(mapping)
}

const VALIDATION_VOCABULARY: &str = "https://json-schema.org/draft/2020-12/vocab/validation";

fn validation_vocabulary_enabled(graph: &ResourceGraph, root: &Ast) -> Result<bool, CompileError> {
    let Ast::Obj(fields) = root else {
        return Ok(true);
    };
    let Some(Ast::Str(dialect)) = get(fields, "$schema") else {
        return Ok(true);
    };
    if is_known_metaschema(dialect) {
        return Ok(true);
    }
    let resource = graph
        .uri_index
        .get(dialect.trim_end_matches('#'))
        .and_then(|id| graph.resources.get(id.get() as usize))
        .ok_or_else(|| {
            CompileError::new(
                ErrorCode::Unsupported,
                Stage::L1,
                "custom metaschema is not registered",
            )
            .with_pointer("/$schema")
            .with_keyword("$schema")
            .with_observed(dialect.clone())
        })?;
    let Ast::Obj(meta) = graph.location_ast(resource.root)? else {
        return Err(malformed(
            "/$schema",
            "$schema",
            "metaschema must be an object",
        ));
    };
    let Some(Ast::Obj(vocabularies)) = get(meta, "$vocabulary") else {
        return Err(CompileError::new(
            ErrorCode::Unsupported,
            Stage::L1,
            "custom metaschema does not declare $vocabulary",
        )
        .with_pointer("/$schema")
        .with_keyword("$schema"));
    };
    let mut validation = false;
    for (uri, required) in vocabularies {
        let Ast::Bool(required) = required else {
            return Err(malformed(
                "/$vocabulary",
                "$vocabulary",
                "vocabulary requirement must be boolean",
            ));
        };
        if uri == VALIDATION_VOCABULARY {
            validation = true;
        } else if *required && !is_known_vocabulary(uri) {
            return Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L1,
                "required vocabulary is not implemented",
            )
            .with_pointer("/$schema")
            .with_keyword("$schema")
            .with_observed(uri.clone()));
        }
    }
    Ok(validation)
}

fn is_known_vocabulary(uri: &str) -> bool {
    matches!(
        uri,
        "https://json-schema.org/draft/2020-12/vocab/core"
            | "https://json-schema.org/draft/2020-12/vocab/applicator"
            | "https://json-schema.org/draft/2020-12/vocab/unevaluated"
            | VALIDATION_VOCABULARY
            | "https://json-schema.org/draft/2020-12/vocab/meta-data"
            | "https://json-schema.org/draft/2020-12/vocab/format-annotation"
            | "https://json-schema.org/draft/2020-12/vocab/content"
    )
}

type DecimalBound = (i128, u32, bool);

enum ContainsLowering {
    Absent,
    Constraint(ContainsConstraint),
    Impossible,
}

type UnsupportedOccurrence = (String, &'static str, UnsupportedReason);

fn unsupported_pattern_reason(pattern: &str) -> UnsupportedReason {
    use crate::frontend::regex::PatternAnalysis;

    match crate::frontend::regex::analyze_pattern(pattern) {
        PatternAnalysis::Unsupported("numeric backreference" | "named backreference") => {
            UnsupportedReason::RegexBackreferenceUnsupported
        }
        PatternAnalysis::Unsupported("look-behind" | "non-leading or nested look-ahead") => {
            UnsupportedReason::RegexAssertionUnsupported
        }
        _ => UnsupportedReason::InvalidPattern,
    }
}

fn preflight_unsupported(ast: &Ast) -> Result<Option<UnsupportedOccurrence>, CompileError> {
    let mut work = Vec::new();
    work.try_reserve(1).map_err(|_| limit("schema preflight"))?;
    work.push((ast, String::new()));
    while let Some((node, pointer)) = work.pop() {
        match node {
            Ast::Obj(fields) => {
                if let Some(Ast::Str(uri)) = get(fields, "$schema") {
                    if !is_known_metaschema(uri) {
                        return Ok(Some((
                            child_ptr(&pointer, "$schema"),
                            "$schema",
                            UnsupportedReason::UnsupportedKeyword,
                        )));
                    }
                }
                work.try_reserve(fields.len())
                    .map_err(|_| limit("schema preflight"))?;
                for (key, child) in fields.iter().rev() {
                    work.push((child, child_ptr(&pointer, key)));
                }
            }
            Ast::Arr(items) => {
                work.try_reserve(items.len())
                    .map_err(|_| limit("schema preflight"))?;
                for (index, child) in items.iter().enumerate().rev() {
                    work.push((child, child_ptr(&pointer, &index.to_string())));
                }
            }
            Ast::Null | Ast::Bool(_) | Ast::Num(_) | Ast::Str(_) => {}
        }
    }
    Ok(None)
}

fn needs_resource_graph(ast: &Ast) -> Result<bool, CompileError> {
    let mut work = Vec::new();
    work.try_reserve(1).map_err(|_| limit("resource routing"))?;
    work.push(ast);
    while let Some(node) = work.pop() {
        match node {
            Ast::Obj(fields) => {
                for (keyword, value) in fields {
                    if matches!(
                        keyword.as_str(),
                        "$id" | "$anchor" | "$dynamicRef" | "$dynamicAnchor"
                    ) || (keyword == "$ref"
                        && matches!(value, Ast::Str(target) if !target.starts_with('#')))
                    {
                        return Ok(true);
                    }
                }
                work.try_reserve(fields.len())
                    .map_err(|_| limit("resource routing"))?;
                work.extend(fields.iter().rev().map(|(_, child)| child));
            }
            Ast::Arr(items) => {
                work.try_reserve(items.len())
                    .map_err(|_| limit("resource routing"))?;
                work.extend(items.iter().rev());
            }
            Ast::Null | Ast::Bool(_) | Ast::Num(_) | Ast::Str(_) => {}
        }
    }
    Ok(false)
}

#[derive(Clone, Copy, Debug, Default)]
struct DialectQuirks {
    ref_exclusive: bool,
    if_then_else_unsupported: bool,
}

fn dialect_quirks(dialect: &str) -> DialectQuirks {
    let dialect = dialect.trim_end_matches('#');
    let path = dialect
        .strip_prefix("https://")
        .or_else(|| dialect.strip_prefix("http://"))
        .unwrap_or(dialect);
    match path {
        "json-schema.org/draft-07/schema" => DialectQuirks {
            ref_exclusive: true,
            if_then_else_unsupported: false,
        },
        "json-schema.org/draft-03/schema"
        | "json-schema.org/draft-04/schema"
        | "json-schema.org/draft-06/schema" => DialectQuirks {
            ref_exclusive: true,
            if_then_else_unsupported: true,
        },
        _ => DialectQuirks::default(),
    }
}

fn declared_dialect(ast: &Ast) -> Option<&str> {
    let Ast::Obj(fields) = ast else {
        return None;
    };
    match get(fields, "$schema") {
        Some(Ast::Str(dialect)) => Some(dialect.as_str()),
        _ => None,
    }
}

fn is_known_metaschema(uri: &str) -> bool {
    matches!(
        uri.trim_end_matches('#'),
        "http://json-schema.org/draft-04/schema"
            | "http://json-schema.org/draft-06/schema"
            | "http://json-schema.org/draft-07/schema"
            | "https://json-schema.org/draft/2019-09/schema"
            | "https://json-schema.org/draft/2020-12/schema"
    )
}

struct Lowerer<'a> {
    builder: Builder,
    root: &'a Ast,
    anchors: FxHashMap<String, &'a Ast>,
    pointers: FxHashMap<String, &'a Ast>,
    def_slots: FxHashMap<String, u32>,
    lowered_refs: FxHashMap<(String, bool, bool), NodeId>,
    recursion_memo: FxHashMap<String, bool>,
    ref_adjacency: FxHashMap<String, Vec<String>>,
    open_under_unevaluated_properties: bool,
    open_under_unevaluated_items: bool,
    budget: u32,
    object_closure: ObjectClosure,
    format_assertion: bool,
    validation_enabled: bool,
    dialect_quirks_root: DialectQuirks,
    dialect_quirks_by_resource: FxHashMap<ResourceId, DialectQuirks>,
    current_resource: Option<ResourceId>,
    graph: Option<&'a ResourceGraph>,
    current_document: DocumentId,
    location_nodes: FxHashMap<SchemaLocationId, NodeId>,
    location_slots: FxHashMap<SchemaLocationId, u32>,
    pending_locations: Vec<SchemaLocationId>,
    assigned_slots: FxHashSet<SchemaLocationId>,
    dynamic_anchor_ids: FxHashMap<AnchorId, AnchorId>,
}

impl<'a> Lowerer<'a> {
    fn lower(&mut self, node: &'a Ast, ptr: &str, depth: u32) -> Result<NodeId, CompileError> {
        let location = self
            .graph
            .and_then(|graph| graph.document_location(self.current_document, ptr));
        let previous_resource = if let (Some(graph), Some(location)) = (self.graph, location) {
            let record = graph
                .locations
                .get(usize::try_from(location.get()).map_err(|_| limit("schema location"))?)
                .ok_or_else(|| limit("schema location"))?;
            let previous_builder_resource = self.builder.set_current_resource(record.resource);
            let previous_lowerer_resource =
                std::mem::replace(&mut self.current_resource, Some(record.resource));
            Some((previous_builder_resource, previous_lowerer_resource))
        } else {
            None
        };
        let result = self.lower_inner(node, ptr, depth);
        if let Some((previous_builder_resource, previous_lowerer_resource)) = previous_resource {
            self.builder.set_current_resource(previous_builder_resource);
            self.current_resource = previous_lowerer_resource;
        }
        result
    }

    fn dialect_quirks_here(&self) -> DialectQuirks {
        match self.current_resource {
            Some(resource) => self
                .dialect_quirks_by_resource
                .get(&resource)
                .copied()
                .unwrap_or(self.dialect_quirks_root),
            None => self.dialect_quirks_root,
        }
    }

    fn lower_inner(
        &mut self,
        node: &'a Ast,
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        if depth > MAX_DEPTH {
            return Err(limit("schema nesting depth"));
        }
        self.budget = self
            .budget
            .checked_sub(1)
            .ok_or_else(|| expanded_node_limit(ptr))?;
        let fields = match node {
            Ast::Obj(fields) => fields,
            Ast::Bool(true) => return self.lower_typeless(&[], ptr, depth),
            Ast::Bool(false) => return self.builder.never(),
            _ => return Err(malformed(ptr, "type", "a schema must be an object")),
        };
        let has_up = obj_has(fields, "unevaluatedProperties");
        let has_ui = obj_has(fields, "unevaluatedItems");
        if has_up || has_ui {
            return self.lower_unevaluated(fields, ptr, depth, has_up, has_ui);
        }
        self.lower_scope(fields, ptr, depth)
    }

    fn lower_location(&mut self, location: SchemaLocationId) -> Result<NodeId, CompileError> {
        if let Some(node) = self.location_nodes.get(&location).copied() {
            if let Some(slot) = self.location_slots.get(&location).copied() {
                if !self.assigned_slots.contains(&location) {
                    self.builder.set_def_target(slot, node)?;
                    self.assigned_slots.insert(location);
                }
            }
            return Ok(node);
        }
        let graph = self
            .graph
            .ok_or_else(|| limit("resource graph is absent"))?;
        let record = graph
            .locations
            .get(usize::try_from(location.get()).map_err(|_| limit("schema location"))?)
            .ok_or_else(|| limit("schema location"))?;
        let document = record.document;
        let pointer = record.document_pointer.to_string();
        let ast = graph.location_ast(location)?;
        let previous_document = std::mem::replace(&mut self.current_document, document);
        let result = self.lower(ast, &pointer, 0);
        self.current_document = previous_document;
        let node = result?;
        self.location_nodes.insert(location, node);
        if let Some(slot) = self.location_slots.get(&location).copied() {
            if !self.assigned_slots.contains(&location) {
                self.builder.set_def_target(slot, node)?;
                self.assigned_slots.insert(location);
            }
        }
        Ok(node)
    }

    fn finish_graph(
        mut self,
        root: NodeId,
        retrieval_uri: Option<String>,
    ) -> Result<SchemaIR, CompileError> {
        let graph = self
            .graph
            .ok_or_else(|| limit("resource graph is absent"))?;
        for anchor in &graph.anchors {
            self.lower_location(anchor.target)?;
        }
        let mut pending_index = 0usize;
        while pending_index < self.pending_locations.len() {
            let location = *self
                .pending_locations
                .get(pending_index)
                .ok_or_else(|| limit("pending reference location"))?;
            pending_index = pending_index
                .checked_add(1)
                .ok_or_else(|| limit("pending reference location"))?;
            self.lower_location(location)?;
        }
        let mut resources = Vec::new();
        resources
            .try_reserve_exact(graph.resources.len())
            .map_err(|_| limit("resource metadata"))?;
        for resource in &graph.resources {
            let root_node = self.lower_location(resource.root)?;
            let mut static_anchors = Vec::new();
            let mut dynamic_anchors = Vec::new();
            for &anchor_id in resource.static_anchors.iter() {
                let anchor = graph
                    .anchors
                    .get(usize::try_from(anchor_id.get()).map_err(|_| limit("anchor ID"))?)
                    .ok_or_else(|| limit("anchor ID"))?;
                static_anchors.push((anchor.name.to_string(), self.lower_location(anchor.target)?));
            }
            for &anchor_id in resource.dynamic_anchors.iter() {
                let anchor = graph
                    .anchors
                    .get(usize::try_from(anchor_id.get()).map_err(|_| limit("anchor ID"))?)
                    .ok_or_else(|| limit("anchor ID"))?;
                dynamic_anchors
                    .push((anchor.name.to_string(), self.lower_location(anchor.target)?));
            }
            static_anchors.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            dynamic_anchors.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            resources.push(ResourceInput {
                canonical_uri: resource.canonical_uri.as_deref().map(str::to_owned),
                root: root_node,
                static_anchors,
                dynamic_anchors,
            });
        }
        let retrieval_uri = retrieval_uri
            .map(|uri| super::resources::resolve_uri_reference(None, &uri))
            .transpose()?
            .map(|uri| uri.resource.to_string());
        self.builder
            .finish_with_resources(root, retrieval_uri, resources)
    }

    fn lower_scope(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let quirks = self.dialect_quirks_here();
        let has_ref = obj_has(fields, "$ref");
        if has_ref && quirks.ref_exclusive {
            let reference = get(fields, "$ref").expect("has_ref implies $ref is present");
            return self.lower_ref(reference, ptr, depth);
        }
        let has_dynamic_ref = obj_has(fields, "$dynamicRef");
        let has_if = obj_has(fields, "if") && !quirks.if_then_else_unsupported;
        let has_combinator = has_if
            || ["anyOf", "allOf", "oneOf", "not"]
                .iter()
                .any(|k| obj_has(fields, k));
        if has_ref || has_dynamic_ref || has_combinator {
            let mut parts = Vec::new();
            if let Some(reference) = get(fields, "$ref") {
                parts.push(self.lower_ref(reference, ptr, depth)?);
            }
            if let Some(reference) = get(fields, "$dynamicRef") {
                parts.push(self.lower_dynamic_ref(reference, ptr, depth)?);
            }
            if let Some(branches) = get(fields, "anyOf") {
                parts.push(self.lower_any_of(branches, ptr, depth)?);
            }
            if let Some(branches) = get(fields, "allOf") {
                parts.push(self.lower_all_of(branches, ptr, depth)?);
            }
            if let Some(branches) = get(fields, "oneOf") {
                parts.push(self.lower_one_of(branches, ptr, depth)?);
            }
            if let Some(sub) = get(fields, "not") {
                let inner = self.lower_isolated(sub, &child_ptr(ptr, "not"), depth + 1)?;
                parts.push(self.builder.not(inner)?);
            }
            if has_if {
                parts.push(self.lower_if_then_else(fields, ptr, depth)?);
            }
            if has_base_assertion(fields) {
                parts.push(self.lower_base(fields, ptr, depth)?);
            }
            return self.builder.intersection_of(parts);
        }
        self.lower_base(fields, ptr, depth)
    }

    fn lower_typeless_isolated(&mut self, ptr: &str, depth: u32) -> Result<NodeId, CompileError> {
        let saved = (
            self.open_under_unevaluated_properties,
            self.open_under_unevaluated_items,
        );
        self.open_under_unevaluated_properties = false;
        self.open_under_unevaluated_items = false;
        let result = self.lower_typeless(&[], ptr, depth);
        (
            self.open_under_unevaluated_properties,
            self.open_under_unevaluated_items,
        ) = saved;
        result
    }

    fn lower_isolated(
        &mut self,
        child: &'a Ast,
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let saved = (
            self.open_under_unevaluated_properties,
            self.open_under_unevaluated_items,
        );
        self.open_under_unevaluated_properties = false;
        self.open_under_unevaluated_items = false;
        let result = self.lower(child, ptr, depth);
        (
            self.open_under_unevaluated_properties,
            self.open_under_unevaluated_items,
        ) = saved;
        result
    }

    fn lower_unevaluated(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
        has_up: bool,
        has_ui: bool,
    ) -> Result<NodeId, CompileError> {
        let saved = (
            self.open_under_unevaluated_properties,
            self.open_under_unevaluated_items,
        );
        if has_up {
            self.open_under_unevaluated_properties = true;
        }
        if has_ui {
            self.open_under_unevaluated_items = true;
        }
        let scope = self.lower_scope(fields, ptr, depth);
        (
            self.open_under_unevaluated_properties,
            self.open_under_unevaluated_items,
        ) = saved;
        let mut scope = scope?;
        if has_up {
            let uneval = get(fields, "unevaluatedProperties").expect("checked by caller");
            let un =
                self.lower_isolated(uneval, &child_ptr(ptr, "unevaluatedProperties"), depth + 1)?;
            scope = self
                .builder
                .unevaluated(UnevaluatedKind::Properties, scope, un)?;
        }
        if has_ui {
            let uneval = get(fields, "unevaluatedItems").expect("checked by caller");
            let un = self.lower_isolated(uneval, &child_ptr(ptr, "unevaluatedItems"), depth + 1)?;
            scope = self
                .builder
                .unevaluated(UnevaluatedKind::Items, scope, un)?;
        }
        Ok(scope)
    }

    fn lower_base(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        if !self.validation_enabled {
            return self.lower_typeless(fields, ptr, depth);
        }
        for (keyword, reason) in UNSUPPORTED_KEYWORDS {
            if obj_has(fields, keyword) {
                return self.unsupported(&child_ptr(ptr, keyword), keyword, *reason);
            }
        }
        match get(fields, "type") {
            None => self.lower_typeless(fields, ptr, depth),
            Some(Ast::Arr(items)) => self.lower_type_union(items, fields, ptr, depth),
            Some(Ast::Str(t)) => {
                if let Some(value) = get(fields, "const") {
                    return self.lower_const(value, ptr, &[t.as_str()]);
                }
                if let Some(values) = get(fields, "enum") {
                    return self.lower_enum(values, ptr, &[t.as_str()]);
                }
                self.lower_type_branch(t, fields, ptr, depth)
            }
            Some(_) => Err(malformed(
                &child_ptr(ptr, "type"),
                "type",
                "type must be a string",
            )),
        }
    }

    fn lower_type_union(
        &mut self,
        items: &'a [Ast],
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let type_ptr = child_ptr(ptr, "type");
        if items.is_empty() {
            return Err(malformed(&type_ptr, "type", "type array must be non-empty"));
        }
        let mut names: Vec<&str> = Vec::with_capacity(items.len());
        for item in items {
            let Ast::Str(name) = item else {
                return Err(malformed(
                    &type_ptr,
                    "type",
                    "each type in the array must be a string",
                ));
            };
            if !VALID_TYPE_NAMES.contains(&name.as_str()) {
                return Err(malformed(
                    &type_ptr,
                    "type",
                    "unrecognized JSON Schema type name",
                ));
            }
            if !names.contains(&name.as_str()) {
                names.push(name.as_str());
            }
        }
        if let Some(value) = get(fields, "const") {
            return self.lower_const(value, ptr, &names);
        }
        if let Some(values) = get(fields, "enum") {
            return self.lower_enum(values, ptr, &names);
        }
        let mut branches = Vec::with_capacity(names.len());
        for name in &names {
            branches.push(self.lower_type_branch(name, fields, ptr, depth)?);
        }
        self.builder.union_of(branches)
    }

    fn lower_typeless(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        if let Some(value) = get(fields, "const") {
            return self.lower_const(value, ptr, &[]);
        }
        if let Some(values) = get(fields, "enum") {
            return self.lower_enum(values, ptr, &[]);
        }
        let object_branch = if OBJECT_ONLY_KEYWORDS.iter().any(|k| obj_has(fields, k)) {
            self.lower_type_branch("object", fields, ptr, depth)?
        } else {
            self.builder.open_object_full(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::Open,
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
            )?
        };
        let branches = vec![
            self.builder.null()?,
            self.builder.boolean()?,
            self.lower_type_branch("string", fields, ptr, depth)?,
            self.lower_type_branch("number", fields, ptr, depth)?,
            self.lower_type_branch("array", fields, ptr, depth)?,
            object_branch,
        ];
        self.builder.union_of(branches)
    }

    fn lower_any_of(
        &mut self,
        branches: &'a Ast,
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let any_of_ptr = child_ptr(ptr, "anyOf");
        let Ast::Arr(items) = branches else {
            return Err(malformed(&any_of_ptr, "anyOf", "anyOf must be an array"));
        };
        if items.is_empty() {
            return Err(malformed(&any_of_ptr, "anyOf", "anyOf must be non-empty"));
        }
        let mut lowered = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let branch_ptr = format!("{any_of_ptr}/{i}");
            lowered.push(self.lower(item, &branch_ptr, depth + 1)?);
        }
        self.builder.union_of(lowered)
    }

    fn lower_all_of(
        &mut self,
        branches: &'a Ast,
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let all_of_ptr = child_ptr(ptr, "allOf");
        let Ast::Arr(items) = branches else {
            return Err(malformed(&all_of_ptr, "allOf", "allOf must be an array"));
        };
        if items.is_empty() {
            return Err(malformed(&all_of_ptr, "allOf", "allOf must be non-empty"));
        }
        let mut lowered = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let branch_ptr = format!("{all_of_ptr}/{i}");
            lowered.push(self.lower(item, &branch_ptr, depth + 1)?);
        }
        self.builder.intersection_of(lowered)
    }

    fn lower_one_of(
        &mut self,
        branches: &'a Ast,
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let one_of_ptr = child_ptr(ptr, "oneOf");
        let Ast::Arr(items) = branches else {
            return Err(malformed(&one_of_ptr, "oneOf", "oneOf must be an array"));
        };
        if items.is_empty() {
            return Err(malformed(&one_of_ptr, "oneOf", "oneOf must be non-empty"));
        }
        let mut lowered = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let branch_ptr = format!("{one_of_ptr}/{i}");
            lowered.push(self.lower(item, &branch_ptr, depth + 1)?);
        }
        self.builder.exactly_one_of(lowered)
    }

    fn lower_if_then_else(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let if_sub = get(fields, "if").expect("caller verified `if` is present");
        if !obj_has(fields, "then") && !obj_has(fields, "else") {
            let under_uneval =
                self.open_under_unevaluated_properties || self.open_under_unevaluated_items;
            if !under_uneval {
                return self.lower_typeless(&[], ptr, depth);
            }
            let cond = self.lower(if_sub, &child_ptr(ptr, "if"), depth + 1)?;
            let not_cond = self.builder.not(cond)?;
            return self.builder.union_of(vec![cond, not_cond]);
        }
        let cond = self.lower(if_sub, &child_ptr(ptr, "if"), depth + 1)?;
        let then_node = match get(fields, "then") {
            Some(sub) => self.lower(sub, &child_ptr(ptr, "then"), depth + 1)?,
            None => self.lower_typeless(&[], ptr, depth)?,
        };
        let else_node = match get(fields, "else") {
            Some(sub) => self.lower(sub, &child_ptr(ptr, "else"), depth + 1)?,
            None => self.lower_typeless(&[], ptr, depth)?,
        };
        let not_cond = self.builder.not(cond)?;
        let then_branch = self.builder.intersection_of(vec![cond, then_node])?;
        let else_branch = self.builder.intersection_of(vec![not_cond, else_node])?;
        self.builder.union_of(vec![then_branch, else_branch])
    }

    fn lower_type_branch(
        &mut self,
        ty: &str,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        self.lower_typed(ty, fields, ptr, depth)
    }

    fn lower_typed(
        &mut self,
        ty: &str,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        match ty {
            "null" => self.builder.null(),
            "boolean" => self.builder.boolean(),
            "integer" => self.lower_integer(fields, ptr),
            "number" => self.lower_number(fields, ptr),
            "string" => self.lower_string(fields, ptr),
            "array" => self.lower_array(fields, ptr, depth),
            "object" => self.lower_object(fields, ptr, depth),
            _ => self.unsupported(
                &child_ptr(ptr, "type"),
                "type",
                UnsupportedReason::UnsupportedKeyword,
            ),
        }
    }

    fn lower_number(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
    ) -> Result<NodeId, CompileError> {
        if !self.validation_enabled {
            return self.builder.number();
        }
        let lexical = self.lower_digit_constraints(fields, ptr, false)?;
        let multiple_of = match self.number_multiple_of(fields, ptr)? {
            Ok(v) => v,
            Err(node) => return Ok(node),
        };
        use std::cmp::Ordering;
        let minimum = match self.number_bound(
            fields,
            ptr,
            "minimum",
            "exclusiveMinimum",
            Ordering::Greater,
        )? {
            Ok(v) => v,
            Err(node) => return Ok(node),
        };
        let maximum =
            match self.number_bound(fields, ptr, "maximum", "exclusiveMaximum", Ordering::Less)? {
                Ok(v) => v,
                Err(node) => return Ok(node),
            };
        if let (Some((lo, ls, lo_excl)), Some((hi, hs, hi_excl))) = (minimum, maximum) {
            let ord = crate::ir::decimal_cmp((lo, ls), (hi, hs));
            if ord == Ordering::Greater || (ord == Ordering::Equal && (lo_excl || hi_excl)) {
                return Err(malformed(
                    ptr,
                    "minimum",
                    "number bounds are an empty range",
                ));
            }
        }
        let semantic = self.builder.number_range(minimum, maximum, multiple_of)?;
        match lexical {
            Some(lexical) => self.builder.intersection_of(vec![semantic, lexical]),
            None => Ok(semantic),
        }
    }

    fn lower_digit_constraints(
        &mut self,
        fields: &[(String, Ast)],
        ptr: &str,
        integer: bool,
    ) -> Result<Option<NodeId>, CompileError> {
        let keys: &[&str] = if integer {
            &["minDigits", "maxDigits"]
        } else {
            &[
                "minDigitsInteger",
                "maxDigitsInteger",
                "minDigitsFraction",
                "maxDigitsFraction",
                "minDigitsExponent",
                "maxDigitsExponent",
            ]
        };
        if !keys.iter().any(|key| get(fields, key).is_some()) {
            return Ok(None);
        }
        let range = |minimum: &str, maximum_key: &str, floor: u32| {
            let minimum = digit_count(fields, minimum, ptr)?
                .unwrap_or(floor)
                .max(floor);
            let maximum = digit_count(fields, maximum_key, ptr)?;
            if maximum.is_some_and(|maximum| maximum < minimum) {
                return Err(malformed(
                    &child_ptr(ptr, maximum_key),
                    maximum_key,
                    "maximum digit count is less than minimum",
                ));
            }
            Ok((minimum, maximum))
        };
        let integer_digits = if integer {
            range("minDigits", "maxDigits", 1)?
        } else {
            range("minDigitsInteger", "maxDigitsInteger", 1)?
        };
        let integer_body = digit_quantifier(
            integer_digits.0.saturating_sub(1),
            integer_digits.1.map(|value| value.saturating_sub(1)),
        );
        let zero = if integer_digits.0 <= 1 && integer_digits.1.is_none_or(|max| max >= 1) {
            "0|"
        } else {
            ""
        };
        let regex = if integer {
            format!(r"-?(?:{zero}[1-9][0-9]{integer_body})")
        } else {
            let fraction = range("minDigitsFraction", "maxDigitsFraction", 1)?;
            let exponent = range("minDigitsExponent", "maxDigitsExponent", 1)?;
            let fraction = optional_digit_part(r"\.", fraction);
            let exponent = optional_digit_part(r"[eE][+-]", exponent);
            format!(r"-?(?:{zero}[1-9][0-9]{integer_body}){fraction}{exponent}")
        };
        self.builder.lexical_number(&regex).map(Some)
    }

    fn number_multiple_of(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
    ) -> Result<Result<Option<(u64, u32)>, NodeId>, CompileError> {
        let Some(value) = get(fields, "multipleOf") else {
            return Ok(Ok(None));
        };
        let Ast::Num(tok) = value else {
            return Err(malformed(
                &child_ptr(ptr, "multipleOf"),
                "multipleOf",
                "multipleOf must be a number",
            ));
        };
        if tok.starts_with('-') {
            return Err(malformed(
                &child_ptr(ptr, "multipleOf"),
                "multipleOf",
                "multipleOf must be strictly positive",
            ));
        }
        match decimal_token_as_coef_exp(tok) {
            Some((0, _)) => Err(malformed(
                &child_ptr(ptr, "multipleOf"),
                "multipleOf",
                "multipleOf must be strictly positive",
            )),
            Some(pair) => Ok(Ok(Some(pair))),
            None => Ok(Err(self.unsupported(
                &child_ptr(ptr, "multipleOf"),
                "multipleOf",
                UnsupportedReason::NumericRangeUnsupported,
            )?)),
        }
    }

    fn number_bound(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        key: &str,
        exclusive_key: &str,
        tighter: std::cmp::Ordering,
    ) -> Result<Result<Option<DecimalBound>, NodeId>, CompileError> {
        let mut bound: Option<DecimalBound> = None;
        for (k, excl) in [(key, false), (exclusive_key, true)] {
            let Some(v) = get(fields, k) else { continue };
            let Ast::Num(tok) = v else {
                return Err(malformed(
                    &child_ptr(ptr, k),
                    k,
                    "numeric bound must be a number",
                ));
            };
            let Some((val, scale)) = decimal_token_scaled(tok) else {
                return Ok(Err(self.unsupported(
                    &child_ptr(ptr, k),
                    k,
                    UnsupportedReason::NumberBoundUnsupported,
                )?));
            };
            let replace = match bound {
                None => true,
                Some((cv, cs, cur_excl)) => {
                    let ord = crate::ir::decimal_cmp((val, scale), (cv, cs));
                    ord == tighter || (ord == std::cmp::Ordering::Equal && excl && !cur_excl)
                }
            };
            if replace {
                bound = Some((val, scale, excl));
            }
        }
        Ok(Ok(bound))
    }

    fn lower_const(
        &mut self,
        value: &Ast,
        ptr: &str,
        types: &[&str],
    ) -> Result<NodeId, CompileError> {
        if let Ast::Num(token) = value {
            if !numeric_literal_matches_types(token, types) {
                return self.builder.never();
            }
            let Some((coefficient, scale)) = decimal_token_scaled(token) else {
                return self.unsupported(
                    &child_ptr(ptr, "const"),
                    "const",
                    UnsupportedReason::NumericRangeUnsupported,
                );
            };
            let bound = Some((coefficient, scale, false));
            return if numeric_type_requires_integer(types) {
                self.builder.integer_number_range(bound, bound, None)
            } else {
                self.builder.number_range(bound, bound, None)
            };
        }
        match scalar_lit(value) {
            Some(lit) => {
                if !types.is_empty() && !types.iter().any(|t| type_matches(t, &lit)) {
                    return self.builder.never();
                }
                match lit {
                    ScalarLit::Str(s) => self.builder.string_const(&s),
                    other => self.builder.enum_values(vec![other]),
                }
            }
            None => self.unsupported(&child_ptr(ptr, "const"), "const", literal_reason(value)),
        }
    }

    fn lower_enum(
        &mut self,
        values: &Ast,
        ptr: &str,
        types: &[&str],
    ) -> Result<NodeId, CompileError> {
        let Ast::Arr(items) = values else {
            return Err(malformed(
                &child_ptr(ptr, "enum"),
                "enum",
                "enum must be an array",
            ));
        };
        if items.is_empty() {
            return self.builder.never();
        }
        let mut lits = Vec::new();
        let mut decimals = Vec::new();
        lits.try_reserve_exact(items.len())
            .map_err(|_| limit("enum allocation"))?;
        decimals
            .try_reserve_exact(items.len())
            .map_err(|_| limit("enum allocation"))?;
        for (i, item) in items.iter().enumerate() {
            if let Ast::Num(token) = item {
                if !numeric_literal_matches_types(token, types) {
                    continue;
                }
                let Some(decimal) = decimal_token_scaled(token) else {
                    let p = format!("{}/{i}", child_ptr(ptr, "enum"));
                    return self.unsupported(
                        &p,
                        "enum",
                        UnsupportedReason::NumericRangeUnsupported,
                    );
                };
                decimals.push(decimal);
                continue;
            }
            match scalar_lit(item) {
                Some(l) => {
                    if !types.is_empty() && !types.iter().any(|t| type_matches(t, &l)) {
                        continue;
                    }
                    lits.push(l);
                }
                None => {
                    let p = format!("{}/{i}", child_ptr(ptr, "enum"));
                    return self.unsupported(&p, "enum", literal_reason(item));
                }
            }
        }
        lits.sort();
        lits.dedup();
        decimals.sort_unstable();
        decimals.dedup();
        if lits.is_empty() && decimals.is_empty() {
            return self.builder.never();
        }
        let mut branches = Vec::new();
        branches
            .try_reserve_exact(decimals.len().saturating_add(usize::from(!lits.is_empty())))
            .map_err(|_| limit("enum allocation"))?;
        if !lits.is_empty() {
            branches.push(self.builder.enum_values(lits)?);
        }
        for (coefficient, scale) in decimals {
            let bound = Some((coefficient, scale, false));
            branches.push(if numeric_type_requires_integer(types) {
                self.builder.integer_number_range(bound, bound, None)?
            } else {
                self.builder.number_range(bound, bound, None)?
            });
        }
        match branches.as_slice() {
            [only] => Ok(*only),
            _ => self.builder.union_of(branches),
        }
    }

    fn lower_integer(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
    ) -> Result<NodeId, CompileError> {
        if !self.validation_enabled {
            return self.builder.integer(None, None);
        }
        let lexical = self.lower_digit_constraints(fields, ptr, true)?;
        use std::cmp::Ordering;
        let minimum = match self.number_bound(
            fields,
            ptr,
            "minimum",
            "exclusiveMinimum",
            Ordering::Greater,
        )? {
            Ok(value) => value,
            Err(node) => return Ok(node),
        };
        let maximum =
            match self.number_bound(fields, ptr, "maximum", "exclusiveMaximum", Ordering::Less)? {
                Ok(value) => value,
                Err(node) => return Ok(node),
            };
        let multiple_of = match self.number_multiple_of(fields, ptr)? {
            Ok(value) => value,
            Err(node) => return Ok(node),
        };
        if let (Some((lo, ls, lo_excl)), Some((hi, hs, hi_excl))) = (minimum, maximum) {
            let ordering = crate::ir::decimal_cmp((lo, ls), (hi, hs));
            if ordering == Ordering::Greater
                || (ordering == Ordering::Equal && (lo_excl || hi_excl))
            {
                return self.builder.never();
            }
        }
        if let (Some(lo), Some(hi), Some(divisor)) = (
            fast_integer_bound(minimum, true),
            fast_integer_bound(maximum, false),
            fast_integer_multiple(multiple_of),
        ) {
            let semantic = self.builder.integer_multiple_of(lo, hi, divisor)?;
            return match lexical {
                Some(lexical) => self.builder.intersection_of(vec![semantic, lexical]),
                None => Ok(semantic),
            };
        }
        let semantic = self
            .builder
            .integer_number_range(minimum, maximum, multiple_of)?;
        match lexical {
            Some(lexical) => self.builder.intersection_of(vec![semantic, lexical]),
            None => Ok(semantic),
        }
    }

    fn lower_string(
        &mut self,
        fields: &[(String, Ast)],
        ptr: &str,
    ) -> Result<NodeId, CompileError> {
        let min_len = match count_field(fields, "minLength", ptr)? {
            Some(CountBound::AboveU32) => return self.builder.never(),
            Some(CountBound::Fits(value)) => Some(value),
            None => None,
        };
        let max_len = match count_field(fields, "maxLength", ptr)? {
            Some(CountBound::Fits(value)) => Some(value),
            Some(CountBound::AboveU32) | None => None,
        };

        let mut parts = Vec::with_capacity(3);
        if let Some(fmt_ast) = get(fields, "format") {
            let Ast::Str(fmt) = fmt_ast else {
                return Err(malformed(
                    &child_ptr(ptr, "format"),
                    "format",
                    "format must be a string",
                ));
            };
            parts.push(self.lower_format(fmt, ptr)?);
        }
        match get(fields, "pattern") {
            Some(Ast::Str(re)) => {
                if let Err(error) = crate::ir::validate_search_pattern(re) {
                    if error.code == ErrorCode::Unsupported {
                        parts.push(self.unsupported(
                            &child_ptr(ptr, "pattern"),
                            "pattern",
                            unsupported_pattern_reason(re),
                        )?);
                        return self.builder.intersection_of(parts);
                    }
                    return Err(error);
                }
                match crate::frontend::regex::analyze_pattern(re) {
                    crate::frontend::regex::PatternAnalysis::Leading(assertions) => {
                        parts.push(crate::frontend::regex::lower_leading_assertions(
                            &mut self.builder,
                            assertions,
                        )?);
                    }
                    crate::frontend::regex::PatternAnalysis::Ordinary => {
                        let search =
                            crate::structured::build_search_pattern(re).map_err(|error| {
                                error
                                    .with_pointer(child_ptr(ptr, "pattern"))
                                    .with_keyword("pattern")
                            })?;
                        let pattern =
                            match self
                                .builder
                                .string_pattern(&search, None, None, Charset::Utf8Any)
                            {
                                Ok(pattern) => pattern,
                                Err(error) if error.code == ErrorCode::Unsupported => self
                                    .unsupported(
                                        &child_ptr(ptr, "pattern"),
                                        "pattern",
                                        UnsupportedReason::InvalidPattern,
                                    )?,
                                Err(error) => return Err(error),
                            };
                        parts.push(pattern);
                    }
                    crate::frontend::regex::PatternAnalysis::Unsupported(_)
                    | crate::frontend::regex::PatternAnalysis::Malformed(_) => {
                        return Err(malformed(
                            &child_ptr(ptr, "pattern"),
                            "pattern",
                            "pattern analysis disagrees with validation",
                        ));
                    }
                }
            }
            Some(_) => {
                return Err(malformed(
                    &child_ptr(ptr, "pattern"),
                    "pattern",
                    "pattern must be a string",
                ))
            }
            None => {}
        }
        if min_len.is_some() || max_len.is_some() {
            parts.push(self.builder.string_pattern(
                JSON_STRING_CODEPOINT,
                min_len,
                max_len,
                Charset::Utf8CountedCodepoints,
            )?);
        }
        match parts.len() {
            0 => self.unconstrained_string(),
            _ => self.builder.intersection_of(parts),
        }
    }

    fn lower_format(&mut self, fmt: &str, ptr: &str) -> Result<NodeId, CompileError> {
        if !self.format_assertion {
            return self.unconstrained_string();
        }
        match fmt {
            "date-time" => self
                .builder
                .string_pattern(DATE_TIME_RE, None, None, Charset::Utf8Any),
            "date" => self
                .builder
                .string_pattern(DATE_RE, None, None, Charset::Utf8Any),
            "time" => self
                .builder
                .string_pattern(TIME_RE, None, None, Charset::Utf8Any),
            "uuid" => self
                .builder
                .string_pattern(UUID_RE, None, None, Charset::Utf8Any),
            "ipv4" => self
                .builder
                .string_pattern(IPV4_RE, None, None, Charset::Utf8Any),
            "email" | "hostname" | "uri" | "regex" | "json-pointer" => self.unconstrained_string(),
            "duration" | "ipv6" => self.unsupported(
                &child_ptr(ptr, "format"),
                "format",
                UnsupportedReason::FormatUnsupported,
            ),
            _ => self.unconstrained_string(),
        }
    }

    fn unconstrained_string(&mut self) -> Result<NodeId, CompileError> {
        self.builder
            .string_pattern(JSON_STRING_BODY, None, None, Charset::Utf8Any)
    }

    fn lower_array(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let min = match count_field(fields, "minItems", ptr)? {
            Some(CountBound::AboveU32) => return self.builder.never(),
            Some(CountBound::Fits(value)) => value,
            None => 0,
        };
        let max = match count_field(fields, "maxItems", ptr)? {
            Some(CountBound::Fits(value)) => Some(value),
            Some(CountBound::AboveU32) | None => None,
        };
        if let Some(m) = max {
            if m < min {
                return self.builder.never();
            }
        }
        let unique_items = match get(fields, "uniqueItems") {
            None | Some(Ast::Bool(false)) => false,
            Some(Ast::Bool(true)) => true,
            Some(_) => {
                return Err(malformed(
                    ptr,
                    "uniqueItems",
                    "uniqueItems must be a boolean",
                ))
            }
        };
        if obj_has(fields, "prefixItems") {
            return self.lower_tuple(fields, ptr, depth, min, max, unique_items);
        }
        let contains = match self.lower_contains(fields, ptr, depth)? {
            ContainsLowering::Absent => None,
            ContainsLowering::Constraint(constraint) => Some(constraint),
            ContainsLowering::Impossible => return self.builder.never(),
        };
        let items_present = obj_has(fields, "items");
        let items = match get(fields, "items") {
            None | Some(Ast::Bool(true)) => crate::ir::ItemsPolicy::AllowAny,
            Some(sub) => crate::ir::ItemsPolicy::Schema(self.lower_isolated(
                sub,
                &child_ptr(ptr, "items"),
                depth + 1,
            )?),
        };
        self.builder
            .array_full(items, min, max, unique_items, contains, items_present)
    }

    fn lower_contains(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<ContainsLowering, CompileError> {
        let Some(contains) = get(fields, "contains") else {
            return Ok(ContainsLowering::Absent);
        };
        let policy = match contains {
            Ast::Bool(true) => ContainsPolicy::Always,
            Ast::Bool(false) => ContainsPolicy::Never,
            sub => ContainsPolicy::Schema(self.lower_isolated(
                sub,
                &child_ptr(ptr, "contains"),
                depth + 1,
            )?),
        };
        let min = match count_field(fields, "minContains", ptr)? {
            Some(CountBound::AboveU32) => return Ok(ContainsLowering::Impossible),
            Some(CountBound::Fits(value)) => value,
            None => 1,
        };
        let max = match count_field(fields, "maxContains", ptr)? {
            Some(CountBound::Fits(value)) => Some(value),
            Some(CountBound::AboveU32) | None => None,
        };
        if let Some(m) = max {
            if m < min {
                return Ok(ContainsLowering::Impossible);
            }
        }
        Ok(ContainsLowering::Constraint(ContainsConstraint {
            policy,
            min,
            max,
        }))
    }

    fn lower_tuple(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
        min: u32,
        max: Option<u32>,
        unique_items: bool,
    ) -> Result<NodeId, CompileError> {
        let prefix_ptr = child_ptr(ptr, "prefixItems");
        let Some(Ast::Arr(items)) = get(fields, "prefixItems") else {
            return Err(malformed(
                &prefix_ptr,
                "prefixItems",
                "prefixItems must be an array",
            ));
        };
        if items.is_empty() {
            return Err(malformed(
                &prefix_ptr,
                "prefixItems",
                "prefixItems must be non-empty",
            ));
        }
        let mut prefix = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            prefix.push(self.lower_isolated(item, &format!("{prefix_ptr}/{i}"), depth + 1)?);
        }
        let items_present = obj_has(fields, "items");
        let tail = match get(fields, "items") {
            Some(Ast::Bool(false)) => None,
            Some(Ast::Bool(true)) | None => Some(self.lower_typeless_isolated(ptr, depth)?),
            Some(sub) => Some(self.lower_isolated(sub, &child_ptr(ptr, "items"), depth + 1)?),
        };
        if tail.is_none() {
            let n = u32::try_from(prefix.len()).map_err(|_| limit("tuple prefix length"))?;
            if min > n {
                return self.builder.never();
            }
        }
        let contains = match self.lower_contains(fields, ptr, depth)? {
            ContainsLowering::Absent => None,
            ContainsLowering::Constraint(constraint) => Some(constraint),
            ContainsLowering::Impossible => return self.builder.never(),
        };
        let tail_annotates = tail.is_some() && items_present;
        self.builder.tuple_full(
            prefix,
            tail,
            min,
            max,
            unique_items,
            contains,
            tail_annotates,
        )
    }

    fn lower_object(
        &mut self,
        fields: &'a [(String, Ast)],
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let required = required_set(fields, ptr)?;
        let props_ptr = child_ptr(ptr, "properties");
        let entries: &[(String, Ast)] = match get(fields, "properties") {
            Some(Ast::Obj(e)) => e,
            Some(_) => {
                return Err(malformed(
                    &props_ptr,
                    "properties",
                    "properties must be an object",
                ))
            }
            None => &[],
        };
        let mut lowered = Vec::with_capacity(entries.len());
        let mut required_flags = Vec::with_capacity(entries.len());
        let mut seen_required: FxHashMap<&str, ()> = FxHashMap::default();
        for (name, sub) in entries {
            let child = self.lower_isolated(
                sub,
                &format!("{}/{}", props_ptr, ptr_escape(name)),
                depth + 1,
            )?;
            if required.contains_key(name.as_str()) {
                seen_required.insert(name.as_str(), ());
            }
            required_flags.push(required.contains_key(name.as_str()));
            lowered.push((name.clone(), child));
        }
        if seen_required.len() != required.len() {
            if let Some(Ast::Arr(items)) = get(fields, "required") {
                for item in items {
                    let Ast::Str(name) = item else {
                        unreachable!("required_set already validated every member is a string")
                    };
                    if seen_required.contains_key(name.as_str()) {
                        continue;
                    }
                    let any_value = self.lower_typeless_isolated(ptr, depth)?;
                    lowered.push((name.clone(), any_value));
                    required_flags.push(true);
                    seen_required.insert(name.as_str(), ());
                }
            }
        }

        let pattern_ptr = child_ptr(ptr, "patternProperties");
        let pattern_entries: &[(String, Ast)] = match get(fields, "patternProperties") {
            Some(Ast::Obj(e)) => e,
            Some(_) => {
                return Err(malformed(
                    &pattern_ptr,
                    "patternProperties",
                    "patternProperties must be an object",
                ))
            }
            None => &[],
        };
        let mut lowered_patterns = Vec::with_capacity(pattern_entries.len());
        for (regex, sub) in pattern_entries {
            if let Err(error) = crate::ir::validate_search_pattern(regex) {
                if error.code == ErrorCode::Unsupported {
                    return self.unsupported(
                        &pattern_ptr,
                        "patternProperties",
                        unsupported_pattern_reason(regex),
                    );
                }
                return Err(error);
            }
            let child = self.lower_isolated(
                sub,
                &format!("{}/{}", pattern_ptr, ptr_escape(regex)),
                depth + 1,
            )?;
            lowered_patterns.push((regex.clone(), child));
        }

        let dependent_ptr = child_ptr(ptr, "dependentSchemas");
        let dependent_entries: &[(String, Ast)] = match get(fields, "dependentSchemas") {
            Some(Ast::Obj(e)) => e,
            Some(_) => {
                return Err(malformed(
                    &dependent_ptr,
                    "dependentSchemas",
                    "dependentSchemas must be an object",
                ))
            }
            None => &[],
        };
        let (legacy_required, legacy_schema_entries) = legacy_dependencies_pairs(fields, ptr)?;
        let mut dependent =
            Vec::with_capacity(dependent_entries.len() + legacy_schema_entries.len());
        for (key, sub) in dependent_entries {
            let child = self.lower(
                sub,
                &format!("{}/{}", dependent_ptr, ptr_escape(key)),
                depth + 1,
            )?;
            dependent.push((key.clone(), child));
        }
        let legacy_ptr = child_ptr(ptr, "dependencies");
        for (key, sub) in legacy_schema_entries {
            let child = self.lower(
                sub,
                &format!("{}/{}", legacy_ptr, ptr_escape(key)),
                depth + 1,
            )?;
            dependent.push((key.to_owned(), child));
        }
        let mut dependent_required = dependent_required_pairs(fields, ptr)?;
        dependent_required.extend(legacy_required);

        let property_names = match get(fields, "propertyNames") {
            Some(sub) => {
                Some(self.lower_isolated(sub, &child_ptr(ptr, "propertyNames"), depth + 1)?)
            }
            None => None,
        };
        let min_properties = match count_field(fields, "minProperties", ptr)? {
            Some(CountBound::AboveU32) => return self.builder.never(),
            Some(CountBound::Fits(value)) => Some(value),
            None => None,
        };
        let max_properties = match count_field(fields, "maxProperties", ptr)? {
            Some(CountBound::Fits(value)) => Some(value),
            Some(CountBound::AboveU32) | None => None,
        };
        if let (Some(lo), Some(hi)) = (min_properties, max_properties) {
            if hi < lo {
                return self.builder.never();
            }
        }
        let has_open_features = !lowered_patterns.is_empty()
            || property_names.is_some()
            || min_properties.is_some()
            || max_properties.is_some();

        match get(fields, "additionalProperties") {
            Some(Ast::Bool(false)) if !has_open_features => self.builder.object_full(
                lowered,
                &required_flags,
                ObjectClosure::Forbidden,
                dependent,
                dependent_required,
            ),
            None if self.open_under_unevaluated_properties => self.builder.open_object_full(
                lowered,
                &required_flags,
                lowered_patterns,
                AdditionalPolicy::Open,
                property_names,
                min_properties,
                max_properties,
                dependent,
                dependent_required,
            ),
            None if !has_open_features
                && self.object_closure == ObjectClosure::AssumeClosedProfile =>
            {
                self.builder.object_full(
                    lowered,
                    &required_flags,
                    ObjectClosure::AssumeClosedProfile,
                    dependent,
                    dependent_required,
                )
            }
            None if !has_open_features
                && self.object_closure == ObjectClosure::RejectOpenObjects =>
            {
                self.unsupported(
                    ptr,
                    "additionalProperties",
                    UnsupportedReason::OpenAdditionalProperties,
                )
            }
            additional => {
                let policy = match additional {
                    Some(Ast::Bool(false)) => AdditionalPolicy::Forbid,
                    Some(Ast::Bool(true)) => AdditionalPolicy::AllowAny,
                    None => AdditionalPolicy::Open,
                    Some(sub) => AdditionalPolicy::Schema(self.lower_isolated(
                        sub,
                        &child_ptr(ptr, "additionalProperties"),
                        depth + 1,
                    )?),
                };
                self.builder.open_object_full(
                    lowered,
                    &required_flags,
                    lowered_patterns,
                    policy,
                    property_names,
                    min_properties,
                    max_properties,
                    dependent,
                    dependent_required,
                )
            }
        }
    }

    fn lower_ref(
        &mut self,
        reference: &Ast,
        ptr: &str,
        depth: u32,
    ) -> Result<NodeId, CompileError> {
        let Ast::Str(target) = reference else {
            return Err(malformed(
                &child_ptr(ptr, "$ref"),
                "$ref",
                "$ref must be a string",
            ));
        };
        if let Some(graph) = self.graph {
            let location = graph
                .document_location(self.current_document, ptr)
                .ok_or_else(|| malformed(ptr, "$ref", "reference location is not indexed"))?;
            let resolved = graph.resolved_reference_from(location, ReferenceKind::Static)?;
            let target_location = graph.target_location(resolved.target)?;
            return self.reference_to_location(target_location);
        }
        let key = target.clone();
        if let Some(&slot) = self.def_slots.get(&key) {
            return self.builder.ref_node(slot);
        }
        if target != "#" && !target.starts_with('#') {
            return self.unsupported(
                &child_ptr(ptr, "$ref"),
                "$ref",
                UnsupportedReason::NonRegularRef,
            );
        }
        let Some(target_ast) = self.resolve_ref_target(target)? else {
            return Err(malformed(
                &child_ptr(ptr, "$ref"),
                "$ref",
                "$ref target not found",
            ));
        };
        if self.is_recursive_ref(&key)? {
            let slot = self.builder.alloc_def_slot()?;
            self.def_slots.insert(key, slot);
            let body = self.lower(target_ast, ptr, depth + 1)?;
            self.builder.set_def_target(slot, body)?;
            return self.builder.ref_node(slot);
        }
        let memo_key = (
            key,
            self.open_under_unevaluated_properties,
            self.open_under_unevaluated_items,
        );
        if let Some(&lowered) = self.lowered_refs.get(&memo_key) {
            return Ok(lowered);
        }
        let lowered = self.lower(target_ast, ptr, depth + 1)?;
        self.lowered_refs
            .try_reserve(1)
            .map_err(|_| limit("lowered reference memo"))?;
        self.lowered_refs.insert(memo_key, lowered);
        Ok(lowered)
    }

    fn lower_dynamic_ref(
        &mut self,
        reference: &Ast,
        ptr: &str,
        _depth: u32,
    ) -> Result<NodeId, CompileError> {
        let Ast::Str(_) = reference else {
            return Err(malformed(
                &child_ptr(ptr, "$dynamicRef"),
                "$dynamicRef",
                "$dynamicRef must be a string",
            ));
        };
        let graph = self.graph.ok_or_else(|| {
            malformed(
                &child_ptr(ptr, "$dynamicRef"),
                "$dynamicRef",
                "dynamic reference requires resource lowering",
            )
        })?;
        let location = graph
            .document_location(self.current_document, ptr)
            .ok_or_else(|| malformed(ptr, "$dynamicRef", "reference location is not indexed"))?;
        let resolved = graph.resolved_reference_from(location, ReferenceKind::Dynamic)?;
        let target_location = graph.target_location(resolved.target)?;
        let initial_target = self.reference_to_location(target_location)?;
        let super::resources::FragmentTarget::Anchor(graph_anchor) = resolved.target else {
            return Ok(initial_target);
        };
        let anchor_record = graph
            .anchors
            .get(usize::try_from(graph_anchor.get()).map_err(|_| limit("dynamic anchor"))?)
            .ok_or_else(|| limit("dynamic anchor"))?;
        if !anchor_record.dynamic {
            return Ok(initial_target);
        }
        let anchor = self
            .dynamic_anchor_ids
            .get(&graph_anchor)
            .copied()
            .ok_or_else(|| limit("dynamic anchor mapping"))?;
        self.builder.dynamic_ref(initial_target, anchor)
    }

    fn reference_to_location(
        &mut self,
        target_location: SchemaLocationId,
    ) -> Result<NodeId, CompileError> {
        let slot = match self.location_slots.get(&target_location).copied() {
            Some(slot) => slot,
            None => {
                let slot = self.builder.alloc_def_slot()?;
                self.location_slots
                    .try_reserve(1)
                    .map_err(|_| limit("reference location slot"))?;
                self.location_slots.insert(target_location, slot);
                self.pending_locations
                    .try_reserve(1)
                    .map_err(|_| limit("pending reference location"))?;
                self.pending_locations.push(target_location);
                slot
            }
        };
        self.builder.ref_node(slot)
    }

    fn resolve_ref_target(&self, target: &str) -> Result<Option<&'a Ast>, CompileError> {
        let fragment = decode_uri_fragment(target)?;
        if fragment.is_empty() {
            Ok(Some(self.root))
        } else if let Some(pointer) = fragment.strip_prefix('/') {
            Ok(self.pointers.get(pointer).copied())
        } else {
            Ok(self.anchors.get(&fragment).copied())
        }
    }

    fn is_recursive_ref(&mut self, key: &str) -> Result<bool, CompileError> {
        if let Some(&v) = self.recursion_memo.get(key) {
            return Ok(v);
        }
        let mut stack = Vec::new();
        self.mark_cycle_membership(key.to_string(), &mut stack)?;
        Ok(self.recursion_memo.get(key).copied().unwrap_or(false))
    }

    fn used_refs_of(&mut self, key: &str) -> Result<Vec<String>, CompileError> {
        if let Some(refs) = self.ref_adjacency.get(key) {
            return Ok(refs.clone());
        }
        let mut refs = Vec::new();
        if let Some(ast) = self.resolve_ref_target(key)? {
            collect_used_refs(ast, &mut refs);
        }
        self.ref_adjacency.insert(key.to_string(), refs.clone());
        Ok(refs)
    }

    fn mark_cycle_membership(
        &mut self,
        node: String,
        stack: &mut Vec<String>,
    ) -> Result<(), CompileError> {
        if self.recursion_memo.contains_key(&node) {
            return Ok(());
        }
        if let Some(i) = stack.iter().position(|n| *n == node) {
            for n in &stack[i..] {
                self.recursion_memo.insert(n.clone(), true);
            }
            return Ok(());
        }
        stack.push(node.clone());
        for r in self.used_refs_of(&node)? {
            self.mark_cycle_membership(r, stack)?;
        }
        stack.pop();
        self.recursion_memo.entry(node).or_insert(false);
        Ok(())
    }

    fn unsupported(
        &mut self,
        ptr: &str,
        keyword: &str,
        reason: UnsupportedReason,
    ) -> Result<NodeId, CompileError> {
        self.builder.unsupported(keyword, reason, ptr)
    }
}

fn decode_uri_fragment(target: &str) -> Result<String, CompileError> {
    let fragment = target.strip_prefix('#').ok_or_else(|| {
        malformed(
            "",
            "$ref",
            "local reference must begin with a fragment marker",
        )
    })?;
    let bytes = fragment.as_bytes();
    let mut decoded = Vec::new();
    decoded.try_reserve_exact(bytes.len()).map_err(|_| {
        CompileError::new(
            ErrorCode::InternalLimitExceeded,
            Stage::L1,
            "reference fragment allocation",
        )
    })?;
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        let high = bytes.get(index + 1).and_then(|byte| hex_digit(*byte));
        let low = bytes.get(index + 2).and_then(|byte| hex_digit(*byte));
        let (Some(high), Some(low)) = (high, low) else {
            return Err(malformed("", "$ref", "invalid percent-encoded fragment"));
        };
        decoded.push((high << 4) | low);
        index += 3;
    }
    String::from_utf8(decoded)
        .map_err(|_| malformed("", "$ref", "reference fragment is not valid UTF-8"))
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[derive(Default)]
struct DocIndex<'a> {
    anchors: FxHashMap<String, &'a Ast>,
    ids: FxHashMap<String, ()>,
    pointers: FxHashMap<String, &'a Ast>,
}

fn index_document<'a>(
    node: &'a Ast,
    path: &str,
    idx: &mut DocIndex<'a>,
) -> Result<(), CompileError> {
    if !path.is_empty() {
        idx.pointers.insert(path.to_string(), node);
    }
    match node {
        Ast::Arr(items) => {
            for (i, item) in items.iter().enumerate() {
                index_document(item, &child_pointer(path, &i.to_string()), idx)?;
            }
            return Ok(());
        }
        Ast::Obj(fields) => {
            let mut seen: FxHashSet<&str> = FxHashSet::default();
            seen.reserve(fields.len());
            for (key, _) in fields {
                if !seen.insert(key.as_str()) {
                    let parent = if path.is_empty() {
                        String::new()
                    } else {
                        format!("/{path}")
                    };
                    return Err(malformed(
                        &child_ptr(&parent, key),
                        key,
                        "duplicate schema keyword",
                    ));
                }
            }
            if let Some(id_ast) = get(fields, "$id") {
                let Ast::Str(id) = id_ast else {
                    return Err(malformed("", "$id", "$id must be a string"));
                };
                if idx.ids.insert(id.clone(), ()).is_some() {
                    return Err(malformed("", "$id", "duplicate $id"));
                }
            }
            if let Some(anchor_ast) = get(fields, "$anchor") {
                let Ast::Str(name) = anchor_ast else {
                    return Err(malformed("", "$anchor", "$anchor must be a string"));
                };
                if idx.anchors.insert(name.clone(), node).is_some() {
                    return Err(malformed("", "$anchor", "duplicate $anchor"));
                }
            }
            for (key, sub) in fields {
                index_document(sub, &child_pointer(path, &ptr_escape(key)), idx)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn child_pointer(path: &str, segment: &str) -> String {
    if path.is_empty() {
        segment.to_string()
    } else {
        format!("{path}/{segment}")
    }
}

fn required_set<'a>(
    fields: &'a [(String, Ast)],
    ptr: &str,
) -> Result<FxHashMap<&'a str, ()>, CompileError> {
    let mut set = FxHashMap::default();
    let Some(req) = get(fields, "required") else {
        return Ok(set);
    };
    let Ast::Arr(items) = req else {
        return Err(malformed(
            &child_ptr(ptr, "required"),
            "required",
            "required must be an array",
        ));
    };
    for item in items {
        let Ast::Str(name) = item else {
            return Err(malformed(
                &child_ptr(ptr, "required"),
                "required",
                "required names must be strings",
            ));
        };
        if set.insert(name.as_str(), ()).is_some() {
            return Err(malformed(
                &child_ptr(ptr, "required"),
                "required",
                "duplicate required name",
            ));
        }
    }
    Ok(set)
}

fn reserve_amplification() -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L1,
        "dependentRequired exceeds the pair cap",
    )
}

fn dependent_required_pairs(
    fields: &[(String, Ast)],
    ptr: &str,
) -> Result<Vec<(String, Vec<String>)>, CompileError> {
    let Some(value) = get(fields, "dependentRequired") else {
        return Ok(Vec::new());
    };
    let keyword_ptr = child_ptr(ptr, "dependentRequired");
    let Ast::Obj(entries) = value else {
        return Err(malformed(
            &keyword_ptr,
            "dependentRequired",
            "dependentRequired must be an object",
        ));
    };
    required_name_groups("dependentRequired", entries, &keyword_ptr)
}

type LegacyDependencyNameGroups = Vec<(String, Vec<String>)>;
type LegacyDependencySchemaEntries<'a> = Vec<(&'a str, &'a Ast)>;

fn legacy_dependencies_pairs<'a>(
    fields: &'a [(String, Ast)],
    ptr: &str,
) -> Result<
    (
        LegacyDependencyNameGroups,
        LegacyDependencySchemaEntries<'a>,
    ),
    CompileError,
> {
    let Some(value) = get(fields, "dependencies") else {
        return Ok((Vec::new(), Vec::new()));
    };
    let keyword_ptr = child_ptr(ptr, "dependencies");
    let Ast::Obj(entries) = value else {
        return Err(malformed(
            &keyword_ptr,
            "dependencies",
            "dependencies must be an object",
        ));
    };
    let mut array_entries = Vec::new();
    let mut schema_entries = Vec::new();
    for (key, value) in entries {
        match value {
            Ast::Arr(_) => array_entries.push((key.clone(), value.clone())),
            Ast::Obj(_) | Ast::Bool(_) => schema_entries.push((key.as_str(), value)),
            _ => {
                return Err(malformed(
                    &child_ptr(&keyword_ptr, key),
                    "dependencies",
                    "a dependencies entry must be an array or a schema",
                ))
            }
        }
    }
    let groups = required_name_groups("dependencies", &array_entries, &keyword_ptr)?;
    Ok((groups, schema_entries))
}

fn required_name_groups(
    keyword: &'static str,
    entries: &[(String, Ast)],
    keyword_ptr: &str,
) -> Result<Vec<(String, Vec<String>)>, CompileError> {
    let mut triggers = FxHashSet::default();
    triggers
        .try_reserve(entries.len())
        .map_err(|_| reserve_amplification())?;
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(entries.len())
        .map_err(|_| reserve_amplification())?;
    let mut edge_count = 0usize;
    for (trigger, value) in entries {
        let trigger_ptr = child_ptr(keyword_ptr, trigger);
        if !triggers.insert(trigger.as_str()) {
            return Err(malformed(
                &trigger_ptr,
                keyword,
                "duplicate dependency trigger",
            ));
        }
        let Ast::Arr(required) = value else {
            return Err(malformed(
                &trigger_ptr,
                keyword,
                "dependency entry must be an array",
            ));
        };
        let mut names = FxHashSet::default();
        names
            .try_reserve(required.len())
            .map_err(|_| reserve_amplification())?;
        let mut owned_names = Vec::new();
        owned_names
            .try_reserve_exact(required.len())
            .map_err(|_| reserve_amplification())?;
        for (index, item) in required.iter().enumerate() {
            let item_ptr = child_ptr(&trigger_ptr, &index.to_string());
            let Ast::Str(required_name) = item else {
                return Err(malformed(
                    &item_ptr,
                    keyword,
                    "dependency names must be strings",
                ));
            };
            if !names.insert(required_name.as_str()) {
                return Err(malformed(&item_ptr, keyword, "duplicate dependency name"));
            }
            edge_count = edge_count
                .checked_add(1)
                .filter(|&n| n <= crate::ir::MAX_DEPENDENT_REQUIRED_PAIRS)
                .ok_or_else(reserve_amplification)?;
            owned_names.push(required_name.clone());
        }
        if !owned_names.is_empty() {
            groups.push((trigger.clone(), owned_names));
        }
    }
    Ok(groups)
}

fn scalar_lit(ast: &Ast) -> Option<ScalarLit> {
    match ast {
        Ast::Null => Some(ScalarLit::Null),
        Ast::Bool(b) => Some(ScalarLit::Bool(*b)),
        Ast::Str(s) => Some(ScalarLit::Str(s.clone())),
        Ast::Num(tok) => numeric_token_as_i64(tok).map(ScalarLit::Int),
        Ast::Arr(_) | Ast::Obj(_) => Some(ScalarLit::Json(canonical_json(ast))),
    }
}

fn literal_reason(ast: &Ast) -> UnsupportedReason {
    match ast {
        Ast::Num(_) => UnsupportedReason::NonIntegerNumericLiteral,
        _ => UnsupportedReason::NonScalarLiteral,
    }
}

fn json_type_of(lit: &ScalarLit) -> &'static str {
    match lit {
        ScalarLit::Null => "null",
        ScalarLit::Bool(_) => "boolean",
        ScalarLit::Int(_) => "integer",
        ScalarLit::Str(_) => "string",
        ScalarLit::Json(s) if s.starts_with('[') => "array",
        ScalarLit::Json(_) => "object",
    }
}

fn type_matches(ty: &str, lit: &ScalarLit) -> bool {
    json_type_of(lit) == ty || (ty == "number" && matches!(lit, ScalarLit::Int(_)))
}

fn numeric_literal_matches_types(token: &str, types: &[&str]) -> bool {
    types.is_empty()
        || types
            .iter()
            .any(|ty| *ty == "number" || (*ty == "integer" && numeric_token_is_integer(token)))
}

fn numeric_type_requires_integer(types: &[&str]) -> bool {
    !types.is_empty() && types.contains(&"integer") && !types.contains(&"number")
}

fn canonical_json(ast: &Ast) -> String {
    let mut out = String::new();
    write_canonical(ast, &mut out);
    out
}

fn write_canonical(ast: &Ast, out: &mut String) {
    match ast {
        Ast::Null => out.push_str("null"),
        Ast::Bool(true) => out.push_str("true"),
        Ast::Bool(false) => out.push_str("false"),
        Ast::Num(tok) => out.push_str(tok),
        Ast::Str(s) => write_json_string(s, out),
        Ast::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Ast::Obj(fields) => {
            out.push('{');
            for (i, (key, value)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(key, out);
                out.push(':');
                write_canonical(value, out);
            }
            out.push('}');
        }
    }
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CountBound {
    Fits(u32),
    AboveU32,
}

fn digit_count(
    fields: &[(String, Ast)],
    key: &str,
    ptr: &str,
) -> Result<Option<u32>, CompileError> {
    match count_field(fields, key, ptr)? {
        Some(CountBound::Fits(value)) => Ok(Some(value)),
        Some(CountBound::AboveU32) => Err(limit("numeric digit count")),
        None => Ok(None),
    }
}

fn digit_quantifier(minimum: u32, maximum: Option<u32>) -> String {
    match maximum {
        Some(maximum) if minimum == maximum => format!("{{{minimum}}}"),
        Some(maximum) => format!("{{{minimum},{maximum}}}"),
        None => format!("{{{minimum},}}"),
    }
}

fn optional_digit_part(prefix: &str, range: (u32, Option<u32>)) -> String {
    format!("(?:{prefix}[0-9]{})?", digit_quantifier(range.0, range.1))
}

fn count_field(
    fields: &[(String, Ast)],
    key: &str,
    ptr: &str,
) -> Result<Option<CountBound>, CompileError> {
    match get(fields, key) {
        None => Ok(None),
        Some(Ast::Num(token)) => {
            if let Some(value) = numeric_token_as_i64(token) {
                if value < 0 {
                    return Err(malformed(
                        &child_ptr(ptr, key),
                        key,
                        "count must be non-negative",
                    ));
                }
                return Ok(Some(match u32::try_from(value) {
                    Ok(value) => CountBound::Fits(value),
                    Err(_) => CountBound::AboveU32,
                }));
            }
            if token.starts_with('-') && !numeric_token_is_zero(token) {
                return Err(malformed(
                    &child_ptr(ptr, key),
                    key,
                    "count must be non-negative",
                ));
            }
            if numeric_token_is_integer(token) {
                Ok(Some(CountBound::AboveU32))
            } else {
                Err(malformed(
                    &child_ptr(ptr, key),
                    key,
                    "count must be an integer",
                ))
            }
        }
        Some(_) => Err(malformed(
            &child_ptr(ptr, key),
            key,
            "count must be a number",
        )),
    }
}

fn numeric_token_is_zero(token: &str) -> bool {
    token
        .split_once(['e', 'E'])
        .map_or(token, |(mantissa, _)| mantissa)
        .bytes()
        .filter(|byte| byte.is_ascii_digit())
        .all(|byte| byte == b'0')
}

fn numeric_token_is_integer(token: &str) -> bool {
    if numeric_token_is_zero(token) {
        return true;
    }
    let rest = token.strip_prefix('-').unwrap_or(token);
    let (mantissa, exponent) = rest.split_once(['e', 'E']).unwrap_or((rest, "0"));
    let exponent = match exponent.parse::<i64>() {
        Ok(value) => value,
        Err(_) => return !exponent.starts_with('-'),
    };
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let coefficient_len = match integer.len().checked_add(fraction.len()) {
        Some(value) => value,
        None => return false,
    };
    let point = match i64::try_from(integer.len())
        .ok()
        .and_then(|value| value.checked_add(exponent))
    {
        Some(value) => value,
        None => return exponent.is_positive(),
    };
    let coefficient_len_i64 = match i64::try_from(coefficient_len) {
        Ok(value) => value,
        Err(_) => return point >= 0,
    };
    if point >= coefficient_len_i64 {
        return true;
    }
    let start = match usize::try_from(point.max(0)) {
        Ok(value) => value,
        Err(_) => return false,
    };
    integer
        .bytes()
        .chain(fraction.bytes())
        .skip(start)
        .all(|byte| byte == b'0')
}

fn decimal_token_scaled(token: &str) -> Option<(i128, u32)> {
    let (neg, rest) = token
        .strip_prefix('-')
        .map_or((false, token), |r| (true, r));
    let (mantissa, exponent) = rest.split_once(['e', 'E']).unwrap_or((rest, "0"));
    let exponent: i64 = exponent.parse().ok()?;
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut digits = String::new();
    digits
        .try_reserve_exact(int_part.len().checked_add(frac_part.len())?)
        .ok()?;
    digits.push_str(int_part);
    digits.push_str(frac_part);
    let mut mag: i128 = digits.parse().ok()?;
    let mut scale = i64::try_from(frac_part.len()).ok()?.checked_sub(exponent)?;
    if mag == 0 {
        return Some((0, 0));
    }
    if scale < 0 {
        let shift = usize::try_from(scale.checked_neg()?).ok()?;
        let significant_digits = digits.trim_start_matches('0').len();
        if significant_digits.checked_add(shift)? > 39 {
            return None;
        }
        for _ in 0..shift {
            mag = mag.checked_mul(10)?;
        }
        scale = 0;
    }
    let mut scale = u32::try_from(scale).ok()?;
    while scale > 0 && mag % 10 == 0 {
        mag /= 10;
        scale -= 1;
    }
    Some((if neg { -mag } else { mag }, scale))
}

fn decimal_token_as_coef_exp(token: &str) -> Option<(u64, u32)> {
    if token.starts_with('-') {
        return None;
    }
    let (mantissa, exponent) = token.split_once(['e', 'E']).unwrap_or((token, "0"));
    let exponent: i64 = exponent.parse().ok()?;
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut coef: u64 = format!("{int_part}{frac_part}").parse().ok()?;
    let mut exp = i64::try_from(frac_part.len()).ok()?.checked_sub(exponent)?;
    if exp < 0 {
        for _ in 0..usize::try_from(-exp).ok()? {
            coef = coef.checked_mul(10)?;
        }
        exp = 0;
    }
    let mut exp = u32::try_from(exp).ok()?;
    while exp > 0 && coef % 10 == 0 {
        coef /= 10;
        exp -= 1;
    }
    Some((coef, exp))
}

fn fast_integer_bound(bound: Option<DecimalBound>, lower: bool) -> Option<Option<i64>> {
    let Some((value, scale, exclusive)) = bound else {
        return Some(None);
    };
    if scale != 0 {
        return None;
    }
    let adjusted = match (lower, exclusive) {
        (true, true) => value.checked_add(1)?,
        (false, true) => value.checked_sub(1)?,
        _ => value,
    };
    i64::try_from(adjusted).ok().map(Some)
}

fn fast_integer_multiple(value: Option<(u64, u32)>) -> Option<Option<u64>> {
    match value {
        Some((_, scale)) if scale != 0 => None,
        Some((coefficient, _)) => Some(Some(coefficient)),
        None => Some(None),
    }
}

fn numeric_token_as_i64(token: &str) -> Option<i64> {
    let (neg, rest) = token
        .strip_prefix('-')
        .map_or((false, token), |r| (true, r));
    let (mantissa, exp_str) = rest.split_once(['e', 'E']).unzip();
    let mantissa = mantissa.unwrap_or(rest);
    let exp: i64 = match exp_str {
        None => 0,
        Some(e) => e.parse().ok()?,
    };
    if exp.checked_abs()?.gt(&1000) {
        return None;
    }
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut digits = String::with_capacity(int_part.len() + frac_part.len());
    digits.push_str(int_part);
    digits.push_str(frac_part);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let point_pos = i64::try_from(int_part.len()).ok()?.checked_add(exp)?;
    let dlen = i64::try_from(digits.len()).ok()?;
    if point_pos < dlen {
        let frac_start = usize::try_from(point_pos.max(0)).ok()?;
        if digits.as_bytes()[frac_start..].iter().any(|&b| b != b'0') {
            return None; // a genuinely nonzero fractional part: not an integer
        }
    }
    let int_digit_count = point_pos.max(0);
    if int_digit_count > 19 {
        return None; // more digits than any i64 magnitude can hold
    }
    let int_digit_count = usize::try_from(int_digit_count).ok()?;
    let mut int_digits = String::with_capacity(int_digit_count);
    if point_pos > 0 {
        let take = usize::try_from(point_pos).ok()?.min(digits.len());
        int_digits.push_str(&digits[..take]);
        for _ in 0..(int_digit_count - int_digits.len()) {
            int_digits.push('0');
        }
    }
    let magnitude: i128 = if int_digits.is_empty() {
        0
    } else {
        int_digits.parse().ok()?
    };
    i64::try_from(if neg { -magnitude } else { magnitude }).ok()
}

const ANNOTATION_KEYWORDS: &[&str] = &[
    "title",
    "description",
    "$comment",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$schema",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$defs",
    "definitions",
    "$vocabulary",
];

fn collect_used_refs(ast: &Ast, out: &mut Vec<String>) {
    match ast {
        Ast::Obj(fields) => {
            for (k, v) in fields {
                if k == "$ref" {
                    if let Ast::Str(target) = v {
                        out.push(target.clone());
                    }
                } else if k != "$defs" && k != "definitions" {
                    collect_used_refs(v, out);
                }
            }
        }
        Ast::Arr(items) => items.iter().for_each(|it| collect_used_refs(it, out)),
        _ => {}
    }
}

fn has_base_assertion(fields: &[(String, Ast)]) -> bool {
    fields.iter().any(|(k, _)| {
        !matches!(
            k.as_str(),
            "$ref" | "$dynamicRef" | "anyOf" | "allOf" | "oneOf" | "not" | "if" | "then" | "else"
        ) && !ANNOTATION_KEYWORDS.contains(&k.as_str())
    })
}

fn get<'a>(fields: &'a [(String, Ast)], key: &str) -> Option<&'a Ast> {
    fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn obj_has(fields: &[(String, Ast)], key: &str) -> bool {
    fields.iter().any(|(k, _)| k == key)
}

fn child_ptr(ptr: &str, segment: &str) -> String {
    format!("{ptr}/{}", ptr_escape(segment))
}

fn ptr_escape(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

fn malformed(ptr: &str, keyword: &str, message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L1, message)
        .with_pointer(ptr.to_string())
        .with_keyword(keyword.to_string())
}

fn limit(what: &'static str) -> CompileError {
    CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L1, what)
}

fn expanded_node_limit(pointer: &str) -> CompileError {
    let mut error = CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L1,
        "expanded schema-node budget exceeded",
    )
    .with_pointer(pointer.to_string())
    .with_observed("lowering expansion at this schema location");
    error.limit = Some((
        crate::error::LimitKind::NodeCount,
        usize::try_from(MAX_EXPANDED_NODES)
            .unwrap_or(usize::MAX)
            .saturating_add(1),
        usize::try_from(MAX_EXPANDED_NODES).unwrap_or(usize::MAX),
    ));
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ir(text: &str) -> Result<SchemaIR, CompileError> {
        schema_to_ir(text, CompileOptions::default())
    }

    enum Matcher<'a> {
        Regular(Box<crate::automaton::RefEngine>),
        Structured(&'a SchemaIR),
    }

    impl Matcher<'_> {
        fn accepts(&self, bytes: &[u8]) -> bool {
            match self {
                Matcher::Regular(e) => e.accepts(bytes),
                Matcher::Structured(ir) => crate::structured::accepts(ir, bytes),
            }
        }

        fn start(&self) -> crate::primitives::StateId {
            match self {
                Matcher::Regular(e) => e.start(),
                Matcher::Structured(_) => unimplemented!(
                    "start state has no structured-backend equivalent in these tests"
                ),
            }
        }

        fn is_dead(&self, state: crate::primitives::StateId) -> bool {
            match self {
                Matcher::Regular(e) => e.is_dead(state),
                Matcher::Structured(_) => {
                    unimplemented!("is_dead has no structured-backend equivalent in these tests")
                }
            }
        }
    }

    fn compile_either(d: &SchemaIR) -> Matcher<'_> {
        if d.requires_structured_backend() {
            Matcher::Structured(d)
        } else {
            Matcher::Regular(Box::new(crate::compile::compile_ir(d).unwrap()))
        }
    }

    fn closed() -> CompileOptions {
        CompileOptions {
            object_closure: ObjectClosure::AssumeClosedProfile,
            ..CompileOptions::default()
        }
    }

    fn strict() -> CompileOptions {
        CompileOptions {
            object_closure: ObjectClosure::RejectOpenObjects,
            ..CompileOptions::default()
        }
    }

    #[test]
    fn lowers_scalars() {
        assert!(ir(r#"{"type":"null"}"#).is_ok());
        assert!(ir(r#"{"type":"boolean"}"#).is_ok());
        assert!(ir(r#"{"type":"integer"}"#).is_ok());
    }

    #[test]
    fn contradictory_cardinality_bounds_compile_to_the_empty_language() {
        for schema in [
            r#"{"type":"array","minItems":2,"maxItems":1}"#,
            r#"{"type":"array","contains":{"const":1},"minContains":3,"maxContains":1}"#,
            r#"{"type":"object","minProperties":2,"maxProperties":1}"#,
            r#"{"type":"array","prefixItems":[true],"items":false,"minItems":2}"#,
        ] {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "{schema}");
            let e = compile_either(&d);
            assert!(!e.accepts(b"null"), "{schema}");
            assert!(!e.accepts(b"[]"), "{schema}");
            assert!(!e.accepts(b"{}"), "{schema}");
        }
    }

    #[test]
    fn oversized_counting_bounds_apply_keyword_specific_semantics() {
        for schema in [
            r#"{"type":"string","minLength":1e100000}"#,
            r#"{"type":"array","minItems":4294967296}"#,
            r#"{"type":"object","minProperties":18446744073709551616}"#,
            r#"{"type":"array","contains":true,"minContains":1e100000}"#,
        ] {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "{schema}");
            let e = compile_either(&d);
            assert!(e.is_dead(e.start()), "{schema}");
        }

        for (schema, instance) in [
            (
                r#"{"type":"string","maxLength":1e100000}"#,
                br#""ok""#.as_slice(),
            ),
            (r#"{"type":"array","maxItems":4294967296}"#, b"[1,2]"),
            (
                r#"{"type":"object","maxProperties":18446744073709551616}"#,
                br#"{"a":1}"#,
            ),
            (
                r#"{"type":"array","contains":true,"maxContains":1e100000}"#,
                b"[1,2]",
            ),
        ] {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "{schema}");
            assert!(compile_either(&d).accepts(instance), "{schema}");
        }
    }

    #[test]
    fn negative_and_fractional_counting_bounds_are_malformed() {
        for schema in [
            r#"{"type":"string","minLength":-1}"#,
            r#"{"type":"array","maxItems":1.5}"#,
            r#"{"type":"object","minProperties":1e-1}"#,
        ] {
            assert_eq!(
                ir(schema).unwrap_err().code,
                ErrorCode::Malformed,
                "{schema}"
            );
        }
        assert!(ir(r#"{"type":"string","minLength":-0}"#).is_ok());
    }

    #[test]
    fn unicode_letter_patterns_match_decoded_scalars() {
        let string = ir(r#"{"type":"string","pattern":"^\\p{Letter}+$"}"#).unwrap();
        assert_eq!(string.diagnostics().len(), 0);
        for value in [
            br#""abc""#.as_slice(),
            br#""caf\u00e9""#,
            "\"café\"".as_bytes(),
        ] {
            assert!(crate::structured::accepts(&string, value), "{value:?}");
        }
        assert!(!crate::structured::accepts(&string, br#""abc1""#));

        let properties = ir(
            r#"{"type":"object","patternProperties":{"^\\p{Letter}+$":{"type":"number"}},"additionalProperties":false}"#,
        )
        .unwrap();
        assert!(crate::structured::accepts(
            &properties,
            br#"{"caf\u00e9":1}"#
        ));
        assert!(!crate::structured::accepts(&properties, br#"{"abc1":1}"#));
    }

    #[test]
    fn integer_bounds_normalize_exclusive() {
        let ir = ir(r#"{"type":"integer","exclusiveMinimum":0,"maximum":10}"#).unwrap();
        assert_eq!(ir.node_count(), 1);
    }

    #[test]
    fn integer_multiple_of_intersects_with_the_range_end_to_end() {
        let d = ir(r#"{"type":"integer","minimum":0,"maximum":100,"multipleOf":10}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        for good in ["0", "10", "50", "100"] {
            assert!(e.accepts(good.as_bytes()), "should accept {good}");
        }
        for bad in ["5", "15", "110", "-10"] {
            assert!(!e.accepts(bad.as_bytes()), "should reject {bad}");
        }
    }

    #[test]
    fn integer_multiple_of_zero_or_negative_is_malformed() {
        assert_eq!(
            ir(r#"{"type":"integer","multipleOf":0}"#).unwrap_err().code,
            ErrorCode::Malformed
        );
        assert_eq!(
            ir(r#"{"type":"integer","multipleOf":-2}"#)
                .unwrap_err()
                .code,
            ErrorCode::Malformed
        );
    }

    #[test]
    fn integer_multiple_of_a_fractional_divisor_uses_exact_value_semantics() {
        let d = ir(r#"{"type":"integer","multipleOf":0.5}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"1e2"));
        assert!(!e.accepts(b"1.5"));
    }

    #[test]
    fn unbounded_number_is_supported_and_matches_the_json_numeric_grammar() {
        let d = ir(r#"{"type":"number"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        for good in [
            "0", "-0", "12", "-3", "3.14", "-2.5", "1e9", "1E+9", "0.5", "2.0e-3",
        ] {
            assert!(e.accepts(good.as_bytes()), "should accept {good}");
        }
        for bad in ["", "01", "1.", ".5", "+1", "1e", "abc", "--1", "1.2.3"] {
            assert!(!e.accepts(bad.as_bytes()), "should reject {bad}");
        }
    }

    #[test]
    fn nonnegative_fractional_bounds_are_now_supported() {
        let d = ir(r#"{"type":"number","minimum":0.5}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"0.5"));
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"12.34"));
        assert!(!e.accepts(b"0.4"));
        assert!(!e.accepts(b"-1"), "below a positive lower bound");

        let d = ir(r#"{"type":"number","maximum":9.9}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"9.9"));
        assert!(
            e.accepts(b"-100.5"),
            "any negative is <= a non-negative max"
        );
        assert!(e.accepts(b"0"));
        assert!(!e.accepts(b"9.91"));
        assert!(!e.accepts(b"10"));
    }

    #[test]
    fn decimal_bounds_end_to_end_cover_real_corpus_values() {
        let d = ir(r#"{"type":"number","maximum":99999999999.99}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"99999999999.99"));
        assert!(e.accepts(b"12345.67"));
        assert!(e.accepts(b"-999999999999"));
        assert!(
            !e.accepts(b"99999999999.999"),
            "beyond two decimal places, above cap"
        );
        assert!(!e.accepts(b"100000000000"));

        let d = ir(r#"{"type":"number","minimum":2.3,"maximum":2.7}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"2.3"));
        assert!(e.accepts(b"2.5"));
        assert!(e.accepts(b"2.700"));
        assert!(!e.accepts(b"2.2"));
        assert!(!e.accepts(b"2.71"));

        let d = ir(r#"{"type":"number","minimum":0.25,"maximum":10}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"0.25"));
        assert!(e.accepts(b"3.5"));
        assert!(e.accepts(b"10"));
        assert!(!e.accepts(b"0.24"));
        assert!(!e.accepts(b"10.1"));
    }

    #[test]
    fn exclusive_and_negative_fractional_bounds_are_exact() {
        for (schema, accepted, rejected) in [
            (
                r#"{"type":"number","minimum":-2.5}"#,
                b"-2.5".as_slice(),
                b"-2.5001".as_slice(),
            ),
            (
                r#"{"type":"number","exclusiveMinimum":1.5}"#,
                b"1.5001".as_slice(),
                b"1.5".as_slice(),
            ),
            (
                r#"{"type":"number","maximum":-2.5}"#,
                b"-2.5".as_slice(),
                b"-2.4999".as_slice(),
            ),
            (
                r#"{"type":"number","exclusiveMaximum":1.5}"#,
                b"1.4999".as_slice(),
                b"1.5".as_slice(),
            ),
        ] {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "{schema}");
            let e = compile_either(&d);
            assert!(e.accepts(accepted), "{schema}");
            assert!(!e.accepts(rejected), "{schema}");
        }
    }

    #[test]
    fn number_with_an_integer_valued_bound_is_supported() {
        let d = ir(r#"{"type":"number","minimum":0,"maximum":10}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"0"));
        assert!(e.accepts(b"0.5"));
        assert!(e.accepts(b"10"));
        assert!(e.accepts(b"10.0"));
        assert!(e.accepts(b"9.999"));
        assert!(!e.accepts(b"-0.1"));
        assert!(!e.accepts(b"10.1"));
        assert!(!e.accepts(b"11"));
    }

    #[test]
    fn decimal_multiple_of_end_to_end_covers_the_real_corpus_divisors() {
        let d = ir(r#"{"type":"number","multipleOf":0.01}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"1.23"));
        assert!(e.accepts(b"1.20"));
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"-0.99"));
        assert!(!e.accepts(b"1.234"));
        assert!(!e.accepts(b"0.005"));

        let d = ir(r#"{"type":"number","multipleOf":0.25}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"3"));
        assert!(e.accepts(b"3.5"));
        assert!(e.accepts(b"0.75"));
        assert!(e.accepts(b"-1.25"));
        assert!(!e.accepts(b"0.3"));
        assert!(!e.accepts(b"3.1"));

        let d = ir(r#"{"type":"number","multipleOf":100}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"100"));
        assert!(e.accepts(b"100.0"));
        assert!(e.accepts(b"-200"));
        assert!(e.accepts(b"0"));
        assert!(!e.accepts(b"100.5"));
        assert!(!e.accepts(b"150"));
    }

    #[test]
    fn large_divisors_use_the_bounded_structured_number_path() {
        for schema in [
            r#"{"type":"number","multipleOf":16}"#,
            r#"{"type":"number","multipleOf":256}"#,
        ] {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "{schema}");
            let e = compile_either(&d);
            assert!(e.accepts(b"256"), "{schema}");
        }
    }

    #[test]
    fn multiple_of_intersects_with_a_number_range() {
        let d = ir(r#"{"type":"number","minimum":0,"maximum":1,"multipleOf":0.25}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"0"));
        assert!(e.accepts(b"0.25"));
        assert!(e.accepts(b"1"));
        assert!(!e.accepts(b"0.3"), "not a multiple of 0.25");
        assert!(!e.accepts(b"1.25"), "outside [0,1]");
    }

    #[test]
    fn typeless_multiple_of_is_now_supported() {
        let d = ir(r#"{"multipleOf":0.01}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, b"1.23"));
        assert!(!crate::structured::accepts(&d, b"1.234"));
        assert!(crate::structured::accepts(
            &d,
            br#""a string is unconstrained""#
        ));
    }

    #[test]
    fn exponent_form_multiple_of_is_exact() {
        let d = ir(r#"{"type":"number","multipleOf":1e-2}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"1.23"));
        assert!(!e.accepts(b"1.234"));
    }

    #[test]
    fn number_exclusive_bounds_exclude_exactly_the_boundary() {
        let d = ir(r#"{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":10}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(!e.accepts(b"0"), "0 itself is excluded");
        assert!(e.accepts(b"0.1"));
        assert!(!e.accepts(b"10"), "10 itself is excluded");
        assert!(e.accepts(b"9.9"));
    }

    #[test]
    fn number_range_spanning_zero_accepts_negative_fractions_near_zero() {
        let d = ir(r#"{"type":"number","minimum":-1,"maximum":1}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(
            e.accepts(b"-0.5"),
            "a fraction strictly between -1 and 0 must be accepted"
        );
        assert!(e.accepts(b"-1"));
        assert!(e.accepts(b"0.9"));
        assert!(!e.accepts(b"-1.1"));
        assert!(!e.accepts(b"1.1"));
    }

    #[test]
    fn number_minimum_and_exclusive_minimum_combine_to_the_stricter_bound() {
        let d = ir(r#"{"type":"number","minimum":0,"exclusiveMinimum":5}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(!e.accepts(b"5"), "exclusiveMinimum:5 is the stricter bound");
        assert!(e.accepts(b"5.1"));
        assert!(!e.accepts(b"3"));
    }

    #[test]
    fn number_bounds_forming_an_empty_range_are_malformed() {
        let e = ir(r#"{"type":"number","minimum":5,"exclusiveMaximum":5}"#).unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn matches_the_official_test_suite_minimum_and_maximum_groups_exactly() {
        let cases: &[(&str, &[(&[u8], bool)])] = &[
            (
                r#"{"type":"number","minimum":-2}"#,
                &[
                    (b"-1" as &[u8], true),
                    (b"0", true),
                    (b"-2", true),
                    (b"-2.0", true),
                    (b"-2.0001", false),
                    (b"-3", false),
                ],
            ),
            (
                r#"{"type":"number","maximum":3.0}"#,
                &[(b"2.6", true), (b"3.0", true), (b"3.5", false)],
            ),
            (
                r#"{"type":"number","maximum":300}"#,
                &[
                    (b"299.97", true),
                    (b"300", true),
                    (b"300.00", true),
                    (b"300.5", false),
                ],
            ),
            (
                r#"{"type":"number","exclusiveMaximum":3.0}"#,
                &[(b"2.2", true), (b"3.0", false), (b"3.5", false)],
            ),
        ];
        for (schema, group) in cases {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "{schema}");
            let e = compile_either(&d);
            for (data, expected) in *group {
                assert_eq!(e.accepts(data), *expected, "schema={schema} data={data:?}");
            }
        }
    }

    #[test]
    fn allof_is_real_intersection() {
        let d = ir(r#"{"allOf":[{"type":"null"}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"null"));
        assert!(!e.accepts(b"true"));
    }

    #[test]
    fn allof_with_a_dynamic_branch_is_lowered_not_dropped() {
        let d = ir(r##"{"$dynamicAnchor":"x","allOf":[{"$dynamicRef":"#x"}]}"##).unwrap();
        assert_eq!(d.diagnostics().count(), 0);
        assert!(d.requires_structured_backend());
    }

    #[test]
    fn dynamic_ref_to_a_static_anchor_lowers_without_runtime_dynamic_metadata() {
        let d = ir(r##"{"$dynamicRef":"#x","$defs":{"target":{"$anchor":"x","type":"integer"}}}"##)
            .unwrap();
        assert!(!d
            .nodes()
            .any(|node| matches!(node, crate::ir::Node::DynamicRef { .. })));
        let executor = compile_either(&d);
        assert!(executor.accepts(b"1"));
        assert!(!executor.accepts(b"false"));
    }

    #[test]
    fn allof_intersects_two_integer_ranges() {
        let d = ir(
            r#"{"allOf":[{"type":"integer","minimum":0,"maximum":10},{"type":"integer","minimum":5,"maximum":20}]}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        for n in -5i64..=25 {
            let want = (0..=10).contains(&n) && (5..=20).contains(&n);
            assert_eq!(e.accepts(n.to_string().as_bytes()), want, "n={n}");
        }
    }

    #[test]
    fn allof_of_disjoint_ranges_compiles_to_the_empty_language() {
        let d = ir(
            r#"{"allOf":[{"type":"integer","minimum":0,"maximum":5},{"type":"integer","minimum":10,"maximum":20}]}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        for n in -5i64..=25 {
            assert!(!e.accepts(n.to_string().as_bytes()), "n={n}");
        }
        assert!(e.is_dead(e.start()));
    }

    #[test]
    fn allof_self_intersection_is_the_same_language() {
        let d = ir(r#"{"allOf":[{"enum":["a","b"]},{"enum":["a","b"]}]}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"\"a\""));
        assert!(e.accepts(b"\"b\""));
        assert!(!e.accepts(b"\"c\""));
    }

    #[test]
    fn nested_allof_compiles() {
        let d = ir(r#"{"allOf":[{"allOf":[{"type":"null"}]}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"null"));
        assert!(!e.accepts(b"true"));
    }

    #[test]
    fn allof_containing_an_anyof_branch_compiles() {
        let d = ir(
            r#"{"allOf":[{"anyOf":[{"type":"integer","minimum":0,"maximum":10},{"type":"string"}]},{"type":"integer","minimum":5,"maximum":20}]}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        for n in -5i64..=25 {
            let want = (0..=10).contains(&n) && (5..=20).contains(&n);
            assert_eq!(e.accepts(n.to_string().as_bytes()), want, "n={n}");
        }
        assert!(
            !e.accepts(b"\"x\""),
            "the type:string anyOf branch never satisfies the integer allOf branch"
        );
    }

    #[test]
    fn anyof_accepts_a_value_of_any_branch() {
        let d = ir(r#"{"anyOf":[{"type":"string"},{"type":"null"}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"\"x\""));
        assert!(e.accepts(b"null"));
        assert!(!e.accepts(b"true"));
    }

    #[test]
    fn anyof_with_an_unsupported_branch_is_unsupported_not_dropped() {
        let d = ir(r#"{"anyOf":[{"type":"string","pattern":"(?=a)"},{"type":"number"}]}"#).unwrap();
        assert_eq!(
            d.diagnostics().next().unwrap().reason,
            UnsupportedReason::RegexAssertionUnsupported
        );
    }

    #[test]
    fn anyof_with_overlapping_languages_still_accepts_the_union() {
        let d = ir(r#"{"anyOf":[{"enum":["a","b"]},{"enum":["b","c"]}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"\"a\""));
        assert!(e.accepts(b"\"b\""));
        assert!(e.accepts(b"\"c\""));
        assert!(!e.accepts(b"\"d\""));
    }

    #[test]
    fn anyof_with_colliding_integer_ranges_still_accepts_the_union() {
        let d = ir(r#"{"anyOf":[{"type":"integer","minimum":0,"maximum":5},{"type":"integer","minimum":3,"maximum":10}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        for n in 0..=10 {
            let want = (0..=5).contains(&n) || (3..=10).contains(&n);
            assert_eq!(
                e.accepts(n.to_string().as_bytes()),
                want,
                "n={n} anyOf([0,5],[3,10])"
            );
        }
        assert!(!e.accepts(b"11"));
    }

    #[test]
    fn nested_anyof_inside_a_property_compiles() {
        use crate::compile::compile_ir;
        let e = compile_ir(
            &ir(r#"{"type":"object","properties":{"a":{"anyOf":[{"type":"null"},{"type":"boolean"}]}},"required":["a"],"additionalProperties":false}"#)
                .unwrap(),
        )
        .unwrap();
        assert!(e.accepts(br#"{"a":null}"#));
        assert!(e.accepts(br#"{"a":true}"#));
        assert!(!e.accepts(br#"{"a":1}"#));
    }

    #[test]
    fn anyof_empty_array_is_malformed() {
        assert_eq!(
            ir(r#"{"anyOf":[]}"#).unwrap_err().code,
            ErrorCode::Malformed
        );
    }

    #[test]
    fn anyof_single_branch_collapses_and_stays_supported() {
        let d = ir(r#"{"anyOf":[{"type":"boolean"}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert_eq!(d.node_count(), 1);
    }

    #[test]
    fn oneof_accepts_a_value_matching_exactly_one_branch() {
        let d = ir(
            r#"{"oneOf":[{"type":"integer","minimum":-100,"maximum":1},{"type":"integer","minimum":2,"maximum":100}]}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"1"), "matches only the first branch");
        assert!(e.accepts(b"2"), "matches only the second branch");
        assert!(!e.accepts(b"5000"), "matches neither branch");
    }

    #[test]
    fn oneof_rejects_a_value_matching_more_than_one_branch() {
        let d = ir(
            r#"{"oneOf":[{"type":"integer","minimum":0,"maximum":10},{"type":"integer","minimum":5,"maximum":20}]}"#,
        )
        .unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"0"), "only the first branch matches");
        assert!(e.accepts(b"15"), "only the second branch matches");
        assert!(!e.accepts(b"7"), "both branches match: must be rejected");
    }

    #[test]
    fn oneof_rejects_a_value_matching_two_identical_branches() {
        let d = ir(r#"{"oneOf":[{"type":"integer","minimum":0,"maximum":9},{"type":"integer","minimum":0,"maximum":9}]}"#)
            .unwrap();
        let e = compile_either(&d);
        for n in 0..=9 {
            assert!(!e.accepts(n.to_string().as_bytes()), "n={n}");
        }
    }

    #[test]
    fn oneof_with_a_dynamic_branch_is_lowered_not_dropped() {
        let d = ir(r##"{"$dynamicAnchor":"x","oneOf":[{"$dynamicRef":"#x"},{"type":"null"}]}"##)
            .unwrap();
        assert_eq!(d.diagnostics().count(), 0);
        assert!(d.requires_structured_backend());
    }

    #[test]
    fn oneof_single_branch_collapses_and_stays_supported() {
        let d = ir(r#"{"oneOf":[{"type":"boolean"}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert_eq!(d.node_count(), 1);
    }

    #[test]
    fn nested_oneof_compiles() {
        let d = ir(r#"{"oneOf":[{"oneOf":[{"type":"null"}]}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"null"));
        assert!(!e.accepts(b"true"));
    }

    #[test]
    fn oneof_empty_array_is_malformed() {
        assert_eq!(
            ir(r#"{"oneOf":[]}"#).unwrap_err().code,
            ErrorCode::Malformed
        );
    }

    #[test]
    fn open_object_is_unsupported_unless_closed_profile() {
        let schema = r#"{"type":"object","properties":{"a":{"type":"null"}}}"#;
        let open = ir(schema).unwrap();
        assert_eq!(open.diagnostics().len(), 0, "default is spec-correct open");
        assert!(crate::structured::accepts(
            &open,
            br#"{"a":null,"extra":1}"#
        ));
        let strict_ir = schema_to_ir(schema, strict()).unwrap();
        assert_eq!(
            strict_ir.diagnostics().next().unwrap().reason,
            UnsupportedReason::OpenAdditionalProperties
        );
        let closed_ir = schema_to_ir(schema, closed()).unwrap();
        assert_eq!(closed_ir.diagnostics().len(), 0);
    }

    #[test]
    fn explicit_additional_properties_false_is_closed() {
        let ir = ir(
            r#"{"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"type":"boolean"}}}"#,
        )
        .unwrap();
        assert_eq!(ir.diagnostics().len(), 0);
    }

    #[test]
    fn duplicate_required_name_is_malformed() {
        let e = ir(r#"{"type":"object","additionalProperties":false,"required":["a","a"]}"#)
            .unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    fn composite_enum_member_is_filtered_out_by_a_declared_scalar_type() {
        let d = ir(r#"{"type":"string","enum":["a",{"x":1}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"\"a\""));
        assert!(!e.accepts(br#"{"x":1}"#));
    }

    #[test]
    fn composite_const_object_accepts_whitespace_but_not_reordered_keys() {
        let d = ir(r#"{"const":{"foo":"bar","baz":"bax"}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#"{"foo":"bar","baz":"bax"}"#));
        assert!(!e.accepts(br#"{"baz":"bax","foo":"bar"}"#));
        assert!(e.accepts(br#"{"foo": "bar", "baz": "bax"}"#));
    }

    #[test]
    fn composite_const_array_accepts_whitespace_variants_of_its_canonical_serialization() {
        let d = ir(r#"{"const":[{"foo":"bar"}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#"[{"foo":"bar"}]"#));
        assert!(e.accepts(br#"[{"foo": "bar"}]"#));
    }

    #[test]
    fn composite_enum_accepts_each_member_canonically() {
        let d = ir(r#"{"enum":[{"x":1},{"y":2},[3,4]]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#"{"x":1}"#));
        assert!(e.accepts(br#"{"y":2}"#));
        assert!(e.accepts(b"[3,4]"));
        assert!(!e.accepts(br#"{"x":2}"#));
    }

    #[test]
    fn composite_const_distinguishes_bool_from_number_spelling() {
        use crate::compile::compile_ir;
        let e = compile_ir(&ir(r#"{"const":{"a":false}}"#).unwrap()).unwrap();
        assert!(e.accepts(br#"{"a":false}"#));
        assert!(!e.accepts(br#"{"a":0}"#));
    }

    #[test]
    fn local_ref_to_defs_inlines() {
        let ir = ir(
            r##"{"type":"array","items":{"$ref":"#/$defs/x"},"$defs":{"x":{"type":"boolean"}}}"##,
        )
        .unwrap();
        assert_eq!(ir.diagnostics().len(), 0);
    }

    #[test]
    fn unresolved_ref_is_malformed() {
        let e = ir(r##"{"$ref":"#/$defs/missing"}"##).unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    fn ref_with_a_const_sibling_intersects_them() {
        let d = ir(r##"{"$defs":{"n":{"type":"integer"}},"$ref":"#/$defs/n","const":5}"##).unwrap();
        assert_eq!(
            d.diagnostics().len(),
            0,
            "the $ref and its const sibling both lower"
        );
        let e = compile_either(&d);
        assert!(
            e.accepts(b"5"),
            "the const sibling constrains the referenced integer"
        );
        assert!(!e.accepts(b"6"), "6 is an integer but not the const value");
    }

    #[test]
    fn ref_to_an_escaped_defs_name_resolves() {
        let d = ir(r##"{"$defs":{"a/b":{"type":"boolean"}},"$ref":"#/$defs/a~1b"}"##).unwrap();
        assert_eq!(
            d.diagnostics().len(),
            0,
            "a~1b decodes to the def named a/b"
        );
        let e = compile_either(&d);
        assert!(e.accepts(b"true"));
        assert!(!e.accepts(b"1"));
    }

    #[test]
    fn local_ref_percent_decodes_the_uri_fragment_before_pointer_lookup() {
        let d = ir(
            r##"{"$defs":{"percent%field":{"type":"integer"},"foo\"bar":{"type":"number"}},"anyOf":[{"$ref":"#/$defs/percent%25field"},{"$ref":"#/$defs/foo%22bar"}]}"##,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"1.5"));
        assert!(!e.accepts(b"true"));
    }

    #[test]
    fn invalid_percent_encoded_local_ref_is_malformed() {
        let error = ir(r##"{"$ref":"#/$defs/bad%2"}"##).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
        assert_eq!(error.keyword.as_deref(), Some("$ref"));
    }

    #[test]
    fn custom_metaschema_is_explicitly_unsupported_but_dynamic_anchor_is_supported() {
        let unsupported =
            ir(r#"{"$schema":"https://example.com/custom","type":"integer"}"#).unwrap();
        assert_eq!(
            unsupported.diagnostics().next().unwrap().reason,
            UnsupportedReason::UnsupportedKeyword
        );
        let dynamic = ir(
            r##"{"$defs":{"x":{"$dynamicAnchor":"node","type":"integer"}},"$ref":"#/$defs/x"}"##,
        )
        .unwrap();
        assert_eq!(dynamic.diagnostics().count(), 0);
    }

    fn lower_with_metaschema(
        root: &str,
        uri: &str,
        metaschema: &str,
    ) -> Result<SchemaIR, CompileError> {
        let limits = SchemaResourceLimits::default();
        let mut registry = SchemaRegistry::new(limits);
        registry.insert(uri, metaschema)?;
        schema_set_to_ir(
            root,
            Some("https://example.com/root"),
            CompileOptions::default(),
            limits,
            registry,
        )
    }

    #[test]
    fn registered_metaschema_controls_validation_and_optional_vocabularies() {
        let no_validation_uri = "https://example.com/meta/no-validation";
        let no_validation = lower_with_metaschema(
            &format!(
                r#"{{"$schema":"{no_validation_uri}","properties":{{"bad":false,"n":{{"minimum":10}}}}}}"#
            ),
            no_validation_uri,
            &format!(
                r#"{{"$id":"{no_validation_uri}","$vocabulary":{{"https://json-schema.org/draft/2020-12/vocab/core":true,"https://json-schema.org/draft/2020-12/vocab/applicator":true}}}}"#
            ),
        )
        .unwrap();
        let executor = compile_either(&no_validation);
        assert!(!executor.accepts(br#"{"bad":1}"#));
        assert!(executor.accepts(br#"{"n":1}"#));

        let optional_uri = "https://example.com/meta/optional";
        let optional = lower_with_metaschema(
            &format!(r#"{{"$schema":"{optional_uri}","type":"number"}}"#),
            optional_uri,
            &format!(
                r#"{{"$id":"{optional_uri}","$vocabulary":{{"https://json-schema.org/draft/2020-12/vocab/core":true,"https://json-schema.org/draft/2020-12/vocab/validation":true,"https://example.com/vocab/optional":false}}}}"#
            ),
        )
        .unwrap();
        let executor = compile_either(&optional);
        assert!(executor.accepts(b"1"));
        assert!(!executor.accepts(br#""x""#));
    }

    #[test]
    fn unknown_required_vocabulary_is_a_typed_error() {
        let uri = "https://example.com/meta/required";
        let error = lower_with_metaschema(
            &format!(r#"{{"$schema":"{uri}","type":"number"}}"#),
            uri,
            &format!(
                r#"{{"$id":"{uri}","$vocabulary":{{"https://json-schema.org/draft/2020-12/vocab/core":true,"https://example.com/vocab/required":true}}}}"#
            ),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert_eq!(error.keyword.as_deref(), Some("$schema"));
        assert_eq!(
            error.observed.as_deref(),
            Some("https://example.com/vocab/required")
        );
    }

    #[test]
    fn nested_resource_reference_uses_its_own_base() {
        let d = ir(
            r##"{"$id":"https://example.com/root","properties":{"x":{"$id":"child","$defs":{"v":{"type":"integer"}},"$ref":"#/$defs/v"}}}"##,
        )
        .unwrap();
        assert_eq!(d.diagnostics().count(), 0);
        assert!(d.resources().len() >= 2);
    }

    #[test]
    fn local_ref_to_legacy_definitions_inlines() {
        let ir = ir(r##"{"type":"array","items":{"$ref":"#/definitions/x"},"definitions":{"x":{"type":"boolean"}}}"##)
            .unwrap();
        assert_eq!(ir.diagnostics().len(), 0);
    }

    #[test]
    fn unresolved_legacy_definitions_ref_is_malformed() {
        let e = ir(r##"{"$ref":"#/definitions/missing"}"##).unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    fn draft07_ref_ignores_a_sibling_required_keyword() {
        let schema = r##"{"$schema":"http://json-schema.org/draft-07/schema#",
            "$ref":"#/definitions/w","required":["configuration"],
            "definitions":{"w":{"properties":{"id":{"type":"string"}}}}}"##;
        let d = ir(schema).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"id":"x"}"#));
    }

    #[test]
    fn draft04_ref_ignores_a_sibling_required_keyword() {
        let schema = r##"{"$schema":"http://json-schema.org/draft-04/schema#",
            "$ref":"#/definitions/w","required":["configuration"],
            "definitions":{"w":{"properties":{"id":{"type":"string"}}}}}"##;
        let d = ir(schema).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"id":"x"}"#));
    }

    #[test]
    fn draft04_if_then_else_is_ignored_as_an_unrecognized_keyword() {
        let schema = r##"{"$schema":"http://json-schema.org/draft-04/schema#",
            "type":"object",
            "if":{"properties":{"kind":{"enum":["action"]}}},
            "then":{"required":["title"]}}"##;
        let d = ir(schema).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"kind":"action"}"#));
    }

    #[test]
    fn draft07_if_then_else_still_applies() {
        let schema = r##"{"$schema":"http://json-schema.org/draft-07/schema#",
            "type":"object",
            "if":{"properties":{"kind":{"enum":["action"]}}},
            "then":{"required":["title"]}}"##;
        let d = ir(schema).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(!crate::structured::accepts(&d, br#"{"kind":"action"}"#));
        assert!(crate::structured::accepts(
            &d,
            br#"{"kind":"action","title":"x"}"#
        ));
    }

    #[test]
    fn ref_to_an_escaped_legacy_definitions_name_resolves() {
        let d = ir(r##"{"definitions":{"a/b":{"type":"boolean"}},"$ref":"#/definitions/a~1b"}"##)
            .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"true"));
        assert!(!e.accepts(b"1"));
    }

    #[test]
    fn defs_and_legacy_definitions_are_independent_namespaces() {
        let d = ir(r##"{
            "$defs": {"x": {"type": "boolean"}},
            "definitions": {"x": {"type": "integer"}},
            "type": "array",
            "items": {"$ref": "#/definitions/x"}
        }"##)
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(
            e.accepts(b"[1]"),
            "#/definitions/x must resolve to the integer, not the same-named $defs entry"
        );
        assert!(!e.accepts(b"[true]"));
    }

    #[test]
    fn deeply_nested_legacy_definitions_reference_resolves() {
        let d = ir(r##"{
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "trait_id": {"$ref": "#/definitions/traits_trait_id_json"}
            },
            "definitions": {
                "traits_trait_id_json": {"type": "string"}
            }
        }"##)
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#"{"trait_id":"x"}"#));
        assert!(!e.accepts(br#"{"trait_id":1}"#));
    }

    #[test]
    fn ref_into_a_nested_definitions_block_walks_the_full_pointer_path() {
        let d = ir(r##"{
            "$ref": "#/definitions/outer/properties/inner",
            "definitions": {
                "outer": {"properties": {"inner": {"type": "boolean"}}}
            }
        }"##)
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"true"));
        assert!(!e.accepts(b"1"));
    }

    #[test]
    fn ref_to_an_arbitrary_pointer_outside_defs_resolves() {
        let d = ir(r##"{
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "a": {"type": "integer"},
                "b": {"$ref": "#/properties/a"}
            }
        }"##)
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#"{"a":1,"b":2}"#));
        assert!(!e.accepts(br#"{"a":1,"b":"x"}"#));
    }

    #[test]
    fn ref_through_an_array_index_segment_resolves() {
        let d = ir(r##"{"allOf":[{"type":"integer"}],"$ref":"#/allOf/0"}"##).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"5"));
        assert!(!e.accepts(br#""x""#));
    }

    #[test]
    fn ref_to_a_non_numeric_array_segment_is_malformed_not_a_panic() {
        let e = ir(r##"{"allOf":[{"type":"integer"}],"$ref":"#/allOf/x"}"##).unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    fn missing_external_ref_is_a_typed_resolution_error() {
        let error = ir(r#"{"$ref":"https://example.com/x"}"#).unwrap_err();
        assert_eq!(error.code, ErrorCode::ReferenceResolution);
        assert_eq!(error.observed.as_deref(), Some("https://example.com/x"));
    }

    #[test]
    fn recursive_ref_compiles_to_a_structured_def() {
        let ir = ir(r##"{"type":"array","items":{"$ref":"#/$defs/node"},"$defs":{"node":{"type":"array","items":{"$ref":"#/$defs/node"}}}}"##)
            .unwrap();
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(ir.requires_structured_backend());
    }

    #[test]
    fn whole_document_self_ref_compiles() {
        let ir = ir(r##"{"type":"array","items":{"$ref":"#"}}"##).unwrap();
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(ir.requires_structured_backend());
    }

    #[test]
    fn unevaluated_properties_lowers_to_a_structured_node() {
        let d = ir(r#"{"type":"object","properties":{"a":{"type":"boolean"}},"unevaluatedProperties":false}"#).unwrap();
        assert_eq!(d.diagnostics().count(), 0);
        assert!(d.requires_structured_backend());
    }

    #[test]
    fn unevaluated_properties_with_dynamic_ref_sibling_lowers() {
        let d = ir(r##"{"$dynamicAnchor":"x","$dynamicRef":"#x","unevaluatedProperties":false}"##)
            .unwrap();
        assert_eq!(d.diagnostics().count(), 0);
        assert!(d.requires_structured_backend());
    }

    #[test]
    fn unevaluated_items_lowers_to_a_structured_node() {
        let d = ir(r#"{"type":"array","unevaluatedItems":false}"#).unwrap();
        assert_eq!(d.diagnostics().count(), 0);
        assert!(d.requires_structured_backend());
    }

    #[test]
    fn non_recursive_ref_still_inlines_without_a_def() {
        let ir = ir(
            r##"{"type":"array","items":{"$ref":"#/$defs/x"},"$defs":{"x":{"type":"boolean"}}}"##,
        )
        .unwrap();
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(!ir.requires_structured_backend());
    }

    #[test]
    fn length_bound_with_pattern_intersects_both_constraints() {
        let ir = ir(r#"{"type":"string","pattern":"^a+$","minLength":2,"maxLength":3}"#).unwrap();
        assert_eq!(ir.diagnostics().len(), 0);
        let e = compile_either(&ir);
        assert!(e.accepts(b"\"aa\""));
        assert!(e.accepts(b"\"aaa\""));
        assert!(!e.accepts(b"\"a\"")); // under minLength
        assert!(!e.accepts(b"\"aaaa\"")); // over maxLength
        assert!(!e.accepts(b"\"ab\"")); // fails the anchored pattern
    }

    #[test]
    fn length_bound_with_an_unanchored_pattern_still_intersects_search_semantics() {
        let ir = ir(r#"{"type":"string","pattern":"a+","minLength":2,"maxLength":3}"#).unwrap();
        assert_eq!(ir.diagnostics().len(), 0);
        let e = compile_either(&ir);
        assert!(
            e.accepts(b"\"ab\""),
            "contains a substring a, length 2 fits"
        );
        assert!(!e.accepts(b"\"bb\""), "no a anywhere");
        assert!(
            !e.accepts(b"\"a\""),
            "under minLength even though it matches"
        );
    }

    #[test]
    fn length_bound_brute_force_matches_independent_codepoint_count() {
        use crate::compile::compile_ir;
        let mut candidates: Vec<String> = vec![
            r#""""#.to_string(),
            r#""a\nb""#.to_string(),
            r#""a\\b""#.to_string(),
            r#""a\"b""#.to_string(),
            r#""aAb""#.to_string(),
            r#""💩""#.to_string(),
            r#""💩💩""#.to_string(),
            "\"\u{e9}\"".to_string(),
            "\"\u{4e2d}\u{6587}\"".to_string(),
            "\"\u{1F600}\"".to_string(),
            "\"a\u{e9}b\u{4e2d}c\"".to_string(),
        ];
        for n in 0..6u32 {
            candidates.push(format!("\"{}\"", "x".repeat(n as usize)));
        }
        for text in &candidates {
            let true_len = serde_json::from_str::<String>(text)
                .unwrap()
                .chars()
                .count() as u32;
            for min in [None, Some(0), Some(1), Some(2), Some(3)] {
                for max in [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(10)] {
                    if let (Some(lo), Some(hi)) = (min, max) {
                        if hi < lo {
                            continue;
                        }
                    }
                    let mut b = Builder::new(CompileOptions::default());
                    let n = b
                        .string_pattern(
                            JSON_STRING_CODEPOINT,
                            min,
                            max,
                            Charset::Utf8CountedCodepoints,
                        )
                        .unwrap();
                    let e = compile_ir(&b.finish(n).unwrap()).unwrap();
                    let expect =
                        min.is_none_or(|lo| true_len >= lo) && max.is_none_or(|hi| true_len <= hi);
                    assert_eq!(
                        e.accepts(text.as_bytes()),
                        expect,
                        "text={text} true_len={true_len} min={min:?} max={max:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn pathological_length_bound_with_pattern_errors_cleanly_not_a_hang() {
        let d = ir(r#"{"type":"string","pattern":"a+","maxLength":50000000}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::compile::compile_ir(&d).is_err());
    }

    #[test]
    fn long_literal_pattern_is_admitted_by_automaton_cost_not_source_length() {
        let huge = "a".repeat(5_000);
        let schema = format!(r#"{{"type":"string","pattern":"{huge}"}}"#);
        let ir = std::sync::Arc::new(ir(&schema).unwrap());
        assert!(ir.requires_structured_backend());
        crate::structured::StructuredProgram::compile(ir).unwrap();
    }

    #[test]
    fn search_semantics_brute_force_matches_independent_substring_check() {
        type Check = fn(&str, &str) -> bool;
        let cases: &[(&str, Check)] = &[
            ("^abc$", |s, p| s == p),
            ("^abc", |s, p| s.starts_with(p)),
            ("abc$", |s, p| s.ends_with(p)),
            ("abc", |s, p| s.contains(p)),
        ];
        let candidates = ["abc", "xabc", "abcx", "xabcx", "ab", "abcabc", "xyz"];
        for (pattern, check) in cases {
            let literal = pattern.trim_start_matches('^').trim_end_matches('$');
            let d = ir(&format!(r#"{{"type":"string","pattern":"{pattern}"}}"#)).unwrap();
            assert_eq!(d.diagnostics().len(), 0);
            let e = compile_either(&d);
            for text in candidates {
                let expect = check(text, literal);
                let json = format!("\"{text}\"");
                assert_eq!(
                    e.accepts(json.as_bytes()),
                    expect,
                    "pattern={pattern} text={text}"
                );
            }
        }
    }

    #[test]
    fn not_rejects_the_negated_type_and_accepts_everything_else() {
        let d = ir(r#"{"not":{"type":"integer"}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#""foo""#));
        assert!(!crate::structured::accepts(&d, b"1"));
    }

    #[test]
    fn not_of_a_type_union_rejects_every_member_of_the_union() {
        let d = ir(r#"{"not":{"type":["integer","boolean"]}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#""foo""#));
        assert!(!crate::structured::accepts(&d, b"1"));
        assert!(!crate::structured::accepts(&d, b"true"));
    }

    #[test]
    fn if_then_else_validates_against_the_correct_branch() {
        let d = ir(
            r#"{"type":"integer","if":{"exclusiveMaximum":0},"then":{"minimum":-10},"else":{"minimum":100}}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, b"-10"), "negative, >= -10");
        assert!(!crate::structured::accepts(&d, b"-11"), "negative, < -10");
        assert!(
            crate::structured::accepts(&d, b"100"),
            "non-negative, >= 100"
        );
        assert!(
            !crate::structured::accepts(&d, b"50"),
            "non-negative, < 100"
        );
        assert!(!crate::structured::accepts(&d, b"0"), "non-negative, < 100");
    }

    #[test]
    fn if_then_without_else_only_constrains_the_matching_branch() {
        let d = ir(r#"{"type":"integer","if":{"minimum":10},"then":{"maximum":20}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, b"15"), ">=10 and <=20");
        assert!(!crate::structured::accepts(&d, b"25"), ">=10 and >20");
        assert!(crate::structured::accepts(&d, b"3"), "<10 is unconstrained");
    }

    #[test]
    fn if_else_without_then_only_constrains_the_non_matching_branch() {
        let d = ir(r#"{"type":"integer","if":{"minimum":10},"else":{"maximum":5}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(
            crate::structured::accepts(&d, b"11"),
            ">=10 is unconstrained"
        );
        assert!(crate::structured::accepts(&d, b"4"), "<10 and <=5");
        assert!(!crate::structured::accepts(&d, b"8"), "<10 and >5");
    }

    #[test]
    fn if_true_always_takes_the_then_branch() {
        let d = ir(r#"{"if":true,"then":{"const":"yes"},"else":{"const":"no"}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#""yes""#));
        assert!(!crate::structured::accepts(&d, br#""no""#));
    }

    #[test]
    fn if_false_always_takes_the_else_branch() {
        let d = ir(r#"{"if":false,"then":{"const":"yes"},"else":{"const":"no"}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#""no""#));
        assert!(!crate::structured::accepts(&d, br#""yes""#));
    }

    #[test]
    fn then_and_else_without_if_are_inert_annotations() {
        for schema in [r#"{"then":{"const":0}}"#, r#"{"else":{"const":0}}"#] {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "{schema}");
            assert!(crate::structured::accepts(&d, br#""anything""#), "{schema}");
        }
    }

    #[test]
    fn if_without_then_or_else_is_inert() {
        let d = ir(r#"{"if":{"const":0}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, b"5"));
        assert!(crate::structured::accepts(&d, br#""x""#));
    }

    #[test]
    fn if_then_else_on_a_discriminated_object() {
        let d = ir(
            r#"{"type":"object","additionalProperties":true,
                "if":{"additionalProperties":true,"required":["kind"],"properties":{"kind":{"const":"a"}}},
                "then":{"additionalProperties":true,"required":["x"]},
                "else":{"additionalProperties":true,"required":["y"]}}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"kind":"a","x":1}"#));
        assert!(
            !crate::structured::accepts(&d, br#"{"kind":"a","y":1}"#),
            "kind a needs x"
        );
        assert!(crate::structured::accepts(&d, br#"{"kind":"b","y":1}"#));
        assert!(
            !crate::structured::accepts(&d, br#"{"kind":"b","x":1}"#),
            "non-a needs y"
        );
    }

    #[test]
    fn not_of_a_closed_object_schema_only_rejects_matching_objects() {
        let d = ir(
            r#"{"not":{"type":"object","properties":{"foo":{"type":"string"}},"additionalProperties":false}}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(
            crate::structured::accepts(&d, b"1"),
            "a non-object always satisfies not"
        );
        assert!(
            crate::structured::accepts(&d, br#"{"foo":1}"#),
            "foo has the wrong type, so the inner schema does not match"
        );
        assert!(
            !crate::structured::accepts(&d, br#"{"foo":"bar"}"#),
            "matches the inner schema exactly, so not must reject it"
        );
    }

    #[test]
    fn not_nested_inside_an_object_property_only_forbids_that_shape_there() {
        let d = ir(r#"{"type":"object","properties":{"foo":{"not":{"type":"integer"}}},"additionalProperties":true}"#)
            .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(
            crate::structured::accepts(&d, b"{}"),
            "foo is optional and absent"
        );
        assert!(crate::structured::accepts(&d, br#"{"foo":"bar"}"#));
        assert!(!crate::structured::accepts(&d, br#"{"foo":1}"#));
    }

    #[test]
    fn not_looks_at_the_whole_value_not_just_its_prefix() {
        let d = ir(r#"{"not":{"type":"array"}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(!crate::structured::accepts(&d, b"[1,[2,3]]"));
        assert!(crate::structured::accepts(&d, br#"{"a":[1,2]}"#));
    }

    #[test]
    fn not_combined_with_a_sibling_one_of_intersects_both() {
        let d = ir(r#"{"oneOf":[{"type":"integer","minimum":0,"maximum":10}],"not":{"const":5}}"#)
            .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, b"3"));
        assert!(
            !crate::structured::accepts(&d, b"5"),
            "excluded by not, even though oneOf allows it"
        );
        assert!(
            !crate::structured::accepts(&d, b"20"),
            "excluded by oneOf's range"
        );
    }

    #[test]
    fn pattern_with_both_edge_anchors_strips_them_and_still_full_matches() {
        let d = ir(r#"{"type":"string","pattern":"^a*$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""aaa""#));
        assert!(e.accepts(br#""""#));
        assert!(!e.accepts(br#""abc""#), "the full string must still be a*");
    }

    #[test]
    fn pattern_with_only_a_leading_anchor_strips_it() {
        let d = ir(r#"{"type":"string","pattern":"^ab"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""ab""#));
        assert!(!e.accepts(br#""xab""#));
    }

    #[test]
    fn pattern_with_only_a_trailing_anchor_strips_it() {
        let d = ir(r#"{"type":"string","pattern":"ab$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""ab""#));
        assert!(!e.accepts(br#""abx""#));
    }

    #[test]
    fn pattern_with_an_escaped_trailing_dollar_keeps_the_literal_character() {
        let d = ir(r#"{"type":"string","pattern":"ab\\$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(
            e.accepts(br#""ab$""#),
            "an escaped dollar is a literal character, not an anchor to strip"
        );
        assert!(!e.accepts(br#""ab""#));
    }

    #[test]
    fn pattern_with_a_caret_inside_a_bracket_class_is_a_literal_not_an_anchor() {
        let d = ir(r#"{"type":"string","pattern":"^[a^]$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""a""#));
        assert!(e.accepts(br#""^""#));
        assert!(!e.accepts(br#""b""#));
    }

    #[test]
    fn pattern_with_a_mid_pattern_anchor_compiles_as_an_empty_language() {
        let d = ir(r#"{"type":"string","pattern":"ab^cd"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(!e.accepts(br#""""#));
        assert!(!e.accepts(br#""ab^cd""#));
    }

    #[test]
    fn pattern_that_is_only_an_anchor_matches_the_empty_string() {
        let d = ir(r#"{"type":"string","pattern":"^$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""""#));
        assert!(!e.accepts(br#""a""#));
    }

    #[test]
    fn pattern_alternation_of_independently_anchored_branches_is_supported() {
        let d = ir(r#"{"type":"string","pattern":"^01$|^07$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""01""#));
        assert!(e.accepts(br#""07""#));
        assert!(!e.accepts(br#""02""#));
        assert!(!e.accepts(br#""010""#));
    }

    #[test]
    fn pattern_alternation_with_one_paren_wrapped_anchored_branch_is_supported() {
        let d = ir(r#"{"type":"string","pattern":"^$|(^a+$)"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""""#));
        assert!(e.accepts(br#""aaa""#));
        assert!(!e.accepts(br#""ab""#));
    }

    #[test]
    fn pattern_alternation_with_an_asymmetrically_anchored_branch_searches_per_branch() {
        let d = ir(r#"{"type":"string","pattern":"^dev-|beta"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""dev-1.0""#));
        assert!(e.accepts(br#""xbetay""#));
        assert!(
            !e.accepts(br#""xdev-y""#),
            "dev- must be at the start, not just present"
        );
        assert!(!e.accepts(br#""nothing""#));
    }

    #[test]
    fn pattern_alternation_with_one_branch_missing_its_trailing_dollar_searches_per_branch() {
        let d = ir(r#"{"type":"string","pattern":"^ab$|^cd"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""ab""#));
        assert!(e.accepts(br#""cdxyz""#));
        assert!(
            !e.accepts(br#""abx""#),
            "the first branch requires an exact match"
        );
        assert!(
            !e.accepts(br#""xcd""#),
            "the second branch requires cd at the start"
        );
    }

    #[test]
    fn pattern_alternation_of_anchored_branches_composes_with_a_length_bound() {
        let d = ir(r#"{"type":"string","pattern":"^01$|^002$","minLength":3}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""002""#));
        assert!(
            !e.accepts(br#""01""#),
            "01 matches the pattern but is below minLength:3"
        );
        assert!(!e.accepts(br#""99""#));
    }

    #[test]
    fn matches_the_official_test_suite_pattern_validation_group_exactly() {
        let d = ir(r#"{"type":"string","pattern":"^a*$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""aaa""#), "a matching pattern is valid");
        assert!(!e.accepts(br#""abc""#), "a non-matching pattern is invalid");
    }

    #[test]
    fn matches_the_official_test_suite_pattern_is_not_anchored_group() {
        let d = ir(r#"{"type":"string","pattern":"a+"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""xxaayy""#), "matches a substring");
    }

    #[test]
    fn corpus_country_code_style_anchored_alternation_is_exact() {
        let d = ir(r#"{"type":"string","pattern":"^(ABW|AFG|AGO|ZWE)$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""ABW""#));
        assert!(e.accepts(br#""ZWE""#));
        assert!(!e.accepts(br#""ABWX""#), "anchored: no trailing chars");
        assert!(!e.accepts(br#""XABW""#), "anchored: no leading chars");
        assert!(!e.accepts(br#""USA""#), "not a listed code");
    }

    #[test]
    fn corpus_prefix_anchored_alternated_with_substring_branch() {
        let d = ir(r#"{"type":"string","pattern":"^v?\\d+|^dev-"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""v12""#), "v?\\d+ at the start");
        assert!(e.accepts(br#""42""#), "optional v, digits at the start");
        assert!(e.accepts(br#""dev-1""#), "dev- at the start");
        assert!(!e.accepts(br#""xdev-""#), "dev- must be at the start");
        assert!(!e.accepts(br#""abc""#));
    }

    #[test]
    fn ungrouped_top_level_alternation_pattern_searches_for_either_branch_anywhere() {
        let d = ir(r#"{"type":"string","pattern":"cat|dog"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""cat""#));
        assert!(e.accepts(br#""cats""#));
        assert!(e.accepts(br#""xdogx""#));
        assert!(!e.accepts(br#""ca""#));
        assert!(!e.accepts(br#""fish""#));
    }

    #[test]
    fn anchored_top_level_alternation_pattern_does_not_leak_into_the_quote_wrapper() {
        let d = ir(r#"{"type":"string","pattern":"^cat$|^dog$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""cat""#));
        assert!(e.accepts(br#""dog""#));
        assert!(!e.accepts(br#""cats""#));
        assert!(!e.accepts(br#""ca""#));
    }

    #[test]
    fn wrapped_alternation_of_mixed_anchored_branches_searches_per_branch() {
        let d = ir(r#"{"type":"string","pattern":"(^Hpt_|^Int_|_Armour_)"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""Hpt_x""#), "Hpt_ prefix");
        assert!(e.accepts(br#""Int_y""#), "Int_ prefix");
        assert!(e.accepts(br#""z_Armour_z""#), "_Armour_ anywhere");
        assert!(!e.accepts(br#""xHpt_""#), "Hpt_ must be at the start");
        assert!(!e.accepts(br#""nothing""#));
    }

    #[test]
    fn long_fully_anchored_pattern_past_the_wildcard_cap_still_compiles() {
        let body = "a".repeat(600);
        let d = ir(&format!(r#"{{"type":"string","pattern":"^{body}$"}}"#)).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        let good = format!("\"{body}\"");
        assert!(e.accepts(good.as_bytes()));
        assert!(!e.accepts(br#""aaa""#));
    }

    #[test]
    fn a_js_regex_literal_wrapper_is_not_stripped() {
        let d = ir(r#"{"type":"string","pattern":"/^[0-9]{4}$/"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(!e.accepts(br#""/1234/""#));
        assert!(!e.accepts(br#""1234""#));
    }

    #[test]
    fn a_leading_slash_pattern_with_no_anchors_is_a_literal_search() {
        let d = ir(r#"{"type":"string","pattern":"/api/v[0-9]+"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(
            e.accepts(br#""x/api/v1y""#),
            "no anchors: a search, not a full match"
        );
        assert!(
            !e.accepts(br#""api/v1""#),
            "the leading / is part of the literal pattern text and still required"
        );
    }

    #[test]
    fn pattern_treats_raw_and_escaped_spellings_of_the_same_string_identically() {
        let raw_a: &[u8] = b"\"a\"";
        let escaped_a: &[u8] = b"\"\\u0061\"";
        let escaped_b: &[u8] = b"\"\\u0062\"";
        let d = ir(r#"{"type":"string","pattern":"^a$"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(raw_a), "raw literal a");
        assert!(e.accepts(escaped_a), "\\u0061 spells the same string");
        assert!(!e.accepts(escaped_b), "\\u0062 spells b, not a");
    }

    #[test]
    fn length_bound_treats_raw_and_escaped_spellings_of_the_same_string_identically() {
        let raw_a: &[u8] = b"\"a\"";
        let escaped_a: &[u8] = b"\"\\u0061\"";
        let escaped_ab: &[u8] = b"\"\\u0061\\u0062\"";
        let d = ir(r#"{"type":"string","minLength":1,"maxLength":1}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(raw_a), "raw literal, length 1");
        assert!(
            e.accepts(escaped_a),
            "\\u0061 is also length 1, not the 6 bytes it spells"
        );
        assert!(
            !e.accepts(escaped_ab),
            "decodes to length 2, over maxLength 1"
        );
    }

    #[test]
    fn unconstrained_string_rejects_a_bare_invalid_utf8_byte() {
        let d = ir(r#"{"type":"string"}"#).unwrap();
        let e = compile_either(&d);
        assert!(!e.accepts(b"\"\xff\""));
    }

    #[test]
    fn unconstrained_string_rejects_a_lone_continuation_byte() {
        let d = ir(r#"{"type":"string"}"#).unwrap();
        let e = compile_either(&d);
        assert!(!e.accepts(b"\"\xc2\x28\""));
    }

    #[test]
    fn unconstrained_string_rejects_an_overlong_two_byte_encoding() {
        let d = ir(r#"{"type":"string"}"#).unwrap();
        let e = compile_either(&d);
        assert!(!e.accepts(b"\"\xc0\xaf\""));
    }

    #[test]
    fn unconstrained_string_rejects_a_utf8_encoded_surrogate() {
        let d = ir(r#"{"type":"string"}"#).unwrap();
        let e = compile_either(&d);
        assert!(!e.accepts(b"\"\xed\xa0\x80\""));
    }

    #[test]
    fn unconstrained_string_rejects_a_codepoint_above_u10ffff() {
        let d = ir(r#"{"type":"string"}"#).unwrap();
        let e = compile_either(&d);
        assert!(!e.accepts(b"\"\xf4\x90\x80\x80\""));
    }

    #[test]
    fn unconstrained_string_accepts_every_valid_utf8_boundary_codepoint() {
        let d = ir(r#"{"type":"string"}"#).unwrap();
        let e = compile_either(&d);
        for cp in [
            0x7fu32, 0x80, 0x7ff, 0x800, 0xd7ff, 0xe000, 0xffff, 0x10000, 0x10ffff,
        ] {
            let ch = char::from_u32(cp).unwrap();
            let mut buf = vec![b'"'];
            buf.extend_from_slice(ch.to_string().as_bytes());
            buf.push(b'"');
            assert!(e.accepts(&buf), "codepoint U+{cp:04X} must be accepted");
        }
    }

    #[test]
    fn malformed_json_is_rejected() {
        assert_eq!(ir("{").unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn each_unsupported_applicator_has_its_reason() {
        let cases = [(
            r#"{"type":"string","pattern":"(?=a)"}"#,
            UnsupportedReason::RegexAssertionUnsupported,
        )];
        for (schema, reason) in cases {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().next().unwrap().reason, reason, "{schema}");
        }
    }

    #[test]
    fn leading_assertions_work_for_schema_patterns_and_pattern_properties() {
        let string = ir(r#"{"type":"string","pattern":"^(?=.{2,4}$)(?!bad$)[a-z]+$"}"#).unwrap();
        let string = compile_either(&string);
        assert!(string.accepts(br#""ok""#));
        assert!(string.accepts(br#""\u006fk""#));
        assert!(!string.accepts(br#""bad""#));
        assert!(!string.accepts(br#""toolong""#));

        let object = ir(
            r#"{"type":"object","patternProperties":{"^(?!@@)[\\w@]+$":{"type":"integer"}},"additionalProperties":false}"#,
        )
        .unwrap();
        let object = std::sync::Arc::new(object);
        assert!(crate::structured::try_accepts(object.clone(), br#"{"ok":1}"#).unwrap());
        assert!(!crate::structured::try_accepts(object.clone(), br#"{"@@meta":1}"#).unwrap());
        assert!(!crate::structured::try_accepts(object.clone(), "{\"λ\":1}".as_bytes()).unwrap());
        assert!(!crate::structured::try_accepts(object, br#"{"ok":"wrong"}"#).unwrap());
    }

    #[test]
    fn type_union_accepts_a_value_of_any_branch() {
        use crate::compile::compile_ir;
        for schema in [
            r#"{"type":["string","null"]}"#,
            r#"{"type":["integer","string"]}"#,
        ] {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "{schema}");
        }
        let e = compile_ir(&ir(r#"{"type":["string","null"]}"#).unwrap()).unwrap();
        assert!(e.accepts(b"\"x\""));
        assert!(e.accepts(b"null"));
        assert!(!e.accepts(b"true"));

        let e = compile_ir(&ir(r#"{"type":["integer","string"]}"#).unwrap()).unwrap();
        assert!(e.accepts(b"12"));
        assert!(e.accepts(b"\"x\""));
        assert!(!e.accepts(b"null"));
    }

    #[test]
    fn type_union_of_object_and_array_agrees_with_both_branches() {
        use crate::compile::compile_ir;
        let e = compile_ir(
            &ir(r#"{"type":["array","object"],"items":{"type":"boolean"},"properties":{"a":{"type":"null"}},"additionalProperties":false}"#)
                .unwrap(),
        )
        .unwrap();
        assert!(e.accepts(b"[true,false]"));
        assert!(e.accepts(br#"{"a":null}"#));
        assert!(!e.accepts(b"1"));
    }

    #[test]
    fn type_union_duplicate_names_are_deduplicated_not_an_error() {
        let a = ir(r#"{"type":["string","string"]}"#).unwrap();
        let b = ir(r#"{"type":["string"]}"#).unwrap();
        assert_eq!(a.diagnostics().len(), 0);
        assert!(a.same_content_address(&b));
    }

    #[test]
    fn type_union_empty_array_is_malformed() {
        assert_eq!(ir(r#"{"type":[]}"#).unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn type_union_invalid_type_name_is_malformed() {
        assert_eq!(
            ir(r#"{"type":["string","banana"]}"#).unwrap_err().code,
            ErrorCode::Malformed
        );
    }

    #[test]
    fn type_union_non_string_member_is_malformed() {
        assert_eq!(
            ir(r#"{"type":["string",1]}"#).unwrap_err().code,
            ErrorCode::Malformed
        );
    }

    #[test]
    fn type_union_with_number_branch_accepts_both_numbers_and_strings() {
        let d = ir(r#"{"type":["number","string"]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"3.14"));
        assert!(e.accepts(b"-7"));
        assert!(e.accepts(b"\"x\""));
        assert!(!e.accepts(b"true"));
    }

    #[test]
    fn type_union_with_const_accepts_a_member_matching_any_branch() {
        use crate::compile::compile_ir;
        let e = compile_ir(&ir(r#"{"type":["integer","string"],"const":"x"}"#).unwrap()).unwrap();
        assert!(e.accepts(b"\"x\""));
        assert!(!e.accepts(b"\"y\""));

        let d = ir(r#"{"type":["integer","string"],"const":true}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(!compile_ir(&d).unwrap().accepts(b"true"));
    }

    #[test]
    fn type_union_with_enum_filters_to_members_matching_any_branch() {
        let d = ir(r#"{"type":["integer","string"],"enum":[1,"x",true,null]}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"1.0"));
        assert!(e.accepts(b"\"x\""));
        assert!(!e.accepts(b"true"));
        assert!(!e.accepts(b"null"));
    }

    #[test]
    fn nested_type_union_inside_a_property_compiles() {
        use crate::compile::compile_ir;
        let e = compile_ir(
            &ir(r#"{"type":"object","properties":{"a":{"type":["null","boolean"]}},"required":["a"],"additionalProperties":false}"#)
                .unwrap(),
        )
        .unwrap();
        assert!(e.accepts(br#"{"a":null}"#));
        assert!(e.accepts(br#"{"a":true}"#));
        assert!(!e.accepts(br#"{"a":1}"#));
    }

    #[test]
    fn closed_tuple_accepts_positional_members_and_shorter_arrays() {
        let d = ir(
            r#"{"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}],"items":false}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#"[1,"foo"]"#));
        assert!(e.accepts(br#"[1]"#), "a shorter array is valid");
        assert!(e.accepts(br#"[]"#), "the empty array is valid");
        assert!(!e.accepts(br#"["foo",1]"#), "positions are type-checked");
        assert!(
            !e.accepts(br#"[1,"foo",true]"#),
            "items:false forbids extra items"
        );
    }

    #[test]
    fn tuple_with_a_uniform_tail_constrains_items_past_the_prefix() {
        let d =
            ir(r#"{"type":"array","prefixItems":[{"type":"string"}],"items":{"type":"integer"}}"#)
                .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#"["a"]"#));
        assert!(e.accepts(br#"["a",1,2,3]"#));
        assert!(e.accepts(br#"[]"#));
        assert!(!e.accepts(br#"["a","b"]"#), "the tail must be an integer");
        assert!(!e.accepts(br#"[1]"#), "the first position must be a string");
    }

    #[test]
    fn tuple_honors_min_and_max_items() {
        let d = ir(r#"{"type":"array","prefixItems":[{"type":"integer"},{"type":"integer"},{"type":"integer"}],"items":false,"minItems":2}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(!e.accepts(br#"[1]"#), "minItems is 2");
        assert!(e.accepts(br#"[1,2]"#));
        assert!(e.accepts(br#"[1,2,3]"#));
        assert!(!e.accepts(br#"[1,2,3,4]"#), "closed at the prefix length");
    }

    #[test]
    fn unique_items_is_now_supported_via_the_structured_backend() {
        let d = ir(r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(d.requires_structured_backend());
        assert!(crate::structured::accepts(&d, b"[1,2,3]"));
        assert!(crate::structured::accepts(&d, b"[]"));
        assert!(
            !crate::structured::accepts(&d, b"[1,2,1]"),
            "1 repeats: not unique"
        );
    }

    #[test]
    fn unique_items_false_or_absent_allows_duplicates() {
        for schema in [
            r#"{"type":"array","items":{"type":"integer"}}"#,
            r#"{"type":"array","items":{"type":"integer"},"uniqueItems":false}"#,
        ] {
            let d = ir(schema).unwrap();
            assert_eq!(d.diagnostics().len(), 0);
            assert!(!d.requires_structured_backend(), "{schema}");
            let e = crate::compile::compile_ir(&d).unwrap();
            assert!(e.accepts(b"[1,1,1]"), "{schema}");
        }
    }

    #[test]
    fn unique_items_non_boolean_value_is_malformed() {
        let err =
            ir(r#"{"type":"array","items":{"type":"integer"},"uniqueItems":"yes"}"#).unwrap_err();
        assert_eq!(err.code, ErrorCode::Malformed);
    }

    #[test]
    fn unique_items_on_object_elements_ignores_key_order() {
        let d = ir(
            r#"{"type":"array","items":{"type":"object","additionalProperties":false,"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]},"uniqueItems":true}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(
            !crate::structured::accepts(&d, br#"[{"a":1,"b":2},{"b":2,"a":1}]"#),
            "same value under a different key order is still a duplicate"
        );
        assert!(!crate::structured::accepts(
            &d,
            br#"[{"a":1,"b":2},{"a":1,"b":2}]"#
        ));
    }

    #[test]
    fn bare_prefix_items_leaves_an_open_tail_via_the_structured_backend() {
        let d = ir(r#"{"type":"array","prefixItems":[{"type":"integer"}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(d.requires_structured_backend());
        assert!(crate::structured::accepts(&d, b"[1]"));
        assert!(crate::structured::accepts(
            &d,
            br#"[1,"a",true,null,{"x":1},[1,2]]"#
        ));
        assert!(
            !crate::structured::accepts(&d, br#"["a"]"#),
            "the prefix position is still checked"
        );
    }

    #[test]
    fn prefix_items_with_an_explicit_bare_true_items_is_the_same_open_tail() {
        let d = ir(r#"{"type":"array","prefixItems":[{"type":"integer"}],"items":true}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(d.requires_structured_backend());
        assert!(crate::structured::accepts(&d, br#"[1,{"x":1}]"#));
    }

    #[test]
    fn array_without_any_item_schema_is_open_tail_via_structured_backend() {
        let d = ir(r#"{"type":"array"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(d.requires_structured_backend());
        assert!(crate::structured::accepts(&d, b"[]"));
        assert!(crate::structured::accepts(
            &d,
            br#"[1,"a",true,null,{"x":1},[1,2]]"#
        ));
        assert!(!crate::structured::accepts(&d, b"not an array"));
    }

    #[test]
    fn items_true_is_the_same_open_tail_as_items_absent() {
        let d = ir(r#"{"type":"array","items":true}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(d.requires_structured_backend());
        assert!(crate::structured::accepts(&d, br#"[1,"a",true]"#));
    }

    #[test]
    fn closed_tuple_with_min_items_past_the_prefix_is_empty() {
        let d = ir(r#"{"prefixItems":[{"type":"integer"}],"items":false,"minItems":3}"#).unwrap();
        assert!(!compile_either(&d).accepts(br#"[1]"#));
    }

    #[test]
    fn empty_prefix_items_is_malformed() {
        let e = ir(r#"{"prefixItems":[],"items":false}"#).unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    fn tuple_round_trips_through_the_wire() {
        let d = ir(r#"{"prefixItems":[{"type":"integer"},{"type":"boolean"}],"items":{"type":"null"},"minItems":1,"maxItems":5}"#).unwrap();
        assert_eq!(SchemaIR::from_wire(&d.to_wire()).unwrap(), d);
    }

    #[test]
    fn additional_properties_schema_is_supported_via_the_open_object_node() {
        let d = ir(r#"{"type":"object","additionalProperties":{"type":"null"}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(matches!(
            d.node(d.root()).unwrap(),
            crate::ir::Node::OpenObject { .. }
        ));
        assert!(crate::structured::accepts(&d, br#"{"a":null,"b":null}"#));
        assert!(!crate::structured::accepts(&d, br#"{"a":true}"#));
    }

    #[test]
    fn duplicate_object_key_in_json_is_rejected() {
        let e = schema_to_ir(
            r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"null"},"a":{"type":"boolean"}}}"#,
            closed(),
        );
        assert_eq!(e.unwrap_err().code, ErrorCode::Malformed);
    }

    #[test]
    fn escaped_property_name_keeps_its_rfc6901_pointer() {
        let error = schema_to_ir(
            r#"{"type":"object","additionalProperties":false,"properties":{"a/b":{"type":1}}}"#,
            closed(),
        )
        .unwrap_err();
        assert_eq!(
            error.json_pointer_path.as_deref(),
            Some("/properties/a~1b/type")
        );
    }

    #[test]
    fn deeply_nested_schema_hits_the_depth_cap_not_a_stack_overflow() {
        let mut s = String::new();
        for _ in 0..(MAX_DEPTH + 50) {
            s.push_str(r#"{"type":"array","items":"#);
        }
        s.push_str(r#"{"type":"null"}"#);
        for _ in 0..(MAX_DEPTH + 50) {
            s.push('}');
        }
        assert_eq!(ir(&s).unwrap_err().code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn length_bound_counts_code_points_including_escapes() {
        use crate::compile::compile_ir;
        let ir = ir(r#"{"type":"string","minLength":1,"maxLength":3}"#).unwrap();
        let e = compile_ir(&ir).unwrap();
        assert!(e.accepts(b"\"ab\""));
        assert!(e.accepts(b"\"a\\b\"")); // `\b` is one escaped code point, so length is 2
        assert!(!e.accepts(b"\"abcd\"")); // over maxLength
        assert!(!e.accepts(b"\"\"")); // under minLength
    }

    #[test]
    fn const_mismatched_with_a_declared_type_is_valid_but_unsatisfiable() {
        use crate::compile::compile_ir;
        let d = ir(r#"{"type":"string","const":1}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(!d.requires_structured_backend());
        let e = compile_either(&d);
        assert!(e.is_dead(e.start()));
        assert!(!e.accepts(b"1"));
        assert!(!e.accepts(br#""1""#));

        let d = ir(r#"{"type":"integer","const":"hello"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(!compile_ir(&d).unwrap().accepts(br#""hello""#));

        assert!(ir(r#"{"type":"null","const":null}"#).is_ok());
    }

    #[test]
    fn enum_is_filtered_by_a_declared_type() {
        let d = ir(r#"{"type":"integer","enum":[1,"hello"]}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"1"));
        assert!(!e.accepts(b"\"hello\""));

        let d = ir(r#"{"type":"boolean","enum":[true,1]}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"true"));
        assert!(!e.accepts(b"1"));
    }

    #[test]
    fn const_whole_number_float_forms_are_numerically_equal_to_the_integer() {
        for lit in ["1", "1.0", "1.00", "1e0", "1e+0"] {
            let d = ir(&format!(r#"{{"type":"integer","const":{lit}}}"#)).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "const:{lit}");
            let e = compile_either(&d);
            assert!(e.accepts(b"1"), "const:{lit} should generate 1");
            assert!(e.accepts(b"1.0"), "const:{lit} should accept 1.0");
            assert!(e.accepts(b"10e-1"), "const:{lit} should accept 10e-1");
        }
        for lit in ["-0", "-0.0"] {
            let d = ir(&format!(r#"{{"type":"integer","const":{lit}}}"#)).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "const:{lit}");
            let e = compile_either(&d);
            assert!(e.accepts(b"0"), "const:{lit} should generate 0");
            assert!(e.accepts(b"-0.0"), "const:{lit} should accept -0.0");
        }
    }

    #[test]
    fn fractional_const_respects_the_declared_integer_type() {
        for lit in ["1.5", "0.1", "1.23e2"] {
            let d = ir(&format!(r#"{{"type":"integer","const":{lit}}}"#)).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "const:{lit}");
            let e = compile_either(&d);
            assert_eq!(e.accepts(b"123"), lit == "1.23e2");
        }
    }

    #[test]
    fn enum_whole_number_float_members_are_numerically_equal_to_the_integer() {
        for lit in ["2.0", "2e0", "2.00"] {
            let d = ir(&format!(r#"{{"enum":[1,{lit}]}}"#)).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "enum member {lit}");
            let e = compile_either(&d);
            assert!(e.accepts(b"1"));
            assert!(e.accepts(b"2"), "enum member {lit} should generate 2");
        }
        let d = ir(r#"{"enum":[1,1.0,1e0]}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"1.0"));
        assert!(e.accepts(b"10e-1"));
        assert!(!e.accepts(b"2"));
        let d = ir(r#"{"enum":[1,1.5]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"1.5"));
        assert!(e.accepts(b"15e-1"));
        assert!(!e.accepts(b"1.5000000001"));

        let d = ir(r#"{"type":"number","enum":[1]}"#).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"1"));
        assert!(e.accepts(b"1.0"));
        assert!(e.accepts(b"100e-2"));
    }

    #[test]
    fn empty_enum_is_an_explicit_empty_language() {
        let d = ir(r#"{"enum":[]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.is_dead(e.start()));
        for value in [b"null".as_slice(), b"0", br#""x""#] {
            assert!(!e.accepts(value));
        }
    }

    #[test]
    fn numeric_bound_whole_number_float_forms_are_accepted_as_the_exact_integer_bound() {
        for lit in ["1.0", "1e0", "1.00"] {
            let d = ir(&format!(
                r#"{{"type":"integer","minimum":{lit},"maximum":{lit}}}"#
            ))
            .unwrap();
            let e = compile_either(&d);
            assert!(e.accepts(b"1"), "minimum/maximum:{lit} should bound to 1");
            assert!(!e.accepts(b"2"));
        }
        let d = ir(r#"{"type":"integer","minimum":1.5}"#).unwrap();
        let e = compile_either(&d);
        assert!(!e.accepts(b"1"));
        assert!(e.accepts(b"2"));
    }

    #[test]
    fn numeric_token_as_i64_agrees_with_an_independent_f64_oracle_over_small_magnitudes() {
        for n in -200i64..=200 {
            for text in [
                n.to_string(),
                format!("{n}.0"),
                format!("{n}.00"),
                format!("{n}e0"),
            ] {
                let want = text.parse::<f64>().ok().filter(|f| f.fract() == 0.0);
                let got = numeric_token_as_i64(&text);
                match (want, got) {
                    (Some(w), Some(g)) => assert!(
                        (w - g as f64).abs() < 0.5,
                        "text={text} f64-oracle={w} numeric_token_as_i64={g}"
                    ),
                    (None, None) => {}
                    _ => panic!("text={text} f64-oracle={want:?} numeric_token_as_i64={got:?}"),
                }
            }
        }
    }

    #[test]
    fn numeric_token_exponent_and_magnitude_edge_cases_never_panic() {
        for lit in ["1e999999999999", "1e-999999999999", "1e400", "9e18", "1e19"] {
            let _ = ir(&format!(r#"{{"type":"integer","const":{lit}}}"#));
        }
    }

    #[test]
    fn enum_with_no_member_satisfying_the_type_is_valid_but_unsatisfiable() {
        let d = ir(r#"{"type":"boolean","enum":[1,2]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.is_dead(e.start()));
        assert!(!e.accepts(b"true"));
        assert!(!e.accepts(b"1"));
    }

    #[test]
    fn unsatisfiable_enum_branch_inside_anyof_just_never_matches() {
        let d = ir(r#"{"anyOf":[{"type":"boolean","enum":[1,2]},{"type":"integer"}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"5"));
        assert!(!e.accepts(b"true"));
    }

    #[test]
    fn not_of_an_unsatisfiable_const_accepts_everything() {
        let d = ir(r#"{"not":{"type":"string","const":1}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, b"1"));
        assert!(crate::structured::accepts(&d, br#""anything""#));
    }

    #[test]
    fn required_name_absent_from_properties_only_asserts_presence() {
        let d = schema_to_ir(
            r#"{"type":"object","additionalProperties":false,"required":["missing"],"properties":{}}"#,
            closed(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"missing":1}"#));
        assert!(crate::structured::accepts(&d, br#"{"missing":null}"#));
        assert!(
            !crate::structured::accepts(&d, b"{}"),
            "the required key must be present"
        );
        assert!(
            !crate::structured::accepts(&d, br#"{"missing":1,"extra":2}"#),
            "additionalProperties:false still forbids unknown keys"
        );
    }

    #[test]
    fn required_name_absent_from_properties_composes_with_a_declared_property() {
        let d = schema_to_ir(
            r#"{"type":"object","additionalProperties":false,"required":["missing","a"],"properties":{"a":{"type":"integer"}}}"#,
            closed(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"a":1,"missing":"x"}"#));
        assert!(
            !crate::structured::accepts(&d, br#"{"a":1}"#),
            "missing is still required"
        );
        assert!(
            !crate::structured::accepts(&d, br#"{"a":"not an integer","missing":1}"#),
            "the declared property's own schema is still enforced"
        );
    }

    #[test]
    fn required_name_absent_from_properties_still_lets_pattern_properties_apply() {
        let d = schema_to_ir(
            r#"{"type":"object","patternProperties":{"^b":{"type":"integer"}},"required":["b"]}"#,
            closed(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"b":1}"#));
        assert!(
            !crate::structured::accepts(&d, b"{}"),
            "b is still required"
        );
    }

    #[test]
    fn required_name_absent_from_properties_via_typeless_dispatch() {
        let d = schema_to_ir(
            r#"{"required":["x"],"additionalProperties":false}"#,
            closed(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"x":null}"#));
        assert!(!crate::structured::accepts(&d, b"{}"));
        for instance in ["null", "true", "1", r#""s""#, "[]"] {
            assert!(
                crate::structured::accepts(&d, instance.as_bytes()),
                "required on a typeless schema must not filter out non-object shape {instance}"
            );
        }
    }

    #[test]
    fn required_name_present_in_properties_compiles() {
        let d = schema_to_ir(
            r#"{"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"type":"null"}}}"#,
            closed(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
    }

    #[test]
    fn multiple_missing_required_names_still_compile_and_assert_presence() {
        let d = schema_to_ir(
            r#"{"type":"object","additionalProperties":false,"required":["a","b"],"properties":{"a":{"type":"null"}}}"#,
            closed(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"a":null,"b":1}"#));
        assert!(!crate::structured::accepts(&d, br#"{"a":null}"#));
    }

    #[test]
    fn missing_required_name_under_ref_still_compiles_and_asserts_presence() {
        let d = schema_to_ir(
            r##"{"$ref":"#/$defs/x","$defs":{"x":{"type":"object","additionalProperties":false,"required":["missing"],"properties":{}}}}"##,
            closed(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"missing":true}"#));
        assert!(!crate::structured::accepts(&d, b"{}"));
    }

    fn asserting() -> CompileOptions {
        CompileOptions {
            format_assertion: true,
            ..CompileOptions::default()
        }
    }

    #[test]
    fn format_is_annotation_only_by_default() {
        let d = ir(r#"{"type":"string","format":"date"}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(b"\"not a date at all\""));
    }

    #[test]
    fn format_date_time_is_enforced_when_asserting() {
        let d = schema_to_ir(r#"{"type":"string","format":"date-time"}"#, asserting()).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"\"2024-01-31T23:59:60.5Z\""));
        assert!(e.accepts(b"\"2024-06-15T12:00:00+05:30\""));
        assert!(!e.accepts(b"\"not-a-date\""));
        assert!(!e.accepts(b"\"2024-13-01T00:00:00Z\""));
    }

    #[test]
    fn format_date_is_enforced_when_asserting() {
        let d = schema_to_ir(r#"{"type":"string","format":"date"}"#, asserting()).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"\"2024-01-31\""));
        assert!(!e.accepts(b"\"2024-1-31\""));
        assert!(!e.accepts(b"\"2024-13-01\""));
        assert!(!e.accepts(b"\"2024-01-32\""));
    }

    #[test]
    fn format_date_rejects_a_day_out_of_range_for_its_month() {
        let d = schema_to_ir(r#"{"type":"string","format":"date"}"#, asserting()).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"\"2024-01-31\""));
        assert!(!e.accepts(b"\"2024-04-31\""));
        assert!(!e.accepts(b"\"2024-06-31\""));
        assert!(!e.accepts(b"\"2024-09-31\""));
        assert!(!e.accepts(b"\"2024-11-31\""));
        assert!(!e.accepts(b"\"2020-02-31\""));
        assert!(!e.accepts(b"\"2020-02-30\""));
    }

    #[test]
    fn format_date_enforces_gregorian_leap_year_rules_for_february_29() {
        let d = schema_to_ir(r#"{"type":"string","format":"date"}"#, asserting()).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"\"2020-02-29\""));
        assert!(e.accepts(b"\"2000-02-29\""));
        assert!(e.accepts(b"\"2024-02-29\""));
        assert!(!e.accepts(b"\"2021-02-29\""));
        assert!(!e.accepts(b"\"2019-02-29\""));
        assert!(!e.accepts(b"\"1900-02-29\""));
        assert!(e.accepts(b"\"2021-02-28\""));
    }

    #[test]
    fn format_date_time_rejects_an_out_of_range_timezone_offset() {
        let d = schema_to_ir(r#"{"type":"string","format":"date-time"}"#, asserting()).unwrap();
        let e = compile_either(&d);
        assert!(!e.accepts(b"\"2022-01-01T12:00:00+25:00\""));
        assert!(e.accepts(b"\"2022-01-01T12:00:00+23:59\""));
    }

    #[test]
    fn format_date_time_rejects_a_calendar_invalid_date_component() {
        let d = schema_to_ir(r#"{"type":"string","format":"date-time"}"#, asserting()).unwrap();
        let e = compile_either(&d);
        assert!(!e.accepts(b"\"2020-02-31T00:00:00Z\""));
        assert!(!e.accepts(b"\"2021-02-29T00:00:00Z\""));
        assert!(e.accepts(b"\"2020-02-29T00:00:00Z\""));
    }

    #[test]
    fn format_time_is_enforced_when_asserting() {
        let d = schema_to_ir(r#"{"type":"string","format":"time"}"#, asserting()).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"\"23:59:60Z\""));
        assert!(!e.accepts(b"\"24:00:00Z\""));
        assert!(!e.accepts(b"\"12:00:00\""));
    }

    #[test]
    fn format_uuid_is_enforced_when_asserting() {
        let d = schema_to_ir(r#"{"type":"string","format":"uuid"}"#, asserting()).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"\"123e4567-e89b-12d3-a456-426614174000\""));
        assert!(!e.accepts(b"\"not-a-uuid\""));
        assert!(!e.accepts(b"\"123e4567e89b12d3a456426614174000\""));
    }

    #[test]
    fn format_ipv4_is_enforced_when_asserting() {
        let d = schema_to_ir(r#"{"type":"string","format":"ipv4"}"#, asserting()).unwrap();
        let e = compile_either(&d);
        assert!(e.accepts(b"\"192.168.1.1\""));
        assert!(e.accepts(b"\"0.0.0.0\""));
        assert!(e.accepts(b"\"255.255.255.255\""));
        assert!(!e.accepts(b"\"256.1.1.1\""));
        assert!(!e.accepts(b"\"1.2.3\""));
    }

    #[test]
    fn format_email_hostname_uri_regex_json_pointer_are_annotation_only_when_asserting() {
        for fmt in ["email", "hostname", "uri", "regex", "json-pointer"] {
            let schema = format!(r#"{{"type":"string","format":"{fmt}"}}"#);
            let d = schema_to_ir(&schema, asserting()).unwrap();
            assert_eq!(d.diagnostics().len(), 0, "{fmt}");
            let e = compile_either(&d);
            assert!(e.accepts(b"\"anything goes\""), "{fmt}");
        }
    }

    #[test]
    fn format_duration_and_ipv6_are_unsupported_when_asserting() {
        for fmt in ["duration", "ipv6"] {
            let schema = format!(r#"{{"type":"string","format":"{fmt}"}}"#);
            let d = schema_to_ir(&schema, asserting()).unwrap();
            assert_eq!(
                d.diagnostics().next().unwrap().reason,
                UnsupportedReason::FormatUnsupported,
                "{fmt}"
            );
        }
    }

    #[test]
    fn unknown_format_is_annotation_only_never_an_error() {
        let d = schema_to_ir(
            r#"{"type":"string","format":"not-a-real-format"}"#,
            asserting(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
    }

    #[test]
    fn duplicate_keyword_at_root_is_malformed() {
        let e = ir(r#"{"type":"integer","type":"string"}"#).unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    fn duplicate_keyword_in_a_nested_schema_is_malformed() {
        let e = ir(r#"{"type":"array","items":{"type":"null","type":"boolean"}}"#).unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    fn duplicate_keyword_inside_a_defs_entry_is_malformed() {
        let e = ir(r##"{"$ref":"#/$defs/x","$defs":{"x":{"type":"null","type":"boolean"}}}"##)
            .unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    fn anchor_ref_resolves() {
        let d = ir(r##"{"type":"array","items":{"$ref":"#x"},"$defs":{"n":{"$anchor":"x","type":"boolean"}}}"##)
            .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
    }

    #[test]
    fn duplicate_anchor_anywhere_in_the_document_is_malformed() {
        let e = ir(r##"{"type":"array","items":{"$ref":"#x"},"$defs":{"a":{"$anchor":"x","type":"boolean"},"b":{"$anchor":"x","type":"null"}}}"##)
            .unwrap_err();
        assert_eq!(e.code, ErrorCode::DuplicateAnchor);
    }

    #[test]
    fn duplicate_id_is_malformed() {
        let e = ir(r##"{"$id":"https://example.com/a","$defs":{"x":{"$id":"https://example.com/a","type":"null"}}}"##).unwrap_err();
        assert_eq!(e.code, ErrorCode::DuplicateResource);
    }

    #[test]
    fn distinct_ids_on_nested_resources_are_accepted() {
        let d = ir(r##"{"$id":"https://example.com/root","type":"array","items":{"$ref":"child"},"$defs":{"x":{"$id":"child","type":"boolean"}}}"##)
            .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
    }

    #[test]
    fn same_defs_name_in_two_unrelated_scopes_resolves_by_exact_pointer_path() {
        let d = ir(r##"{"type":"object","additionalProperties":false,"properties":{"a":{"$ref":"#/$defs/x"}},"$defs":{"x":{"type":"null"}},"b":{"$defs":{"x":{"type":"boolean"}}}}"##)
            .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#"{"a":null}"#));
        assert!(!e.accepts(br#"{"a":true}"#));
    }

    #[test]
    fn same_definitions_name_nested_under_two_different_parents_each_resolve_independently() {
        let d = ir(r##"{
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "a": {"$ref": "#/definitions/Foo/definitions/field"},
                "b": {"$ref": "#/definitions/Bar/definitions/field"}
            },
            "definitions": {
                "Foo": {"definitions": {"field": {"type": "string"}}},
                "Bar": {"definitions": {"field": {"type": "integer"}}}
            }
        }"##)
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#"{"a":"x","b":1}"#));
        assert!(!e.accepts(br#"{"a":1,"b":1}"#));
        assert!(!e.accepts(br#"{"a":"x","b":"x"}"#));
    }

    #[test]
    fn diamond_shaped_acyclic_ref_inlines_twice() {
        let d = schema_to_ir(
            r##"{"type":"object","additionalProperties":false,"properties":{"a":{"$ref":"#/$defs/leaf"},"b":{"$ref":"#/$defs/leaf"}},"$defs":{"leaf":{"type":"boolean"}}}"##,
            closed(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
    }

    #[test]
    fn format_with_pattern_now_intersects_correctly() {
        let d = schema_to_ir(
            r#"{"type":"string","format":"date","pattern":"^2023"}"#,
            asserting(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        let e = compile_either(&d);
        assert!(e.accepts(br#""2023-01-15""#));
        assert!(
            !e.accepts(br#""2024-01-15""#),
            "fails the year prefix pattern"
        );
    }

    #[test]
    fn format_with_a_length_bound_errors_cleanly_not_a_hang() {
        let d = schema_to_ir(
            r#"{"type":"string","format":"date","minLength":10,"maxLength":10}"#,
            asserting(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::compile::compile_ir(&d).is_err());
    }

    #[test]
    fn format_with_an_unanchored_pattern_that_can_never_match_is_empty_not_a_hang() {
        let d = schema_to_ir(
            r#"{"type":"string","format":"date","pattern":"a"}"#,
            asserting(),
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        for date in ["\"2024-01-01\"", "\"1999-12-31\"", "\"0001-01-01\""] {
            assert!(!crate::structured::accepts(&d, date.as_bytes()));
        }
    }

    fn accepts(schema: &str, instance: &str) -> bool {
        let ir = schema_to_ir(schema, CompileOptions::default()).unwrap();
        assert_eq!(
            ir.diagnostics().len(),
            0,
            "schema should be fully supported: {schema}"
        );
        crate::structured::accepts(&ir, instance.as_bytes())
    }

    #[test]
    fn github_workflow_style_env_map_with_pattern_and_additional_false() {
        let schema = r#"{
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "patternProperties": {"^[A-Z_]+$": {"type": "string"}},
            "additionalProperties": false
        }"#;
        assert!(accepts(schema, r#"{"name":"build","PATH":"x","HOME":"y"}"#));
        assert!(
            accepts(schema, r#"{"PATH":"x","name":"build"}"#),
            "order independent"
        );
        assert!(
            !accepts(schema, r#"{"name":"build","lowercase":"x"}"#),
            "no matching pattern, forbidden"
        );
        assert!(
            !accepts(schema, r#"{"name":"build","PATH":123}"#),
            "wrong pattern value type"
        );
        assert!(!accepts(schema, r#"{"PATH":"x"}"#), "required name missing");
    }

    #[test]
    fn kubernetes_style_metadata_with_open_labels() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "apiVersion": {"type": "string"},
                "kind": {"type": "string"}
            },
            "required": ["apiVersion", "kind"],
            "additionalProperties": true
        }"#;
        assert!(accepts(
            schema,
            r#"{"apiVersion":"v1","kind":"Pod","metadata":{"name":"x"},"spec":[1,2,3]}"#
        ));
        assert!(
            accepts(schema, r#"{"kind":"Pod","apiVersion":"v1"}"#),
            "no extras, order swapped"
        );
        assert!(!accepts(schema, r#"{"kind":"Pod"}"#), "apiVersion missing");
    }

    #[test]
    fn kitchen_sink_open_object_all_keywords_together() {
        let schema = r#"{
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
            "patternProperties": {"^opt_": {"type": "boolean"}},
            "additionalProperties": {"type": "string"},
            "propertyNames": {"type": "string", "pattern": "^[a-z_]+$"},
            "minProperties": 1,
            "maxProperties": 4
        }"#;
        assert!(accepts(schema, r#"{"id":1,"opt_x":true,"note":"hi"}"#));
        assert!(
            !accepts(schema, r#"{"id":1,"opt_x":"not bool"}"#),
            "opt_ pattern needs boolean"
        );
        assert!(
            !accepts(schema, r#"{"id":1,"note":123}"#),
            "additional needs string"
        );
        assert!(
            !accepts(schema, r#"{"id":1,"Bad":"x"}"#),
            "uppercase key violates propertyNames"
        );
        assert!(
            !accepts(schema, r#"{"id":1,"a":"1","b":"2","c":"3","d":"4"}"#),
            "6 properties exceeds maxProperties 4"
        );
    }

    #[test]
    fn nested_open_objects_two_levels_deep_via_real_schema_text() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner_id": {"type": "integer"}},
                    "required": ["inner_id"],
                    "additionalProperties": {"type": "boolean"}
                }
            },
            "required": ["outer"],
            "additionalProperties": false
        }"#;
        assert!(accepts(schema, r#"{"outer":{"inner_id":1,"flag":true}}"#));
        assert!(!accepts(
            schema,
            r#"{"outer":{"inner_id":1,"flag":"nope"}}"#
        ));
        assert!(!accepts(schema, r#"{"outer":{"inner_id":1},"extra":1}"#));
    }

    #[test]
    fn open_object_nested_in_array_via_real_schema_text() {
        let schema = r#"{
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": true
            }
        }"#;
        let ir = schema_to_ir(schema, CompileOptions::default()).unwrap();
        assert_eq!(ir.diagnostics().len(), 0);
        assert!(crate::structured::accepts(
            &ir,
            br#"[{"ok":true},{"ok":false,"note":"x"}]"#
        ));
        assert!(!crate::structured::accepts(&ir, br#"[{"note":"x"}]"#));
    }

    #[test]
    fn pattern_properties_malformed_shape_is_rejected() {
        let e = ir(r#"{"type":"object","patternProperties":["not","an","object"]}"#).unwrap_err();
        assert_eq!(e.code, ErrorCode::Malformed);
    }

    #[test]
    fn min_properties_above_max_properties_is_empty() {
        let d = ir(r#"{"type":"object","minProperties":5,"maxProperties":2}"#).unwrap();
        assert!(!compile_either(&d).accepts(br#"{}"#));
    }

    #[test]
    fn open_additional_properties_is_open_by_default_and_rejectable_in_strict_mode() {
        let schema = r#"{"type":"object","properties":{"a":{"type":"boolean"}}}"#;
        let d = ir(schema).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"a":true,"b":1}"#));
        let strict_d = schema_to_ir(schema, strict()).unwrap();
        assert_eq!(
            strict_d.diagnostics().next().unwrap().reason,
            UnsupportedReason::OpenAdditionalProperties
        );
    }

    #[test]
    fn open_by_default_composes_with_allof_anyof_ref() {
        let d = ir(r##"{"allOf":[{"properties":{"a":{"type":"boolean"}}}],"$defs":{"x":{"properties":{"c":{"type":"integer"}}}},"anyOf":[{"properties":{"b":{"type":"string"}}},{"$ref":"#/$defs/x"}]}"##).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(
            &d,
            br#"{"a":true,"b":"x","stray":1}"#
        ));
        assert!(crate::structured::accepts(
            &d,
            br#"{"a":true,"c":5,"stray":1}"#
        ));
    }

    #[test]
    fn open_by_default_still_isolated_under_unevaluated_properties_scope() {
        let d =
            ir(r#"{"properties":{"a":{"type":"boolean"}},"unevaluatedProperties":false}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"a":true}"#));
        assert!(
            !crate::structured::accepts(&d, br#"{"a":true,"stray":1}"#),
            "unevaluatedProperties:false must still close the object, not inherit open-by-default"
        );
    }

    #[test]
    fn open_by_default_nested_property_value_stays_its_own_scope() {
        let d = ir(r#"{"properties":{"inner":{"properties":{"x":{"type":"boolean"}},"additionalProperties":false}}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(
            &d,
            br#"{"inner":{"x":true},"outer_stray":1}"#
        ));
        assert!(
            !crate::structured::accepts(&d, br#"{"inner":{"x":true,"inner_stray":1}}"#),
            "the nested object's own additionalProperties:false must still forbid its stray key"
        );
    }

    #[test]
    fn pattern_properties_alone_implies_open_regardless_of_default_policy() {
        let d = ir(r#"{"type":"object","patternProperties":{"^x":{"type":"boolean"}}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(
            &d,
            br#"{"xa":true,"anything":123}"#
        ));
    }

    #[test]
    fn bare_empty_schema_accepts_every_instance_shape() {
        let d = ir("{}").unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        for instance in [
            "null",
            "true",
            "false",
            r#""hello""#,
            "42",
            "-3.5",
            "[]",
            "[1,2,3]",
            "{}",
            r#"{"a":1}"#,
        ] {
            assert!(
                crate::structured::accepts(&d, instance.as_bytes()),
                "typeless schema should accept {instance}"
            );
        }
    }

    #[test]
    fn typeless_min_length_constrains_strings_only_and_ignores_other_shapes() {
        let d = ir(r#"{"minLength":3}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#""abc""#));
        assert!(!crate::structured::accepts(&d, br#""ab""#));
        for instance in ["null", "true", "1", "[]", "{}", "[1]"] {
            assert!(
                crate::structured::accepts(&d, instance.as_bytes()),
                "minLength must not filter out non-string shape {instance}"
            );
        }
    }

    #[test]
    fn typeless_minimum_constrains_numbers_only_and_ignores_other_shapes() {
        let d = ir(r#"{"minimum":10}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, b"10"));
        assert!(crate::structured::accepts(&d, b"11"));
        assert!(!crate::structured::accepts(&d, b"9"));
        for instance in ["null", "true", r#""x""#, "[]", "{}"] {
            assert!(
                crate::structured::accepts(&d, instance.as_bytes()),
                "minimum must not filter out non-number shape {instance}"
            );
        }
    }

    #[test]
    fn typeless_min_items_constrains_arrays_only_and_ignores_other_shapes() {
        let d = ir(r#"{"minItems":2}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, b"[1,2]"));
        assert!(!crate::structured::accepts(&d, b"[1]"));
        for instance in ["null", "true", "1", r#""x""#, "{}"] {
            assert!(
                crate::structured::accepts(&d, instance.as_bytes()),
                "minItems must not filter out non-array shape {instance}"
            );
        }
    }

    #[test]
    fn typeless_prefix_items_still_only_filters_array_shaped_instances() {
        let d = ir(r#"{"prefixItems":[{"type":"integer"}]}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, b"[1]"));
        assert!(!crate::structured::accepts(&d, br#"["a"]"#));
        assert!(
            crate::structured::accepts(&d, br#""not an array""#),
            "a typeless schema with prefixItems still accepts non-array shapes"
        );
    }

    #[test]
    fn typeless_object_only_keyword_is_open_by_default_and_rejectable_in_strict_mode() {
        let schema = r#"{"properties":{"a":{"type":"boolean"}}}"#;
        let d = ir(schema).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"a":true,"b":1}"#));
        let strict_d = schema_to_ir(schema, strict()).unwrap();
        assert_eq!(
            strict_d.diagnostics().next().unwrap().reason,
            UnsupportedReason::OpenAdditionalProperties
        );
    }

    #[test]
    fn typeless_closed_object_keyword_composes_with_every_other_shape() {
        let d =
            ir(r#"{"properties":{"a":{"type":"boolean"}},"additionalProperties":false}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"a":true}"#));
        assert!(!crate::structured::accepts(&d, br#"{"a":true,"b":1}"#));
        for instance in ["null", "1", r#""x""#, "[]"] {
            assert!(
                crate::structured::accepts(&d, instance.as_bytes()),
                "the object-only keyword must not filter out non-object shape {instance}"
            );
        }
    }

    #[test]
    fn typeless_const_and_enum_ignore_the_union_entirely() {
        assert!(crate::structured::accepts(
            &ir(r#"{"const":5}"#).unwrap(),
            b"5"
        ));
        assert!(!crate::structured::accepts(
            &ir(r#"{"const":5}"#).unwrap(),
            b"6"
        ));
        let d = ir(r#"{"enum":[1,"a",null]}"#).unwrap();
        assert!(crate::structured::accepts(&d, b"1"));
        assert!(crate::structured::accepts(&d, b"null"));
        assert!(!crate::structured::accepts(&d, b"2"));
    }

    #[test]
    fn bare_true_schema_root_accepts_everything() {
        let d = ir("true").unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        for instance in ["null", "true", "1", r#""x""#, "[]", "{}"] {
            assert!(crate::structured::accepts(&d, instance.as_bytes()));
        }
    }

    #[test]
    fn bare_false_schema_root_accepts_nothing() {
        let d = ir("false").unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        for instance in ["null", "true", "1", r#""x""#, "[]", "{}"] {
            assert!(!crate::structured::accepts(&d, instance.as_bytes()));
        }
    }

    #[test]
    fn bare_boolean_as_a_property_value_gates_that_property_only() {
        let d = ir(
            r#"{"type":"object","properties":{"ok":true,"never":false},"additionalProperties":false}"#,
        )
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"ok":"anything"}"#));
        assert!(crate::structured::accepts(&d, br#"{}"#));
        assert!(!crate::structured::accepts(&d, br#"{"never":1}"#));
    }

    #[test]
    fn bare_boolean_as_an_array_item_schema() {
        let allow_any = ir(r#"{"type":"array","items":true}"#).unwrap();
        assert_eq!(allow_any.diagnostics().len(), 0);
        assert!(crate::structured::accepts(
            &allow_any,
            br#"[1,"a",null,{}]"#
        ));

        let allow_none = ir(r#"{"type":"array","items":false}"#).unwrap();
        assert_eq!(allow_none.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&allow_none, b"[]"));
        assert!(!crate::structured::accepts(&allow_none, b"[1]"));
    }

    #[test]
    fn bare_boolean_as_a_not_target() {
        let never = ir(r#"{"not":true}"#).unwrap();
        assert_eq!(never.diagnostics().len(), 0);
        assert!(!crate::structured::accepts(&never, b"1"));

        let always = ir(r#"{"not":false}"#).unwrap();
        assert_eq!(always.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&always, b"1"));
        assert!(crate::structured::accepts(&always, br#"{"a":1}"#));
    }

    #[test]
    fn bare_boolean_as_a_dependent_schema() {
        let d =
            ir(r#"{"type":"object","dependentSchemas":{"a":false},"additionalProperties":true}"#)
                .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"b":1}"#));
        assert!(!crate::structured::accepts(&d, br#"{"a":1}"#));
    }

    #[test]
    fn legacy_dependencies_array_form_is_a_dependent_required_alias() {
        let d = ir(r#"{"type":"object","dependencies":{"a":["b"]}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"a":1,"b":2}"#));
        assert!(!crate::structured::accepts(&d, br#"{"a":1}"#));
        assert!(crate::structured::accepts(&d, br#"{"c":1}"#));
    }

    #[test]
    fn legacy_dependencies_schema_form_is_a_dependent_schemas_alias() {
        let d = ir(r#"{"type":"object","dependencies":{"a":{"required":["b"]}}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"a":1,"b":2}"#));
        assert!(!crate::structured::accepts(&d, br#"{"a":1}"#));
        assert!(crate::structured::accepts(&d, br#"{"c":1}"#));
    }

    #[test]
    fn legacy_dependencies_mixes_array_and_schema_entries() {
        let d =
            ir(r#"{"type":"object","dependencies":{"a":["b"],"c":{"required":["d"]}}}"#).unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&d, br#"{"a":1,"b":2}"#));
        assert!(!crate::structured::accepts(&d, br#"{"a":1}"#));
        assert!(crate::structured::accepts(&d, br#"{"c":1,"d":2}"#));
        assert!(!crate::structured::accepts(&d, br#"{"c":1}"#));
    }

    #[test]
    fn bare_boolean_as_a_contains_target() {
        let must_be_empty_or_all_reject = ir(r#"{"type":"array","contains":false}"#).unwrap();
        assert_eq!(must_be_empty_or_all_reject.diagnostics().len(), 0);
        assert!(!crate::structured::accepts(
            &must_be_empty_or_all_reject,
            b"[1]"
        ));

        let any_nonempty_array = ir(r#"{"type":"array","contains":true}"#).unwrap();
        assert_eq!(any_nonempty_array.diagnostics().len(), 0);
        assert!(crate::structured::accepts(&any_nonempty_array, b"[1]"));
        assert!(!crate::structured::accepts(&any_nonempty_array, b"[]"));
    }

    #[test]
    fn dependent_required_frontend_shapes_are_strict_and_precise() {
        let cases = [
            (r#"{"dependentRequired":[]}"#, "/dependentRequired"),
            (
                r#"{"dependentRequired":{"a":true}}"#,
                "/dependentRequired/a",
            ),
            (
                r#"{"dependentRequired":{"a":[1]}}"#,
                "/dependentRequired/a/0",
            ),
            (
                r#"{"dependentRequired":{"a/b~c":[1]}}"#,
                "/dependentRequired/a~1b~0c/0",
            ),
            (
                r#"{"dependentRequired":{"a":["b","b"]}}"#,
                "/dependentRequired/a/1",
            ),
        ];
        for (schema, pointer) in cases {
            let error = ir(schema).unwrap_err();
            assert_eq!(error.code, ErrorCode::Malformed, "{schema}");
            assert_eq!(
                error.json_pointer_path.as_deref(),
                Some(pointer),
                "{schema}"
            );
        }
        let duplicate = ir(r#"{"dependentRequired":{"a":[],"a":["b"]}}"#).unwrap_err();
        assert_eq!(duplicate.code, ErrorCode::Malformed);
        assert_eq!(
            duplicate.json_pointer_path.as_deref(),
            Some("/dependentRequired/a")
        );
    }

    #[test]
    fn dependent_required_semantics_cover_order_unicode_escapes_and_non_objects() {
        let d = ir(r#"{
            "dependentRequired":{
                "a":["b","c"],
                "self":["self"],
                "é":["雪"],
                "line\nname":["slash/name"]
            }
        }"#)
        .unwrap();
        assert_eq!(d.diagnostics().len(), 0);
        assert!(d.requires_structured_backend());
        for valid in [
            br#"{}"#.as_slice(),
            br#"{"b":1}"#,
            br#"{"b":1,"c":2,"a":3}"#,
            br#"{"a":1,"c":2,"b":3}"#,
            br#"{"self":1}"#,
            br#"{"\u00e9":1,"\u96ea":2}"#,
            br#"{"line\nname":1,"slash/name":2}"#,
            br#"[]"#,
            br#""text""#,
            br#"null"#,
        ] {
            assert!(crate::structured::accepts(&d, valid), "{valid:?}");
        }
        for invalid in [
            br#"{"a":1}"#.as_slice(),
            br#"{"a":1,"b":2}"#,
            br#"{"a":1,"c":2}"#,
            "{\"é\":1}".as_bytes(),
            br#"{"line\nname":1}"#,
        ] {
            assert!(!crate::structured::accepts(&d, invalid), "{invalid:?}");
        }
    }

    #[test]
    fn dependent_required_empty_array_is_noop_and_typeless_keeps_non_object_branches() {
        let empty = ir(r#"{"dependentRequired":{"a":[]}}"#).unwrap();
        assert_eq!(empty.diagnostics().len(), 0);
        for instance in [br#"{"a":1}"#.as_slice(), br#"42"#, br#"false"#] {
            assert!(crate::structured::accepts(&empty, instance));
        }
    }

    #[test]
    fn lexical_integer_digit_bounds_compile_on_regular_and_structured_paths() {
        let schema = r#"{"type":"integer","minDigits":3,"maxDigits":5}"#;
        let direct = ir(schema).unwrap();
        let engine = crate::compile::compile_ir(&direct).unwrap();
        for valid in [b"100".as_slice(), b"-9999", b"10000"] {
            assert!(engine.accepts(valid), "{valid:?}");
        }
        for invalid in [b"0".as_slice(), b"10", b"100000", b"01"] {
            assert!(!engine.accepts(invalid), "{invalid:?}");
        }

        let nested = ir(&format!(
            r#"{{"type":"object","properties":{{"n":{schema}}},"required":["n"]}}"#
        ))
        .unwrap();
        assert!(nested.requires_structured_backend());
        let program = crate::StructuredProgram::compile(std::sync::Arc::new(nested)).unwrap();
        let mut matcher = program.new_matcher().unwrap();
        assert!(matcher.advance(br#"{"n":123}"#).unwrap());
        assert!(matcher.eos_legal());
    }

    #[test]
    fn lexical_number_component_digit_bounds_are_exact() {
        let schema = r#"{"type":"number","minDigitsInteger":3,"maxDigitsInteger":5,"minDigitsFraction":3,"maxDigitsFraction":5,"minDigitsExponent":3,"maxDigitsExponent":5}"#;
        let compiled = ir(schema).unwrap();
        let engine = crate::compile::compile_ir(&compiled).unwrap();
        for valid in [
            b"100".as_slice(),
            b"100.005",
            b"100e+001",
            b"100.005e-00001",
        ] {
            assert!(engine.accepts(valid), "{valid:?}");
        }
        for invalid in [b"10.005".as_slice(), b"100.05", b"100e+01", b"100e001"] {
            assert!(!engine.accepts(invalid), "{invalid:?}");
        }
    }

    #[test]
    fn lexical_digit_bounds_reject_invalid_ranges_and_types() {
        for schema in [
            r#"{"type":"integer","minDigits":5,"maxDigits":3}"#,
            r#"{"type":"number","minDigitsFraction":3,"maxDigitsFraction":0}"#,
            r#"{"type":"integer","minDigits":1.5}"#,
        ] {
            assert_eq!(
                ir(schema).unwrap_err().code,
                ErrorCode::Malformed,
                "{schema}"
            );
        }
    }
}
