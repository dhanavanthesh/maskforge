//! Discovers and indexes Draft 2020-12 schema resources.

use std::mem::size_of;
use std::sync::Arc;

use fluent_uri::{Uri, UriRef};
use rustc_hash::{FxHashMap, FxHashSet};

use super::ast::{self, Ast};
use crate::error::{CompileError, ErrorCode, LimitKind, Stage};
use crate::ir::{AnchorId, ResourceId};

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DocumentId(pub(crate) u32);

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SchemaLocationId(u32);

impl SchemaLocationId {
    pub(crate) const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SchemaResourceLimits {
    pub max_documents: usize,
    pub max_resources: usize,
    pub max_total_input_bytes: usize,
    pub max_document_bytes: usize,
    pub max_ast_nodes: usize,
    pub max_ast_bytes: usize,
    pub max_uri_bytes: usize,
    pub max_total_uri_bytes: usize,
    pub max_anchors_per_resource: usize,
    pub max_total_anchors: usize,
    pub max_references: usize,
    pub max_schema_locations: usize,
    pub max_resource_depth: u32,
    pub max_resolution_work: usize,
    pub max_retained_graph_bytes: usize,
    pub max_peak_build_bytes: usize,
}

impl Default for SchemaResourceLimits {
    fn default() -> Self {
        Self {
            max_documents: 128,
            max_resources: 4_096,
            max_total_input_bytes: 16 << 20,
            max_document_bytes: 4 << 20,
            max_ast_nodes: 1_000_000,
            max_ast_bytes: 128 << 20,
            max_uri_bytes: 16 << 10,
            max_total_uri_bytes: 4 << 20,
            max_anchors_per_resource: 4_096,
            max_total_anchors: 65_536,
            max_references: 1_000_000,
            max_schema_locations: 1_000_000,
            max_resource_depth: 1_024,
            max_resolution_work: 4_000_000,
            max_retained_graph_bytes: 128 << 20,
            max_peak_build_bytes: 256 << 20,
        }
    }
}

#[derive(Clone, Debug)]
struct RegisteredDocument {
    retrieval_uri: Option<Arc<str>>,
    text: Arc<str>,
}

#[derive(Debug)]
pub struct SchemaRegistry {
    documents: Vec<RegisteredDocument>,
    retrieval_index: FxHashMap<Arc<str>, DocumentId>,
    limits: SchemaResourceLimits,
    total_input_bytes: usize,
    total_uri_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct ParsedDocument {
    pub(crate) retrieval_uri: Option<Arc<str>>,
    pub(crate) text: Arc<str>,
    pub(crate) ast: Ast,
    pub(crate) locations: FxHashMap<Box<str>, SchemaLocationId>,
    ast_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct SchemaLocation {
    pub(crate) document: DocumentId,
    pub(crate) resource: ResourceId,
    pub(crate) document_pointer: Box<str>,
    pub(crate) base_uri: Option<Arc<str>>,
}

#[derive(Debug)]
pub(crate) struct SchemaResource {
    pub(crate) canonical_uri: Option<Arc<str>>,
    pub(crate) document: DocumentId,
    pub(crate) root: SchemaLocationId,
    pub(crate) static_anchors: Box<[AnchorId]>,
    pub(crate) dynamic_anchors: Box<[AnchorId]>,
    pub(crate) locations: FxHashMap<Box<str>, SchemaLocationId>,
    pub(crate) anchor_names: FxHashMap<Box<str>, AnchorId>,
}

#[derive(Debug)]
pub(crate) struct AnchorRecord {
    pub(crate) resource: ResourceId,
    pub(crate) name: Box<str>,
    pub(crate) target: SchemaLocationId,
    pub(crate) dynamic: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ReferenceKind {
    Static,
    Dynamic,
}

impl ReferenceKind {
    const fn index(self) -> usize {
        match self {
            Self::Static => 0,
            Self::Dynamic => 1,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ReferenceRecord {
    pub(crate) location: SchemaLocationId,
    pub(crate) kind: ReferenceKind,
    pub(crate) text: Box<str>,
    pub(crate) base_uri: Option<Arc<str>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedReferenceRecord {
    pub(crate) source: SchemaLocationId,
    pub(crate) kind: ReferenceKind,
    pub(crate) resolved: ResolvedReference,
}

#[derive(Debug)]
pub(crate) struct ResourceGraph {
    pub(crate) documents: Box<[ParsedDocument]>,
    pub(crate) resources: Box<[SchemaResource]>,
    pub(crate) locations: Box<[SchemaLocation]>,
    pub(crate) anchors: Box<[AnchorRecord]>,
    pub(crate) resolved_references: Box<[ResolvedReferenceRecord]>,
    reference_index: Box<[[u32; 2]]>,
    pub(crate) uri_index: FxHashMap<Arc<str>, ResourceId>,
    pub(crate) retained_bytes: usize,
    pub(crate) peak_build_bytes: usize,
}

#[cfg(feature = "bench-internals")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ResourceDiscoveryProfile {
    pub(crate) document_parse_ns: u64,
    pub(crate) resource_uri_anchor_index_ns: u64,
    pub(crate) canonicalize_ns: u64,
    pub(crate) reference_resolution_ns: u64,
    pub(crate) graph_freeze_ns: u64,
}

#[derive(Debug)]
pub(crate) struct ResolvedUri {
    pub(crate) resource: Arc<str>,
    pub(crate) fragment: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FragmentTarget {
    ResourceRoot(SchemaLocationId),
    JsonPointer(SchemaLocationId),
    Anchor(AnchorId),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ResolvedReference {
    pub(crate) resource: ResourceId,
    pub(crate) target: FragmentTarget,
}

impl ResourceGraph {
    pub(crate) fn root_location(
        &self,
        retrieval_uri: Option<&str>,
        root_text: &str,
    ) -> Result<SchemaLocationId, CompileError> {
        if let Some(uri) = retrieval_uri {
            let canonical = parse_absolute_resource_uri(uri, "retrieval URI")?;
            let resource = self
                .uri_index
                .get(canonical.as_str())
                .copied()
                .ok_or_else(|| {
                    resource_error(
                        ErrorCode::ReferenceResolution,
                        "root retrieval URI is not registered",
                        Some(canonical),
                    )
                })?;
            return self
                .resources
                .get(usize::try_from(resource.get()).map_err(|_| limit_overflow())?)
                .map(|record| record.root)
                .ok_or_else(|| malformed("root resource ID is invalid"));
        }
        let mut found = None;
        for (index, document) in self.documents.iter().enumerate() {
            if document.retrieval_uri.is_none() && document.text.as_ref() == root_text {
                if found.is_some() {
                    return Err(malformed("anonymous root document is ambiguous"));
                }
                let _ = u32::try_from(index).map_err(|_| limit_overflow())?;
                found = document.locations.get("").copied();
            }
        }
        found.ok_or_else(|| malformed("anonymous root document is not registered"))
    }

    pub(crate) fn document_location(
        &self,
        document: DocumentId,
        pointer: &str,
    ) -> Option<SchemaLocationId> {
        self.documents
            .get(usize::try_from(document.0).ok()?)?
            .locations
            .get(pointer)
            .copied()
    }

    pub(crate) fn location_ast(&self, location: SchemaLocationId) -> Result<&Ast, CompileError> {
        let location = self
            .locations
            .get(usize::try_from(location.0).map_err(|_| limit_overflow())?)
            .ok_or_else(|| malformed("schema location ID is invalid"))?;
        let document = self
            .documents
            .get(usize::try_from(location.document.0).map_err(|_| limit_overflow())?)
            .ok_or_else(|| malformed("document ID is invalid"))?;
        ast_at_pointer(&document.ast, &location.document_pointer)
    }

    pub(crate) fn target_location(
        &self,
        target: FragmentTarget,
    ) -> Result<SchemaLocationId, CompileError> {
        match target {
            FragmentTarget::ResourceRoot(location) | FragmentTarget::JsonPointer(location) => {
                Ok(location)
            }
            FragmentTarget::Anchor(anchor) => self
                .anchors
                .get(usize::try_from(anchor.get()).map_err(|_| limit_overflow())?)
                .map(|record| record.target)
                .ok_or_else(|| malformed("anchor ID is invalid")),
        }
    }

    #[cfg(test)]
    pub(crate) fn resolve_reference(
        &self,
        base_uri: Option<&str>,
        reference: &str,
    ) -> Result<ResolvedReference, CompileError> {
        let resolved = resolve_uri_reference(base_uri, reference)?;
        let resource = self.uri_index.get(resolved.resource.as_ref()).copied();
        resolve_fragment_target(
            &self.resources,
            resource,
            resolved.fragment.as_deref(),
            reference,
        )
    }

    pub(crate) fn resolved_reference_from(
        &self,
        location: SchemaLocationId,
        kind: ReferenceKind,
    ) -> Result<ResolvedReference, CompileError> {
        let index = self
            .reference_index
            .get(usize::try_from(location.0).map_err(|_| limit_overflow())?)
            .ok_or_else(|| malformed("reference location index is invalid"))?[kind.index()];
        let record = self
            .resolved_references
            .get(usize::try_from(index).map_err(|_| limit_overflow())?)
            .ok_or_else(|| malformed("resolved reference occurrence is not indexed"))?;
        if record.source != location || record.kind != kind {
            return Err(malformed("resolved reference occurrence is inconsistent"));
        }
        Ok(record.resolved)
    }
}

#[cfg(test)]
fn resolve_fragment_target(
    resources: &[SchemaResource],
    resource: Option<ResourceId>,
    fragment: Option<&str>,
    reference: &str,
) -> Result<ResolvedReference, CompileError> {
    let resource = resource.ok_or_else(|| {
        resource_error(
            ErrorCode::ReferenceResolution,
            "schema resource is not registered",
            Some(reference.to_owned()),
        )
    })?;
    let resource_record = resources
        .get(usize::try_from(resource.0).map_err(|_| limit_overflow())?)
        .ok_or_else(|| malformed("resource ID is invalid"))?;
    let target = match fragment {
        None | Some("") => FragmentTarget::ResourceRoot(resource_record.root),
        Some(pointer) if pointer.starts_with('/') => {
            validate_json_pointer(pointer)?;
            let location = resource_record
                .locations
                .get(pointer)
                .copied()
                .ok_or_else(|| {
                    resource_error(
                        ErrorCode::ReferenceResolution,
                        "JSON Pointer does not identify a schema location",
                        Some(reference.to_owned()),
                    )
                })?;
            FragmentTarget::JsonPointer(location)
        }
        Some(anchor) => {
            if !valid_anchor(anchor) {
                return Err(resource_error(
                    ErrorCode::ReferenceResolution,
                    "plain-name fragment is not a valid anchor",
                    Some(reference.to_owned()),
                ));
            }
            let anchor = resource_record
                .anchor_names
                .get(anchor)
                .copied()
                .ok_or_else(|| {
                    resource_error(
                        ErrorCode::ReferenceResolution,
                        "plain-name anchor is not registered in the resource",
                        Some(reference.to_owned()),
                    )
                })?;
            FragmentTarget::Anchor(anchor)
        }
    };
    Ok(ResolvedReference { resource, target })
}

impl SchemaRegistry {
    #[must_use]
    pub fn new(limits: SchemaResourceLimits) -> Self {
        Self {
            documents: Vec::new(),
            retrieval_index: FxHashMap::default(),
            limits,
            total_input_bytes: 0,
            total_uri_bytes: 0,
        }
    }

    pub fn insert(
        &mut self,
        retrieval_uri: impl AsRef<str>,
        schema_text: impl Into<Arc<str>>,
    ) -> Result<DocumentId, CompileError> {
        self.insert_inner(Some(retrieval_uri.as_ref()), schema_text.into())
    }

    pub fn insert_anonymous(
        &mut self,
        schema_text: impl Into<Arc<str>>,
    ) -> Result<DocumentId, CompileError> {
        self.insert_inner(None, schema_text.into())
    }

    fn insert_inner(
        &mut self,
        retrieval_uri: Option<&str>,
        text: Arc<str>,
    ) -> Result<DocumentId, CompileError> {
        check_limit(
            LimitKind::DocumentCount,
            self.documents.len().checked_add(1),
            self.limits.max_documents,
        )?;
        check_limit(
            LimitKind::InputBytes,
            Some(text.len()),
            self.limits.max_document_bytes,
        )?;
        let total_input = self.total_input_bytes.checked_add(text.len());
        check_limit(
            LimitKind::InputBytes,
            total_input,
            self.limits.max_total_input_bytes,
        )?;

        let retrieval_uri = match retrieval_uri {
            Some(uri) => {
                let canonical = parse_absolute_resource_uri(uri, "retrieval URI")?;
                check_limit(
                    LimitKind::UriBytes,
                    Some(canonical.len()),
                    self.limits.max_uri_bytes,
                )?;
                let total_uri = self.total_uri_bytes.checked_add(canonical.len());
                check_limit(
                    LimitKind::UriBytes,
                    total_uri,
                    self.limits.max_total_uri_bytes,
                )?;
                if self.retrieval_index.contains_key(canonical.as_str()) {
                    return Err(resource_error(
                        ErrorCode::DuplicateResource,
                        "duplicate retrieval URI",
                        Some(canonical),
                    ));
                }
                self.total_uri_bytes = total_uri.ok_or_else(limit_overflow)?;
                Some(Arc::<str>::from(canonical))
            }
            None => None,
        };

        let id = DocumentId(u32::try_from(self.documents.len()).map_err(|_| limit_overflow())?);
        self.documents
            .try_reserve_exact(1)
            .map_err(|_| limit_overflow())?;
        if let Some(uri) = &retrieval_uri {
            self.retrieval_index
                .try_reserve(1)
                .map_err(|_| limit_overflow())?;
            self.retrieval_index.insert(uri.clone(), id);
        }
        self.documents.push(RegisteredDocument {
            retrieval_uri,
            text,
        });
        self.total_input_bytes = total_input.ok_or_else(limit_overflow)?;
        Ok(id)
    }

    pub(crate) fn discover(self) -> Result<ResourceGraph, CompileError> {
        ResourceBuilder::new(self)?.discover()
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn discover_profiled(
        self,
    ) -> Result<(ResourceGraph, ResourceDiscoveryProfile), CompileError> {
        let mut profile = ResourceDiscoveryProfile::default();
        let started = std::time::Instant::now();
        let builder = ResourceBuilder::new(self)?;
        profile.document_parse_ns = elapsed_ns(started);
        builder.discover_profiled(profile)
    }
}

#[cfg(all(test, feature = "alloc-probe"))]
pub(crate) fn discover_memory_for_probe(
    registry: SchemaRegistry,
) -> Result<(usize, usize), CompileError> {
    let graph = registry.discover()?;
    Ok((graph.peak_build_bytes, graph.retained_bytes))
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new(SchemaResourceLimits::default())
    }
}

struct MutableResource {
    canonical_uri: Option<Arc<str>>,
    document: DocumentId,
    root: SchemaLocationId,
    static_anchors: Vec<AnchorId>,
    dynamic_anchors: Vec<AnchorId>,
    locations: FxHashMap<Box<str>, SchemaLocationId>,
    anchor_names: FxHashMap<Box<str>, AnchorId>,
}

struct WorkItem<'a> {
    schema: &'a Ast,
    document_pointer: String,
    resource_pointer: String,
    base_uri: Option<Arc<str>>,
    resource: ResourceId,
    resource_depth: u32,
    root: bool,
    legacy_ids: bool,
}

struct ResourceBuilder {
    limits: SchemaResourceLimits,
    documents: Vec<ParsedDocument>,
    resources: Vec<MutableResource>,
    locations: Vec<SchemaLocation>,
    anchors: Vec<AnchorRecord>,
    references: Vec<ReferenceRecord>,
    resolved_references: Vec<ResolvedReferenceRecord>,
    reference_index: Vec<[u32; 2]>,
    uri_index: FxHashMap<Arc<str>, ResourceId>,
    counted_uris: FxHashSet<Arc<str>>,
    memory: ResourceMemory,
    total_uri_bytes: usize,
    work: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct ResourceMemory {
    current_build: usize,
    peak_build: usize,
    retained_frozen: usize,
}

impl ResourceMemory {
    fn observe_build(
        &mut self,
        retained: usize,
        temporary: usize,
        limit: usize,
    ) -> Result<(), CompileError> {
        let total = retained.checked_add(temporary).ok_or_else(limit_overflow)?;
        self.current_build = retained;
        self.peak_build = self.peak_build.max(total);
        check_limit(LimitKind::ResourceBuildBytes, Some(total), limit)
    }

    fn freeze(&mut self, bytes: usize, limit: usize) -> Result<(), CompileError> {
        self.retained_frozen = bytes;
        check_limit(LimitKind::ResourceGraphBytes, Some(bytes), limit)
    }
}

impl ResourceBuilder {
    fn new(registry: SchemaRegistry) -> Result<Self, CompileError> {
        let input_floor = registry_build_bytes(&registry)?;
        check_limit(
            LimitKind::ResourceBuildBytes,
            Some(input_floor),
            registry.limits.max_peak_build_bytes,
        )?;
        let mut registered = registry.documents;
        registered.sort_by(|left, right| {
            left.retrieval_uri
                .as_deref()
                .cmp(&right.retrieval_uri.as_deref())
                .then_with(|| left.text.as_ref().cmp(right.text.as_ref()))
        });
        let mut documents = Vec::new();
        documents
            .try_reserve_exact(registered.len())
            .map_err(|_| limit_overflow())?;
        let parse_floor = input_floor
            .checked_add(vec_bytes(&documents)?)
            .ok_or_else(limit_overflow)?;
        check_limit(
            LimitKind::ResourceBuildBytes,
            Some(parse_floor),
            registry.limits.max_peak_build_bytes,
        )?;
        let mut ast_nodes = 0usize;
        let mut ast_bytes = 0usize;
        for document in registered {
            let remaining_nodes = registry
                .limits
                .max_ast_nodes
                .checked_sub(ast_nodes)
                .ok_or_else(limit_overflow)?;
            let remaining_bytes = registry
                .limits
                .max_ast_bytes
                .checked_sub(ast_bytes)
                .ok_or_else(limit_overflow)?;
            let remaining_build = registry
                .limits
                .max_peak_build_bytes
                .checked_sub(parse_floor)
                .and_then(|bytes| bytes.checked_sub(ast_bytes))
                .ok_or_else(|| {
                    let observed = parse_floor
                        .checked_add(ast_bytes)
                        .map_or(usize::MAX, |value| value);
                    resource_limit_error(
                        LimitKind::ResourceBuildBytes,
                        observed,
                        registry.limits.max_peak_build_bytes,
                    )
                })?;
            let (parsed, parsed_memory) = ast::parse_bounded(
                &document.text,
                remaining_nodes,
                remaining_bytes.min(remaining_build),
            )?;
            ast_nodes = ast_nodes
                .checked_add(parsed_memory.nodes)
                .ok_or_else(limit_overflow)?;
            ast_bytes = ast_bytes
                .checked_add(parsed_memory.bytes)
                .ok_or_else(limit_overflow)?;
            documents.push(ParsedDocument {
                retrieval_uri: document.retrieval_uri,
                text: document.text,
                ast: parsed,
                locations: FxHashMap::default(),
                ast_bytes: parsed_memory.bytes,
            });
        }
        let mut counted_uris = FxHashSet::default();
        counted_uris
            .try_reserve(documents.len())
            .map_err(|_| limit_overflow())?;
        for document in &documents {
            if let Some(uri) = &document.retrieval_uri {
                counted_uris.insert(uri.clone());
            }
        }
        let mut builder = Self {
            limits: registry.limits,
            documents,
            resources: Vec::new(),
            locations: Vec::new(),
            anchors: Vec::new(),
            references: Vec::new(),
            resolved_references: Vec::new(),
            reference_index: Vec::new(),
            uri_index: FxHashMap::default(),
            counted_uris,
            memory: ResourceMemory {
                current_build: 0,
                peak_build: parse_floor
                    .checked_add(ast_bytes)
                    .ok_or_else(limit_overflow)?,
                retained_frozen: 0,
            },
            total_uri_bytes: registry.total_uri_bytes,
            work: 0,
        };
        builder.observe_build_memory(0)?;
        Ok(builder)
    }

    fn discover(mut self) -> Result<ResourceGraph, CompileError> {
        for document_index in 0..self.documents.len() {
            let document = DocumentId(u32::try_from(document_index).map_err(|_| limit_overflow())?);
            self.discover_document(document)?;
        }
        self.materialize_pointer_targets()?;
        self.canonicalize_resource_ids()?;
        self.validate_references()?;
        self.freeze()
    }

    #[cfg(feature = "bench-internals")]
    fn discover_profiled(
        mut self,
        mut profile: ResourceDiscoveryProfile,
    ) -> Result<(ResourceGraph, ResourceDiscoveryProfile), CompileError> {
        let started = std::time::Instant::now();
        for document_index in 0..self.documents.len() {
            let document = DocumentId(u32::try_from(document_index).map_err(|_| limit_overflow())?);
            self.discover_document(document)?;
        }
        self.materialize_pointer_targets()?;
        profile.resource_uri_anchor_index_ns = elapsed_ns(started);

        let started = std::time::Instant::now();
        self.canonicalize_resource_ids()?;
        profile.canonicalize_ns = elapsed_ns(started);

        let started = std::time::Instant::now();
        self.validate_references()?;
        profile.reference_resolution_ns = elapsed_ns(started);

        let started = std::time::Instant::now();
        let graph = self.freeze()?;
        profile.graph_freeze_ns = elapsed_ns(started);
        Ok((graph, profile))
    }

    fn freeze(mut self) -> Result<ResourceGraph, CompileError> {
        let resource_output = self
            .resources
            .len()
            .checked_mul(size_of::<SchemaResource>())
            .ok_or_else(limit_overflow)?;
        self.observe_build_memory(resource_output)?;
        let mut resources = Vec::new();
        resources
            .try_reserve_exact(self.resources.len())
            .map_err(|_| limit_overflow())?;
        for resource in self.resources {
            resources.push(SchemaResource {
                canonical_uri: resource.canonical_uri,
                document: resource.document,
                root: resource.root,
                static_anchors: resource.static_anchors.into_boxed_slice(),
                dynamic_anchors: resource.dynamic_anchors.into_boxed_slice(),
                locations: resource.locations,
                anchor_names: resource.anchor_names,
            });
        }
        let retained_bytes = frozen_graph_bytes(
            &self.documents,
            &resources,
            &self.locations,
            &self.anchors,
            &self.resolved_references,
            &self.reference_index,
            &self.uri_index,
        )?;
        self.memory
            .freeze(retained_bytes, self.limits.max_retained_graph_bytes)?;
        let peak_build_bytes = self.memory.peak_build.max(retained_bytes);
        Ok(ResourceGraph {
            documents: self.documents.into_boxed_slice(),
            resources: resources.into_boxed_slice(),
            locations: self.locations.into_boxed_slice(),
            anchors: self.anchors.into_boxed_slice(),
            resolved_references: self.resolved_references.into_boxed_slice(),
            reference_index: self.reference_index.into_boxed_slice(),
            uri_index: self.uri_index,
            retained_bytes,
            peak_build_bytes,
        })
    }

    fn canonicalize_resource_ids(&mut self) -> Result<(), CompileError> {
        let old = std::mem::take(&mut self.resources);
        let old_bytes = vec_bytes(&old)?;
        let mut indexed = Vec::new();
        indexed
            .try_reserve_exact(old.len())
            .map_err(|_| limit_overflow())?;
        for entry in old.into_iter().enumerate() {
            indexed.push(entry);
        }
        self.observe_build_memory(
            old_bytes
                .checked_add(vec_bytes(&indexed)?)
                .ok_or_else(limit_overflow)?,
        )?;
        indexed.sort_by(|left, right| {
            left.1
                .canonical_uri
                .as_deref()
                .cmp(&right.1.canonical_uri.as_deref())
        });
        let mut remap = Vec::new();
        remap
            .try_reserve_exact(indexed.len())
            .map_err(|_| limit_overflow())?;
        remap.resize(indexed.len(), ResourceId(u32::MAX));
        self.resources
            .try_reserve_exact(indexed.len())
            .map_err(|_| limit_overflow())?;
        self.observe_build_memory(checked_sum([vec_bytes(&indexed)?, vec_bytes(&remap)?])?)?;
        for (new_index, (old_index, resource)) in indexed.into_iter().enumerate() {
            let new = ResourceId(u32::try_from(new_index).map_err(|_| limit_overflow())?);
            remap[old_index] = new;
            self.resources.push(resource);
        }
        for location in &mut self.locations {
            location.resource = *remap
                .get(usize::try_from(location.resource.get()).map_err(|_| limit_overflow())?)
                .ok_or_else(|| malformed("resource remap is invalid"))?;
        }
        for anchor in &mut self.anchors {
            anchor.resource = *remap
                .get(usize::try_from(anchor.resource.get()).map_err(|_| limit_overflow())?)
                .ok_or_else(|| malformed("resource remap is invalid"))?;
        }
        for resource in self.uri_index.values_mut() {
            *resource = *remap
                .get(usize::try_from(resource.get()).map_err(|_| limit_overflow())?)
                .ok_or_else(|| malformed("resource remap is invalid"))?;
        }
        self.observe_build_memory(0)?;
        Ok(())
    }

    fn validate_references(&mut self) -> Result<(), CompileError> {
        self.resolved_references
            .try_reserve_exact(self.references.len())
            .map_err(|_| limit_overflow())?;
        self.reference_index
            .try_reserve_exact(self.locations.len())
            .map_err(|_| limit_overflow())?;
        self.reference_index
            .resize(self.locations.len(), [u32::MAX; 2]);
        for index in 0..self.references.len() {
            self.bump_work(1)?;
            let reference = self
                .references
                .get(index)
                .ok_or_else(|| malformed("reference index is invalid"))?;
            let location = self
                .locations
                .get(usize::try_from(reference.location.0).map_err(|_| limit_overflow())?)
                .ok_or_else(|| malformed("reference location is invalid"))?;
            let resource = self
                .resources
                .get(usize::try_from(location.resource.0).map_err(|_| limit_overflow())?)
                .ok_or_else(|| malformed("reference resource is invalid"))?;
            if resource.document != location.document {
                return Err(malformed("resource and schema location documents differ"));
            }
            let keyword = match reference.kind {
                ReferenceKind::Static => "$ref",
                ReferenceKind::Dynamic => "$dynamicRef",
            };
            let pointer = if location.document_pointer.is_empty() {
                format!("/{keyword}")
            } else {
                format!("{}/{keyword}", location.document_pointer)
            };
            let result = (|| {
                let (target_resource, fragment) = if let Some(base) = reference.base_uri.as_deref()
                {
                    let resolved = resolve_uri_reference(Some(base), &reference.text)?;
                    let resource = self
                        .uri_index
                        .get(resolved.resource.as_ref())
                        .copied()
                        .ok_or_else(|| {
                            resource_error(
                                ErrorCode::ReferenceResolution,
                                "schema resource is not registered",
                                Some(reference.text.to_string()),
                            )
                        })?;
                    (resource, resolved.fragment)
                } else if UriRef::parse(reference.text.as_ref())
                    .map_err(|_| {
                        resource_error(
                            ErrorCode::InvalidUri,
                            "invalid URI reference",
                            Some(reference.text.to_string()),
                        )
                    })?
                    .has_scheme()
                {
                    let resolved = resolve_uri_reference(None, &reference.text)?;
                    let resource = self
                        .uri_index
                        .get(resolved.resource.as_ref())
                        .copied()
                        .ok_or_else(|| {
                            resource_error(
                                ErrorCode::ReferenceResolution,
                                "schema resource is not registered",
                                Some(reference.text.to_string()),
                            )
                        })?;
                    (resource, resolved.fragment)
                } else {
                    (location.resource, same_document_fragment(&reference.text)?)
                };
                let target = self
                    .resources
                    .get(usize::try_from(target_resource.0).map_err(|_| limit_overflow())?)
                    .ok_or_else(|| malformed("resolved resource ID is invalid"))?;
                let target = match fragment.as_deref() {
                    None | Some("") => FragmentTarget::ResourceRoot(target.root),
                    Some(pointer) if pointer.starts_with('/') => {
                        validate_json_pointer(pointer)?;
                        let location = target.locations.get(pointer).copied().ok_or_else(|| {
                            resource_error(
                                ErrorCode::ReferenceResolution,
                                "JSON Pointer does not identify a schema location",
                                Some(reference.text.to_string()),
                            )
                        })?;
                        FragmentTarget::JsonPointer(location)
                    }
                    Some(anchor) => {
                        if !valid_anchor(anchor) {
                            return Err(resource_error(
                                ErrorCode::ReferenceResolution,
                                "plain-name anchor is not registered in the resource",
                                Some(reference.text.to_string()),
                            ));
                        }
                        let anchor = target.anchor_names.get(anchor).copied().ok_or_else(|| {
                            resource_error(
                                ErrorCode::ReferenceResolution,
                                "plain-name anchor is not registered in the resource",
                                Some(reference.text.to_string()),
                            )
                        })?;
                        FragmentTarget::Anchor(anchor)
                    }
                };
                Ok(ResolvedReference {
                    resource: target_resource,
                    target,
                })
            })();
            let resolved = result
                .map_err(|error: CompileError| error.with_pointer(pointer).with_keyword(keyword))?;
            let resolved_index =
                u32::try_from(self.resolved_references.len()).map_err(|_| limit_overflow())?;
            self.resolved_references.push(ResolvedReferenceRecord {
                source: reference.location,
                kind: reference.kind,
                resolved,
            });
            let location_index =
                usize::try_from(reference.location.0).map_err(|_| limit_overflow())?;
            let slot = self
                .reference_index
                .get_mut(location_index)
                .ok_or_else(|| malformed("reference location index is invalid"))?;
            if slot[reference.kind.index()] != u32::MAX {
                return Err(malformed(
                    "duplicate reference occurrence at one schema location",
                ));
            }
            slot[reference.kind.index()] = resolved_index;
        }
        self.observe_build_memory(0)?;
        Ok(())
    }

    fn discover_document(&mut self, document: DocumentId) -> Result<(), CompileError> {
        let index = usize::try_from(document.0).map_err(|_| limit_overflow())?;
        let retrieval_uri = self
            .documents
            .get(index)
            .ok_or_else(|| malformed("document ID is invalid"))?
            .retrieval_uri
            .clone();
        let ast = {
            let parsed = self
                .documents
                .get_mut(index)
                .ok_or_else(|| malformed("document ID is invalid"))?;
            std::mem::replace(&mut parsed.ast, Ast::Bool(false))
        };
        let result = (|| {
            let legacy_ids = object_string(&ast, "$schema")?.is_some_and(is_legacy_metaschema);
            let root_id = object_string(&ast, if legacy_ids { "id" } else { "$id" })?;
            let root_uri = match (root_id, legacy_ids) {
                (Some(id), true) => {
                    Some(resolve_uri_reference(retrieval_uri.as_deref(), id)?.resource)
                }
                (Some(id), false) => Some(Arc::<str>::from(resolve_resource_id(
                    retrieval_uri.as_deref(),
                    id,
                )?)),
                (None, _) => retrieval_uri.clone(),
            };
            let resource = self.push_resource(document, root_uri.clone(), 0)?;
            if let Some(retrieval) = &retrieval_uri {
                if root_uri.as_deref() != Some(retrieval.as_ref()) {
                    self.push_uri_alias(retrieval.clone(), resource)?;
                }
            }
            let mut stack = Vec::new();
            stack.try_reserve(1).map_err(|_| limit_overflow())?;
            stack.push(WorkItem {
                schema: &ast,
                document_pointer: String::new(),
                resource_pointer: String::new(),
                base_uri: root_uri.or(retrieval_uri),
                resource,
                resource_depth: 0,
                root: true,
                legacy_ids,
            });
            self.discover_stack(document, stack)
        })();
        let parsed = self
            .documents
            .get_mut(index)
            .ok_or_else(|| malformed("document ID is invalid"))?;
        parsed.ast = ast;
        result
    }

    fn discover_stack(
        &mut self,
        document: DocumentId,
        mut stack: Vec<WorkItem<'_>>,
    ) -> Result<(), CompileError> {
        while let Some(item) = stack.pop() {
            self.bump_work(1)?;
            let mut active_resource = item.resource;
            let mut active_base = item.base_uri;
            let mut resource_pointer = item.resource_pointer;
            let mut resource_depth = item.resource_depth;
            if !item.root {
                let id_keyword = if item.legacy_ids { "id" } else { "$id" };
                if let Some(id) = object_string(item.schema, id_keyword)? {
                    let legacy = item
                        .legacy_ids
                        .then(|| resolve_uri_reference(active_base.as_deref(), id))
                        .transpose()?;
                    if let Some(resolved) = legacy.filter(|value| {
                        value
                            .fragment
                            .as_deref()
                            .is_some_and(|fragment| !fragment.is_empty())
                    }) {
                        let uri = resolved.resource;
                        self.push_uri_alias(uri.clone(), active_resource)?;
                        active_base = Some(uri);
                    } else {
                        resource_depth =
                            resource_depth.checked_add(1).ok_or_else(limit_overflow)?;
                        check_limit(
                            LimitKind::ResourceDepth,
                            usize::try_from(resource_depth).ok(),
                            usize::try_from(self.limits.max_resource_depth)
                                .map_err(|_| limit_overflow())?,
                        )?;
                        let canonical = resolve_resource_id(active_base.as_deref(), id)?;
                        let uri = Arc::<str>::from(canonical);
                        active_resource =
                            self.push_resource(document, Some(uri.clone()), resource_depth)?;
                        active_base = Some(uri);
                        resource_pointer.clear();
                    }
                }
            }
            if let Some(existing) = self
                .documents
                .get(document.0 as usize)
                .and_then(|entry| entry.locations.get(item.document_pointer.as_str()))
                .and_then(|id| self.locations.get(id.get() as usize))
            {
                if existing.resource == active_resource {
                    continue;
                }
            }
            let location = self.push_location(
                document,
                active_resource,
                &item.document_pointer,
                &resource_pointer,
                active_base.clone(),
            )?;
            if resource_pointer.is_empty() {
                let resource_index =
                    usize::try_from(active_resource.0).map_err(|_| limit_overflow())?;
                let entry = self
                    .resources
                    .get_mut(resource_index)
                    .ok_or_else(|| malformed("resource ID is invalid"))?;
                entry.root = location;
            }
            self.index_keywords(
                item.schema,
                location,
                active_resource,
                active_base.clone(),
                item.legacy_ids,
            )?;
            let placement = ChildPlacement {
                base_uri: active_base,
                resource: active_resource,
                resource_depth,
                legacy_ids: item.legacy_ids,
            };
            push_schema_children(
                item.schema,
                &item.document_pointer,
                &resource_pointer,
                &placement,
                &mut stack,
            )?;
            self.observe_build_memory(work_stack_bytes(&stack)?)?;
        }
        Ok(())
    }

    fn materialize_pointer_targets(&mut self) -> Result<(), CompileError> {
        let mut index = 0usize;
        while index < self.references.len() {
            let reference = self
                .references
                .get(index)
                .ok_or_else(|| malformed("reference index is invalid"))?;
            let source = self
                .locations
                .get(reference.location.get() as usize)
                .ok_or_else(|| malformed("reference location is invalid"))?;
            let source_resource = source.resource;
            let text = reference.text.clone();
            let base = reference.base_uri.clone();
            index = index.checked_add(1).ok_or_else(limit_overflow)?;
            let (resource, fragment) = if let Some(base) = base.as_deref() {
                let resolved = resolve_uri_reference(Some(base), &text)?;
                let Some(resource) = self.uri_index.get(resolved.resource.as_ref()).copied() else {
                    continue;
                };
                (resource, resolved.fragment)
            } else if UriRef::parse(text.as_ref())
                .map_err(|_| resource_error(ErrorCode::InvalidUri, "invalid URI reference", None))?
                .has_scheme()
            {
                let resolved = resolve_uri_reference(None, &text)?;
                let Some(resource) = self.uri_index.get(resolved.resource.as_ref()).copied() else {
                    continue;
                };
                (resource, resolved.fragment)
            } else {
                (source_resource, same_document_fragment(&text)?)
            };
            let Some(pointer) = fragment.as_deref().filter(|value| value.starts_with('/')) else {
                continue;
            };
            validate_json_pointer(pointer)?;
            if self
                .resources
                .get(resource.get() as usize)
                .is_some_and(|entry| entry.locations.contains_key(pointer))
            {
                continue;
            }
            self.materialize_pointer_schema(resource, pointer)?;
        }
        Ok(())
    }

    fn materialize_pointer_schema(
        &mut self,
        resource: ResourceId,
        pointer: &str,
    ) -> Result<(), CompileError> {
        let resource_record = self
            .resources
            .get(resource.get() as usize)
            .ok_or_else(|| malformed("resource ID is invalid"))?;
        let root = self
            .locations
            .get(resource_record.root.get() as usize)
            .ok_or_else(|| malformed("resource root is invalid"))?;
        let document = root.document;
        let document_pointer = if root.document_pointer.is_empty() {
            pointer.to_owned()
        } else {
            format!("{}{pointer}", root.document_pointer)
        };
        let base_uri = resource_record.canonical_uri.clone();
        let document_index = document.0 as usize;
        if let Some(existing) = self
            .documents
            .get(document_index)
            .and_then(|entry| entry.locations.get(document_pointer.as_str()))
            .copied()
        {
            return self.push_location_alias(resource, pointer, existing);
        }
        let ast = {
            let parsed = self
                .documents
                .get_mut(document_index)
                .ok_or_else(|| malformed("document ID is invalid"))?;
            std::mem::replace(&mut parsed.ast, Ast::Bool(false))
        };
        let legacy_ids = object_string(&ast, "$schema")?.is_some_and(is_legacy_metaschema);
        let result = (|| match ast_at_pointer(&ast, &document_pointer) {
            Ok(schema) if matches!(schema, Ast::Obj(_) | Ast::Bool(_)) => {
                let mut stack = Vec::new();
                stack.try_reserve_exact(1).map_err(|_| limit_overflow())?;
                stack.push(WorkItem {
                    schema,
                    document_pointer,
                    resource_pointer: pointer.to_owned(),
                    base_uri,
                    resource,
                    resource_depth: 0,
                    root: true,
                    legacy_ids,
                });
                self.discover_stack(document, stack)
            }
            Ok(_) | Err(_) => Ok(()),
        })();
        self.documents
            .get_mut(document_index)
            .ok_or_else(|| malformed("document ID is invalid"))?
            .ast = ast;
        result
    }

    fn push_resource(
        &mut self,
        document: DocumentId,
        canonical_uri: Option<Arc<str>>,
        _depth: u32,
    ) -> Result<ResourceId, CompileError> {
        check_limit(
            LimitKind::ResourceCount,
            self.resources.len().checked_add(1),
            self.limits.max_resources,
        )?;
        let id = ResourceId(u32::try_from(self.resources.len()).map_err(|_| limit_overflow())?);
        if let Some(uri) = &canonical_uri {
            if self.uri_index.contains_key(uri.as_ref()) {
                return Err(resource_error(
                    ErrorCode::DuplicateResource,
                    "duplicate canonical schema-resource URI",
                    Some(uri.to_string()),
                ));
            }
            check_limit(
                LimitKind::UriBytes,
                Some(uri.len()),
                self.limits.max_uri_bytes,
            )?;
            if self.counted_uris.insert(uri.clone()) {
                self.total_uri_bytes = self
                    .total_uri_bytes
                    .checked_add(uri.len())
                    .ok_or_else(limit_overflow)?;
                check_limit(
                    LimitKind::UriBytes,
                    Some(self.total_uri_bytes),
                    self.limits.max_total_uri_bytes,
                )?;
            }
            self.uri_index
                .try_reserve(1)
                .map_err(|_| limit_overflow())?;
            self.uri_index.insert(uri.clone(), id);
        }
        self.charge(size_of::<MutableResource>())?;
        self.resources
            .try_reserve(1)
            .map_err(|_| limit_overflow())?;
        self.resources.push(MutableResource {
            canonical_uri,
            document,
            root: SchemaLocationId(0),
            static_anchors: Vec::new(),
            dynamic_anchors: Vec::new(),
            locations: FxHashMap::default(),
            anchor_names: FxHashMap::default(),
        });
        self.observe_build_memory(0)?;
        Ok(id)
    }

    fn push_uri_alias(&mut self, uri: Arc<str>, resource: ResourceId) -> Result<(), CompileError> {
        if let Some(existing) = self.uri_index.get(uri.as_ref()) {
            if *existing == resource {
                return Ok(());
            }
            return Err(resource_error(
                ErrorCode::DuplicateResource,
                "retrieval URI collides with another schema resource",
                Some(uri.to_string()),
            ));
        }
        self.uri_index
            .try_reserve(1)
            .map_err(|_| limit_overflow())?;
        self.uri_index.insert(uri, resource);
        self.observe_build_memory(0)?;
        Ok(())
    }

    fn push_location(
        &mut self,
        document: DocumentId,
        resource: ResourceId,
        document_pointer: &str,
        resource_pointer: &str,
        base_uri: Option<Arc<str>>,
    ) -> Result<SchemaLocationId, CompileError> {
        check_limit(
            LimitKind::SchemaLocationCount,
            self.locations.len().checked_add(1),
            self.limits.max_schema_locations,
        )?;
        let bytes = size_of::<SchemaLocation>()
            .checked_add(document_pointer.len())
            .and_then(|value| value.checked_add(resource_pointer.len()))
            .ok_or_else(limit_overflow)?;
        self.charge(bytes)?;
        let id =
            SchemaLocationId(u32::try_from(self.locations.len()).map_err(|_| limit_overflow())?);
        self.locations
            .try_reserve(1)
            .map_err(|_| limit_overflow())?;
        let document_index = usize::try_from(document.0).map_err(|_| limit_overflow())?;
        let parsed_document = self
            .documents
            .get_mut(document_index)
            .ok_or_else(|| malformed("document ID is invalid"))?;
        parsed_document
            .locations
            .try_reserve(1)
            .map_err(|_| limit_overflow())?;
        if parsed_document
            .locations
            .insert(document_pointer.into(), id)
            .is_some()
        {
            return Err(malformed("duplicate physical schema location"));
        }
        let index_key: Box<str> = resource_pointer.into();
        let resource_index = usize::try_from(resource.0).map_err(|_| limit_overflow())?;
        let resource_record = self
            .resources
            .get_mut(resource_index)
            .ok_or_else(|| malformed("resource ID is invalid"))?;
        resource_record
            .locations
            .try_reserve(1)
            .map_err(|_| limit_overflow())?;
        match resource_record.locations.entry(index_key) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(malformed("duplicate schema location inside one resource"));
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(id);
            }
        }
        self.locations.push(SchemaLocation {
            document,
            resource,
            document_pointer: document_pointer.into(),
            base_uri,
        });
        self.observe_build_memory(0)?;
        Ok(id)
    }

    fn push_location_alias(
        &mut self,
        resource: ResourceId,
        pointer: &str,
        location: SchemaLocationId,
    ) -> Result<(), CompileError> {
        self.charge(
            pointer
                .len()
                .checked_add(size_of::<(Box<str>, SchemaLocationId)>())
                .ok_or_else(limit_overflow)?,
        )?;
        let record = self
            .resources
            .get_mut(resource.get() as usize)
            .ok_or_else(|| malformed("resource ID is invalid"))?;
        record
            .locations
            .try_reserve(1)
            .map_err(|_| limit_overflow())?;
        match record.locations.entry(pointer.into()) {
            std::collections::hash_map::Entry::Occupied(entry) if *entry.get() == location => {}
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(malformed("duplicate schema location inside one resource"));
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(location);
            }
        }
        self.observe_build_memory(0)
    }

    fn index_keywords(
        &mut self,
        schema: &Ast,
        location: SchemaLocationId,
        resource: ResourceId,
        base_uri: Option<Arc<str>>,
        legacy_ids: bool,
    ) -> Result<(), CompileError> {
        if let Some(anchor) = object_string(schema, "$anchor")? {
            self.push_anchor(resource, location, anchor, false)?;
        }
        if let Some(anchor) = object_string(schema, "$dynamicAnchor")? {
            self.push_anchor(resource, location, anchor, true)?;
        }
        if legacy_ids {
            if let Some(id) = object_string(schema, "id")? {
                let resolved = resolve_uri_reference(base_uri.as_deref(), id)?;
                if let Some(anchor) = resolved
                    .fragment
                    .as_deref()
                    .filter(|name| valid_anchor(name))
                {
                    self.push_anchor(resource, location, anchor, false)?;
                }
            }
        }
        for (keyword, kind) in [
            ("$ref", ReferenceKind::Static),
            ("$dynamicRef", ReferenceKind::Dynamic),
        ] {
            if let Some(reference) = object_string(schema, keyword)? {
                check_limit(
                    LimitKind::ReferenceCount,
                    self.references.len().checked_add(1),
                    self.limits.max_references,
                )?;
                check_limit(
                    LimitKind::UriBytes,
                    Some(reference.len()),
                    self.limits.max_uri_bytes,
                )?;
                UriRef::parse(reference).map_err(|_| {
                    resource_error(
                        ErrorCode::InvalidUri,
                        "invalid URI reference",
                        Some(reference.to_owned()),
                    )
                    .with_keyword(keyword)
                })?;
                self.total_uri_bytes = self
                    .total_uri_bytes
                    .checked_add(reference.len())
                    .ok_or_else(limit_overflow)?;
                check_limit(
                    LimitKind::UriBytes,
                    Some(self.total_uri_bytes),
                    self.limits.max_total_uri_bytes,
                )?;
                self.charge(
                    size_of::<ReferenceRecord>()
                        .checked_add(reference.len())
                        .ok_or_else(limit_overflow)?,
                )?;
                self.references
                    .try_reserve(1)
                    .map_err(|_| limit_overflow())?;
                self.references.push(ReferenceRecord {
                    location,
                    kind,
                    text: reference.into(),
                    base_uri: base_uri.clone(),
                });
                self.observe_build_memory(0)?;
            }
        }
        Ok(())
    }

    fn push_anchor(
        &mut self,
        resource: ResourceId,
        target: SchemaLocationId,
        name: &str,
        dynamic: bool,
    ) -> Result<(), CompileError> {
        if !valid_anchor(name) {
            return Err(resource_error(
                ErrorCode::Malformed,
                "invalid schema anchor name",
                Some(name.to_owned()),
            ));
        }
        let resource_index = usize::try_from(resource.0).map_err(|_| limit_overflow())?;
        let anchor_count = {
            let entry = self
                .resources
                .get(resource_index)
                .ok_or_else(|| malformed("resource ID is invalid"))?;
            entry
                .static_anchors
                .len()
                .checked_add(entry.dynamic_anchors.len())
        };
        check_limit(
            LimitKind::AnchorCount,
            anchor_count.and_then(|count| count.checked_add(1)),
            self.limits.max_anchors_per_resource,
        )?;
        check_limit(
            LimitKind::AnchorCount,
            self.anchors.len().checked_add(1),
            self.limits.max_total_anchors,
        )?;
        if self
            .resources
            .get(resource_index)
            .ok_or_else(|| malformed("resource ID is invalid"))?
            .anchor_names
            .contains_key(name)
        {
            return Err(resource_error(
                ErrorCode::DuplicateAnchor,
                "duplicate static or dynamic anchor in one resource",
                Some(name.to_owned()),
            ));
        }
        self.charge(
            size_of::<AnchorRecord>()
                .checked_add(name.len())
                .ok_or_else(limit_overflow)?,
        )?;
        let id = AnchorId(u32::try_from(self.anchors.len()).map_err(|_| limit_overflow())?);
        self.anchors.try_reserve(1).map_err(|_| limit_overflow())?;
        self.anchors.push(AnchorRecord {
            resource,
            name: name.into(),
            target,
            dynamic,
        });
        let entry = self
            .resources
            .get_mut(resource_index)
            .ok_or_else(|| malformed("resource ID is invalid"))?;
        let run = if dynamic {
            &mut entry.dynamic_anchors
        } else {
            &mut entry.static_anchors
        };
        run.try_reserve(1).map_err(|_| limit_overflow())?;
        run.push(id);
        entry
            .anchor_names
            .try_reserve(1)
            .map_err(|_| limit_overflow())?;
        entry.anchor_names.insert(name.into(), id);
        self.observe_build_memory(0)?;
        Ok(())
    }

    fn bump_work(&mut self, amount: usize) -> Result<(), CompileError> {
        self.work = self.work.checked_add(amount).ok_or_else(limit_overflow)?;
        check_limit(
            LimitKind::ResolutionWork,
            Some(self.work),
            self.limits.max_resolution_work,
        )
    }

    fn charge(&mut self, bytes: usize) -> Result<(), CompileError> {
        let projected = self
            .memory
            .current_build
            .checked_add(bytes)
            .ok_or_else(limit_overflow)?;
        check_limit(
            LimitKind::ResourceBuildBytes,
            Some(projected),
            self.limits.max_peak_build_bytes,
        )
    }

    fn observe_build_memory(&mut self, temporary: usize) -> Result<(), CompileError> {
        let retained = mutable_builder_bytes(self)?;
        self.memory
            .observe_build(retained, temporary, self.limits.max_peak_build_bytes)
    }
}

fn vec_bytes<T>(values: &Vec<T>) -> Result<usize, CompileError> {
    values
        .capacity()
        .checked_mul(size_of::<T>())
        .ok_or_else(limit_overflow)
}

fn registry_build_bytes(registry: &SchemaRegistry) -> Result<usize, CompileError> {
    let mut document_strings = 0usize;
    for document in &registry.documents {
        document_strings = checked_sum([
            document_strings,
            arc_str_bytes(document.text.len())?,
            document
                .retrieval_uri
                .as_deref()
                .map_or(Ok(0), |uri| arc_str_bytes(uri.len()))?,
        ])?;
    }
    checked_sum([
        vec_bytes(&registry.documents)?,
        document_strings,
        map_bytes(&registry.retrieval_index)?,
        arc_str_map_keys(&registry.retrieval_index)?,
    ])
}

fn slice_bytes<T>(values: &[T]) -> Result<usize, CompileError> {
    values
        .len()
        .checked_mul(size_of::<T>())
        .ok_or_else(limit_overflow)
}

fn map_bytes<K, V>(map: &FxHashMap<K, V>) -> Result<usize, CompileError> {
    map.capacity()
        .checked_mul(2)
        .and_then(|buckets| buckets.checked_mul(size_of::<(K, V)>().checked_add(1)?))
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(limit_overflow)
}

fn set_bytes<T>(set: &FxHashSet<T>) -> Result<usize, CompileError> {
    set.capacity()
        .checked_mul(2)
        .and_then(|buckets| buckets.checked_mul(size_of::<T>().checked_add(1)?))
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(limit_overflow)
}

fn box_str_map_keys<V>(map: &FxHashMap<Box<str>, V>) -> Result<usize, CompileError> {
    map.keys()
        .try_fold(0usize, |total, key| total.checked_add(key.len()))
        .ok_or_else(limit_overflow)
}

fn arc_str_map_keys<V>(map: &FxHashMap<Arc<str>, V>) -> Result<usize, CompileError> {
    map.keys().try_fold(0usize, |total, key| {
        total
            .checked_add(arc_str_bytes(key.len())?)
            .ok_or_else(limit_overflow)
    })
}

fn arc_str_set_keys(set: &FxHashSet<Arc<str>>) -> Result<usize, CompileError> {
    set.iter().try_fold(0usize, |total, key| {
        total
            .checked_add(arc_str_bytes(key.len())?)
            .ok_or_else(limit_overflow)
    })
}

fn arc_str_bytes(len: usize) -> Result<usize, CompileError> {
    len.checked_add(
        size_of::<usize>()
            .checked_mul(2)
            .ok_or_else(limit_overflow)?,
    )
    .ok_or_else(limit_overflow)
}

fn work_stack_bytes(stack: &Vec<WorkItem<'_>>) -> Result<usize, CompileError> {
    let strings = stack.iter().try_fold(0usize, |total, item| {
        total
            .checked_add(item.document_pointer.capacity())?
            .checked_add(item.resource_pointer.capacity())
    });
    vec_bytes(stack)?
        .checked_add(strings.ok_or_else(limit_overflow)?)
        .ok_or_else(limit_overflow)
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize, CompileError> {
    values
        .into_iter()
        .try_fold(0usize, |total, value| total.checked_add(value))
        .ok_or_else(limit_overflow)
}

fn mutable_builder_bytes(builder: &ResourceBuilder) -> Result<usize, CompileError> {
    let mut document_heaps = 0usize;
    for document in &builder.documents {
        document_heaps = checked_sum([
            document_heaps,
            arc_str_bytes(document.text.len())?,
            document
                .retrieval_uri
                .as_deref()
                .map_or(Ok(0), |uri| arc_str_bytes(uri.len()))?,
            document.ast_bytes,
            map_bytes(&document.locations)?,
            box_str_map_keys(&document.locations)?,
        ])?;
    }
    let mut resource_heaps = 0usize;
    for resource in &builder.resources {
        resource_heaps = checked_sum([
            resource_heaps,
            vec_bytes(&resource.static_anchors)?,
            vec_bytes(&resource.dynamic_anchors)?,
            map_bytes(&resource.locations)?,
            box_str_map_keys(&resource.locations)?,
            map_bytes(&resource.anchor_names)?,
            box_str_map_keys(&resource.anchor_names)?,
            resource
                .canonical_uri
                .as_deref()
                .map_or(Ok(0), |uri| arc_str_bytes(uri.len()))?,
        ])?;
    }
    let mut location_strings = 0usize;
    for location in &builder.locations {
        location_strings = checked_sum([
            location_strings,
            location.document_pointer.len(),
            location
                .base_uri
                .as_deref()
                .map_or(Ok(0), |uri| arc_str_bytes(uri.len()))?,
        ])?;
    }
    let anchor_strings = builder.anchors.iter().try_fold(0usize, |total, anchor| {
        total
            .checked_add(arc_str_bytes(anchor.name.len())?)
            .ok_or_else(limit_overflow)
    })?;
    let mut reference_strings = 0usize;
    for reference in &builder.references {
        reference_strings = checked_sum([
            reference_strings,
            arc_str_bytes(reference.text.len())?,
            reference
                .base_uri
                .as_deref()
                .map_or(Ok(0), |uri| arc_str_bytes(uri.len()))?,
        ])?;
    }
    checked_sum([
        vec_bytes(&builder.documents)?,
        document_heaps,
        vec_bytes(&builder.resources)?,
        resource_heaps,
        vec_bytes(&builder.locations)?,
        location_strings,
        vec_bytes(&builder.anchors)?,
        anchor_strings,
        vec_bytes(&builder.references)?,
        reference_strings,
        vec_bytes(&builder.resolved_references)?,
        vec_bytes(&builder.reference_index)?,
        map_bytes(&builder.uri_index)?,
        arc_str_map_keys(&builder.uri_index)?,
        set_bytes(&builder.counted_uris)?,
        arc_str_set_keys(&builder.counted_uris)?,
    ])
}

fn frozen_graph_bytes(
    documents: &[ParsedDocument],
    resources: &[SchemaResource],
    locations: &[SchemaLocation],
    anchors: &[AnchorRecord],
    resolved_references: &[ResolvedReferenceRecord],
    reference_index: &[[u32; 2]],
    uri_index: &FxHashMap<Arc<str>, ResourceId>,
) -> Result<usize, CompileError> {
    let mut document_heaps = 0usize;
    for document in documents {
        document_heaps = checked_sum([
            document_heaps,
            arc_str_bytes(document.text.len())?,
            document
                .retrieval_uri
                .as_deref()
                .map_or(Ok(0), |uri| arc_str_bytes(uri.len()))?,
            document.ast_bytes,
            map_bytes(&document.locations)?,
        ])?;
    }
    let mut resource_heaps = 0usize;
    for resource in resources {
        resource_heaps = checked_sum([
            resource_heaps,
            slice_bytes(&resource.static_anchors)?,
            slice_bytes(&resource.dynamic_anchors)?,
            map_bytes(&resource.locations)?,
            box_str_map_keys(&resource.locations)?,
            map_bytes(&resource.anchor_names)?,
            box_str_map_keys(&resource.anchor_names)?,
            resource
                .canonical_uri
                .as_deref()
                .map_or(Ok(0), |uri| arc_str_bytes(uri.len()))?,
        ])?;
    }
    let mut location_strings = 0usize;
    for location in locations {
        location_strings = checked_sum([
            location_strings,
            location.document_pointer.len(),
            location
                .base_uri
                .as_deref()
                .map_or(Ok(0), |uri| arc_str_bytes(uri.len()))?,
        ])?;
    }
    let anchor_strings = anchors.iter().try_fold(0usize, |total, anchor| {
        total
            .checked_add(arc_str_bytes(anchor.name.len())?)
            .ok_or_else(limit_overflow)
    })?;
    checked_sum([
        slice_bytes(documents)?,
        document_heaps,
        slice_bytes(resources)?,
        resource_heaps,
        slice_bytes(locations)?,
        location_strings,
        slice_bytes(anchors)?,
        anchor_strings,
        slice_bytes(resolved_references)?,
        slice_bytes(reference_index)?,
        map_bytes(uri_index)?,
        arc_str_map_keys(uri_index)?,
    ])
}

pub(crate) fn resolve_uri_reference(
    base: Option<&str>,
    reference: &str,
) -> Result<ResolvedUri, CompileError> {
    let reference_uri = UriRef::parse(reference).map_err(|_| {
        resource_error(
            ErrorCode::InvalidUri,
            "invalid URI reference",
            Some(reference.to_owned()),
        )
    })?;
    let resolved = if let Some(base) = base {
        let base_uri = Uri::parse(base).map_err(|_| {
            resource_error(
                ErrorCode::InvalidUri,
                "invalid base URI",
                Some(base.to_owned()),
            )
        })?;
        reference_uri.resolve_against(&base_uri).map_err(|_| {
            resource_error(
                ErrorCode::InvalidUri,
                "URI reference cannot be resolved against its base",
                Some(reference.to_owned()),
            )
        })?
    } else if reference_uri.has_scheme() {
        Uri::parse(reference)
            .map_err(|_| {
                resource_error(
                    ErrorCode::InvalidUri,
                    "invalid absolute URI reference",
                    Some(reference.to_owned()),
                )
            })?
            .to_owned()
    } else {
        return Err(resource_error(
            ErrorCode::ReferenceResolution,
            "relative URI reference has no usable base",
            Some(reference.to_owned()),
        ));
    };
    let normalized = resolved.normalize();
    let fragment = normalized
        .fragment()
        .map(|value| {
            value
                .decode()
                .to_string()
                .map(|value| value.into_owned())
                .map_err(|_| {
                    resource_error(
                        ErrorCode::InvalidUri,
                        "URI fragment is not valid UTF-8 after percent decoding",
                        Some(reference.to_owned()),
                    )
                })
        })
        .transpose()?;
    Ok(ResolvedUri {
        resource: Arc::<str>::from(normalized.strip_fragment().as_str()),
        fragment,
    })
}

fn same_document_fragment(reference: &str) -> Result<Option<String>, CompileError> {
    let parsed = UriRef::parse(reference).map_err(|_| {
        resource_error(
            ErrorCode::InvalidUri,
            "invalid URI reference",
            Some(reference.to_owned()),
        )
    })?;
    if parsed.has_scheme() || !parsed.path().is_empty() || parsed.query().is_some() {
        return Err(resource_error(
            ErrorCode::ReferenceResolution,
            "relative URI reference has no usable base",
            Some(reference.to_owned()),
        ));
    }
    parsed
        .fragment()
        .map(|fragment| {
            fragment
                .decode()
                .to_string()
                .map(|value| value.into_owned())
                .map_err(|_| {
                    resource_error(
                        ErrorCode::InvalidUri,
                        "URI fragment is not valid UTF-8 after percent decoding",
                        Some(reference.to_owned()),
                    )
                })
        })
        .transpose()
}

fn resolve_resource_id(base: Option<&str>, value: &str) -> Result<String, CompileError> {
    let resolved = resolve_uri_reference(base, value)?;
    if resolved
        .fragment
        .as_deref()
        .is_some_and(|fragment| !fragment.is_empty())
    {
        return Err(resource_error(
            ErrorCode::InvalidUri,
            "$id must not contain a non-empty fragment",
            Some(value.to_owned()),
        )
        .with_keyword("$id"));
    }
    Ok(resolved.resource.to_string())
}

fn parse_absolute_resource_uri(value: &str, label: &'static str) -> Result<String, CompileError> {
    let uri = Uri::parse(value)
        .map_err(|_| resource_error(ErrorCode::InvalidUri, label, Some(value.to_owned())))?;
    if uri.fragment().is_some_and(|fragment| !fragment.is_empty()) {
        return Err(resource_error(
            ErrorCode::InvalidUri,
            "retrieval URI must not contain a non-empty fragment",
            Some(value.to_owned()),
        ));
    }
    Ok(uri.normalize().strip_fragment().as_str().to_owned())
}

fn object_string<'a>(schema: &'a Ast, keyword: &str) -> Result<Option<&'a str>, CompileError> {
    let Ast::Obj(fields) = schema else {
        return Ok(None);
    };
    let mut found = None;
    for (name, value) in fields {
        if name != keyword {
            continue;
        }
        if found.is_some() {
            return Err(malformed("duplicate schema keyword").with_keyword(keyword));
        }
        let Ast::Str(value) = value else {
            return Err(
                malformed("schema identifier, anchor, or reference must be a string")
                    .with_keyword(keyword),
            );
        };
        found = Some(value.as_str());
    }
    Ok(found)
}

fn push_schema_children<'a>(
    schema: &'a Ast,
    document_pointer: &str,
    resource_pointer: &str,
    placement: &ChildPlacement,
    stack: &mut Vec<WorkItem<'a>>,
) -> Result<(), CompileError> {
    let Ast::Obj(fields) = schema else {
        return Ok(());
    };
    for (keyword, value) in fields.iter().rev() {
        if single_schema_keyword(keyword) {
            if matches!(value, Ast::Obj(_) | Ast::Bool(_)) {
                push_child(
                    stack,
                    value,
                    document_pointer,
                    resource_pointer,
                    keyword,
                    placement,
                )?;
            }
        } else if map_schema_keyword(keyword) {
            if let Ast::Obj(children) = value {
                for (name, child) in children.iter().rev() {
                    if matches!(child, Ast::Obj(_) | Ast::Bool(_)) {
                        let segment = pointer_pair(keyword, name)?;
                        push_child(
                            stack,
                            child,
                            document_pointer,
                            resource_pointer,
                            &segment,
                            placement,
                        )?;
                    }
                }
            }
        } else if array_schema_keyword(keyword) {
            if let Ast::Arr(children) = value {
                for (index, child) in children.iter().enumerate().rev() {
                    if matches!(child, Ast::Obj(_) | Ast::Bool(_)) {
                        let segment = format!("{keyword}/{index}");
                        push_child(
                            stack,
                            child,
                            document_pointer,
                            resource_pointer,
                            &segment,
                            placement,
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

struct ChildPlacement {
    base_uri: Option<Arc<str>>,
    resource: ResourceId,
    resource_depth: u32,
    legacy_ids: bool,
}

fn push_child<'a>(
    stack: &mut Vec<WorkItem<'a>>,
    schema: &'a Ast,
    document_pointer: &str,
    resource_pointer: &str,
    segment: &str,
    placement: &ChildPlacement,
) -> Result<(), CompileError> {
    stack.try_reserve(1).map_err(|_| limit_overflow())?;
    stack.push(WorkItem {
        schema,
        document_pointer: append_pointer(document_pointer, segment)?,
        resource_pointer: append_pointer(resource_pointer, segment)?,
        base_uri: placement.base_uri.clone(),
        resource: placement.resource,
        resource_depth: placement.resource_depth,
        root: false,
        legacy_ids: placement.legacy_ids,
    });
    Ok(())
}

fn append_pointer(parent: &str, segment: &str) -> Result<String, CompileError> {
    let capacity = parent
        .len()
        .checked_add(segment.len())
        .and_then(|len| len.checked_add(1))
        .ok_or_else(limit_overflow)?;
    let mut output = new_string_with_capacity(capacity)?;
    output.push_str(parent);
    output.push('/');
    output.push_str(segment);
    Ok(output)
}

#[cfg(test)]
fn pointer_segment(value: &str) -> Result<String, CompileError> {
    let capacity = pointer_segment_len(value)?;
    let mut output = new_string_with_capacity(capacity)?;
    push_pointer_segment(&mut output, value);
    Ok(output)
}

fn pointer_pair(first: &str, second: &str) -> Result<String, CompileError> {
    let capacity = pointer_segment_len(first)?
        .checked_add(pointer_segment_len(second)?)
        .and_then(|len| len.checked_add(1))
        .ok_or_else(limit_overflow)?;
    let mut output = new_string_with_capacity(capacity)?;
    push_pointer_segment(&mut output, first);
    output.push('/');
    push_pointer_segment(&mut output, second);
    Ok(output)
}

fn pointer_segment_len(value: &str) -> Result<usize, CompileError> {
    let escapes = value
        .bytes()
        .filter(|byte| matches!(byte, b'~' | b'/'))
        .count();
    value.len().checked_add(escapes).ok_or_else(limit_overflow)
}

fn new_string_with_capacity(capacity: usize) -> Result<String, CompileError> {
    let mut output = String::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| limit_overflow())?;
    Ok(output)
}

fn push_pointer_segment(output: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '~' => output.push_str("~0"),
            '/' => output.push_str("~1"),
            _ => output.push(ch),
        }
    }
}

fn is_legacy_metaschema(uri: &str) -> bool {
    matches!(
        uri.trim_end_matches('#'),
        "http://json-schema.org/draft-04/schema"
            | "http://json-schema.org/draft-06/schema"
            | "http://json-schema.org/draft-07/schema"
    )
}

fn single_schema_keyword(keyword: &str) -> bool {
    matches!(
        keyword,
        "additionalProperties"
            | "unevaluatedProperties"
            | "propertyNames"
            | "contains"
            | "unevaluatedItems"
            | "items"
            | "not"
            | "if"
            | "then"
            | "else"
            | "contentSchema"
    )
}

fn map_schema_keyword(keyword: &str) -> bool {
    matches!(
        keyword,
        "$defs" | "definitions" | "properties" | "patternProperties" | "dependentSchemas"
    )
}

fn array_schema_keyword(keyword: &str) -> bool {
    matches!(keyword, "prefixItems" | "allOf" | "anyOf" | "oneOf")
}

fn valid_anchor(value: &str) -> bool {
    let mut bytes = value.bytes();
    matches!(bytes.next(), Some(byte) if byte.is_ascii_alphabetic())
        && bytes
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b':' | b'.'))
}

fn validate_json_pointer(pointer: &str) -> Result<(), CompileError> {
    for token in pointer[1..].split('/') {
        let mut bytes = token.bytes();
        while let Some(byte) = bytes.next() {
            if byte == b'~' && !matches!(bytes.next(), Some(b'0' | b'1')) {
                return Err(resource_error(
                    ErrorCode::ReferenceResolution,
                    "JSON Pointer contains an invalid escape",
                    Some(pointer.to_owned()),
                ));
            }
        }
    }
    Ok(())
}

fn ast_at_pointer<'a>(root: &'a Ast, pointer: &str) -> Result<&'a Ast, CompileError> {
    if pointer.is_empty() {
        return Ok(root);
    }
    let Some(path) = pointer.strip_prefix('/') else {
        return Err(malformed("document pointer is not absolute"));
    };
    let mut current = root;
    for encoded in path.split('/') {
        let token = decode_pointer_token(encoded)?;
        current = match current {
            Ast::Obj(fields) => fields
                .iter()
                .find_map(|(name, value)| (name == &token).then_some(value))
                .ok_or_else(|| malformed("document pointer object key is missing"))?,
            Ast::Arr(items) => {
                if token.starts_with('0') && token.len() > 1 {
                    return Err(malformed("document pointer array index has a leading zero"));
                }
                let index = token
                    .parse::<usize>()
                    .map_err(|_| malformed("document pointer array index is invalid"))?;
                items
                    .get(index)
                    .ok_or_else(|| malformed("document pointer array index is out of range"))?
            }
            _ => return Err(malformed("document pointer traverses a scalar")),
        };
    }
    Ok(current)
}

fn decode_pointer_token(token: &str) -> Result<String, CompileError> {
    let mut output = String::new();
    output
        .try_reserve_exact(token.len())
        .map_err(|_| limit_overflow())?;
    let mut chars = token.chars();
    while let Some(ch) = chars.next() {
        if ch != '~' {
            output.push(ch);
            continue;
        }
        match chars.next() {
            Some('0') => output.push('~'),
            Some('1') => output.push('/'),
            _ => return Err(malformed("document pointer escape is invalid")),
        }
    }
    Ok(output)
}

fn check_limit(kind: LimitKind, observed: Option<usize>, limit: usize) -> Result<(), CompileError> {
    let observed = observed.ok_or_else(limit_overflow)?;
    if observed <= limit {
        return Ok(());
    }
    Err(resource_limit_error(kind, observed, limit))
}

fn resource_limit_error(kind: LimitKind, observed: usize, limit: usize) -> CompileError {
    let mut error = CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L1,
        "schema resource limit exceeded",
    );
    error.limit = Some((kind, observed, limit));
    error
}

#[cfg(feature = "bench-internals")]
fn elapsed_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn limit_overflow() -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L1,
        "schema resource size overflow or allocation failure",
    )
}

fn malformed(message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L1, message)
}

fn resource_error(
    code: ErrorCode,
    message: &'static str,
    observed: Option<String>,
) -> CompileError {
    let mut error = CompileError::new(code, Stage::L1, message);
    error.observed = observed;
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_helpers_escape_in_one_pass_and_preserve_unicode() {
        assert_eq!(pointer_segment("a/b~c/λ").unwrap(), "a~1b~0c~1λ");
        assert_eq!(
            pointer_pair("properties", "a/b~c").unwrap(),
            "properties/a~1b~0c"
        );
        assert_eq!(
            append_pointer("/properties", "a~1b").unwrap(),
            "/properties/a~1b"
        );
        let error = new_string_with_capacity(usize::MAX).expect_err("capacity must fail");
        assert_eq!(error.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn uri_resolution_keeps_fragment_kinds_distinct_and_decodes_percent_encoding() {
        let pointer = resolve_uri_reference(
            Some("https://example.com/schemas/root.json"),
            "child.json#/properties/a%7E1b",
        )
        .expect("pointer");
        assert_eq!(
            pointer.resource.as_ref(),
            "https://example.com/schemas/child.json"
        );
        assert_eq!(pointer.fragment.as_deref(), Some("/properties/a~1b"));

        let anchor =
            resolve_uri_reference(Some("https://example.com/root"), "#node").expect("anchor");
        assert_eq!(anchor.fragment.as_deref(), Some("node"));
        let root = resolve_uri_reference(Some("https://example.com/root"), "#").expect("root");
        assert_eq!(root.fragment.as_deref(), Some(""));
    }

    #[test]
    fn discovery_indexes_nested_resources_and_separate_anchor_namespaces() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert(
                "https://example.com/root",
                r##"{
                    "$anchor":"root",
                    "$defs":{
                        "child":{"$id":"child","$anchor":"same","type":"string"},
                        "other":{"$id":"other","$anchor":"same","$dynamicAnchor":"dyn"}
                    }
                }"##,
            )
            .expect("insert");
        let graph = registry.discover().expect("discover");
        assert_eq!(graph.documents.len(), 1);
        assert_eq!(graph.resources.len(), 3);
        assert_eq!(graph.anchors.len(), 4);
        assert!(graph.uri_index.contains_key("https://example.com/child"));
        assert!(graph.uri_index.contains_key("https://example.com/other"));
    }

    #[test]
    fn duplicate_anchor_across_static_and_dynamic_tables_is_rejected() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert(
                "https://example.com/root",
                r##"{"$anchor":"same","$dynamicAnchor":"same"}"##,
            )
            .expect("insert");
        let error = registry.discover().expect_err("duplicate anchor");
        assert_eq!(error.code, ErrorCode::DuplicateAnchor);
    }

    #[test]
    fn fragment_resolution_distinguishes_root_pointer_and_anchor() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert(
                "https://example.com/root",
                r##"{
                    "$anchor":"rootAnchor",
                    "properties":{"a/b~c":{"$anchor":"valueAnchor","type":"string"}}
                }"##,
            )
            .expect("insert");
        let graph = registry.discover().expect("discover");
        let root = graph
            .resolve_reference(Some("https://example.com/root"), "#")
            .expect("root");
        assert!(matches!(root.target, FragmentTarget::ResourceRoot(_)));
        let pointer = graph
            .resolve_reference(Some("https://example.com/root"), "#%2Fproperties%2Fa~1b~0c")
            .expect("pointer");
        assert!(matches!(pointer.target, FragmentTarget::JsonPointer(_)));
        let anchor = graph
            .resolve_reference(Some("https://example.com/root"), "#valueAnchor")
            .expect("anchor");
        assert!(matches!(anchor.target, FragmentTarget::Anchor(_)));
        let error = graph
            .resolve_reference(Some("https://example.com/root"), "#/properties/a~2b")
            .expect_err("invalid pointer escape");
        assert_eq!(error.code, ErrorCode::ReferenceResolution);
    }

    #[test]
    fn reference_materializes_a_schema_outside_known_keyword_paths() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert(
                "https://example.com/root",
                r##"{
                    "$schema":"http://json-schema.org/draft-04/schema#",
                    "port":{"type":"string"},
                    "properties":{"port":{"$ref":"#/port"}}
                }"##,
            )
            .expect("insert");
        let graph = registry.discover().expect("discover");
        let resolved = graph
            .resolve_reference(Some("https://example.com/root"), "#/port")
            .expect("pointer target");
        assert!(matches!(resolved.target, FragmentTarget::JsonPointer(_)));
    }

    #[test]
    fn containing_resource_pointer_aliases_an_embedded_resource_root() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert(
                "https://example.com/root",
                r##"{
                    "$defs":{"child":{"$id":"child","type":"string"}},
                    "properties":{"value":{"$ref":"#/$defs/child"}}
                }"##,
            )
            .expect("insert");
        let graph = registry.discover().expect("discover");
        let resolved = graph
            .resolve_reference(Some("https://example.com/root"), "#/$defs/child")
            .expect("embedded root");
        assert!(matches!(resolved.target, FragmentTarget::JsonPointer(_)));
        assert_eq!(graph.resources.len(), 2);
    }

    #[test]
    fn legacy_fragment_id_is_indexed_as_a_plain_name_anchor() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert(
                "https://example.com/root",
                r##"{
                    "$schema":"http://json-schema.org/draft-04/schema#",
                    "id":"https://example.com/root#root",
                    "definitions":{"capability":{"id":"https://example.com/root#capability","type":"string"}},
                    "properties":{"value":{"$ref":"#capability"}}
                }"##,
            )
            .expect("insert");
        let graph = registry.discover().expect("discover");
        let resolved = graph
            .resolve_reference(Some("https://example.com/root"), "#capability")
            .expect("legacy anchor");
        assert!(matches!(resolved.target, FragmentTarget::Anchor(_)));
        let root = graph
            .resolve_reference(Some("https://example.com/root"), "#root")
            .expect("root anchor");
        assert!(matches!(root.target, FragmentTarget::Anchor(_)));
    }

    #[test]
    fn retrieval_alias_and_nested_id_both_identify_the_root_location() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert(
                "https://retrieval.example/schema",
                r##"{"$id":"https://canonical.example/root","$anchor":"node"}"##,
            )
            .expect("insert");
        let graph = registry.discover().expect("discover");
        let retrieval = graph
            .resolve_reference(None, "https://retrieval.example/schema")
            .expect("retrieval alias");
        let canonical = graph
            .resolve_reference(None, "https://canonical.example/root")
            .expect("canonical identifier");
        assert_eq!(retrieval.resource, canonical.resource);
    }

    #[test]
    fn duplicate_canonical_resource_uri_is_rejected() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert(
                "https://example.com/a",
                r##"{"$id":"https://example.com/shared"}"##,
            )
            .expect("a");
        registry
            .insert(
                "https://example.com/b",
                r##"{"$id":"https://example.com/shared"}"##,
            )
            .expect("b");
        let error = registry.discover().expect_err("duplicate canonical URI");
        assert_eq!(error.code, ErrorCode::DuplicateResource);
    }

    #[test]
    fn registry_discovery_order_is_independent_of_insertion_order() {
        fn build(reverse: bool) -> Vec<Option<String>> {
            let mut registry = SchemaRegistry::default();
            let documents = [
                ("https://example.com/a", r##"{"$defs":{"x":{"$id":"x"}}}"##),
                ("https://example.com/b", "true"),
            ];
            for index in if reverse { [1, 0] } else { [0, 1] } {
                registry
                    .insert(documents[index].0, documents[index].1)
                    .expect("insert");
            }
            registry
                .discover()
                .expect("discover")
                .resources
                .iter()
                .map(|resource| resource.canonical_uri.as_deref().map(str::to_owned))
                .collect()
        }
        assert_eq!(build(false), build(true));
    }

    #[test]
    fn anonymous_document_needs_a_base_only_for_relative_references() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert_anonymous(r##"{"$ref":"https://example.com/target"}"##)
            .expect("anonymous");
        registry
            .insert("https://example.com/target", "true")
            .expect("target");
        registry.discover().expect("absolute reference");

        let mut registry = SchemaRegistry::default();
        registry
            .insert_anonymous(r##"{"$ref":"relative"}"##)
            .expect("anonymous");
        let error = registry
            .discover()
            .expect_err("relative reference without base");
        assert_eq!(error.code, ErrorCode::ReferenceResolution);
    }

    #[test]
    fn document_count_limit_is_exact() {
        let limits = SchemaResourceLimits {
            max_documents: 1,
            ..SchemaResourceLimits::default()
        };
        let mut registry = SchemaRegistry::new(limits);
        registry
            .insert("https://example.com/a", "true")
            .expect("exact cap");
        let error = registry
            .insert("https://example.com/b", "true")
            .expect_err("cap plus one");
        assert_eq!(error.limit, Some((LimitKind::DocumentCount, 2, 1)));
    }

    fn assert_limit(error: CompileError, kind: LimitKind) {
        assert_eq!(error.code, ErrorCode::InternalLimitExceeded);
        assert_eq!(error.limit.map(|entry| entry.0), Some(kind));
    }

    #[test]
    fn input_byte_limits_are_exact() {
        let limits = SchemaResourceLimits {
            max_document_bytes: 4,
            max_total_input_bytes: 8,
            ..SchemaResourceLimits::default()
        };
        let mut registry = SchemaRegistry::new(limits);
        registry.insert_anonymous("true").expect("per-document cap");
        registry.insert_anonymous("null").expect("aggregate cap");

        let mut per_document = SchemaRegistry::new(SchemaResourceLimits {
            max_document_bytes: 3,
            ..limits
        });
        assert_limit(
            per_document
                .insert_anonymous("true")
                .expect_err("cap plus one"),
            LimitKind::InputBytes,
        );
        let mut aggregate = SchemaRegistry::new(SchemaResourceLimits {
            max_total_input_bytes: 7,
            ..limits
        });
        aggregate.insert_anonymous("true").expect("first document");
        assert_limit(
            aggregate
                .insert_anonymous("null")
                .expect_err("cap plus one"),
            LimitKind::InputBytes,
        );
    }

    #[test]
    fn uri_byte_limits_are_exact() {
        let uri = "https://e.test/a";
        let exact = uri.len();
        let mut registry = SchemaRegistry::new(SchemaResourceLimits {
            max_uri_bytes: exact,
            max_total_uri_bytes: exact,
            ..SchemaResourceLimits::default()
        });
        registry.insert(uri, "true").expect("exact URI cap");
        registry.discover().expect("exact aggregate URI cap");

        let mut one_short = SchemaRegistry::new(SchemaResourceLimits {
            max_uri_bytes: exact - 1,
            ..SchemaResourceLimits::default()
        });
        assert_limit(
            one_short.insert(uri, "true").expect_err("URI cap plus one"),
            LimitKind::UriBytes,
        );
        let mut total_short = SchemaRegistry::new(SchemaResourceLimits {
            max_total_uri_bytes: exact - 1,
            ..SchemaResourceLimits::default()
        });
        assert_limit(
            total_short
                .insert(uri, "true")
                .expect_err("total URI cap plus one"),
            LimitKind::UriBytes,
        );
    }

    #[test]
    fn resource_and_depth_limits_are_exact() {
        let schema = r##"{"$defs":{"x":{"$id":"x"}}}"##;
        let mut registry = SchemaRegistry::new(SchemaResourceLimits {
            max_resources: 2,
            max_resource_depth: 1,
            ..SchemaResourceLimits::default()
        });
        registry
            .insert("https://e.test/root", schema)
            .expect("insert");
        registry.discover().expect("exact resource and depth caps");

        for (kind, limits) in [
            (
                LimitKind::ResourceCount,
                SchemaResourceLimits {
                    max_resources: 1,
                    ..SchemaResourceLimits::default()
                },
            ),
            (
                LimitKind::ResourceDepth,
                SchemaResourceLimits {
                    max_resource_depth: 0,
                    ..SchemaResourceLimits::default()
                },
            ),
        ] {
            let mut registry = SchemaRegistry::new(limits);
            registry
                .insert("https://e.test/root", schema)
                .expect("insert");
            assert_limit(registry.discover().expect_err("cap plus one"), kind);
        }
    }

    #[test]
    fn anchor_limits_are_exact() {
        let one = r##"{"$anchor":"a"}"##;
        let two = r##"{"$anchor":"a","$defs":{"x":{"$anchor":"b"}}}"##;
        let mut exact = SchemaRegistry::new(SchemaResourceLimits {
            max_anchors_per_resource: 2,
            max_total_anchors: 2,
            ..SchemaResourceLimits::default()
        });
        exact.insert("https://e.test/root", two).expect("insert");
        exact.discover().expect("exact anchor caps");

        let mut per_resource = SchemaRegistry::new(SchemaResourceLimits {
            max_anchors_per_resource: 1,
            ..SchemaResourceLimits::default()
        });
        per_resource
            .insert("https://e.test/root", two)
            .expect("insert");
        assert_limit(
            per_resource
                .discover()
                .expect_err("per-resource cap plus one"),
            LimitKind::AnchorCount,
        );

        let mut total = SchemaRegistry::new(SchemaResourceLimits {
            max_total_anchors: 1,
            ..SchemaResourceLimits::default()
        });
        total.insert("https://e.test/root", one).expect("root");
        total.insert("https://e.test/other", one).expect("other");
        assert_limit(
            total.discover().expect_err("total cap plus one"),
            LimitKind::AnchorCount,
        );
    }

    #[test]
    fn reference_location_and_work_limits_are_exact() {
        let schema = r##"{"$ref":"#","$defs":{"x":true}}"##;
        let mut exact = SchemaRegistry::new(SchemaResourceLimits {
            max_references: 1,
            max_schema_locations: 2,
            max_resolution_work: 3,
            ..SchemaResourceLimits::default()
        });
        exact.insert("https://e.test/root", schema).expect("insert");
        exact
            .discover()
            .expect("exact reference/location/work caps");

        for (kind, limits) in [
            (
                LimitKind::ReferenceCount,
                SchemaResourceLimits {
                    max_references: 0,
                    ..SchemaResourceLimits::default()
                },
            ),
            (
                LimitKind::SchemaLocationCount,
                SchemaResourceLimits {
                    max_schema_locations: 1,
                    ..SchemaResourceLimits::default()
                },
            ),
            (
                LimitKind::ResolutionWork,
                SchemaResourceLimits {
                    max_resolution_work: 2,
                    ..SchemaResourceLimits::default()
                },
            ),
        ] {
            let mut registry = SchemaRegistry::new(limits);
            registry
                .insert("https://e.test/root", schema)
                .expect("insert");
            assert_limit(registry.discover().expect_err("cap plus one"), kind);
        }
    }

    #[test]
    fn retained_graph_limit_is_exact() {
        let uri = "https://e.test/root";
        let mut probe = SchemaRegistry::default();
        probe.insert(uri, "true").expect("insert");
        let retained = probe.discover().expect("probe").retained_bytes;
        assert!(retained > 0);

        let mut exact = SchemaRegistry::new(SchemaResourceLimits {
            max_retained_graph_bytes: retained,
            ..SchemaResourceLimits::default()
        });
        exact.insert(uri, "true").expect("insert");
        exact.discover().expect("exact retained cap");

        let mut one_short = SchemaRegistry::new(SchemaResourceLimits {
            max_retained_graph_bytes: retained - 1,
            ..SchemaResourceLimits::default()
        });
        one_short.insert(uri, "true").expect("insert");
        assert_limit(
            one_short.discover().expect_err("retained cap plus one"),
            LimitKind::ResourceGraphBytes,
        );
    }

    #[test]
    fn peak_build_limit_is_exact() {
        let uri = "https://e.test/root";
        let schema = r##"{"$defs":{"x":{"type":"string"}},"$ref":"#/$defs/x"}"##;
        let mut probe = SchemaRegistry::default();
        probe.insert(uri, schema).expect("insert");
        let peak = probe.discover().expect("probe").peak_build_bytes;
        assert!(peak > schema.len());

        let mut exact = SchemaRegistry::new(SchemaResourceLimits {
            max_peak_build_bytes: peak,
            ..SchemaResourceLimits::default()
        });
        exact.insert(uri, schema).expect("insert");
        exact.discover().expect("exact peak cap");

        let mut one_short = SchemaRegistry::new(SchemaResourceLimits {
            max_peak_build_bytes: peak - 1,
            ..SchemaResourceLimits::default()
        });
        one_short.insert(uri, schema).expect("insert");
        assert_limit(
            one_short.discover().expect_err("peak cap plus one"),
            LimitKind::ResourceBuildBytes,
        );
    }

    #[test]
    fn invalid_anchor_and_missing_fragment_targets_fail_during_freeze() {
        for schema in [
            r##"{"$anchor":"1bad"}"##,
            r##"{"$ref":"#/missing"}"##,
            r##"{"$ref":"#missing"}"##,
            r##"{"$ref":"https://missing.test/schema"}"##,
            r##"{"$ref":"#%GG"}"##,
        ] {
            let mut registry = SchemaRegistry::default();
            registry
                .insert("https://e.test/root", schema)
                .expect("insert");
            assert!(
                registry.discover().is_err(),
                "schema unexpectedly froze: {schema}"
            );
        }
    }

    #[test]
    fn unresolved_references_report_keyword_pointer_and_reference_text() {
        for (schema, keyword) in [
            (r##"{"$ref":"missing.json"}"##, "$ref"),
            (r##"{"$dynamicRef":"#missing"}"##, "$dynamicRef"),
        ] {
            let mut registry = SchemaRegistry::default();
            registry
                .insert("https://e.test/root", schema)
                .expect("insert");
            let error = registry.discover().expect_err("unresolved reference");
            let expected_pointer = format!("/{keyword}");
            assert_eq!(error.code, ErrorCode::ReferenceResolution);
            assert_eq!(error.keyword.as_deref(), Some(keyword));
            assert_eq!(
                error.json_pointer_path.as_deref(),
                Some(expected_pointer.as_str())
            );
            assert!(error
                .observed
                .as_deref()
                .is_some_and(|text| !text.is_empty()));
        }
    }

    #[test]
    fn same_anchor_name_in_distinct_resources_is_valid() {
        let mut registry = SchemaRegistry::default();
        registry
            .insert(
                "https://e.test/root",
                r##"{"$anchor":"same","$defs":{"x":{"$id":"x","$anchor":"same"}}}"##,
            )
            .expect("insert");
        registry.discover().expect("distinct anchor namespaces");
    }
}
