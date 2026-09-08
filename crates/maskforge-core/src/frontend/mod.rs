//! Converts JSON Schema text into validated schema IR.

mod ast;
mod json_schema;
pub(crate) mod regex;
mod resources;

pub use json_schema::{schema_to_ir, schema_to_ir_timed};
pub(crate) use json_schema::{JSON_STRING_BODY, JSON_STRING_CODEPOINT};
pub use regex::regex_to_ir;
#[cfg(all(test, feature = "alloc-probe"))]
pub(crate) use resources::discover_memory_for_probe;
pub use resources::{SchemaRegistry, SchemaResourceLimits};

use crate::error::CompileError;
use crate::ir::{CompileOptions, SchemaIR};

#[doc(hidden)]
#[cfg(feature = "bench-internals")]
#[derive(Clone, Copy, Debug)]
pub struct SchemaSetProfile {
    pub registry_insert_ns: u64,
    pub document_parse_ns: u64,
    pub resource_uri_anchor_index_ns: u64,
    pub resource_canonicalize_ns: u64,
    pub reference_resolution_ns: u64,
    pub graph_freeze_ns: u64,
    pub ir_lowering_and_construction_validation_ns: u64,
    pub ir_revalidation_ns: u64,
    pub retained_graph_bytes: usize,
    pub peak_resource_build_bytes: usize,
}

pub fn schema_to_ir_with_external_refs(
    text: &str,
    options: CompileOptions,
    resolver: &rustc_hash::FxHashMap<String, String>,
) -> Result<SchemaIR, CompileError> {
    let limits = SchemaResourceLimits::default();
    let mut registry = SchemaRegistry::new(limits);
    for (uri, schema) in resolver {
        registry.insert(uri, std::sync::Arc::<str>::from(schema.as_str()))?;
    }
    json_schema::schema_set_to_ir(text, None, options, limits, registry)
}

pub fn schema_to_ir_with_resources(
    text: &str,
    retrieval_uri: Option<&str>,
    options: CompileOptions,
    limits: SchemaResourceLimits,
    registry: SchemaRegistry,
) -> Result<SchemaIR, CompileError> {
    json_schema::schema_set_to_ir(text, retrieval_uri, options, limits, registry)
}

#[doc(hidden)]
#[cfg(feature = "bench-internals")]
pub fn schema_to_ir_with_resources_profiled(
    text: &str,
    retrieval_uri: Option<&str>,
    options: CompileOptions,
    limits: SchemaResourceLimits,
    registry: SchemaRegistry,
) -> Result<(SchemaIR, SchemaSetProfile), CompileError> {
    json_schema::schema_set_to_ir_profiled(text, retrieval_uri, options, limits, registry)
}

#[cfg(test)]
mod resource_lowering_tests {
    use super::*;

    fn compile(root: &str, retrieval: Option<&str>, resources: &[(&str, &str)]) -> SchemaIR {
        let limits = SchemaResourceLimits::default();
        let mut registry = SchemaRegistry::new(limits);
        for (uri, schema) in resources {
            registry.insert(uri, *schema).expect("resource");
        }
        schema_to_ir_with_resources(root, retrieval, CompileOptions::default(), limits, registry)
            .expect("schema set")
    }

    #[test]
    fn external_static_root_pointer_anchor_and_transitive_refs_lower_once() {
        let ir = compile(
            r##"{
                "type":"object",
                "properties":{
                    "a":{"$ref":"defs.json#/$defs/value"},
                    "b":{"$ref":"defs.json#value"}
                },
                "required":["a","b"],
                "additionalProperties":false
            }"##,
            Some("https://example.com/root.json"),
            &[(
                "https://example.com/defs.json",
                r##"{"$defs":{"value":{"$anchor":"value","type":"integer","minimum":5}}}"##,
            )],
        );
        assert_eq!(ir.resources().len(), 2);
        assert!(crate::structured::try_accepts(
            std::sync::Arc::new(ir.clone()),
            br#"{"a":5,"b":6}"#
        )
        .expect("accept"));
        assert!(
            !crate::structured::try_accepts(std::sync::Arc::new(ir), br#"{"a":4,"b":6}"#)
                .expect("reject")
        );
    }

    #[test]
    fn productive_external_mutual_recursion_is_not_a_resolver_cycle() {
        let ir = compile(
            r##"{"$ref":"a.json"}"##,
            Some("https://example.com/root.json"),
            &[
                (
                    "https://example.com/a.json",
                    r##"{"type":"object","properties":{"b":{"$ref":"b.json"}},"additionalProperties":false}"##,
                ),
                (
                    "https://example.com/b.json",
                    r##"{"type":"object","properties":{"a":{"$ref":"a.json"}},"additionalProperties":false}"##,
                ),
            ],
        );
        assert!(
            crate::structured::try_accepts(std::sync::Arc::new(ir), br#"{"b":{"a":{}}}"#)
                .expect("accept")
        );
    }

    #[test]
    fn nested_id_changes_the_active_reference_base() {
        let ir = compile(
            r##"{
                "$defs":{"nested":{"$id":"folder/child.json","$ref":"value.json"}},
                "$ref":"folder/child.json"
            }"##,
            Some("https://example.com/root.json"),
            &[(
                "https://example.com/folder/value.json",
                r##"{"type":"string"}"##,
            )],
        );
        assert!(
            crate::structured::try_accepts(std::sync::Arc::new(ir), br#""ok""#).expect("accept")
        );
    }

    #[test]
    fn registry_insertion_order_does_not_change_the_frozen_hash() {
        fn build(reverse: bool) -> [u8; 32] {
            let limits = SchemaResourceLimits::default();
            let mut registry = SchemaRegistry::new(limits);
            let resources = [
                (
                    "https://example.com/a",
                    r##"{"$anchor":"a","type":"string"}"##,
                ),
                (
                    "https://example.com/b",
                    r##"{"$anchor":"b","type":"integer"}"##,
                ),
            ];
            for index in if reverse { [1, 0] } else { [0, 1] } {
                registry
                    .insert(resources[index].0, resources[index].1)
                    .expect("resource");
            }
            schema_to_ir_with_resources(
                r##"{"allOf":[{"$ref":"https://example.com/a#a"},{"$ref":"https://example.com/b#b"}]}"##,
                Some("https://example.com/root"),
                CompileOptions::default(),
                limits,
                registry,
            )
            .expect("schema set")
            .canonical_hash()
        }
        assert_eq!(build(false), build(true));
    }

    #[test]
    fn anonymous_root_resolves_empty_pointer_and_plain_name_fragments() {
        let ir = compile(
            r##"{
                "$defs":{"value":{"$anchor":"value","type":"integer"}},
                "allOf":[{"$ref":"#/$defs/value"},{"$ref":"#value"}]
            }"##,
            None,
            &[],
        );
        assert!(crate::structured::try_accepts(std::sync::Arc::new(ir.clone()), b"7").unwrap());
        assert!(!crate::structured::try_accepts(std::sync::Arc::new(ir), br#""x""#).unwrap());
    }
}
