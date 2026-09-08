"""Advanced API: raw IR, index and matcher handles, and the process-wide cache controls.

Most callers want `maskforge.Compiler` and `maskforge.Generator` instead. Everything here is
supported, but the shapes are closer to the Rust core and change more freely than the root API.
"""

from ._native import (
    PyIndex,
    PySchemaIR,
    PyStructuredMatcher,
    PyVocabulary,
    clear_executable_cache,
    compile_ir,
    compile_ir_with_vocabulary,
    compile_json_schema,
    compile_json_schema_with_vocabulary,
    configure_vocab_build_budget,
    configure_vocab_serving_budget,
    executable_cache_stats,
    ir_from_wire,
    regex_ir,
    schema_to_ir,
    schema_to_ir_with_external_refs,
    schema_to_ir_with_resources,
    trie_cache_stats,
    vocab_build_budget_stats,
    vocab_serving_budget_stats,
)
from .generation import ConstraintSession
from .logits_processor import MaskforgeLogitsProcessor

__all__ = [
    "ConstraintSession",
    "MaskforgeLogitsProcessor",
    "PyIndex",
    "PySchemaIR",
    "PyStructuredMatcher",
    "PyVocabulary",
    "clear_executable_cache",
    "compile_ir",
    "compile_ir_with_vocabulary",
    "compile_json_schema",
    "compile_json_schema_with_vocabulary",
    "configure_vocab_build_budget",
    "configure_vocab_serving_budget",
    "executable_cache_stats",
    "ir_from_wire",
    "regex_ir",
    "schema_to_ir",
    "schema_to_ir_with_external_refs",
    "schema_to_ir_with_resources",
    "trie_cache_stats",
    "vocab_build_budget_stats",
    "vocab_serving_budget_stats",
]
