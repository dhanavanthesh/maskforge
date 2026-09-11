# Changelog

All notable changes to MaskForge are documented here. Versioning follows
[Semantic Versioning](https://semver.org/); before `1.0.0`, `0.y.z` releases may change the
public API without a major bump (see the README's versioning-policy section).

## [0.1.1] - 2026-09-08

### Fixed

- Removed the `License :: OSI Approved :: Apache Software License` classifier. PyPI rejects a
  distribution that declares both a PEP 639 `License-Expression` and a legacy license classifier,
  which blocked the wheel upload. The SPDX `license = "Apache-2.0"` field is unchanged.

## [0.1.0] - 2026-09-08

First public alpha. Packaging identity, error contract, and documentation brought up to
release-candidate quality; see `docs/development/program5-report.md` for the underlying
compiler-correctness evidence this release rests on.

### Changed

- **Python packaging has a single source.** The duplicate
  `crates/maskforge-py/pyproject.toml` is removed; the repository-root `pyproject.toml` is the
  only manifest, and every workflow, Make target, and documented command builds from it. The
  shipped feature set now includes `tokenizer-processing`, so
  `PyVocabulary.from_tokenizer_json` (which `from_transformers` requires) is present in every
  released wheel. CI asserts this against the installed wheel.
- **`Generator` compiles and binds once.** The signature is now
  `Generator(model, output_type)`; each call creates only session state. The previous
  call-time `generator(prompts, output_type, ...)` form is gone, because it recompiled and
  rebound on every call.
- **`Generator` never mutates the caller's tokenizer.** Batch prompts are encoded
  independently and left-padded into a local tensor, instead of temporarily reassigning
  `padding_side` and `pad_token` on a possibly shared tokenizer.
- The package root exports only the stable high-level API. Raw IR, index and matcher handles,
  the process-wide cache controls, `ConstraintSession` and `MaskforgeLogitsProcessor` now live
  in `maskforge.low_level`. The old root names still resolve for one release and emit a
  `DeprecationWarning`.
- `CompilerOptions` gained `with_executable_cache_bytes` / `with_trie_cache_bytes` /
  `with_bind_policy` builder setters, so a `#[non_exhaustive]` struct is constructible
  downstream; `BindPolicy` and `TrieCacheStats` are re-exported at the crate root.
- Python `Compiler` accepts `trie_cache_bytes` and `bind_policy` alongside
  `executable_cache_bytes`.
- The torch tensor adapter unpacks packed masks through a cached byte lookup into a
  thread-local buffer rather than widening to `int64` and allocating five temporaries per
  step. Measured on this machine, the processor call drops 2.7x at 50K vocabulary/batch 1
  (889 -> 330 us p50) and 3.9x at 250K/batch 32 (40.8 -> 10.6 ms p50).

### Added

- `Session::mask_byte_count()` (Rust): the single checked source of the mask byte length.
- `MAX_TOKENIZER_JSON_BYTES` and a byte cap on `Vocabulary::from_tokenizer_json`, so
  caller-supplied tokenizer JSON cannot drive parser allocation. Reported as
  `InternalLimitExceeded`, distinct from `MalformedTokenizer`.
- Generic `OutputSpec[T]`, `SchemaProgram[T]`, `BoundSchema[T]`, `Session[T]`, `Generator[T]`
  with `str -> T` / `Sequence[str] -> list[T]` overloads, and annotations on every public
  Python callable (enforced by a test, since `py.typed` ships).
- `GenerationError`, raised when generation stops before the constrained value is complete,
  instead of leaking a raw `JSONDecodeError`.
- `integration/program6_processor_latency.py`: processor-only p50/p95/p99 across
  50K/150K/250K vocabularies and batch 1/8/32, for both backends.

### Removed

- Beam search, `num_return_sequences > 1`, `assistant_model`, `prompt_lookup_num_tokens`,
  `assistant_early_exit`, `custom_generate`, and per-call `eos_token_id`/`pad_token_id`
  overrides are rejected up front rather than silently mis-masked. Non-CPU, sharded, and
  meta-device placements are rejected by `from_transformers`.

### Added

- Rust `Compiler` / `SchemaProgram` / `BoundSchema` / `Session` facade
  (`maskforge_core::api`): compile once, bind to a vocabulary, drive a session, without the
  caller inspecting which backend a schema needs. `Compiler` owns a configurable, clearable
  executable cache; `Compiler::without_cache()` for deterministic tests.
- `ExecutableCache::clear()` (Rust) and `maskforge.low_level.clear_executable_cache()` /
  `maskforge.low_level.executable_cache_stats()` (Python), previously bench-internals-only.
  `PyVocabulary.clear_artifact_cache()` (Python), previously bench-internals-only under a
  different name.
- `maskforge.Compiler` / `Generator` as the documented Python entry point: selects the
  byte-DFA or structured backend internally.
- Stable `MaskforgeError` shape: `.code`, `.stage`, `.pointer`, `.keyword`, `.observed`,
  `.message`, `.limit`, consistent between native and pure-Python raise sites.
- `test-utils` Cargo feature gating the correctness corpus (`corpus_*`) out of production
  Python wheels.
- Python type stubs (`py.typed`, `_native.pyi`) for the stable surface.
- `examples/python/` and `examples/rust/`, each verified to run and produce correct output.
- `docs/development/json-schema-support.md`, a keyword-by-keyword support matrix checked
  directly against the built extension.
- `THIRD_PARTY_NOTICES.md` documenting vendored and locally-used third-party data.

### Fixed

- `ConstraintSession`: committing EOS did not transition the session to a terminal state.
  Ordinary tokens committed afterward were silently accepted instead of raising, and
  `mask_into()`/`allowed_ids()` kept showing the pre-stop mask. Added an explicit
  Active/Stopped lifecycle; `reset()` clears it. `replay()` is now transactional: on failure
  the session is left at the empty prefix, never partially replayed.
- The Rust facade's `Session::advance` for the structured backend silently treated a rejected
  token as success (`StructuredMatcher::advance` signals rejection via `Ok(false)`, not `Err`,
  and the return value was being discarded); now surfaces `SessionError::IllegalToken`.
- Root package identity (was `outlines_core`, now `maskforge`) in `pyproject.toml`,
  `Cargo.toml`, CI workflows, and the Makefile.
- `hugginface-hub` Cargo feature typo, renamed to `huggingface-hub`.
- `MaskforgeLogitsProcessor` O(T²) per-step cost: the append-only decode path now reads only
  the newest token instead of re-copying the full sequence every step.
- CI wheel matrix: now builds one ABI3 wheel per platform for Python 3.11-3.14 (was
  contradicting the `abi3-py311` PyO3 configuration by also testing 3.9/3.10).
- `cargo publish`/`cargo package` now target `maskforge-core` explicitly; `maskforge-py` is
  `publish = false`.
- `crates/maskforge-core`'s published package now excludes the vendored JSON Schema Test
  Suite and other test-only data (verified via `cargo package --list`).

### Known limitations

See the README's "What's not finished yet" section: single-sequence generation only, no full
Draft 2020-12 Format-Assertion vocabulary, Hugging Face integration measured for
mask-application only.
