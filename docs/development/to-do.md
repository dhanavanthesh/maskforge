# Development roadmap

This page tracks unfinished engineering work for MaskForge. It covers correctness first, then
generation reliability, serving, and broader structured-output formats. Current JSON Schema support
and known limits are documented in [JSON Schema support](json-schema-support.md).

Status:

- `[ ]` not implemented
- `[~]` partly implemented or proved only on a limited corpus
- `[x]` complete and verified

## Work order

```text
+--------------------+     +---------------------+     +------------------------+
| Prefix correctness | --> | Independent oracles | --> | Generation reliability |
+--------------------+     +---------------------+     +------------------------+
                                                                  |
                                                                  v
+--------------------+     +---------------------+     +------------------------+
| Advanced semantics | <-- | Reasoning and tools | <-- | Branching and serving  |
+--------------------+     +---------------------+     +------------------------+
```

Correctness work blocks performance claims and new decoding modes. A feature is complete only after
its focused regression, differential gate, resource checks, and public documentation pass.

## JSON Schema correctness

### [ ] Conditional object liveness

An object key can be legal while `if` is undecided, then make the selected `then` or `else` branch
impossible to close.

**Solution:** track branch residuals together with seen properties and annotations. Admit a key,
value, comma, or closing brace only when at least one compatible branch still has a completion.

```text
object prefix -> update condition candidates -> apply branch obligations -> test closure -> mask
```

**Done when:** property-order, nested-combinator, escaped-key, and dependency cases pass
mask-versus-advance checks and an independently validated completion search.

### [ ] Numeric residual languages

Complete numbers validate correctly, but some prefixes under `const`, ranges, exclusive bounds, and
`multipleOf` can be admitted after every valid completion has become impossible.

**Solution:** represent the remaining numeric language over sign, integer, fraction, and exponent
components. Intersect syntax with exact rational constraints before admitting each byte.

```text
numeric prefix -> lexical state -> rational interval/residue -> completion test -> mask
```

**Done when:** integers, decimals, exponents, exclusive bounds, and `multipleOf` combinations have
two-way differential tests against complete-instance validation, including long-prefix limits.

### [ ] `unevaluatedItems` prefix liveness

`unevaluatedItems: false` can admit a separator before proving that the next item can receive an
evaluation annotation.

**Solution:** carry per-index annotations from `prefixItems`, `items`, `contains`, and adjacent
applicators. Check remaining annotated capacity before opening another item.

```text
array prefix -> merge successful annotations -> compute next-index owners -> test capacity -> mask
```

**Done when:** overlapping `contains`, conditional applicators, tuple tails, and recursive references
cannot enter an unclosable array prefix.

## Correctness testing

### [ ] Admitted-frontier liveness gate

Walking one valid witness proves that a good path remains available. It does not prove that every
admitted branch remains completable.

**Solution:** expand bytes actually admitted by the mask, prioritize structural closers, and search
each successor for an accepting completion. Exhausted budgets report `Inconclusive`, never success or
failure.

```text
reachable prefix -> enumerate admitted bytes -> create successor states -> closer-first search
                 -> accepting | proven trap | inconclusive
```

**Done when:** the finite keyword corpus explores admitted branches, catches seeded trap states, and
records stable node and depth budgets.

### [ ] Independent semantic oracle

Mask routes and `advance` share engine semantics, so agreement between them cannot detect a shared
interpretation error.

**Solution:** generate bounded complete instances from selected prefixes and validate them with an
independent Draft 2020-12 implementation. Test both valid reachability and invalid rejection.

```text
schema + prefix -> MaskForge candidates -> bounded completions -> external validator -> compare
```

**Done when:** the official JSON Schema Test Suite, adversarial keyword fixtures, and a pinned
JSONSchemaBench sample run reproducibly with categorized results.

### [ ] Stateful fuzzing across token boundaries

Escapes, UTF-8 fragments, duplicate values, and multi-byte vocabulary records expose behavior that
complete-document tests miss.

**Solution:** fuzz schemas, raw vocabulary records, aliases, chunk boundaries, repeated masks,
rollback, and resource limits. Minimize every disagreement to a schema, prefix, and token record.

```text
seed corpus -> mutate schema/prefix/vocabulary -> mask and advance -> minimize -> regression
```

**Done when:** continuous fuzz targets cover regular, structured, and reference routes without state
mutation, accounting drift, or unreproducible failures.

## Runtime and decoding

### [ ] Session fork, rollback, and speculative decoding

Sessions currently support append-only generation. Beam search and speculative decoding need cheap,
exact state branching.

**Solution:** add bounded token checkpoints, `rollback(n)`, and copy-on-write session forks. Traverse
draft chains and trees transactionally, then commit the accepted prefix once.

```text
session -> fork/checkpoint -> verify draft branches -> select accepted prefix -> commit or rollback
```

**Done when:** beam reorder, draft-tree verification, rejection rollback, cache ownership, and memory
accounting pass under regular and structured schemas.

### [ ] Batch-native and device-native masks

Python batching still coordinates per-session work, and model placement is limited by the high-level
adapter.

**Solution:** fill selected mask rows in one native call, reuse packed buffers, overlap CPU mask work
with model inference, and add tested CUDA/Triton application kernels without moving session state to
the GPU prematurely.

```text
active sessions -> native parallel mask fill -> selected device rows -> in-place logits mask
```

**Done when:** continuous batches can join, leave, and reorder rows; CPU and GPU paths are bit-identical;
latency and retained memory are reported for GPT-2, Qwen, and Bloom vocabularies.

### [ ] Deterministic spans and jump-forward decoding

Some states have a single forced byte sequence, but the model is still called for every token.

**Solution:** find the longest byte string accepted by every live path, tokenize it exactly, and
return only a prefix whose tokenization and state transition are proven equivalent.

```text
matcher state -> common forced bytes -> exact tokenization -> transactional advance -> appended span
```

**Done when:** ambiguous tokenizations, stop tokens, UTF-8, and structured boundaries fall back safely,
with measured end-to-end benefit before the feature is enabled by default.

## Output formats

### [ ] Mixed reasoning, text, and tool calls

MaskForge constrains JSON and regex output, but it does not yet describe model-specific streams such
as `<think>...</think>`, plain text, and one or more tool calls in the same response.

**Solution:** add a phase IR with literal tags, free-text regions, JSON Schema regions, and explicit
transitions. Provide adapters for common tool envelopes without baking model names into the core.

```text
model profile + tools -> phase IR -> text/reasoning/tool dispatcher -> active constraint -> mask
```

**Done when:** `auto`, `required`, `none`, and named-tool choices work for single, parallel, and repeated
calls; reasoning can be enabled or disabled; tool arguments retain full schema semantics.

### [ ] CFG and composable grammar frontend

Regex and JSON Schema do not cover arbitrary programming languages, templates, or mixed protocols.

**Solution:** introduce a typed grammar IR with EBNF or Lark-style rules, embedded regex terminals,
embedded JSON Schema nonterminals, special-token terminals, captures, and bounded recursion.

```text
CFG/EBNF + embedded JSON/regex -> grammar IR -> parser executable -> vocabulary binding -> session
```

**Done when:** left recursion, ambiguity, nullable rules, Unicode terminals, resource limits, and error
locations have reference tests and never silently change the accepted language.

### [ ] Captures, stops, and safe streaming

Applications need completed fields and tool arguments before the entire response ends, but an
arbitrary byte prefix is not always safe to expose or act on.

**Solution:** attach named captures to grammar regions and emit events only after a region is sealed.
Track a safe-to-render watermark separately from parser acceptance and stop-token handling.

```text
committed bytes -> capture boundaries -> sealed value -> validated event -> safe stream watermark
```

**Done when:** chunking cannot change capture values, escaped data cannot forge delimiters, and no tool
action fires before its complete arguments validate.

## Tools and application constraints

### [ ] Schema inspection and observability

Typed errors exist, but users still need to understand routing, unsupported constructs, resource cost,
and why a token was forced or rejected.

**Solution:** provide `inspect` and `explain` APIs plus a small CLI. Report dialect, route, references,
limits, cache use, mask density, forced spans, and rejection provenance without exposing mutable
internals.

```text
schema/session -> structured trace events -> bounded collector -> CLI/API report
```

**Done when:** diagnostics are stable, bounded, disabled at zero runtime cost, and redact caller data by
default.

### [ ] Semantic policies and abstention

JSON validity does not guarantee that a value is useful or truthful. Hard constraints can also force a
model to fabricate a value when `null` or refusal would be safer.

**Solution:** layer explicit application predicates over grammar masks. Support cross-field checks,
allowed-value lookups, and a schema-declared abstention branch while preserving a clear boundary
between JSON Schema semantics and application policy.

```text
grammar mask -> semantic policy -> abstention availability -> final mask -> audited decision
```

**Done when:** policies are deterministic, transactional, separately testable, and cannot silently
weaken the schema or remove every completion.

### [ ] Portable compiled artifacts and schema containment

Large deployments need reproducible cold starts and a way to decide whether a schema update broadens or
narrows accepted output.

**Solution:** version compiled artifacts with source and vocabulary fingerprints, validate them before
loading, and compare regular-language fragments for equivalence or containment. Return `Unknown` for
structured cases without a proof.

```text
schema + vocabulary -> versioned artifact -> integrity check -> load
old schema + new schema -> containment proof -> equal | narrower | broader | unknown
```

**Done when:** corrupted or incompatible artifacts fail closed, concurrent writers are atomic, and every
containment result includes a counterexample or a proof record.

## Completion standard

An item is checked off only when all applicable evidence exists:

1. A minimal failing or feature-defining fixture.
2. Complete-instance validation against an independent implementation.
3. Mask-versus-advance equality over raw vocabulary records and token aliases.
4. Repeated-mask, rejected-advance, rollback, and retained-memory checks.
5. Focused tests, formatting, strict Clippy, full Rust and Python suites, and the official profile.
6. Public documentation with limitations stated next to the feature.
7. Performance numbers only from a reproducible benchmark after correctness passes.

## Research basis

The roadmap uses the [JSON Schema Draft 2020-12 specification](https://json-schema.org/draft/2020-12),
the language-agnostic [JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite),
and the [JSONSchemaBench methodology](https://arxiv.org/abs/2501.10868) for semantics and evaluation.
