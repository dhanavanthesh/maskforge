# JSON Schema support

MaskForge compiles supported JSON Schema constraints into token masks. A recognized validation
construct outside the supported subset fails compilation with a typed diagnostic. It is not removed
or treated as an unconstrained branch.

Status meanings:

- **Supported**: complete-instance semantics and the tested incremental paths are implemented.
- **Partial**: complete instances are validated, but a documented generation or assertion limit
  remains.
- **Rejected**: compilation fails with `Unsupported` and identifies the keyword and JSON Pointer.

## Quick reference

| Keyword or feature | Drafts | Status | Notes |
| --- | --- | --- | --- |
| `type` | 04 to 2020-12 | Supported | All seven JSON types |
| `enum`, `const` | 04 to 2020-12 | Supported | Composite values and decoded string equality |
| `minLength`, `maxLength` | 04 to 2020-12 | Supported | Counts decoded Unicode scalar values |
| `pattern` | 04 to 2020-12 | Partial | Common ECMA-style syntax; exclusions below |
| `format` | 07 to 2020-12 | Partial | Annotation by default; opt-in assertions for `date`, `date-time`, `time`, `uuid`, and `ipv4` |
| `minimum`, `maximum`, exclusive bounds, `multipleOf` | 04 to 2020-12 | Partial | Complete-instance validation; numeric prefix limit below |
| `items`, `prefixItems` | 2020-12 | Supported | Homogeneous arrays and tuples |
| `minItems`, `maxItems` | 04 to 2020-12 | Supported | Array length bounds |
| `contains`, `minContains`, `maxContains` | 06 to 2020-12 | Supported | Incremental counts and completion checks |
| `uniqueItems` | 04 to 2020-12 | Supported | Semantic JSON equality, including decoded escapes |
| `properties`, `required` | 04 to 2020-12 | Supported | Required and optional properties |
| `additionalProperties` | 04 to 2020-12 | Supported | Boolean and schema forms |
| `propertyNames`, `patternProperties` | 06 to 2020-12 | Supported | Decoded keys and overlapping patterns |
| `minProperties`, `maxProperties` | 04 to 2020-12 | Supported | Object property count |
| `dependentRequired`, `dependentSchemas` | 2019-09, 2020-12 | Supported | Structured backend |
| `dependencies` | 04 to 07 | Supported | Legacy combined form |
| `unevaluatedProperties` | 2019-09, 2020-12 | Supported | Uses successful annotations from adjacent applicators |
| `unevaluatedItems` | 2019-09, 2020-12 | Partial | Complete-instance validation; prefix limit below |
| `allOf`, `anyOf`, `oneOf`, `not` | 04 to 2020-12 | Supported | `oneOf` keeps exactly-one semantics |
| `if`, `then`, `else` | 07 to 2020-12 | Partial | Complete-instance validation; cross-property prefix limit below |
| `$defs`, `$ref`, `$anchor` | 04 to 2020-12 | Supported | Internal references; external resources may be supplied by the caller |
| `$dynamicRef`, `$dynamicAnchor` | 2020-12 | Supported | Dynamic scope with bounded execution resources |
| Legacy array-form `items`, `additionalItems` | 04 to 07 | Rejected | Use 2020-12 `prefixItems` and `items` |
| `$recursiveRef`, `$recursiveAnchor` | 2019-09 | Rejected | Use 2020-12 dynamic references |
| Full Format-Assertion vocabulary | 2019-09, 2020-12 | Rejected | Only the opt-in subset above is asserted |
| `contentEncoding`, `contentMediaType`, `contentSchema` validation | 07 to 2020-12 | Partial | Preserved as annotations; content is not decoded or independently validated |

## Regular expressions

`pattern`, `patternProperties`, and regex entry points reject these constructs:

- numeric and named backreferences
- lookbehind
- non-leading or nested lookahead
- incompatible ECMA group constructs
- malformed groups, classes, and escapes

Supported leading lookaheads are bounded by compilation resource limits.

## Prefix-generation limits

Schema validation and generation liveness are different properties. Validation asks whether a
complete instance is valid. Prefix liveness asks whether every admitted prefix still has a valid
completion.

Current limits:

- `unevaluatedItems: false` can admit an item separator before proving that the next item will receive
  an evaluation annotation.
- A property admitted while `if` is undecided can later make the selected `then` or `else` branch
  unclosable. Preselect or discriminate the branch for unattended generation.
- Numeric constraints, including a numeric `const`, can admit some decimal prefixes that cannot
  complete. A weak model can extend such a prefix until the number-byte resource limit is reached.
  Use a string `const` or string `enum` for a fixed numeric-looking value.
- Recursive schemas are valid and supported, but a model can keep selecting the recursive branch.
  A token budget bounds execution time; it does not prove termination.

The finite payment example avoids these shapes. Internal tests retain them as correctness fixtures.

## Formatting policy

JSON permits arbitrary insignificant whitespace. `Generator(..., max_json_whitespace=N)` optionally
caps consecutive whitespace characters outside strings. Omitting the option preserves the full JSON
whitespace language. The option does not alter whitespace inside strings or rewrite Unicode escapes.

## Errors and resources

Compilation errors expose stable fields through `MaskforgeError`: `code`, `stage`, `pointer`,
`keyword`, `observed`, `message`, and `limit`.

Relevant codes include:

| Code | Meaning |
| --- | --- |
| `Malformed` | Invalid schema syntax or shape |
| `Unsupported` | Recognized validation construct outside the supported subset |
| `ReferenceResolution` | A referenced resource was not supplied or could not be resolved |
| `InternalLimitExceeded` | A documented compiler, matcher, or memory bound was reached |
| `IllegalToken` | The caller attempted to commit a token not allowed at the current prefix |

External references are never fetched implicitly. Document size, total supplied resource bytes,
compiled graph size, active matcher state, and retained caches are bounded. Exceeding a bound returns
a typed error instead of truncating the schema or mask.
