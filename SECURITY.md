# Security policy

## Reporting a vulnerability

Please report security issues privately via GitHub's
["Report a vulnerability"](https://github.com/dhanavanthesh/maskforge/security/advisories/new)
flow rather than a public issue. Include the schema or input that triggers the problem, the
observed behavior, and your MaskForge version.

## Scope

MaskForge processes caller-supplied JSON Schema text and vocabulary data. Configurable resource
limits bound compilation and matching; inputs that exceed them should return a typed diagnostic.

Please report any supported input that causes unbounded resource use, non-termination, a panic,
memory-safety failure, or a token mask that permits an invalid transition.

External `$ref` targets require the caller to supply the referenced resource text; MaskForge
performs no network I/O, so it has no SSRF surface of its own.

## Supported versions

Only the latest `0.1.x` release is supported during the alpha period; there is no
long-term-support branch yet.
