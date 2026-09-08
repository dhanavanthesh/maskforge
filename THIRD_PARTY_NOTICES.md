# Third-party notices

## Committed to this repository

### outlines-core (Apache-2.0)

`maskforge-core` began as a reorganized fork of
[outlines-core](https://github.com/dottxt-ai/outlines-core) (Apache-2.0). File-by-file
provenance (which files are verbatim, hardened-over-keep, extracted, or new) is tracked in
`crates/maskforge-core/PROVENANCE.md`.

### JSON Schema Test Suite (MIT)

`crates/maskforge-core/tools/json-schema-test-suite/` vendors the official
[json-schema-org/JSON-Schema-Test-Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite)
(MIT, Copyright (c) 2012 Julian Berman), pinned at commit `6648e8194c69697b2e1a15fe76a06a480b183a51`.
Full provenance, the pin date, and the one excluded file (`refRemote.json`, which needs a
network fetcher this repository does not have) are documented in
`crates/maskforge-core/tools/json-schema-test-suite/PROVENANCE.md`.

This directory is excluded from the published `maskforge-core` crate (see the `include` list
in `crates/maskforge-core/Cargo.toml`) and from the Python wheel (outside `maturin`'s
`python-source` root); it exists only for `cargo test`.
