# Vendored JSON Schema Test Suite

Source: https://github.com/json-schema-org/JSON-Schema-Test-Suite
License: MIT (Copyright (c) 2012 Julian Berman; see the suite repository's `LICENSE`).

All 46 `draft2020-12/*.json` files are vendored here verbatim. `correctness::official_suite`
runs every one of them and classifies each group as supported, unsupported, or malformed, and
every case in a supported group as passed or failed against the suite's own `valid` field:

```
cargo test -p maskforge-core official_suite -- --nocapture
```

No group or case is hand-picked. The classification is produced mechanically from the
frontend's own diagnostics, and a test asserts the vendored file set matches the list the
runner expects, so adding or removing a file fails the build rather than silently changing
coverage.

`refRemote.json` is excluded: its cases resolve `$ref` targets from other HTTP-served schema
documents. MaskForge resolves `$ref` locally by design and has no network-fetch component to
serve them, even for testing.
