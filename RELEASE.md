# Release process

## Pre-flight

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p maskforge-core --release
cargo test -p maskforge-core --release --all-features
cargo test -p maskforge-py --features python-bindings,test-utils,bench-internals
cargo package -p maskforge-core --list   # inspect: no corpora, no evidence, no stray binaries
cargo publish -p maskforge-core --dry-run
uv build
python -m twine check --strict dist/*
```

Then a clean-venv install proof:

```bash
python -m venv .release-venv
.release-venv/bin/pip install --no-index --find-links dist maskforge
.release-venv/bin/python -c "import maskforge; print(maskforge.__version__)"
```

## Cutting a release

The published artifacts are a byte-for-byte product of the tagged commit. CI never rewrites the
manifest: it validates that the tag, `Cargo.toml`'s committed version and the checked-out commit
all agree, and refuses to build otherwise. The version change is therefore committed *before*
tagging. The tag must be `v` followed by valid SemVer (e.g. `v0.1.0`).

1. Update `Cargo.toml`'s `[workspace.package] version` to the release version.
2. Update `CHANGELOG.md`: move `[Unreleased]` entries under the new version and date.
3. Commit both. Confirm `dry_run_publish.yml` is green on `main`.
4. Tag that exact commit (`git tag v0.1.0`) and push the tag.
5. Creating a GitHub release from the tag triggers `publish.yml`, which:
   - validates tag == committed version == HEAD, on a clean worktree;
   - builds wheels for the supported platform matrix and runs the end-to-end wheel smoke test;
   - runs `twine check --strict`, `cargo package` and `cargo publish --dry-run`, and records
     SHA-256 for every artifact, all *before* any upload;
   - publishes `maskforge-core` to crates.io first (the immutable one), then `maskforge` to PyPI.
     `maskforge-py` stays `publish = false`.

Both upload jobs run in protected environments (`crates-io`, `pypi`) and require approval. PyPI
uses Trusted Publishing, so no long-lived API token is stored.

## Supported platform matrix (0.1.0)

Linux x86_64/aarch64, macOS x86_64/arm64, Windows x86_64; Python 3.11-3.14; Rust MSRV 1.86,
stable. Do not claim support for a platform whose wheel has not actually been built,
installed, and import-tested in CI.

## Required credentials

`CARGO_REGISTRY_TOKEN`, scoped to the `maskforge-core` project and held in the
`crates-io` environment. PyPI needs no stored token: the `pypi` environment publishes
through Trusted Publishing (OIDC).

## Manual publish (if `publish.yml` needs to be re-run)

```bash
gh workflow run publish.yml --ref main -f tag=v0.1.0
```
