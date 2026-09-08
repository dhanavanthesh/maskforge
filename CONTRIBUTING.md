# Contributing to MaskForge

## Setup

```bash
git clone https://github.com/dhanavanthesh/maskforge.git
cd maskforge
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -e ".[test]"
```

## Building the Python extension

```bash
make build-extension-debug     # debug build
make build-extension-release   # release build
```

Both enable `test-utils` (the correctness corpus) and `bench-internals`, so the local test
suite has everything it needs. Neither feature ships in a released wheel.

## Running tests

```bash
make test           # Rust + Python
make test-rust       # cargo test -p maskforge-core, -p maskforge-py
make test-python     # pytest crates/maskforge-py/tests
```

## Before sending a change

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
make test
```

Read `project-instruction.log` before touching `crates/maskforge-core/src/` — it has the
project's hygiene rules (comment length, no AI-sounding language, no internal planning
references in shipped code) enforced on every review.

## Reporting a bug

Open an issue with: the schema (if JSON-Schema-related), the exact error (`MaskforgeError`'s
`.code`/`.stage`/`.message` if applicable), and your MaskForge version
(`python -c "import maskforge; print(maskforge.__version__)"` or the crate version).

## Third-party data

If your change adds a new vendored test corpus or dataset, update `THIRD_PARTY_NOTICES.md`
with its source, license, and pinned revision before it can be committed.
