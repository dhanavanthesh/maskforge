"""_native.pyi must declare exactly the stable native surface: no missing, no stale names.

Run with: uv run pytest crates/maskforge-py/tests -q
"""

import ast
from pathlib import Path

import pytest

maskforge = pytest.importorskip("maskforge")
native = pytest.importorskip("maskforge._native")

# Feature-gated test APIs are omitted from published stubs.
_FEATURE_GATED = {
    "compile_schema_executable_for_bench",
    "schema_to_ir_with_resources_profile_for_bench",
    "corpus_index",
    "corpus_json_schema",
    "corpus_names",
    "corpus_samples",
}


def _stub_names():
    stub_path = Path(native.__file__).parent / "_native.pyi"
    tree = ast.parse(stub_path.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _native_public_names():
    return {n for n in dir(native) if not n.startswith("_")} - _FEATURE_GATED


def test_every_stable_native_name_is_declared_in_the_stub():
    missing = _native_public_names() - _stub_names()
    assert not missing, f"declared in _native but missing from _native.pyi: {sorted(missing)}"


def test_the_stub_declares_no_stale_names():
    stale = _stub_names() - _native_public_names() - _FEATURE_GATED
    assert not stale, f"declared in _native.pyi but absent from the built extension: {sorted(stale)}"


def test_documented_stable_classes_are_exported_from_the_package_root():
    for name in ("Compiler", "SchemaProgram", "BoundSchema", "Session", "Vocabulary", "OutputSpec"):
        assert name in maskforge.__all__, f"{name} is missing from maskforge.__all__"
        assert hasattr(maskforge, name), f"{name} is missing from the maskforge package"


def test_the_public_surface_type_checks_cleanly():
    """py.typed is shipped, so a type checker must find no error in the package or examples."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    if shutil.which("mypy") is None:
        try:
            import mypy  # noqa: F401
        except ImportError:
            pytest.skip("mypy is not installed in this environment")

    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable, "-m", "mypy",
            str(root / "crates/maskforge-py/python/maskforge"),
            str(root / "examples/python"),
            "--ignore-missing-imports",
            "--no-error-summary",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, f"mypy reported errors:\n{result.stdout}\n{result.stderr}"
