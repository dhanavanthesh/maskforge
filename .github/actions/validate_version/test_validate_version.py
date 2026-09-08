"""The version validator must refuse every way a tag can disagree with the tree.

Run: python -m pytest .github/actions/validate_version/test_validate_version.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from validate_version import (  # noqa: E402
    ValidationError,
    manifest_version,
    normalized_version,
    validate,
)

WORKSPACE_MANIFEST = '[workspace.package]\nversion = "0.1.0"\nedition = "2021"\n'
PACKAGE_MANIFEST = '[package]\nname = "x"\nversion = "2.3.4"\n'


def write(tmp_path: Path, text: str) -> Path:
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text(text, encoding="utf-8")
    return manifest


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.1.0", "0.1.0"),
        ("v1.2.3", "1.2.3"),
        ("v0.1.0-alpha.1", "0.1.0-alpha.1"),
        ("v0.1.0+build.5", "0.1.0+build.5"),
    ],
)
def test_a_valid_tag_normalizes(tag, expected):
    assert normalized_version(tag) == expected


@pytest.mark.parametrize(
    "tag",
    ["0.1.0", "release-0.1.0", "v0.1", "v1.2.3.4", "vX.Y.Z", "", "main", "v0.1.0 "],
)
def test_a_malformed_tag_is_rejected(tag):
    with pytest.raises(ValidationError):
        normalized_version(tag)


def test_the_workspace_version_is_preferred(tmp_path):
    assert manifest_version(write(tmp_path, WORKSPACE_MANIFEST)) == "0.1.0"


def test_a_plain_package_version_is_read(tmp_path):
    assert manifest_version(write(tmp_path, PACKAGE_MANIFEST)) == "2.3.4"


def test_a_manifest_without_a_version_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="no version field"):
        manifest_version(write(tmp_path, '[package]\nname = "x"\n'))


def test_a_matching_tag_and_manifest_validate(tmp_path):
    assert validate("v0.1.0", write(tmp_path, WORKSPACE_MANIFEST), check_git=False) == "0.1.0"


def test_a_tag_ahead_of_the_manifest_is_rejected(tmp_path):
    """The exact failure the old bump-at-release-time flow hid by rewriting the manifest."""
    with pytest.raises(ValidationError, match="commit the version change before tagging"):
        validate("v0.2.0", write(tmp_path, WORKSPACE_MANIFEST), check_git=False)


def test_a_prerelease_tag_against_a_release_manifest_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        validate("v0.1.0-alpha.1", write(tmp_path, WORKSPACE_MANIFEST), check_git=False)


def test_the_repository_manifest_matches_a_tag_for_its_own_version():
    """Guards against the checked-in manifest drifting out of SemVer shape."""
    repo_manifest = Path(__file__).resolve().parents[3] / "Cargo.toml"
    version = manifest_version(repo_manifest)
    assert validate(f"v{version}", repo_manifest, check_git=False) == version
