"""Prove the checked-out tree is exactly the tagged commit, at the tagged version.

Release artifacts must be a byte-for-byte product of the tag. Nothing here mutates the source:
the version is committed before tagging, and this script only refuses to proceed when the tag,
the manifest and the working tree disagree.
"""

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

# v-prefixed tag -> bare SemVer, e.g. "v0.1.0" -> "0.1.0".
TAG_RE = re.compile(r"^v(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)$")


class ValidationError(RuntimeError):
    """The tree does not match the tag it claims to build."""


def normalized_version(tag: str) -> str:
    """Strips the leading 'v', rejecting anything that is not SemVer."""
    match = TAG_RE.match(tag)
    if not match:
        raise ValidationError(f"release tag {tag!r} must be 'v' followed by valid SemVer")
    return match.group(1)


def manifest_version(manifest: Path) -> str:
    """Reads `[workspace.package] version`, falling back to `[package] version`."""
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    workspace = data.get("workspace", {}).get("package", {})
    if "version" in workspace:
        return workspace["version"]
    package = data.get("package", {})
    if "version" in package:
        return package["version"]
    raise ValidationError(f"no version field found in {manifest}")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ValidationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def check_clean_worktree() -> None:
    """Refuses to build from a tree with uncommitted changes."""
    if _git("status", "--porcelain"):
        raise ValidationError("working tree is dirty; release builds must not modify source")


def check_head_is_tag(tag: str) -> None:
    """Refuses to build when HEAD is not the commit the tag points at."""
    head = _git("rev-parse", "HEAD")
    tagged = _git("rev-list", "-n", "1", tag)
    if head != tagged:
        raise ValidationError(f"HEAD {head} is not the tagged commit {tagged} for {tag}")


def validate(tag: str, manifest: Path, check_git: bool = True) -> str:
    """Runs every check and returns the validated version."""
    version = normalized_version(tag)
    declared = manifest_version(manifest)
    if version != declared:
        raise ValidationError(
            f"tag {tag} implies version {version}, but {manifest} declares {declared}; "
            "commit the version change before tagging"
        )
    if check_git:
        check_clean_worktree()
        check_head_is_tag(tag)
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", type=str, help="release tag, e.g. v0.1.0")
    parser.add_argument("--manifest", "-m", type=Path, default=Path("Cargo.toml"))
    parser.add_argument(
        "--skip-git", action="store_true", help="validate tag/manifest agreement only"
    )
    args = parser.parse_args()
    try:
        version = validate(args.tag, args.manifest, check_git=not args.skip_git)
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"validated {args.tag} against {args.manifest}: version {version}")


if __name__ == "__main__":
    main()
