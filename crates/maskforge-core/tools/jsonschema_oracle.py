# /// script
# requires-python = ">=3.9"
# dependencies = ["jsonschema>=4"]
# ///
"""Minimal standalone Python jsonschema oracle.

Reads a JSON array of {"schema": <json-schema>, "instance": <json-value>} from stdin and
prints {"python", "jsonschema", "verdicts": [bool, ...]} to stdout. Runs as an external
process; never linked into the Rust crate.
"""
import json
import sys
from importlib.metadata import version

from jsonschema import Draft202012Validator


def main() -> None:
    data = json.load(sys.stdin)
    verdicts = [bool(Draft202012Validator(item["schema"]).is_valid(item["instance"])) for item in data]
    json.dump(
        {
            "python": ".".join(str(p) for p in sys.version_info[:3]),
            "jsonschema": version("jsonschema"),
            "verdicts": verdicts,
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
