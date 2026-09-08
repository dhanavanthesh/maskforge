"""OutputSpec: normalizes a schema-like input into one immutable (schema, parser) pair."""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")

# Accepted output types for `Compiler.compile` and `Generator`; `object` covers typing constructs.
# `Any` is excluded so downstream validation remains effective.
OutputInput = typing.Union[
    "OutputSpec[T]",
    str,
    "dict[str, object]",
    "type[T]",
    object,
]


@dataclasses.dataclass(frozen=True)
class OutputSpec(Generic[T]):
    """Immutable (schema_text, parser) pair; `parse_json` turns generated text back into a value."""

    schema_text: str
    parse_json: Callable[[str], T]
    output_type: object | None = None


def _identity_parse(text: str) -> Any:
    return json.loads(text)


def _looks_like_a_type(value: object) -> bool:
    """True for a class or typing construct; guards TypeAdapter against an arbitrary callable."""
    if isinstance(value, type):
        return True
    if typing.get_origin(value) is not None:
        return True
    if isinstance(value, types.UnionType):
        return True
    return value is None


def _unsupported(value: object) -> TypeError:
    return TypeError(
        f"{value!r} is not a supported output type; expected an OutputSpec, JSON Schema "
        "string, dict, or a Pydantic-compatible type (BaseModel, dataclass, TypedDict, "
        "Enum, Literal, a union, or list[T])"
    )


def normalize(value: OutputInput[T]) -> OutputSpec[T]:
    """Normalizes `value` into an OutputSpec; raises TypeError for an unsupported input."""
    if isinstance(value, OutputSpec):
        return value
    if isinstance(value, str):
        # No JSON validation here: malformed text is left for the native compiler to reject
        # as a proper MaskforgeError, not a raw json.JSONDecodeError.
        return OutputSpec(schema_text=value, parse_json=_identity_parse, output_type=None)
    if isinstance(value, dict):
        return OutputSpec(
            schema_text=json.dumps(value, sort_keys=True, ensure_ascii=False),
            parse_json=_identity_parse,
            output_type=None,
        )

    if not _looks_like_a_type(value):
        raise _unsupported(value)

    try:
        import pydantic
    except ImportError as exc:
        raise ImportError(
            'constraining to a Python type requires Pydantic: pip install "maskforge[pydantic]"'
        ) from exc

    try:
        adapter: pydantic.TypeAdapter[T] = pydantic.TypeAdapter(value)
        schema_text = json.dumps(
            adapter.json_schema(mode="validation"), sort_keys=True, ensure_ascii=False
        )
    except Exception as exc:
        raise _unsupported(value) from exc

    return OutputSpec(schema_text=schema_text, parse_json=adapter.validate_json, output_type=value)
