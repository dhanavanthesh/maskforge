"""OutputSpec.normalize: JSON Schema text/dict, Pydantic types, and the hard cases around them.

Run with: uv run pytest crates/maskforge-py/tests -q
"""

import dataclasses
import enum
import json
import typing

import pytest

maskforge = pytest.importorskip("maskforge")
pydantic = pytest.importorskip("pydantic")

from maskforge.constraints import OutputSpec, normalize  # noqa: E402


def test_json_schema_string_passes_through_unchanged():
    text = json.dumps({"type": "boolean"})
    assert normalize(text).schema_text == text


def test_dict_is_serialized_deterministically():
    spec = normalize({"type": "integer"})
    assert json.loads(spec.schema_text) == {"type": "integer"}


def test_output_spec_passes_through_unchanged():
    spec = OutputSpec(schema_text="{}", parse_json=lambda t: t)
    assert normalize(spec) is spec


def test_nested_base_model():
    class Inner(pydantic.BaseModel):
        x: int

    class Outer(pydantic.BaseModel):
        inner: Inner

    assert "inner" in normalize(Outer).schema_text


def test_recursive_model():
    class Node(pydantic.BaseModel):
        value: int
        children: list["Node"] = []

    Node.model_rebuild()
    assert normalize(Node) is not None


def test_root_model():
    class Root(pydantic.RootModel[list[int]]):
        pass

    assert normalize(Root) is not None


def test_dataclass_round_trips_through_parse_json():
    @dataclasses.dataclass
    class DC:
        a: int
        b: str

    spec = normalize(DC)
    assert spec.parse_json(json.dumps({"a": 1, "b": "x"})) == DC(a=1, b="x")


def test_typed_dict_with_optional_keys():
    # Pydantic requires the typing_extensions backport for TypedDict before 3.12.
    typing_extensions = pytest.importorskip("typing_extensions")

    class TD(typing_extensions.TypedDict, total=False):
        a: int
        b: str

    assert normalize(TD) is not None


def test_literal_and_enum():
    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    assert normalize(typing.Literal["a", "b"]) is not None
    assert normalize(Color) is not None


def test_ordinary_union():
    assert normalize(typing.Union[int, str]) is not None


def test_pep604_union():
    assert normalize(int | str) is not None


def test_field_alias_appears_in_schema():
    class Aliased(pydantic.BaseModel):
        field_name: int = pydantic.Field(alias="fieldName")

    assert "fieldName" in normalize(Aliased).schema_text


def test_optional_vs_required_nullable():
    class OptModel(pydantic.BaseModel):
        a: typing.Optional[int] = None
        b: int

    assert normalize(OptModel) is not None


def test_extra_allow_and_forbid():
    class Allow(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow")
        a: int

    class Forbid(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")
        a: int

    assert normalize(Allow) is not None
    assert normalize(Forbid) is not None


def test_unicode_property_names_are_not_ascii_escaped():
    class Unicode(pydantic.BaseModel):
        名前: str

    assert "名前" in normalize(Unicode).schema_text


def test_constrained_numbers_and_strings():
    class Constrained(pydantic.BaseModel):
        n: int = pydantic.Field(ge=0, le=100)
        s: str = pydantic.Field(min_length=1, max_length=10)

    assert normalize(Constrained) is not None


def test_custom_validator_not_expressible_in_json_schema_still_runs_at_parse_time():
    class Validated(pydantic.BaseModel):
        n: int

        @pydantic.field_validator("n")
        @classmethod
        def check(cls, v):
            if v % 2 != 0:
                raise ValueError("must be even")
            return v

    spec = normalize(Validated)
    with pytest.raises(pydantic.ValidationError):
        spec.parse_json(json.dumps({"n": 3}))


def test_unsupported_callable_raises_type_error_not_a_bogus_empty_schema():
    # pydantic.TypeAdapter silently accepts a bare lambda by introspecting its call
    # signature; normalize() must reject it explicitly instead of forwarding that.
    with pytest.raises(TypeError):
        normalize(lambda: None)


def test_unsupported_plain_instance_raises_type_error():
    class Foo:
        pass

    with pytest.raises(TypeError):
        normalize(Foo())

    with pytest.raises(TypeError):
        normalize(42)
