"""ConstraintSession lifecycle: Active -> Stopped, transactional replay, error shape.

Run with: uv run pytest crates/maskforge-py/tests -q
"""

import json

import pytest

maskforge = pytest.importorskip("maskforge")

EOS = 256
BOOLEAN_SCHEMA = json.dumps({"type": "boolean"})


def byte_vocab():
    return [(bytes([b]), [b]) for b in range(256)]


def new_session(schema=BOOLEAN_SCHEMA):
    vocabulary = maskforge.PyVocabulary(EOS, byte_vocab())
    return maskforge.ConstraintSession.from_json_schema(vocabulary, schema)


def test_eos_before_acceptance_raises_illegal_token():
    session = new_session()
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        session.commit_token(EOS)
    assert excinfo.value.code == "IllegalToken"
    assert not session.is_stopped


def test_eos_at_acceptance_transitions_to_stopped():
    session = new_session()
    for byte in b"true":
        session.commit_token(byte)
    assert not session.is_stopped
    session.commit_token(EOS)
    assert session.is_stopped


def test_ordinary_token_after_stopped_raises():
    session = new_session()
    for byte in b"true":
        session.commit_token(byte)
    session.commit_token(EOS)
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        session.commit_token(ord(" "))
    assert excinfo.value.code == "IllegalToken"


def test_mask_into_after_stopped_is_eos_only():
    session = new_session()
    for byte in b"true":
        session.commit_token(byte)
    session.commit_token(EOS)
    assert session.allowed_ids() == [EOS]


def test_reset_clears_stopped_state():
    session = new_session()
    for byte in b"true":
        session.commit_token(byte)
    session.commit_token(EOS)
    session.reset()
    assert not session.is_stopped
    assert session.token_ids == ()
    # session is usable again after reset
    for byte in b"false":
        session.commit_token(byte)
    assert session.is_finished


def test_replay_is_transactional_on_failure():
    session = new_session()
    with pytest.raises(maskforge.MaskforgeError):
        session.replay([ord("t"), ord("r"), ord("!"), ord("e")])
    assert session.token_ids == ()
    assert not session.is_stopped


def test_replay_succeeds_and_matches_manual_commit():
    session = new_session()
    session.replay([ord(c) for c in "true"])
    assert session.is_finished
    assert session.token_ids == tuple(ord(c) for c in "true")


def test_error_shape_is_consistent_across_raise_sites():
    unsupported = None
    try:
        maskforge.ConstraintSession.from_json_schema(
            maskforge.PyVocabulary(EOS, byte_vocab()),
            json.dumps({"type": "string", "pattern": "(a"}),
        )
    except maskforge.MaskforgeError as exc:
        unsupported = exc
    illegal = None
    try:
        new_session().commit_token(EOS)
    except maskforge.MaskforgeError as exc:
        illegal = exc
    for err in (unsupported, illegal):
        assert err is not None
        assert isinstance(err.code, str)
        assert isinstance(err.stage, str)
        assert isinstance(err.message, str)
        assert len(err.args) == 7
