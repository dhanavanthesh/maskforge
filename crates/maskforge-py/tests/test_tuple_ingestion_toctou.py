"""The direct tuple-to-CSR ingestion path (PyVocabulary/compile_ir/compile_json_schema) must
never let its packed buffers grow past what the pre-scan admitted, even when a token component
reports a smaller length on the pre-scan pass than it actually yields on the copy pass. This is a
single-threaded, deterministic reproduction of the GIL-release race the real admission gate closes
against a genuinely concurrent mutator - it exercises the same check-before-copy code path."""

import pytest

maskforge = pytest.importorskip("maskforge")

EOS = 999


class GrowingBytes:
    """A proper sequence whose `__len__` reports a small size on its FIRST call (the pre-scan) and
    the real, much larger size on every call after (the copy pass) - reproducing what a genuinely
    concurrent mutation between pre-scan and copy would look like, without needing real threads."""

    def __init__(self, real: bytes, lied_len: int):
        self._real = real
        self._lied_len = lied_len
        self._calls = 0

    def __len__(self):
        self._calls += 1
        return self._lied_len if self._calls == 1 else len(self._real)

    def __getitem__(self, i):
        return self._real[i]


class GrowingIds:
    """Same growth-after-first-call trick as `GrowingBytes`, for the ids component."""

    def __init__(self, real, lied_len: int):
        self._real = real
        self._lied_len = lied_len
        self._calls = 0

    def __len__(self):
        self._calls += 1
        return self._lied_len if self._calls == 1 else len(self._real)

    def __getitem__(self, i):
        return self._real[i]


def test_bytes_component_growing_after_prescan_is_rejected_not_overrun():
    # First len() call (pre-scan) reports 1 byte; every call after reports the real 4096.
    real = bytes(range(256)) * 16
    tokens = [(GrowingBytes(real, 1), [0]), (b"ok", [1])]
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.PyVocabulary(EOS, tokens)


def test_ids_component_growing_after_prescan_is_rejected_not_overrun():
    real = list(range(2, 2 + 4096))
    tokens = [(b"a", GrowingIds(real, 1)), (b"ok", [1])]
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.PyVocabulary(EOS, tokens)


def test_well_formed_tuple_input_still_succeeds():
    tokens = [(b"a", [0]), (b"b", [1]), (b"c", [2])]
    h = maskforge.PyVocabulary(EOS, tokens)
    assert h.len() == 4
