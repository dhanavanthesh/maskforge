"""Unified stateful generation API over MaskForge's two execution backends."""

from ._native import (
    MaskforgeError,
    PyStructuredMatcher,
    compile_ir_with_vocabulary,
    schema_to_ir,
)


def _session_error(code, stage, message):
    """Builds a MaskforgeError with the same 7-field shape native errors carry."""
    return MaskforgeError(code, stage, None, None, None, message, None)


class ConstraintSession:
    """One constrained generation sequence.

    ``from_json_schema`` selects the packed byte-DFA for regular schemas and the
    structured matcher for schemas that need value memory or order-independent
    objects. Callers therefore do not need to understand MaskForge's backend split.
    """

    def __init__(self, *, vocabulary, ir, index=None, structured=None):
        if (index is None) == (structured is None):
            raise ValueError("exactly one MaskForge backend is required")
        self._vocabulary = vocabulary
        self._ir = ir
        self._index = index
        self._structured = structured
        self._state = index.start() if index is not None else None
        self._tokens = []
        self._stopped = False
        self._mask = bytearray(4 * ((vocabulary.mask_vocab_size() + 31) // 32))

    @classmethod
    def from_json_schema(
        cls, vocabulary, schema, *, assume_closed=False, bind_mode="byte_trie"
    ):
        """Compile ``schema`` and return a fresh stateful generation session."""
        ir = schema_to_ir(schema, assume_closed)
        if not ir.is_supported():
            details = ", ".join(f"{code} at {pointer or '/'}" for code, pointer in ir.diagnostics())
            raise _session_error("Unsupported", "schema_compile", details or "schema is unsupported")
        if ir.requires_structured_backend():
            return cls(
                vocabulary=vocabulary,
                ir=ir,
                structured=PyStructuredMatcher(ir, vocabulary),
            )
        return cls(
            vocabulary=vocabulary,
            ir=ir,
            index=compile_ir_with_vocabulary(vocabulary, ir.to_wire(), bind_mode),
        )

    @property
    def backend(self):
        """``"dfa"`` or ``"structured"``."""
        return "dfa" if self._index is not None else "structured"

    @property
    def token_ids(self):
        """Committed generated token ids, as an immutable tuple."""
        return tuple(self._tokens)

    @property
    def mask_vocab_size(self):
        return self._vocabulary.mask_vocab_size()

    @property
    def eos_token_id(self):
        return self._vocabulary.eos_token_id()

    @property
    def is_finished(self):
        """Stopping now (committing EOS) would yield a complete, valid instance."""
        if self._index is not None:
            return self._index.is_accepting(self._state)
        return self._structured.is_finished()

    @property
    def is_stopped(self):
        """EOS has been committed; no further token may be committed until ``reset()``."""
        return self._stopped

    @property
    def is_dead(self):
        if self._index is not None:
            return self._index.is_dead(self._state)
        return self._structured.is_dead()

    def reset(self):
        """Return to the empty generated prefix while retaining compiled artifacts."""
        self._tokens.clear()
        self._stopped = False
        if self._index is not None:
            self._state = self._index.start()
        else:
            self._structured.reset()

    def commit_token(self, token_id):
        """Commit one sampled token. EOS transitions to ``Stopped``; anything after raises."""
        token_id = int(token_id)
        if self._stopped:
            raise _session_error("IllegalToken", "commit_token", "session is stopped; call reset() first")
        if token_id == self.eos_token_id:
            if not self.is_finished:
                raise _session_error("IllegalToken", "commit_token", "EOS is not legal at this prefix")
            self._stopped = True
            return
        if self._index is not None:
            self._state = self._index.advance(self._state, token_id)
        elif not self._structured.commit_token(token_id):
            raise _session_error(
                "IllegalToken", "commit_token", f"token {token_id} is not legal at this prefix"
            )
        self._tokens.append(token_id)

    def replay(self, token_ids):
        """Reset and replay a prefix; transactional, resets to empty again on any failure."""
        self.reset()
        try:
            for token_id in token_ids:
                self.commit_token(token_id)
        except Exception:
            self.reset()
            raise

    def mask_into(self, out=None):
        """Fill and return a packed little-endian allowed-token bitmask; EOS-only once stopped."""
        if out is None:
            out = self._mask
        if len(out) != len(self._mask):
            raise ValueError(f"mask buffer must contain exactly {len(self._mask)} bytes")
        if self._stopped:
            out[:] = bytes(len(out))
            eos = self.eos_token_id
            offset = 4 * (eos // 32)
            out[offset : offset + 4] = (1 << (eos % 32)).to_bytes(4, "little")
            return out
        if self._index is not None:
            self._index.mask_word_into_state(self._state, out)
        else:
            self._structured.compute_mask_into(out)
        if self.is_finished:
            eos = self.eos_token_id
            if eos >= self.mask_vocab_size:
                raise ValueError("EOS token id lies outside the configured logits vocabulary")
            offset = 4 * (eos // 32)
            word = int.from_bytes(out[offset : offset + 4], "little")
            out[offset : offset + 4] = (word | (1 << (eos % 32))).to_bytes(4, "little")
        return out

    def allowed_ids(self):
        """Debugging convenience: an O(vocab) Python scan of the mask. Use mask_into() in a hot loop."""
        mask = self.mask_into()
        return [
            token_id
            for token_id in range(self.mask_vocab_size)
            if mask[4 * (token_id // 32) + ((token_id % 32) // 8)]
            & (1 << (token_id % 8))
        ]
