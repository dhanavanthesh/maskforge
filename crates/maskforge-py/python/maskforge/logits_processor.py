"""Hugging Face-compatible constrained-generation logits processor."""

from .generation import ConstraintSession


class MaskforgeLogitsProcessor:
    """Apply a :class:`ConstraintSession` to one Hugging Face decode sequence.

    The first call records the prompt length. Later calls commit newly generated
    ids before masking the next-token scores. Decoder rewinds are handled by
    resetting and replaying the changed generated prefix. Beam/batched generation
    needs one independent session per row and is deliberately rejected for now.
    """

    def __init__(self, session, prompt_length=None):
        if not isinstance(session, ConstraintSession):
            raise TypeError("session must be a maskforge.ConstraintSession")
        self._session = session
        self._prompt_length = prompt_length
        self._seen = []

    @property
    def session(self):
        return self._session

    def reset(self, prompt_length=None):
        self._session.reset()
        self._prompt_length = prompt_length
        self._seen.clear()

    def __call__(self, input_ids, scores):
        import torch

        if input_ids.ndim == 1:
            row_ids = input_ids
        elif input_ids.ndim == 2 and input_ids.shape[0] == 1:
            row_ids = input_ids[0]
        else:
            raise ValueError("MaskforgeLogitsProcessor currently supports exactly one sequence")

        current_length = int(row_ids.shape[-1])
        if self._prompt_length is None:
            self._prompt_length = current_length
        if current_length < self._prompt_length:
            raise ValueError("input_ids is shorter than the configured prompt length")

        generated_length = current_length - self._prompt_length
        if generated_length == len(self._seen) + 1:
            # Append-only path: read only the one new token, not the whole sequence.
            token_id = int(row_ids[-1].item())
            self._session.commit_token(token_id)
            self._seen.append(token_id)
        elif generated_length != len(self._seen):
            # Rewind, replacement, or a multi-token jump: replay the changed suffix.
            generated = row_ids[self._prompt_length :].tolist()
            self._session.replay(generated)
            self._seen = generated

        if scores.ndim == 1:
            row = scores.unsqueeze(0)
        elif scores.ndim == 2 and scores.shape[0] == 1:
            row = scores
        else:
            raise ValueError("scores must contain exactly one vocabulary row")
        if row.shape[1] != self._session.mask_vocab_size:
            raise ValueError(
                f"scores vocabulary width {row.shape[1]} does not match "
                f"MaskForge width {self._session.mask_vocab_size}"
            )

        packed = self._session.mask_into()
        mask = torch.frombuffer(packed, dtype=torch.int32).reshape(1, -1)
        if mask.device != row.device:
            mask = mask.to(row.device)
        from .tensor_adapters.torch import apply_token_bitmask_inplace

        apply_token_bitmask_inplace(row, mask)
        return scores
