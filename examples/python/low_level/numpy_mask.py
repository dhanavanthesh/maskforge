"""Apply a MaskForge mask to a NumPy logits array. Dependencies: pip install maskforge numpy"""

import json

import numpy as np

import maskforge
from maskforge.tensor_adapters import get_adapter

EOS_TOKEN_ID = 256


def byte_vocabulary():
    return [(bytes([b]), [b]) for b in range(256)]


def main():
    vocabulary = maskforge.low_level.PyVocabulary(EOS_TOKEN_ID, byte_vocabulary())
    session = maskforge.low_level.ConstraintSession.from_json_schema(vocabulary, json.dumps({"type": "boolean"}))

    logits = np.random.default_rng(0).normal(size=session.mask_vocab_size).astype(np.float32)
    mask_words = np.frombuffer(session.mask_into(), dtype="<u4")  # packed bytes -> u32 words

    apply_mask = get_adapter("numpy")
    masked = apply_mask(logits, mask_words, session.mask_vocab_size)

    finite = np.isfinite(masked)
    print(f"{finite.sum()} of {masked.size} logits kept finite (schema-legal next bytes)")
    assert set(np.flatnonzero(finite).tolist()) == set(session.allowed_ids())


if __name__ == "__main__":
    main()
