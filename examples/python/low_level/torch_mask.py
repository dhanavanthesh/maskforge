"""Apply a MaskForge mask to a PyTorch logits tensor. Dependencies: pip install maskforge torch"""

import json

import torch

import maskforge
from maskforge.tensor_adapters import get_adapter

EOS_TOKEN_ID = 256


def byte_vocabulary():
    return [(bytes([b]), [b]) for b in range(256)]


def main():
    vocabulary = maskforge.low_level.PyVocabulary(EOS_TOKEN_ID, byte_vocabulary())
    session = maskforge.low_level.ConstraintSession.from_json_schema(vocabulary, json.dumps({"type": "boolean"}))

    logits = torch.randn(session.mask_vocab_size)
    mask_words = torch.frombuffer(bytearray(session.mask_into()), dtype=torch.int32)  # packed bytes -> u32 words

    apply_mask = get_adapter("torch")
    masked = apply_mask(logits, mask_words, session.mask_vocab_size)

    finite = torch.isfinite(masked)
    print(f"{finite.sum().item()} of {masked.numel()} logits kept finite (schema-legal next bytes)")
    assert set(torch.nonzero(finite).flatten().tolist()) == set(session.allowed_ids())


if __name__ == "__main__":
    main()
