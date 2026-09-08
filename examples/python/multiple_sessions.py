"""Compile once, bind once, start many independent sessions. Dependencies: pip install maskforge[transformers]"""

import json

from transformers import AutoTokenizer

import maskforge

MODEL_ID = "gpt2"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    vocabulary = maskforge.tokenizers.from_transformers(tokenizer)

    compiler = maskforge.Compiler()
    program = compiler.compile(json.dumps({"type": "boolean"}))
    bound = program.bind(vocabulary)  # one compile, one bind, shared by every session below

    sessions = [bound.start_session() for _ in range(3)]

    # Advance them out of step; each session's mask reflects only its own committed prefix.
    for token_id in tokenizer.encode("true", add_special_tokens=False):
        sessions[0].advance(token_id)
    for token_id in tokenizer.encode("false", add_special_tokens=False):
        sessions[1].advance(token_id)

    for i, session in enumerate(sessions):
        print(f"session {i}: is_accepting={session.is_accepting}")


if __name__ == "__main__":
    main()
