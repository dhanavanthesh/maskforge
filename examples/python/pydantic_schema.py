"""Pass a Pydantic type directly to Compiler.compile and parse the result back into it. Dependencies: pip install maskforge[transformers] pydantic"""

import pydantic
from transformers import AutoTokenizer

import maskforge

MODEL_ID = "gpt2"


class QuestionAnswer(pydantic.BaseModel):
    question: str
    answer: str


class Profile(pydantic.BaseModel):
    bio: str
    interests: list[str]
    qna: QuestionAnswer


def bit_set(mask, token_id):
    return mask[4 * (token_id // 32) + (token_id % 32) // 8] & (1 << (token_id % 8)) != 0


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    vocabulary = maskforge.tokenizers.from_transformers(tokenizer)

    compiler = maskforge.Compiler()
    program = compiler.compile(Profile)  # OutputSpec derives the schema from the type itself
    session = program.bind(vocabulary).start_session()

    # A real LLM samples one legal token at a time; here we replay a known-valid instance instead.
    instance = Profile(
        bio="constrained decoding enthusiast",
        interests=["compilers", "type systems"],
        qna=QuestionAnswer(question="favorite data structure?", answer="a trie"),
    )
    payload = instance.model_dump_json()

    token_ids = tokenizer.encode(payload, add_special_tokens=False)
    for token_id in token_ids:
        mask = bytearray(4 * session.mask_word_count)
        session.write_mask(mask)
        assert bit_set(mask, token_id), f"token {token_id} unexpectedly rejected mid-instance"
        session.advance(token_id)
    assert session.is_accepting

    parsed = program.parse(payload)
    assert parsed == instance
    print(f"replayed {len(token_ids)} tokens; parsed back into {type(parsed).__name__}: {parsed!r}")


if __name__ == "__main__":
    main()
