"""Generate a finite, validated payment instruction with a cached CPU model."""

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import maskforge  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://example.test/payment-instruction",
    "type": "object",
    "additionalProperties": False,
    "required": ["rail", "amountMinor", "currency", "iban", "approvals", "allocation"],
    "properties": {
        "rail": {"const": "wire"},
        "amountMinor": {"const": "500"},
        "currency": {"enum": ["USD", "EUR"]},
        "iban": {"type": "string", "pattern": "^[A-Z]{2}[0-9]{4}$"},
        "approvals": {
            "type": "array",
            "items": {"enum": ["treasury", "ops", "risk"]},
            "contains": {"const": "treasury"},
            "uniqueItems": True,
            "minItems": 1,
            "maxItems": 3,
        },
        "allocation": {"$ref": "#/$defs/allocation"},
    },
    "$defs": {
        "allocation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["pct"],
            "properties": {
                "pct": {"const": "60"},
            },
        }
    },
}

PROMPT = (
    "Return a JSON payment: wire 500 minor units in EUR to IBAN DE1234, "
    "approved by treasury, with 60 percent allocation."
)


def load_model():
    """Load the requested cached model, otherwise try the two documented small models."""
    requested = os.environ.get("MASKFORGE_MODEL_ID")
    model_ids = (requested,) if requested else (
        "Qwen/Qwen2.5-0.5B",
        "openai-community/gpt2",
    )
    for model_id in model_ids:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True).cpu()
            return model_id, model, tokenizer
        except Exception:
            continue
    raise SystemExit("no supported model is present in the local Hugging Face cache")


def main() -> None:
    model_id, model, tokenizer = load_model()
    generate: maskforge.Generator[dict] = maskforge.Generator(
        maskforge.from_transformers(model, tokenizer), SCHEMA
    )
    payment = generate(
        PROMPT,
        max_new_tokens=1200,
        do_sample=False,
        max_json_whitespace=1,
    )

    assert payment["rail"] == "wire"
    assert payment["amountMinor"] == "500"
    assert payment["currency"] in ("USD", "EUR")
    assert "treasury" in payment["approvals"]
    assert len(payment["approvals"]) == len(set(payment["approvals"]))
    assert payment["allocation"]["pct"] == "60"

    print(f"model: {model_id}")
    print(json.dumps(payment, indent=2))


if __name__ == "__main__":
    main()
