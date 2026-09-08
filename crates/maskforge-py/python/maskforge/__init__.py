"""MaskForge: fast, bounded JSON Schema constrained decoding.

```python
import maskforge
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").cpu()
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

constrained = maskforge.from_transformers(model, tokenizer)
generate = maskforge.Generator(constrained, Profile)
profile = generate("Create a profile", max_new_tokens=128)
```

Raw IR, index handles and the process-wide cache controls live in `maskforge.low_level`.
"""

import importlib.metadata

from ._native import MaskforgeError

try:
    __version__ = importlib.metadata.version("maskforge")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import low_level, tokenizers
from .compiler import BoundSchema, Compiler, SchemaProgram, Session
from .constraints import OutputSpec
from .generator import GenerationError, Generator
from .models import TransformersModel, from_transformers
from .vocabulary import Vocabulary

# Named access into MaskforgeError.args, the same 7-tuple every raise site fills in order.
for _index, _field in enumerate(
    ("code", "stage", "pointer", "keyword", "observed", "message", "limit")
):
    setattr(
        MaskforgeError,
        _field,
        property(
            lambda self, _i=_index: self.args[_i] if len(self.args) > _i else None  # type: ignore[misc]
        ),
    )
del _index, _field

__all__ = [
    "__version__",
    "BoundSchema",
    "Compiler",
    "GenerationError",
    "Generator",
    "MaskforgeError",
    "OutputSpec",
    "SchemaProgram",
    "Session",
    "TransformersModel",
    "Vocabulary",
    "from_transformers",
    "low_level",
    "tokenizers",
]

# Names that used to live at the package root. They still resolve, with a warning, so existing
# code keeps working for one release; `maskforge.low_level` is the supported home.
_MOVED_TO_LOW_LEVEL = frozenset(low_level.__all__)


def __getattr__(name: str) -> object:
    if name in _MOVED_TO_LOW_LEVEL:
        import warnings

        warnings.warn(
            f"maskforge.{name} has moved to maskforge.low_level.{name} and will stop "
            "resolving from the package root in a future release",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(low_level, name)
    # corpus_* exists only in a test-utils build, never in a released wheel.
    if name.startswith("corpus_"):
        from . import _native

        try:
            return getattr(_native, name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
