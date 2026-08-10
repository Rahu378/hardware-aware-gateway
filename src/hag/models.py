"""Model loading that survives the transformers 4.x / 5.x split.

`from_pretrained` took `torch_dtype` through transformers 4 and renamed it to
`dtype` in 5. This matters more than a rename usually would, because Colab
ships a 4.x preinstalled and a `transformers>=4.44` dependency is already
satisfied by it, so pip quietly declines to upgrade and the newer spelling
fails at runtime, on the machine you were counting on.
"""

from __future__ import annotations

from typing import Any


def load_model_and_tokenizer(name: str, dtype: Any, device: str):
    """Load a causal LM and its tokenizer onto `device`, in `dtype`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)

    return model.to(device).eval(), tokenizer


def vocab_size(tokenizer, model) -> int:
    """Vocabulary size, preferring the model's embedding over the tokenizer.

    Tokenizers sometimes report a smaller vocabulary than the embedding matrix
    (added special tokens land above `vocab_size`), and sometimes a larger one
    (the embedding is padded down). Sampling ids above the embedding row count
    is an index error inside the forward pass, so take the smaller of the two.
    """
    sizes = [s for s in (getattr(tokenizer, "vocab_size", None),
                         getattr(getattr(model, "config", None), "vocab_size", None)) if s]
    return min(sizes) if sizes else 32000
