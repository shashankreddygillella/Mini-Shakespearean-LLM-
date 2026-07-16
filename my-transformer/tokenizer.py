"""
tokenizer.py — fixed byte-level tokenizer.

Locks vocab_size to exactly 256 (every possible byte value). This replaces
the microgpt.py approach of building a character dictionary from the
dataset (uchars = sorted(set(''.join(docs)))). No dataset-dependent
vocabulary, no OOV handling needed — every UTF-8 file encodes to bytes 0-255.
"""


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    # errors="replace" guards against a generated sequence that ends mid
    # multi-byte UTF-8 character (common with a tiny undertrained model).
    return bytes(tokens).decode("utf-8", errors="replace")


VOCAB_SIZE = 256
