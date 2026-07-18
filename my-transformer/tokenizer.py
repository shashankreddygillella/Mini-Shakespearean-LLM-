"""Byte-level tokenizer for Tiny Shakespeare."""

VOCAB_SIZE = 256

def encode(text: str) -> list[int]:
    """Convert text into UTF-8 byte token IDs."""
    return list(text.encode("utf-8"))

def decode(tokens: list[int]) -> str:
    """Convert UTF-8 byte token IDs back into readable text."""
    return bytes(tokens).decode("utf-8", errors="replace")
