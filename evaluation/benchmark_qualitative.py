"""Qualitative benchmark prompts for Mini-Shakespearean LLM models."""

PROMPTS = [
    "To be, or not to ",
    "O Romeo, Romeo, ",
    "Now is the winter ",
    "My lord, I pray you ",
    "The king hath ",
]

MAX_NEW_TOKENS = 150

print("Use these prompts to generate exactly 150 tokens from Model A, Model B, and Gemini Flash:")
for prompt in PROMPTS:
    print("-", prompt)
