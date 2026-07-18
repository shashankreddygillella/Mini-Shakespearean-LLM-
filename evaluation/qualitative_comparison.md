# Qualitative Comparison

| Prompt | Model A Output Quality | Model B Output Quality | Gemini Flash Output Quality |
|---|---|---|---|
| To be, or not to | Fragmented text, some Shakespeare-like rhythm | More stable wording, fewer broken patterns | Fluent and coherent |
| O Romeo, Romeo, | Repeats some short patterns | Better style imitation | Strong Shakespearean style |
| Now is the winter | Understands some character-level patterns | More readable than Model A | Best grammar and meaning |
| My lord, I pray you | Short phrases but weak structure | Better sentence-like fragments | Strongest completion |
| The king hath | Some old-style words appear | More consistent old-style phrasing | Most polished output |

## Interpretation

The local mini-models are expected to produce imperfect text because they are very small and trained for a limited number of steps. Model B is generally better than Model A because it has more capacity. Gemini Flash performs best because it is a large production model trained on far more data and compute.

The purpose of this comparison is not to prove that the local model is perfect. The purpose is to understand how model size, context length, and training affect output quality.
