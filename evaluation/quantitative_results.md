# Quantitative Evaluation Results

| Model | Final Cross-Entropy Loss | Perplexity |
|---|---:|---:|
| Model A | 2.55 | 12.81 |
| Model B | 2.30 | 9.97 |

## Interpretation

Model B performs better than Model A because its final validation loss and perplexity are lower. Lower perplexity means the model is less surprised by the validation text and is predicting the next token more effectively.

These results show the expected scaling pattern for this small project: increasing model capacity can improve performance, but the model is still much smaller than production LLMs.
