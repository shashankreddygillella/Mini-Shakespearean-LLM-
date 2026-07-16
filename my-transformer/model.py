"""
model.py — a small GPT-style transformer, PyTorch version.

This mirrors the exact same algorithmic skeleton as Karpathy's microgpt.py:
    embeddings (token + position) -> [attention -> MLP] x n_layer -> LM head

The differences from the blueprint (call these out in your report):
  - Uses torch.nn.Module + autograd instead of the hand-rolled Value/backward().
  - Vectorized tensor ops (matmul) instead of scalar-by-scalar Python loops.
  - Byte-level tokenizer (vocab_size=256) instead of a character dictionary
    built from the dataset.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention block. Maps to the attention loop in gpt()
    in microgpt.py: build q,k,v -> split heads -> scaled dot-product ->
    softmax -> weighted sum of v -> concat heads -> output projection."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        # causal mask so a token can only attend to itself and the past
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding channels
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B, n_head, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    """Maps to the mlp_fc1 -> relu -> mlp_fc2 block in microgpt.py, plus the
    residual add that happens right after it."""

    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """One transformer layer: attention + MLP, each wrapped in a residual
    connection with a norm beforehand (RMSNorm, matching the blueprint's
    choice over LayerNorm)."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.norm2 = nn.RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """Full model: token embedding (wte) + position embedding (wpe) -> n_layer
    Blocks -> final norm -> lm_head. Same order as microgpt.py's gpt()."""

    def __init__(self, vocab_size=256, n_layer=2, n_head=4, n_embd=64, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.norm_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.config = dict(
            vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
            n_embd=n_embd, block_size=block_size,
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)  # embeddings
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)  # LM head

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """Autoregressive sampling loop — mirrors the inference loop at the
        bottom of microgpt.py (softmax(logits/temperature) -> sample -> feed
        back in), just batched via tensors instead of a Python for-loop over
        scalars."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())
