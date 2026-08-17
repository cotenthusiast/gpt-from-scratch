import torch
import torch.nn as nn
from torch.nn import functional as F

# Vectorized Multi-head attention 
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.keys = nn.Linear(n_embed,num_heads*head_size, bias=False)
        self.queries = nn.Linear(n_embed,num_heads*head_size, bias=False)
        self.values = nn.Linear(n_embed,num_heads*head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape
        H = self.num_heads
        D = self.head_size

        k = self.keys(x) # B, T, C
        q = self.queries(x) 
        v = self.values(x) 

        k = k.reshape(B, T, H, D).transpose(1, 2) # B, T, C  --->  B, T, H, D  --->  B, H, T, D
        q = q.reshape(B, T, H, D).transpose(1, 2)
        v = v.reshape(B, T, H, D).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # B, H, T, D  @  (B, H, T, D).H  --->  B, H, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, H, T, T
        wei = F.softmax(wei, dim=-1) # B, H, T, T
        wei = self.dropout(wei) # B, H, T, T

        out = wei @ v # B, H, T, T  @  B, H, T, D  --->  B, H, T, D

        return self.dropout(
            self.proj(
                (out.transpose(1, 2)).reshape(B, T, C) # Switching H and T, reshaping the matrix from B, T, H, D to B, T, C, carrying out the learned projection transformation then doing dropout
                )
            )

# Feed-forward network
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Transformer block
class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_embed, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Language model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)],
            nn.LayerNorm(n_embed),
        )

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
