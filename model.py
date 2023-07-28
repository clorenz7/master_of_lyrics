
import math

import torch
from torch import nn

class AttentionHead(nn.Module):

    def __init__(self, n_embed_in, n_embed_out, max_tokens=2048, dropout=0.2):
        super().__init__()
        self.n_embed = n_embed_in

        self.query_xform = nn.Linear(n_embed_in, n_embed_out, bias=False)
        self.key_xform = nn.Linear(n_embed_in, n_embed_out, bias=False)
        self.value_xform = nn.Linear(n_embed_in, n_embed_out, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('no_look_ahead', torch.triu(torch.full((max_tokens, max_tokens), float('-inf')), diagonal=1))
        # Andrej's way:
        # self.register_buffer('tril', torch.tril(torch.ones(max_tokens, max_tokens)))

    def forward(self, x):
        """
        expected shape:
            batch x tokens x embed
        """
        Q = self.query_xform(x)
        K = self.key_xform(x).transpose(-2, -1)
        V = self.value_xform(x)

        n_tokens = x.shape[1]

        NLA = self.no_look_ahead[:n_tokens, :n_tokens]

        dk = 1.0 / math.sqrt(Q.shape[-1])

        W = (Q @ K) * dk
        attention = torch.softmax(W + NLA, dim=2)

        # Andrej's way:
        # W2 = W.masked_fill(self.tril[:n_tokens, :n_tokens] == 0, float('-inf'))
        # A2 = torch.softmax(W2, dim=2)

        attention = self.dropout(attention)

        output = attention @ V

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embed, n_heads=6, n_inner=None, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_embed = n_embed

        n_inner = n_inner or (4 * n_embed)
        head_size = n_embed // n_heads

        self.heads = nn.ModuleList(
            [AttentionHead(n_embed, head_size, dropout=dropout) for _ in range(n_heads)]
        )

        self.out_proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        attentions = [head(x) for head in self.heads]
        y = torch.concat(attentions, dim=2)
        y = self.out_proj(y)

        return y

class FeedForward(nn.Module):

    def __init__(self, n_embed, n_inner=None, dropout=0.0):
        super().__init__()
        n_inner = n_inner or (4 * n_embed)
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_inner),
            nn.ReLU(),
            # nn.GELU(),
            nn.Linear(n_inner, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class CorBlock(nn.Module):
    def __init__(self, n_embed, n_heads=6, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_embed = n_embed

        self.mha = MultiHeadAttention(n_embed, n_heads, dropout=dropout)
        self.ln_1 = nn.LayerNorm(n_embed)
        self.ln_2 = nn.LayerNorm(n_embed)

        self.ffw = FeedForward(n_embed, dropout=dropout)

    def forward(self, x):
        # Just added this residual connection and out projection
        x = x + self.mha(self.ln_1(x))
        # Add residual
        x = x + self.ffw(self.ln_2(x))

        return x

class TransCORmer(nn.Module):

    def __init__(self, n_vocab, n_embed, n_blocks=4, n_positions=2048,
                 n_heads=4, dropout=0.2):
        super().__init__()
        # Create position and word embeddings
        self.token_embed = nn.Embedding(n_vocab, n_embed)
        self.pos_embed = nn.Embedding(n_positions, n_embed)

        self.attention_blocks = nn.Sequential(
            *[CorBlock(n_embed, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)]
        )
        # self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed, n_vocab)

    def forward(self, x):
        pos = torch.arange(0, x.shape[1], device=x.device)
        e = self.token_embed(x) + self.pos_embed(pos)

        y = self.attention_blocks(e)

        # y = self.dropout(y)
        y = self.ln(y)

        y = self.lm_head(y)
        # I think the loss function will do this.
        # y = torch.softmax(y, dim=2)

        return y