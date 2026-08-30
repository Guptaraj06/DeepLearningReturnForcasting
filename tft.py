import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = None,
        dropout: float = 0.1,
        context_dim: int = None,
    ):
        super().__init__()
        output_dim = output_dim or input_dim

        self.fc1 = nn.Linear(input_dim + (context_dim or 0), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim * 2)
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

        self.skip = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = self.skip(x)
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        eta2 = F.elu(self.fc1(x))
        eta1 = self.drop(self.fc2(eta2))

        g = self.gate(eta1)
        gate, values = g.chunk(2, dim=-1)
        glu_out = torch.sigmoid(gate) * values

        return self.norm(residual + glu_out)


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self, n_vars: int, input_dim: int, hidden_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(n_vars)
            ]
        )
        self.vsn_grn = GatedResidualNetwork(
            n_vars * input_dim, hidden_dim, n_vars, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embeddings: list) -> tuple:
        processed = [grn(emb) for grn, emb in zip(self.var_grns, embeddings)]
        stacked = torch.stack(processed, dim=-2)
        flat = torch.cat(embeddings, dim=-1)
        sel_weights = self.softmax(self.vsn_grn(flat))
        combined = (sel_weights.unsqueeze(-1) * stacked).sum(dim=-2)
        return combined, sel_weights


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_w = self.drop(F.softmax(scores, dim=-1))
        out = (attn_w @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), attn_w


class TFTForecaster(nn.Module):
    QUANTILES = [0.1, 0.5, 0.9]

    def __init__(
        self,
        n_features: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_features)]
        )

        self.vsn = VariableSelectionNetwork(n_features, d_model, d_model, dropout)

        self.lstm_enc = nn.LSTM(
            d_model, d_model, n_lstm_layers, batch_first=True, dropout=dropout
        )

        self.post_lstm_grn = GatedResidualNetwork(d_model, d_model, dropout=dropout)

        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.attn_grn = GatedResidualNetwork(d_model, d_model, dropout=dropout)

        self.ff_grn = GatedResidualNetwork(d_model, d_model * 4, d_model, dropout)

        self.quantile_heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in self.QUANTILES]
        )

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, F = x.shape

        embeddings = [proj(x[..., i : i + 1]) for i, proj in enumerate(self.input_proj)]

        vsn_out, var_weights = self.vsn(embeddings)

        lstm_out, _ = self.lstm_enc(vsn_out)
        lstm_out = self.post_lstm_grn(lstm_out)

        attn_out, attn_weights = self.attn(lstm_out)
        attn_out = self.attn_grn(attn_out + lstm_out)

        final = self.ff_grn(attn_out)

        last = final[:, -1, :]

        quantiles = torch.cat([h(last) for h in self.quantile_heads], dim=-1)

        return quantiles, var_weights, attn_weights
