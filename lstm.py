import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class FinancialSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 60):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.L = seq_len

    def __len__(self):
        return len(self.y) - self.L

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.L]
        label = self.y[idx + self.L]
        return x_seq, label


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> tuple:
        scores = self.attn(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)
        return context, weights


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int = 20,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attn = use_attention

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )

        if use_attention:
            self.attention = TemporalAttention(hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                p.data.fill_(0)
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1)

    def forward(self, x: torch.Tensor) -> tuple:
        lstm_out, (h_n, _) = self.lstm(x)

        if self.use_attn:
            context, attn_w = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]
            attn_w = None

        out = self.out(self.drop(self.norm(context)))
        return out.squeeze(-1), attn_w
