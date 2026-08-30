import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast


def quantile_loss(
    predictions: torch.Tensor, targets: torch.Tensor, quantiles: list = [0.1, 0.5, 0.9]
) -> torch.Tensor:
    losses = []
    for i, q in enumerate(quantiles):
        pred_q = predictions[:, i]
        err = targets - pred_q
        loss = torch.max(q * err, (q - 1) * err)
        losses.append(loss.mean())
    return sum(losses)


def ic_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    T = 0.1
    scores = predictions - predictions.mean()
    ranks_pred = torch.softmax(scores / T, dim=0)
    ranks_true = torch.softmax(targets / T, dim=0)
    loss = -(ranks_true * torch.log(ranks_pred + 1e-8)).sum()
    return loss


class FinancialTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        model_type: str = "lstm",
    ):
        self.model = model
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.patience = patience
        self.model_type = model_type
        self.scaler = GradScaler()

    def train_epoch(self, loader: DataLoader, loss_fn: str = "mse") -> float:
        self.model.train()
        total_loss = 0

        for x_batch, y_batch in loader:
            self.optimizer.zero_grad()

            with autocast():
                if self.model_type == "tft":
                    preds, _, _ = self.model(x_batch)
                    loss = quantile_loss(preds, y_batch)
                else:
                    preds, _ = self.model(x_batch)
                    if loss_fn == "mse":
                        loss = F.mse_loss(preds, y_batch)
                    else:
                        loss = ic_loss(preds, y_batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(loader)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int = 200
    ) -> dict:
        best_val_ic = -np.inf
        best_state = None
        no_improve = 0
        history = {"train_loss": [], "val_ic": [], "lr": []}

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_ic = self.eval_ic(val_loader)

            history["train_loss"].append(train_loss)
            history["val_ic"].append(val_ic)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d} | loss={train_loss:.5f} | val_IC={val_ic:.4f} | best={best_val_ic:.4f}"
                )

            if no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(best_state)
        return history

    def eval_ic(self, loader: DataLoader) -> float:
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in loader:
                if self.model_type == "tft":
                    preds, _, _ = self.model(x)
                    preds = preds[:, 1]
                else:
                    preds, _ = self.model(x)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        from scipy.stats import spearmanr

        ic, _ = spearmanr(all_preds, all_targets)
        return float(ic) if not np.isnan(ic) else 0.0
