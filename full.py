import numpy as np
import pandas as pd
import shap


class FinancialSHAPExplainer:
    def __init__(self, model: nn.Module, feature_names: list, model_type: str = "lstm"):
        self.model = model.eval()
        self.feat_names = feature_names
        self.mtype = model_type

    def _model_wrapper(self, x: np.ndarray) -> np.ndarray:
        """Wrap model to accept numpy and return numpy for SHAP."""
        x_t = torch.FloatTensor(x)
        with torch.no_grad():
            if self.mtype == "tft":
                out, _, _ = self.model(x_t)
                return out[:, 1].numpy()
            else:
                out, _ = self.model(x_t)
                return out.numpy()

    def compute_gradient_shap(
        self, X_background: np.ndarray, X_explain: np.ndarray, n_samples: int = 200
    ) -> np.ndarray:
        background = torch.FloatTensor(X_background[:n_samples])
        explainer = shap.GradientExplainer(self.model, background)
        shap_vals = explainer.shap_values(torch.FloatTensor(X_explain), nsamples=50)
        return np.array(shap_vals)

    def feature_importance_summary(self, shap_vals: np.ndarray) -> pd.DataFrame:
        mean_abs = np.abs(shap_vals).mean(axis=(0, 1))
        df = pd.DataFrame(
            {
                "feature": self.feat_names,
                "mean_abs_shap": mean_abs,
                "rank": np.argsort(np.argsort(-mean_abs)) + 1,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        return df.reset_index(drop=True)

    def temporal_importance(self, shap_vals: np.ndarray) -> pd.DataFrame:
        seq_len = shap_vals.shape[1]
        lag_importance = np.abs(shap_vals).mean(axis=(0, 2))
        return pd.DataFrame(
            {"lag_days_ago": np.arange(seq_len, 0, -1), "importance": lag_importance}
        )
