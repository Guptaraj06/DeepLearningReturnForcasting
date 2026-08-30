from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class PurgedWalkForwardCV:
    def __init__(
        self, n_splits: int = 5, purge_gap: int = 20, embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.purge = purge_gap
        self.embargo = embargo_pct

    def split(self, X: pd.DataFrame, y=None, groups=None):
        T = len(X)
        embargo_days = int(T * self.embargo)
        fold_size = T // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = (i + 1) * fold_size
            test_end = test_start + fold_size
            test_idx = np.arange(test_start, min(test_end, T))

            train_end = test_start - self.purge
            if train_end <= 0:
                continue
            train_idx = np.arange(0, train_end)

            if embargo_days > 0:
                embargo_start = test_end
                embargo_end = min(test_end + embargo_days, T)
                embargoed = np.arange(embargo_start, embargo_end)

            yield train_idx, test_idx

    def combinatorial_split(self, X: pd.DataFrame, k_test_groups: int = 2):
        T = len(X)
        fold_size = T // self.n_splits
        folds = [
            np.arange(i * fold_size, min((i + 1) * fold_size, T))
            for i in range(self.n_splits)
        ]

        for test_combo in combinations(range(self.n_splits), k_test_groups):
            test_folds = [folds[i] for i in test_combo]
            test_idx = np.concatenate(test_folds)

            all_train = []
            for j in range(self.n_splits):
                if j in test_combo:
                    continue
                fold = folds[j]
                keep = [
                    idx
                    for idx in fold
                    if all(abs(idx - t) > self.purge for t in test_idx)
                ]
                all_train.extend(keep)

            train_idx = np.array(sorted(set(all_train)))
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


class FeatureLeakageAuditor:

    def check_future_info(
        self, features: pd.DataFrame, returns: pd.Series, max_lag: int = 5
    ) -> pd.DataFrame:
        results = []
        for col in features.columns:
            feat = features[col].dropna()
            for lag in range(-max_lag, max_lag + 1):
                shifted_ret = returns.shift(-lag)  # negative lag = future
                idx = feat.index.intersection(shifted_ret.dropna().index)
                corr = feat.loc[idx].corr(shifted_ret.loc[idx])
                results.append({"feature": col, "lag": lag, "corr": corr})

        df = pd.DataFrame(results)
        suspicious = df[(df["lag"] < 0) & (df["corr"].abs() > 0.1)]
        if len(suspicious) > 0:
            print(f"LEAKAGE DETECTED: {suspicious['feature'].unique().tolist()}")
        return df.pivot(index="lag", columns="feature", values="corr")
