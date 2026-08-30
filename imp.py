import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

if __name__ == "__main__":
    TICKERS = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "JPM",
        "JNJ",
        "XOM",
        "KO",
        "PEP",
        "WMT",
        "HD",
        "CAT",
        "GS",
        "BAC",
        "COP",
        "CVX",
        "LLY",
        "UNH",
        "TMO",
        "ABBV",
        "AVGO",
        "AMD",
        "QCOM",
        "TXN",
    ]

    raw = yf.download(TICKERS, start="2013-01-01", end="2024-12-31", auto_adjust=True)
    prices = raw["Close"].dropna(axis=1, thresh=int(0.95 * len(raw))).ffill()
    volumes = raw["Volume"].reindex(columns=prices.columns)

    engine = AlphaFeatureEngine(prices, volumes)
    feat_dict, labels = engine.build_all(target_horizon=5)

    auditor = FeatureLeakageAuditor()
    for name, feat_df in feat_dict.items():
        for ticker in prices.columns[:3]:  # spot-check 3 stocks
            auditor.check_future_info(
                feat_df[[ticker]].dropna(), labels[ticker].dropna()
            )

    feature_list = list(feat_dict.keys())
    N_FEAT = len(feature_list)

    wf_cv = PurgedWalkForwardCV(n_splits=5, purge_gap=21, embargo_pct=0.01)
    all_preds = []
    all_labels = []

    for ticker in prices.columns:
        feat_matrix = pd.concat(
            [feat_dict[f][ticker].rename(f) for f in feature_list], axis=1
        ).dropna()
        label_series = labels[ticker].reindex(feat_matrix.index).dropna()
        feat_matrix = feat_matrix.reindex(label_series.index)

        X = feat_matrix.values.astype(np.float32)
        y = label_series.values.astype(np.float32)

        for train_idx, test_idx in wf_cv.split(feat_matrix):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0) + 1e-8
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

            train_ds = FinancialSequenceDataset(X_train, y_train, seq_len=60)
            train_ld = DataLoader(train_ds, batch_size=128, shuffle=False)

            model = LSTMForecaster(n_features=N_FEAT)
            trainer = FinancialTrainer(model, lr=1e-3)
            trainer.fit(train_ld, train_ld, n_epochs=50)

            test_ds = FinancialSequenceDataset(X_test, y_test, seq_len=60)
            test_ld = DataLoader(test_ds, batch_size=256, shuffle=False)
            model.eval()
            with torch.no_grad():
                fold_preds = []
                for xb, _ in test_ld:
                    p, _ = model(xb)
                    fold_preds.extend(p.numpy())
            all_preds.extend(fold_preds)
            all_labels.extend(y_test[: len(fold_preds)])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    ic_oos, _ = spearmanr(all_preds, all_labels)
    print(f"\nOOS Rank IC (all folds, all stocks): {ic_oos:.4f}")
