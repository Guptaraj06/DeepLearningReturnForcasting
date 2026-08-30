import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class AlphaEvaluator:
    def compute_ic_series(
        self, signals: pd.DataFrame, returns: pd.DataFrame, horizon: int = 5
    ) -> pd.Series:
        fwd = returns.shift(-horizon)
        ic_series = pd.Series(index=signals.index, dtype=float)

        for date in signals.index:
            sig_row = signals.loc[date].dropna()
            ret_row = fwd.loc[date].reindex(sig_row.index).dropna()
            common = sig_row.index.intersection(ret_row.index)
            if len(common) < 10:
                continue
            ic, _ = spearmanr(sig_row[common], ret_row[common])
            ic_series[date] = ic

        return ic_series.dropna()

    def ic_summary(self, ic_series: pd.Series) -> dict:
        return {
            "mean_ic": round(ic_series.mean(), 4),
            "std_ic": round(ic_series.std(), 4),
            "icir": round(ic_series.mean() / (ic_series.std() + 1e-8), 3),
            "pct_pos": round((ic_series > 0).mean() * 100, 1),
            "t_stat": round(
                ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series))), 3
            ),
        }

    def quintile_backtest(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        horizon: int = 5,
        tc_bps: float = 10,
    ) -> pd.DataFrame:
        fwd = returns.shift(-horizon)
        q_returns = {q: [] for q in range(1, 6)}

        prev_weights = {q: None for q in range(1, 6)}

        for date in signals.index[:-horizon]:
            sig_row = signals.loc[date].dropna()
            ret_row = fwd.loc[date].reindex(sig_row.index).dropna()
            common = sig_row.index.intersection(ret_row.index)
            if len(common) < 10:
                continue

            ranks = sig_row[common].rank(pct=True)

            for q in range(1, 6):
                lower = (q - 1) / 5
                upper = q / 5
                mask = (ranks >= lower) & (ranks < upper)
                stocks = ranks[mask].index
                if len(stocks) == 0:
                    continue

                w_new = 1 / len(stocks)
                q_ret = float(ret_row[stocks].mean())

                if prev_weights[q] is not None:
                    turnover = abs(len(stocks) - len(prev_weights[q])) / max(
                        len(stocks), len(prev_weights[q])
                    )
                    q_ret -= turnover * tc_bps / 10_000

                q_returns[q].append({"date": date, "ret": q_ret})
                prev_weights[q] = stocks

        results = {}
        for q in range(1, 6):
            s = pd.DataFrame(q_returns[q]).set_index("date")["ret"]
            ann_ret = ((1 + s).prod()) ** (252 / len(s)) - 1
            sharpe = s.mean() / s.std() * np.sqrt(252)
            results[f"Q{q}"] = {"ann_return": ann_ret, "sharpe": sharpe}

        ls_ret = (
            pd.DataFrame(q_returns[5]).set_index("date")["ret"]
            - pd.DataFrame(q_returns[1]).set_index("date")["ret"]
        ).dropna()
        ann_ret = ((1 + ls_ret).prod()) ** (252 / len(ls_ret)) - 1
        sharpe = ls_ret.mean() / ls_ret.std() * np.sqrt(252)
        results["L/S Q5-Q1"] = {"ann_return": ann_ret, "sharpe": sharpe}

        return pd.DataFrame(results).T
