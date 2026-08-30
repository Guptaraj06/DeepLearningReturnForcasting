import numpy as np
import pandas as pd
from scipy.stats import rankdata


class AlphaFeatureEngine:
    def __init__(self, prices: pd.DataFrame, volumes: pd.DataFrame = None):
        self.px = prices
        self.vol = volumes
        self.ret = np.log(prices / prices.shift(1))

    def momentum(self, windows=(5, 10, 21, 63, 126, 252)) -> dict:
        feats = {}
        for w in windows:
            feats[f"mom_{w}d"] = self.ret.rolling(w).sum()
        feats["mom_12_1"] = self.ret.rolling(252).sum() - self.ret.rolling(21).sum()
        return feats

    def reversal(self) -> dict:
        return {
            "rev_1w": -self.ret.rolling(5).sum(),
            "rev_1m": -self.ret.rolling(21).sum(),
        }

    def volatility(self, windows=(5, 21, 63)) -> dict:
        feats = {}
        for w in windows:
            feats[f"rvol_{w}d"] = self.ret.rolling(w).std() * np.sqrt(252)
        feats["vol_ratio"] = feats["rvol_5d"] / (feats["rvol_63d"] + 1e-8)
        return feats

    def microstructure(self) -> dict:
        if self.vol is None:
            return {}
        dollar_vol = self.px * self.vol
        amihud = (self.ret.abs() / (dollar_vol + 1)).rolling(21).mean()

        vol_mom = self.vol / (self.vol.rolling(21).mean() + 1e-8)

        turnover = self.vol / (self.vol.rolling(252).mean() + 1e-8)

        return {"amihud": amihud, "vol_mom": vol_mom, "turnover": turnover}

    def technical(self) -> dict:
        delta = self.px.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + up / (down + 1e-8)))

        sma_20 = self.px.rolling(20).mean()
        std_20 = self.px.rolling(20).std()
        bb_z = (self.px - sma_20) / (std_20 + 1e-8)

        high_52w = self.px.rolling(252).max()
        dist_52w = (self.px - high_52w) / (high_52w + 1e-8)

        return {"rsi_14": rsi, "bb_zscore": bb_z, "dist_52w_high": dist_52w}

    def cross_section_standardize(
        self, feat_df: pd.DataFrame, method: str = "rank"
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=feat_df.index, columns=feat_df.columns, dtype=float)
        for date in feat_df.index:
            row = feat_df.loc[date].dropna()
            if len(row) < 3:
                continue
            if method == "rank":
                # Rank to uniform [0,1] then standardize
                ranks = rankdata(row, method="average") / (len(row) + 1)
                result.loc[date, row.index] = ranks
            else:
                result.loc[date, row.index] = (row - row.mean()) / (row.std() + 1e-8)
        return result

    def build_all(self, target_horizon: int = 5) -> tuple:
        all_feats = {}
        all_feats.update(self.momentum())
        all_feats.update(self.reversal())
        all_feats.update(self.volatility())
        all_feats.update(self.microstructure())
        all_feats.update(self.technical())

        feat_panels = {}
        for name, df in all_feats.items():
            feat_panels[name] = self.cross_section_standardize(df, method="rank")

        fwd_ret = self.ret.rolling(target_horizon).sum().shift(-target_horizon)

        return feat_panels, fwd_ret
