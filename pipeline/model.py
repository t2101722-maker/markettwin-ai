"""Stress-aware multi-factor model.

For every asset we learn how its weekly return responds to a small set of
market factors (the market, its own sector, rates and gold). Weeks when the
market was falling get extra weight, because the relationship between assets
changes in a crisis (correlations rise), and crises are what we want to predict.

A scenario ("tech falls 20%") only fixes one or two factors. We fill in the
rest with the conditional expectation under the factors' covariance measured
in past stress weeks:  E[F_rest | F_S = s] = S_rS · S_SS^-1 · s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from universe import ASSETS, FACTORS

FACTOR_LIST = list(FACTORS)


def weekly_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    weekly = prices.ffill(limit=5).resample("W-FRI").last()
    return np.log(weekly).diff().iloc[1:]


def stress_mask(rets: pd.DataFrame, q: float = 0.2) -> pd.Series:
    """Weeks in the worst `q` share for the S&P 500."""
    return rets["SPY"] <= rets["SPY"].quantile(q)


def features_for(ticker: str) -> list[str]:
    """Which factors drive this asset. Sector enters as excess over the market."""
    sector = ASSETS[ticker][1]
    feats = ["SPY"]
    if sector not in ("SPY",):
        feats.append(sector)
    for f in ("TLT", "GLD"):
        if f not in feats:
            feats.append(f)
    return feats


def design(rets: pd.DataFrame, ticker: str) -> pd.DataFrame:
    feats = features_for(ticker)
    X = pd.DataFrame(index=rets.index)
    for f in feats:
        if f == "SPY":
            X["SPY"] = rets["SPY"]
        elif f in ("TLT", "GLD"):
            X[f] = rets[f]
        else:  # sector excess return over market
            X[f] = rets[f] - rets["SPY"]
    return X


def to_factor_coefs(ticker: str, coef: dict[str, float]) -> np.ndarray:
    """Map coefficients on (SPY, sector-SPY, TLT, GLD) back to raw factor space."""
    out = dict.fromkeys(FACTOR_LIST, 0.0)
    for name, b in coef.items():
        if name in ("SPY", "TLT", "GLD"):
            out[name] += b
        else:  # sector excess: b*(sector - SPY)
            out[name] += b
            out["SPY"] -= b
    return np.array([out[f] for f in FACTOR_LIST])


def fit_asset(rets: pd.DataFrame, ticker: str, alpha: float, stress_w: float):
    """Returns (factor coefficient vector, weekly residual std)."""
    if ticker in FACTOR_LIST:  # a factor is trivially explained by itself
        v = np.zeros(len(FACTOR_LIST))
        v[FACTOR_LIST.index(ticker)] = 1.0
        return v, 0.0
    X = design(rets, ticker)
    y = rets[ticker]
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    w = np.where(stress_mask(rets)[ok], 1.0 + stress_w, 1.0)
    model = Ridge(alpha=alpha, fit_intercept=True).fit(X.values, y.values, sample_weight=w)
    resid = y.values - model.predict(X.values)
    coefs = dict(zip(X.columns, model.coef_))
    return to_factor_coefs(ticker, coefs), float(np.std(resid))


def fit_capm(rets: pd.DataFrame, ticker: str) -> float:
    """Baseline 2: classic single beta to the market from ordinary least squares."""
    if ticker == "SPY":
        return 1.0
    df = rets[["SPY", ticker]].dropna()
    c = np.cov(df["SPY"], df[ticker])
    return float(c[0, 1] / c[0, 0])


def stress_cov(rets: pd.DataFrame) -> np.ndarray:
    sub = rets.loc[stress_mask(rets), FACTOR_LIST].dropna()
    return np.cov(sub.values, rowvar=False)


def propagate(shocks: dict[str, float], cov: np.ndarray) -> np.ndarray:
    """Fill in un-shocked factors with their conditional expectation (log-return space)."""
    idx_s = [FACTOR_LIST.index(k) for k in shocks]
    s = np.array([shocks[k] for k in shocks])
    full = cov[:, idx_s] @ np.linalg.solve(cov[np.ix_(idx_s, idx_s)], s)
    full[idx_s] = s  # keep the user's shocks exactly
    return full


def fit_all(rets: pd.DataFrame, alpha: float, stress_w: float):
    coefs, resid = {}, {}
    for t in ASSETS:
        coefs[t], resid[t] = fit_asset(rets, t, alpha, stress_w)
    return coefs, resid
