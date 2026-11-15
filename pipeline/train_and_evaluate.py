"""Step 2: tune, train and honestly evaluate the stress model, then export it for the website.

Protocol (no look-ahead):
  * Tuning:     train 2010-2017, pick hyper-parameters on two 2018 sell-offs.
  * Testing:    train 2010-2019, test on four crises from 2020-2025 the model never saw.
  * Deployment: refit on all data up to Oct 2025 with the chosen hyper-parameters.

Three prediction modes per episode:
  * full        - every factor's real move is known (how well do the betas explain assets?)
  * market_only - only the S&P 500 move is known (realistic "market falls X%" scenario)
  * sector_only - only the tech sector move is known ("tech falls X%" scenario)
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import model as M
from universe import ASSETS, EPISODES, FACTORS, TRAIN_END, TRAIN_START

HERE = Path(__file__).parent
WEB_DATA = HERE.parent / "web" / "data"
WEB_DATA.mkdir(parents=True, exist_ok=True)

prices = pd.read_csv(HERE / "data" / "prices.csv", index_col=0, parse_dates=True)
rets_all = M.weekly_log_returns(prices)
FL = M.FACTOR_LIST
TICKERS = list(ASSETS)

VAL_EPISODES = {
    "feb_2018": ("Feb 2018 volatility shock", "2018-01-26", "2018-02-08"),
    "q4_2018": ("Q4 2018 sell-off", "2018-09-20", "2018-12-24"),
}


def episode_logret(start: str, end: str) -> pd.Series:
    p = prices.ffill()
    p0 = p.loc[:start].iloc[-1]
    p1 = p.loc[:end].iloc[-1]
    return np.log(p1 / p0)


def predict_episode(coefs, capm, cov, ep_lr: pd.Series, mode: str) -> pd.DataFrame:
    if mode == "full":
        f = ep_lr[FL].values
    elif mode == "market_only":
        f = M.propagate({"SPY": ep_lr["SPY"]}, cov)
    elif mode == "sector_only":
        f = M.propagate({"XLK": ep_lr["XLK"]}, cov)
    else:
        raise ValueError(mode)
    spy_known = f[FL.index("SPY")]  # baselines only see the (given or implied) market move
    rows = []
    for t in TICKERS:
        actual = np.expm1(ep_lr[t])
        if np.isnan(actual):
            continue
        if mode == "full" and t in FL:
            continue  # SPY/TLT/GLD are inputs in this mode: predicting them would be trivial, so excluded
        rows.append({
            "ticker": t,
            "actual": actual,
            "ours": np.expm1(coefs[t] @ f),
            "baseline_market": np.expm1(spy_known),
            "baseline_capm": np.expm1(capm[t] * spy_known),
        })
    return pd.DataFrame(rows).set_index("ticker")


def fit(train: pd.DataFrame, alpha: float, stress_w: float):
    coefs, resid = M.fit_all(train, alpha, stress_w)
    capm = {t: M.fit_capm(train, t) for t in TICKERS}
    return coefs, resid, capm, M.stress_cov(train)


def mae(df, col):
    return float((df[col] - df["actual"]).abs().mean())


# ---------------------------------------------------------------- 1. tuning
print("== Tuning on 2018 validation sell-offs (train 2010-2017) ==")
train_tune = rets_all.loc[TRAIN_START:"2017-12-31"]
grid = []
for alpha, sw in product([0.0, 1e-3, 1e-2, 3e-2], [0.0, 1.0, 3.0, 6.0]):
    coefs, _, capm, cov = fit(train_tune, alpha, sw)
    scores = []
    for _, s, e in VAL_EPISODES.values():
        lr = episode_logret(s, e)
        for mode in ("full", "market_only"):
            scores.append(mae(predict_episode(coefs, capm, cov, lr, mode), "ours"))
    grid.append({"alpha": alpha, "stress_weight": sw, "val_mae": float(np.mean(scores))})
    print(f"  alpha={alpha:<6} stress_w={sw:<4} val MAE={np.mean(scores)*100:.2f} pp")
best = min(grid, key=lambda g: g["val_mae"])
ALPHA, SW = best["alpha"], best["stress_weight"]
print(f"Best: alpha={ALPHA}, stress_weight={SW}")

# ---------------------------------------------------------------- 2. testing
print("\n== Test on unseen crises (train 2010-2019) ==")
train = rets_all.loc[TRAIN_START:TRAIN_END]
coefs, resid, capm, cov = fit(train, ALPHA, SW)
coefs_nostress, *_ = fit(train, ALPHA, 0.0)  # ablation: same model without stress weighting

results, all_rows, episode_preds = {}, [], {}
for key, (label, s, e) in EPISODES.items():
    lr = episode_logret(s, e)
    results[key] = {"label": label, "start": s, "end": e, "spy_move": float(np.expm1(lr["SPY"])), "modes": {}}
    for mode in ("full", "market_only", "sector_only"):
        df = predict_episode(coefs, capm, cov, lr, mode)
        df_ns = predict_episode(coefs_nostress, capm, cov, lr, mode)
        df["ours_no_stress"] = df_ns["ours"]
        rho = spearmanr(df["ours"], df["actual"]).statistic
        rho_capm = spearmanr(df["baseline_capm"], df["actual"]).statistic
        results[key]["modes"][mode] = {
            "mae_ours": mae(df, "ours"),
            "mae_ours_no_stress": mae(df, "ours_no_stress"),
            "mae_baseline_market": mae(df, "baseline_market"),
            "mae_baseline_capm": mae(df, "baseline_capm"),
            "rank_corr_ours": float(rho),
            "rank_corr_capm": float(rho_capm),
            "n_assets": int(len(df)),
        }
        tmp = df.reset_index().assign(episode=key, mode=mode)
        all_rows.append(tmp)
        if mode == "market_only":
            episode_preds[key] = {t: round(float(v), 4) for t, v in df["ours"].items()}
        m = results[key]["modes"][mode]
        print(f"  {label:<34} {mode:<12} MAE ours={m['mae_ours']*100:5.2f}  no-stress={m['mae_ours_no_stress']*100:5.2f}"
              f"  market={m['mae_baseline_market']*100:5.2f}  capm={m['mae_baseline_capm']*100:5.2f}"
              f"  rank-corr ours={rho:.2f} capm={rho_capm:.2f}")

rows = pd.concat(all_rows, ignore_index=True)

# Overall averages per mode
summary = {}
for mode, g in rows.groupby("mode"):
    summary[mode] = {
        col: float((g[col] - g["actual"]).abs().mean())
        for col in ("ours", "ours_no_stress", "baseline_market", "baseline_capm")
    }
print("\nOverall MAE (pp):", {m: {k: round(v * 100, 2) for k, v in d.items()} for m, d in summary.items()})

# Win rate: share of (asset, episode) pairs where ours beats the CAPM baseline
mo = rows[rows["mode"] == "market_only"]
win_rate = float(((mo["ours"] - mo["actual"]).abs() < (mo["baseline_capm"] - mo["actual"]).abs()).mean())

# Random portfolios: does the model get the *portfolio* loss right?
rng = np.random.default_rng(42)
port_err = {"ours": [], "baseline_market": [], "baseline_capm": []}
for _ in range(2000):
    ep = rng.choice(list(EPISODES))
    g = mo[mo["episode"] == ep].set_index("ticker")
    pick = rng.choice(g.index, size=4, replace=False)
    w = rng.dirichlet(np.ones(4))
    act = float(w @ g.loc[pick, "actual"])
    for c in port_err:
        port_err[c].append(abs(float(w @ g.loc[pick, c]) - act))
portfolio_mae = {k: float(np.mean(v)) for k, v in port_err.items()}
print("Random 4-asset portfolios, market-only scenario MAE (pp):",
      {k: round(v * 100, 2) for k, v in portfolio_mae.items()})

# Biggest failures (market-only mode, the realistic scenario)
mo = mo.assign(err=(mo["ours"] - mo["actual"]))
fails = mo.reindex(mo["err"].abs().sort_values(ascending=False).index).head(8)
failures = [{
    "ticker": r.ticker, "episode": EPISODES[r.episode][0],
    "predicted": round(float(r.ours), 4), "actual": round(float(r.actual), 4),
} for r in fails.itertuples()]
print("\nLargest misses:")
for f in failures:
    print(f"  {f['ticker']:<8} {f['episode']:<34} predicted {f['predicted']*100:6.1f}%  actual {f['actual']*100:6.1f}%")

# Per-asset typical error in past crises -> shown in the app as a reliability badge
asset_err = mo.groupby("ticker")["err"].apply(lambda x: float(x.abs().mean())).to_dict()

# Who does the model fail for? Error by asset group (full mode = best case, market_only = realistic)
def group_of(t: str) -> str:
    if t == "BTC-USD":
        return "Crypto"
    if t in ("SPY", "QQQ", "TLT", "GLD"):
        return "Funds (index, bonds, gold)"
    if ASSETS[t][1] in ("XLK",) or t in ("AMZN", "TSLA", "NFLX"):
        return "Tech & growth stocks"
    if ASSETS[t][1] in ("XLP", "XLV", "XLU"):
        return "Defensive stocks"
    return "Cyclical stocks (banks, energy, industry)"


group_err = {}
for mode in ("full", "market_only"):
    g = rows[rows["mode"] == mode].assign(group=lambda d: d["ticker"].map(group_of))
    out = {}
    for name, gg in g.groupby("group"):
        out[name] = {
            "n_assets": int(gg["ticker"].nunique()),
            "mae_ours": float((gg["ours"] - gg["actual"]).abs().mean()),
            "mae_capm": float((gg["baseline_capm"] - gg["actual"]).abs().mean()),
        }
    group_err[mode] = out
print("\nError by asset group (pp):")
for mode, d in group_err.items():
    for k, v in d.items():
        print(f"  {mode:<12} {k:<42} ours={v['mae_ours']*100:5.1f}  capm={v['mae_capm']*100:5.1f}  n={v['n_assets']}")

# Preset portfolios: which kind of beginner investor gets the most reliable answer?
PRESETS = {
    "Tech fan": {"AAPL": .30, "NVDA": .25, "MSFT": .20, "TSLA": .15, "META": .10},
    "Balanced starter": {"SPY": .40, "TLT": .20, "JNJ": .15, "KO": .15, "GLD": .10},
    "Dividends & defensives": {"KO": .25, "PG": .25, "JNJ": .20, "NEE": .15, "XOM": .15},
    "Crypto + growth": {"BTC-USD": .30, "NVDA": .25, "TSLA": .25, "AMD": .20},
}
preset_err = {}
for name, w in PRESETS.items():
    errs, errs_c = [], []
    for ep in EPISODES:
        g = rows[(rows["mode"] == "market_only") & (rows["episode"] == ep)].set_index("ticker")
        act = sum(wt * g.loc[t, "actual"] for t, wt in w.items())
        errs.append(abs(sum(wt * g.loc[t, "ours"] for t, wt in w.items()) - act))
        errs_c.append(abs(sum(wt * g.loc[t, "baseline_capm"] for t, wt in w.items()) - act))
    preset_err[name] = {"mae_ours": float(np.mean(errs)), "mae_capm": float(np.mean(errs_c))}
print("Preset portfolios, market-only (pp):", {k: round(v["mae_ours"] * 100, 1) for k, v in preset_err.items()})

# ---------------------------------------------------------------- 3. deploy
print("\n== Refit on all data for the live app ==")
coefs_live, resid_live, capm_live, cov_live = fit(rets_all, ALPHA, SW)

recent = rets_all.loc[rets_all.index[-156]:]  # last ~3 years of weekly returns
p = prices.ffill()
assets_out = {}
for t, (name, sector) in ASSETS.items():
    s = p[t].dropna()
    dd = float((s / s.cummax() - 1).min())
    assets_out[t] = {
        "name": name,
        "sector": sector,
        "sector_name": FACTORS[sector],
        "coefs": [round(float(c), 4) for c in coefs_live[t]],
        "capm_beta": round(capm_live[t], 3),
        "weekly_resid_vol": round(resid_live[t], 4),
        "annual_vol": round(float(recent[t].std() * np.sqrt(52)), 4),
        "max_drawdown": round(dd, 4),
        "crisis_error": round(asset_err.get(t, float("nan")), 4),
        "history_start": str(s.index[0].date()),
    }

corr = recent[TICKERS].corr().round(3)
model_out = {
    "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "data_end": str(prices.index[-1].date()),
    "factors": [{"id": f, "name": FACTORS[f]} for f in FL],
    "stress_cov": np.round(cov_live, 8).tolist(),
    "assets": assets_out,
    "corr": {"tickers": TICKERS, "matrix": corr.values.tolist()},
    "hyperparams": {"ridge_alpha": ALPHA, "stress_weight": SW},
}

episodes_out = {}
for key, (label, s, e) in EPISODES.items():
    lr = episode_logret(s, e)
    episodes_out[key] = {
        "label": label, "start": s, "end": e,
        "actual": {t: round(float(np.expm1(lr[t])), 4) for t in TICKERS if not np.isnan(lr[t])},
        "factor_moves": {f: round(float(np.expm1(lr[f])), 4) for f in FL},
        "model_pred_market_only": episode_preds[key],
    }

eval_out = {
    "protocol": {
        "train": f"{TRAIN_START} to {TRAIN_END}",
        "tuning": "train 2010-2017, validate on Feb 2018 and Q4 2018 sell-offs",
        "test_episodes": {k: v[0] for k, v in EPISODES.items()},
    },
    "tuning_grid": grid,
    "best": best,
    "episodes": results,
    "overall_mae": summary,
    "win_rate_vs_capm_market_only": win_rate,
    "portfolio_mae_market_only": portfolio_mae,
    "failures": failures,
    "group_error": group_err,
    "preset_portfolio_error": preset_err,
}

(WEB_DATA / "model.json").write_text(json.dumps(model_out))
(WEB_DATA / "episodes.json").write_text(json.dumps(episodes_out))
(WEB_DATA / "evaluation.json").write_text(json.dumps(eval_out, indent=1))
# Single JS bundle so the website also works when opened straight from disk (file://)
bundle = {"model": model_out, "episodes": episodes_out, "evaluation": eval_out}
(WEB_DATA / "bundle.js").write_text("window.MT_DATA = " + json.dumps(bundle) + ";\n")
rows.to_csv(HERE / "data" / "test_predictions.csv", index=False)
print("Exported model.json, episodes.json, evaluation.json to", WEB_DATA)
