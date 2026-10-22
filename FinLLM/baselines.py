"""
TradeWiseAI - Step 4: baselines with walk-forward evaluation.
Run after step1_data_pipeline.py (needs dataset_prices_indicators.csv).
Saves per-fold metrics and all test-set predictions (for the trading simulator).
pip install scikit-learn pandas numpy
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from FinLLM.data_pipeline import walk_forward_splits

FEATURES = ["ret_1d", "rsi_14", "macd", "macd_signal", "macd_diff",
            "bb_pct", "bb_width", "sma_10", "sma_50", "ema_20", "vol_chg"]
SEEDS = [0, 1, 2, 3, 4]

data = pd.read_csv("dataset_prices_indicators.csv", parse_dates=["Date"])
data = data.sort_values(["Date", "ticker"]).reset_index(drop=True)
data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)


def metrics(y, pred):
    return dict(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
                f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred))


def make_models(seed):
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                     n_jobs=-1, random_state=seed),
        "gboost": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                 max_iter=200, random_state=seed),
    }


rows, preds = [], []
for fold, (tr, va, te) in enumerate(walk_forward_splits(data["Date"])):
    train = data[tr & data["label"].notna().values]
    test = data[te & data["label"].notna().values]
    if len(train) == 0 or len(test) == 0:
        continue
    ytr, yte = train["label"].astype(int), test["label"].astype(int)
    period = f"{test['Date'].min().date()}..{test['Date'].max().date()}"

    # Non-learned baselines
    maj = int(ytr.mean() >= 0.5)
    rows.append(dict(fold=fold, period=period, model="always_majority", seed=-1,
                     **metrics(yte, np.full(len(yte), maj))))
    rows.append(dict(fold=fold, period=period, model="prev_day_direction", seed=-1,
                     **metrics(yte, (test["ret_1d"] > 0).astype(int))))

    # Learned baselines (multiple seeds; logreg is deterministic so one is enough)
    for seed in SEEDS:
        for name, m in make_models(seed).items():
            if name == "logreg" and seed != SEEDS[0]:
                continue
            m.fit(train[FEATURES], ytr)
            p = m.predict_proba(test[FEATURES])[:, 1]
            rows.append(dict(fold=fold, period=period, model=name, seed=seed,
                             **metrics(yte, (p >= 0.5).astype(int))))
            if seed == SEEDS[0]:
                preds.append(pd.DataFrame({"Date": test["Date"].values, "ticker": test["ticker"].values,
                                           "fold": fold, "model": name, "prob_up": p,
                                           "label": yte.values, "fwd_ret": test["fwd_ret"].values}))

res = pd.DataFrame(rows)
res.to_csv("baseline_fold_metrics.csv", index=False)
pd.concat(preds).to_csv("baseline_test_predictions.csv", index=False)

# Average over seeds within a fold first, then mean/std across folds
per_fold = res.groupby(["model", "fold"])[["acc", "bal_acc", "f1", "mcc"]].mean()
summary = per_fold.groupby("model").agg(["mean", "std"]).round(4)
print(summary.to_string())