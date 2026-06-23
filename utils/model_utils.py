from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss

# Lag (months) between the application feature snapshot and the maturity month at which the 6-month-on-book default label is observed
LABEL_LAG_MONTHS = 6

# Identifier / target columns that must never be used as features.
ID_COLS = ["Customer_ID", "loan_id", "snapshot_date", "feat_date",
           "label", "label_def", "model_name", "model_version", "predicted_pd"]

# Categorical feature columns produced by the Assignment-1 gold feature store.
CATEGORICAL_COLS = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]


# Leakage-safe modeling table
def _first_of_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def build_modeling_table(label_df: pd.DataFrame, feature_df: pd.DataFrame, lag_months: int = LABEL_LAG_MONTHS) -> pd.DataFrame:
    lab = label_df.copy()
    feat = feature_df.copy()
    lab["snapshot_date"] = _first_of_month(lab["snapshot_date"])
    feat["snapshot_date"] = _first_of_month(feat["snapshot_date"])
    lab["feat_date"] = lab["snapshot_date"] - pd.offsets.DateOffset(months=lag_months)
    feat = feat.rename(columns={"snapshot_date": "feat_date"})
    feat = feat.drop_duplicates(subset=["Customer_ID", "feat_date"], keep="last")
    return lab.merge(feat, on=["Customer_ID", "feat_date"], how="inner")


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ID_COLS]


def split_feature_types(feature_cols: list):
    cat = [c for c in feature_cols if c in CATEGORICAL_COLS]
    num = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    return num, cat


# Pre-processing + different models
def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])


def candidate_models(random_state: int = 88) -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=random_state),
    }
    try:
        import xgboost as xgb
        models["xgboost"] = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=random_state, n_jobs=-1)
    except Exception as exc:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingClassifier
        print(f"[model_utils] xgboost unavailable ({exc}); using HistGradientBoosting")
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300, random_state=random_state)
    return models


# Metrics
def ks_statistic(y_true, y_score) -> float:
    order = np.argsort(y_score)
    y = np.asarray(y_true)[order]
    pos = np.cumsum(y) / max(y.sum(), 1)
    neg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    out = {"n": int(len(y_true)), "default_rate": float(y_true.mean()) if len(y_true) else np.nan}
    if len(np.unique(y_true)) < 2:
        out.update(auc=np.nan, gini=np.nan, ks=np.nan, precision=np.nan, recall=np.nan, f1=np.nan, brier=float(brier_score_loss(y_true, y_score)) if len(y_true) else np.nan)
        return out
    auc = roc_auc_score(y_true, y_score)
    out.update(auc=float(auc), gini=float(2 * auc - 1), ks=ks_statistic(y_true, y_score),
               precision=float(precision_score(y_true, y_pred, zero_division=0)),
               recall=float(recall_score(y_true, y_pred, zero_division=0)),
               f1=float(f1_score(y_true, y_pred, zero_division=0)),
               brier=float(brier_score_loss(y_true, y_score)))
    return out


def population_stability_index(expected, actual, bin_edges) -> float:
    eps = 1e-6
    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)
    exp_pct = exp_counts / max(exp_counts.sum(), 1) + eps
    act_pct = act_counts / max(act_counts.sum(), 1) + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# Train candidates and select champion
def train_and_select(modeling_df, train_test_end, oot_start, oot_end, train_test_ratio: float = 0.8, random_state: int = 88, model_names=None) -> dict:
    df = modeling_df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    tte, oos, ooe = pd.Timestamp(train_test_end), pd.Timestamp(oot_start), pd.Timestamp(oot_end)

    feature_cols = get_feature_columns(df)
    num_cols, cat_cols = split_feature_types(feature_cols)
    train_test = df[df["snapshot_date"] <= tte]
    oot = df[(df["snapshot_date"] >= oos) & (df["snapshot_date"] <= ooe)]

    from sklearn.model_selection import train_test_split
    X_tt, y_tt = train_test[feature_cols], train_test["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt, test_size=1 - train_test_ratio, random_state=random_state, shuffle=True, stratify=y_tt)
    X_oot, y_oot = oot[feature_cols], oot["label"].astype(int)

    all_candidates = candidate_models(random_state)
    if model_names:
        candidates = {k: v for k, v in all_candidates.items() if k in model_names}
        if not candidates:
            candidates = all_candidates
    else:
        candidates = all_candidates

    results, fitted = {}, {}
    for name, est in candidates.items():
        pipe = Pipeline([("pre", build_preprocessor(num_cols, cat_cols)), ("clf", est)])
        pipe.fit(X_train, y_train)
        m_train = classification_metrics(y_train, pipe.predict_proba(X_train)[:, 1])
        m_test = classification_metrics(y_test, pipe.predict_proba(X_test)[:, 1])
        m_oot = classification_metrics(y_oot, pipe.predict_proba(X_oot)[:, 1]) if len(X_oot) else {"auc": np.nan}
        results[name] = {"train": m_train, "test": m_test, "oot": m_oot}
        fitted[name] = pipe
        print(f"[train] {name:22s} AUC train={m_train['auc']:.3f} "f"test={m_test['auc']:.3f} oot={m_oot.get('auc', float('nan')):.3f}")

    def _score(n):
        a = results[n]["oot"].get("auc")
        return a if a == a else results[n]["test"]["auc"]
    champion = max(results, key=_score)
    champ_pipe = fitted[champion]
    print(f"[train] CHAMPION = {champion}")

    train_scores = champ_pipe.predict_proba(X_train)[:, 1]
    psi_bins = np.unique(np.quantile(train_scores, np.linspace(0, 1, 11)))
    if len(psi_bins) < 3:
        psi_bins = np.linspace(0, 1, 11)
    psi_bins[0], psi_bins[-1] = -np.inf, np.inf

    return {
        "pipeline": champ_pipe,
        "champion_name": champion,
        "candidate_names": list(candidates.keys()),
        "feature_cols": feature_cols,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "all_candidate_results": results,
        "champion_results": results[champion],
        "psi_bin_edges": psi_bins,
        "train_score_baseline": train_scores,
        "split": {"train_test_end": train_test_end, "oot_start": oot_start, "oot_end": oot_end, "n_train": int(len(X_train)), "n_test": int(len(X_test)), "n_oot": int(len(X_oot)), "default_rate_train": float(y_train.mean())},
    }


# Governance to control retraining
def compute_windows(train_month, oot_months: int = 3):
    m = pd.Timestamp(train_month).to_period("M").to_timestamp()
    oot_start = m - pd.offsets.DateOffset(months=oot_months - 1)
    train_test_end = oot_start - pd.offsets.DateOffset(months=1)
    f = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")
    return f(train_test_end), f(oot_start), f(m)


def months_between(a, b) -> int:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (b.year - a.year) * 12 + (b.month - a.month)


def decide_training_action(month, champion_artefact, monitoring_df, initial_anchor, refresh_months: int = 3, auc_floor: float = 0.70, psi_retrain: float = 0.25):
    month = pd.Timestamp(month)
    anchor = pd.Timestamp(initial_anchor)
    if month < anchor:
        return None, "before_anchor"
    if champion_artefact is None:
        return "bootstrap", "bootstrap_initial"
    last_train = pd.Timestamp(champion_artefact.get("train_date", anchor))
    if months_between(last_train, month) >= refresh_months:
        return "refresh", "scheduled_cadence"
    if monitoring_df is not None and len(monitoring_df):
        mdf = monitoring_df.copy()
        mdf["snapshot_date"] = pd.to_datetime(mdf["snapshot_date"])
        labelled = mdf.sort_values("snapshot_date").dropna(subset=["auc"])
        if len(labelled):
            last = labelled.iloc[-1]
            if float(last["auc"]) < auc_floor:
                return "refresh", "trigger_auc_below_floor"
            if "psi_score" in last and float(last["psi_score"]) >= psi_retrain:
                return "refresh", "trigger_psi_breach"
    return None, "within_sla"


# Inference
def score_frame(artefact: dict, df: pd.DataFrame) -> np.ndarray:
    X = df.reindex(columns=artefact["feature_cols"])
    return artefact["pipeline"].predict_proba(X)[:, 1]


# Monitoring (Performance & Stability)
def monitor_over_time(pred_df: pd.DataFrame, baselines_by_version: dict,
                      threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    df = pred_df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    if "model_version" not in df.columns:
        df["model_version"] = "unknown"
    for month, g in df.groupby("snapshot_date"):
        version = g["model_version"].iloc[0]
        name = g["model_name"].iloc[0] if "model_name" in g.columns else version
        rec = {"snapshot_date": pd.Timestamp(month), "model_version": version, "model_name": name}
        labelled = g.dropna(subset=["label"])
        if len(labelled):
            rec.update(classification_metrics(labelled["label"], labelled["predicted_pd"], threshold))
        else:
            rec.update(n=int(len(g)), default_rate=np.nan, auc=np.nan, gini=np.nan, ks=np.nan, precision=np.nan, recall=np.nan, f1=np.nan, brier=np.nan)
        if version in baselines_by_version:
            bins, baseline = baselines_by_version[version]
            rec["psi_score"] = population_stability_index(np.asarray(baseline), g["predicted_pd"].values, np.asarray(bins))
        else:
            rec["psi_score"] = np.nan
        rec["avg_predicted_pd"] = float(g["predicted_pd"].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("snapshot_date").reset_index(drop=True)
