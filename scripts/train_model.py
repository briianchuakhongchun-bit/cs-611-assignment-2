## Training of model
## Initial anchor on 2024-06-01, then follow the governance model 

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from spark_io import get_spark, load_label_store, load_feature_store
import utils.model_utils as mu

MODEL_BANK = "model_bank"
ARTIFACTS = "artifacts"
MON_PATH = "datamart/gold/model_monitoring/model_monitoring.parquet"
REGISTRY = os.path.join(MODEL_BANK, "model_registry.csv")


def _load_champion():
    p = os.path.join(MODEL_BANK, "champion_latest.pkl")
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[train] WARN could not read champion_latest.pkl: {e}")
    return None


def _load_monitoring(spark):
    if os.path.exists(MON_PATH):
        try:
            return spark.read.parquet(MON_PATH).toPandas()
        except Exception:
            return None
    return None


def _append_registry(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(REGISTRY):
        df = pd.concat([pd.read_csv(REGISTRY), df], ignore_index=True)
    df.to_csv(REGISTRY, index=False)


def main(snapshotdate, initial_anchor, initial_model, refresh_months, auc_floor, psi_retrain, oot_months):
    os.makedirs(MODEL_BANK, exist_ok=True)
    os.makedirs(ARTIFACTS, exist_ok=True)

    champion = _load_champion()

    spark = get_spark("train_model")
    monitoring_df = _load_monitoring(spark)

    action, reason = mu.decide_training_action(snapshotdate, champion, monitoring_df, initial_anchor=initial_anchor, refresh_months=refresh_months, auc_floor=auc_floor, psi_retrain=psi_retrain)
    print(f"[train] {snapshotdate}: action={action} reason={reason}")
    if action is None:
        spark.stop(); return

    label_df = load_label_store(spark)
    feature_df = load_feature_store(spark)
    spark.stop()
    if label_df is None or feature_df is None:
        raise RuntimeError("Gold label/feature store empty - run data pipeline first.")

    modeling_df = mu.build_modeling_table(label_df, feature_df)
    tte, oos, ooe = mu.compute_windows(snapshotdate, oot_months=oot_months)
    model_names = [initial_model] if action == "bootstrap" else None
    print(f"[train] windows train/test<= {tte} | OOT {oos}..{ooe} | "
          f"candidates={'['+initial_model+']' if model_names else 'ALL'}")

    artefact = mu.train_and_select(modeling_df, train_test_end=tte, oot_start=oos, oot_end=ooe, model_names=model_names)
    artefact["train_date"] = pd.Timestamp(snapshotdate)
    artefact["model_version"] = "credit_model_" + pd.Timestamp(snapshotdate).strftime("%Y_%m_%d")
    artefact["action"] = action
    artefact["reason"] = reason

    vpath = os.path.join(MODEL_BANK, artefact["model_version"] + ".pkl")
    lpath = os.path.join(MODEL_BANK, "champion_latest.pkl")
    for p in (vpath, lpath):
        with open(p, "wb") as f:
            pickle.dump(artefact, f)
    print(f"[train] promoted '{artefact['champion_name']}' as {artefact['model_version']}")

    res = artefact["champion_results"]
    _append_registry({
        "train_date": pd.Timestamp(snapshotdate).strftime("%Y-%m-%d"),
        "model_version": artefact["model_version"],
        "action": action, "reason": reason,
        "champion_model": artefact["champion_name"],
        "candidates_considered": "|".join(artefact["candidate_names"]),
        "n_train": artefact["split"]["n_train"],
        "n_test": artefact["split"]["n_test"],
        "n_oot": artefact["split"]["n_oot"],
        "auc_train": round(res["train"]["auc"], 4),
        "auc_test": round(res["test"]["auc"], 4),
        "auc_oot": round(res["oot"].get("auc", float("nan")), 4),
        "gini_oot": round(res["oot"].get("gini", float("nan")), 4),
    })
    print(f"[train] registry updated -> {REGISTRY}")

    # per-version training summary
    summary = {
        "model_version": artefact["model_version"], "train_date": str(snapshotdate),
        "action": action, "reason": reason, "champion": artefact["champion_name"],
        "split": artefact["split"],
        "candidate_results": {
            k: {s: {m: (None if (isinstance(v, float) and np.isnan(v)) else v)
                    for m, v in metr.items()} for s, metr in r.items()}
            for k, r in artefact["all_candidate_results"].items()},
    }
    with open(os.path.join(ARTIFACTS, f"training_results_{artefact['model_version']}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # also keep a 'latest' copy for the dashboard
    with open(os.path.join(ARTIFACTS, "training_results.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    ap.add_argument("--initial_anchor", default="2024-06-01")
    ap.add_argument("--initial_model", default="logistic_regression")
    ap.add_argument("--refresh_months", type=int, default=3)
    ap.add_argument("--auc_floor", type=float, default=0.70)
    ap.add_argument("--psi_retrain", type=float, default=0.25)
    ap.add_argument("--oot_months", type=int, default=3)
    a = ap.parse_args()
    main(a.snapshotdate, a.initial_anchor, a.initial_model, a.refresh_months, a.auc_floor, a.psi_retrain, a.oot_months)
