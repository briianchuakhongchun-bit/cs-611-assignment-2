## Inference
import argparse
import glob
import os
import pickle
import re

import pandas as pd

from spark_io import get_spark, load_label_store, load_feature_store
import utils.model_utils as mu

MODEL_BANK = "model_bank"
PRED_DIR = "datamart/gold/model_predictions"


def _existing_months():
    months = set()
    for f in glob.glob(os.path.join(PRED_DIR, "predictions_*.parquet")):
        m = re.search(r"predictions_(\d{4})_(\d{2})\.parquet$", os.path.basename(f))
        if m:
            months.add(f"{m.group(1)}-{m.group(2)}")
    return months


def main(snapshotdate):
    champ_path = os.path.join(MODEL_BANK, "champion_latest.pkl")
    if not os.path.exists(champ_path):
        print(f"[infer] no champion yet at {snapshotdate}; skipping (bootstrap pending).")
        return
    with open(champ_path, "rb") as f:
        artefact = pickle.load(f)

    os.makedirs(PRED_DIR, exist_ok=True)
    spark = get_spark("run_inference")
    label_df = load_label_store(spark)
    feature_df = load_feature_store(spark)
    if label_df is None or feature_df is None:
        spark.stop(); print("[infer] gold store empty; skipping."); return

    model_df = mu.build_modeling_table(label_df, feature_df)
    model_df["snapshot_date"] = pd.to_datetime(model_df["snapshot_date"])

    MONITOR_START = "2024-04"
    done = _existing_months()
    all_months = sorted(model_df["snapshot_date"].dt.strftime("%Y-%m").unique())
    todo = [m for m in all_months if m not in done and m >= MONITOR_START]
    if not todo:
        spark.stop(); print(f"[infer] all {len(all_months)} months already scored; nothing to do."); return

    model_df["predicted_pd"] = mu.score_frame(artefact, model_df)
    model_df["model_name"] = artefact.get("champion_name", "model")
    model_df["model_version"] = artefact.get("model_version", "legacy_unversioned")
    out_cols = ["loan_id", "Customer_ID", "snapshot_date", "feat_date",
                "label", "predicted_pd", "model_name", "model_version"]

    written = 0
    for ym in todo:
        g = model_df[model_df["snapshot_date"].dt.strftime("%Y-%m") == ym][out_cols]
        if not len(g):
            continue
        path = os.path.join(PRED_DIR, f"predictions_{ym.replace('-', '_')}.parquet")
        spark.createDataFrame(g).write.mode("overwrite").parquet(path)
        written += 1
    spark.stop()
    print(f"[infer] champion {artefact.get('model_version','legacy')} ({artefact.get('champion_name','model')}) "
          f"gap-filled {written} new month(s); {len(done)} already existed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    main(ap.parse_args().snapshotdate)
