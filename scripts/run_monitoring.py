## Performance and Monitoring
import argparse
import glob
import os
import pickle

import pandas as pd

from spark_io import get_spark
import utils.model_utils as mu

MODEL_BANK = "model_bank"
PRED_DIR = "datamart/gold/model_predictions"
MON_DIR = "datamart/gold/model_monitoring"


def _load_baselines(versions):
    out = {}
    for v in versions:
        p = os.path.join(MODEL_BANK, f"{v}.pkl")
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    art = pickle.load(f)
                out[v] = (art["psi_bin_edges"], art["train_score_baseline"])
            except Exception as e:
                print(f"[monitor] WARN baseline load failed for {v}: {e}")
    return out


def main(snapshotdate):
    pred_files = glob.glob(os.path.join(PRED_DIR, "predictions_*.parquet"))
    if not pred_files:
        print(f"[monitor] no predictions yet at {snapshotdate}; skipping.")
        return

    spark = get_spark("run_monitoring")
    preds = spark.read.parquet(*pred_files).toPandas()
    versions = sorted(preds["model_version"].dropna().unique()) \
        if "model_version" in preds.columns else []
    baselines = _load_baselines(versions)

    monitoring = mu.monitor_over_time(preds, baselines)
    monitoring["run_date"] = snapshotdate

    os.makedirs(MON_DIR, exist_ok=True)
    out_path = os.path.join(MON_DIR, "model_monitoring.parquet")
    spark.createDataFrame(monitoring).write.mode("overwrite").parquet(out_path)
    spark.stop()

    pd.set_option("display.width", 220)
    print(f"[monitor] wrote {len(monitoring)} rows ({len(versions)} version(s)) -> {out_path}")
    cols = ["snapshot_date", "model_version", "n", "default_rate", "auc", "gini", "psi_score"]
    print(monitoring[cols].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    main(ap.parse_args().snapshotdate)
