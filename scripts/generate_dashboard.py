## Monitoring and Governance Visualisations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from spark_io import get_spark

MODEL_BANK = "model_bank"
MON_DIR = "datamart/gold/model_monitoring"
ARTIFACTS = "artifacts"
REGISTRY = os.path.join(MODEL_BANK, "model_registry.csv")

NAVY = "#1f2a44"; BLUE = "#2e6fb5"; AMBER = "#e0a526"; RED = "#c0392b"
GREEN = "#2e7d54"; GREY = "#8a93a6"; PURPLE = "#6b4c9a"
ERA_COLORS = [BLUE, GREEN, AMBER, PURPLE]


def _fmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _registry():
    if os.path.exists(REGISTRY):
        try:
            return pd.read_csv(REGISTRY)
        except Exception:
            pass
    return pd.DataFrame()


def _eras(ax, mon):
    if "model_version" not in mon.columns:
        return
    for i, (ver, g) in enumerate(mon.groupby("model_version")):
        lo, hi = g["snapshot_date"].min(), g["snapshot_date"].max()
        col = ERA_COLORS[i % len(ERA_COLORS)]
        ax.axvspan(lo, hi, color=col, alpha=0.07)
        label = g["model_name"].iloc[0] if "model_name" in g else ver
        ax.text(lo, ax.get_ylim()[1], f" {label}", color=col, fontsize=8, va="top")


def _markers(ax, reg):
    for _, r in reg.iterrows():
        d = pd.Timestamp(r["train_date"])
        ax.axvline(d, color=NAVY, ls=":", lw=1.1, alpha=0.8)
        tag = "bootstrap" if r.get("action") == "bootstrap" else "refresh"
        ax.text(d, ax.get_ylim()[0], f" {tag}", color=NAVY, fontsize=7.5, rotation=90, va="bottom", ha="right")


def main(snapshotdate):
    os.makedirs(ARTIFACTS, exist_ok=True)
    spark = get_spark("dashboard")
    mon = spark.read.parquet(os.path.join(MON_DIR, "model_monitoring.parquet")).toPandas()
    spark.stop()
    mon["snapshot_date"] = pd.to_datetime(mon["snapshot_date"])
    mon = mon.sort_values("snapshot_date")
    reg = _registry()

    # Model Performance over Time
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_ylim(0.4, 1.0)
    ax.plot(mon["snapshot_date"], mon["auc"], "-o", color=NAVY, lw=2, label="AUC")
    ax.plot(mon["snapshot_date"], mon["gini"], "-s", color=BLUE, lw=1.5, label="Gini")
    _eras(ax, mon)
    if len(reg): _markers(ax, reg)
    ax.axhline(0.70, color=RED, ls="--", lw=1, label="AUC floor 0.70")
    ax.set_title("Model Performance over Time", fontsize=12, color=NAVY, weight="bold")
    ax.set_ylabel("score"); ax.legend(loc="lower left", fontsize=8, ncol=2); _fmt(ax)
    fig.tight_layout(); fig.savefig(f"{ARTIFACTS}/performance_over_time.png", dpi=140); plt.close(fig)

    # PSI
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(mon["snapshot_date"], mon["psi_score"], width=20, color=BLUE, alpha=0.85)
    ax.axhline(0.10, color=AMBER, ls="--", lw=1.2, label="PSI 0.10 (watch)")
    ax.axhline(0.25, color=RED, ls="--", lw=1.2, label="PSI 0.25 (retrain)")
    if len(reg): _markers(ax, reg)
    ax.set_title("Score stability — PSI vs champion model training baseline", fontsize=12, color=NAVY, weight="bold")
    ax.set_ylabel("PSI"); ax.legend(loc="upper left", fontsize=8); _fmt(ax)
    fig.tight_layout(); fig.savefig(f"{ARTIFACTS}/stability_psi_over_time.png", dpi=140); plt.close(fig)

    # Drift
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(mon["snapshot_date"], mon["default_rate"], "-o", color=RED, lw=2, label="Actual Default Rate")
    ax.plot(mon["snapshot_date"], mon["avg_predicted_pd"], "-o", color=BLUE, lw=2, label="Avg Predicted PD")
    _eras(ax, mon)
    if len(reg): _markers(ax, reg)
    ax.set_title("Actual Default Rate vs Predicted PD", fontsize=12, color=NAVY, weight="bold")
    ax.set_ylabel("rate"); ax.legend(loc="upper left", fontsize=8); _fmt(ax)
    fig.tight_layout(); fig.savefig(f"{ARTIFACTS}/default_rate_drift.png", dpi=140); plt.close(fig)

    # Model Comparison
    try:
        res = json.load(open(os.path.join(ARTIFACTS, "training_results.json")))["candidate_results"]
        names = list(res); x = np.arange(len(names)); w = 0.25
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, sset in enumerate(["train", "test", "oot"]):
            vals = [res[n][sset].get("auc") or 0 for n in names]
            ax.bar(x + (i - 1) * w, vals, w, label=sset.upper(), color=[GREY, BLUE, GREEN][i])
            for xi, v in zip(x + (i - 1) * w, vals):
                ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=7, color="#333")
        ax.set_xticks(x); ax.set_xticklabels([n.replace("_", " ").title() for n in names], fontsize=9)
        ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC")
        ax.set_title("Latest training — candidate AUC by split", fontsize=12, color=NAVY, weight="bold")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{ARTIFACTS}/model_comparison.png", dpi=140); plt.close(fig)
    except Exception as e:
        print("[dashboard] comparison skipped:", e)

    # Model Registry Table
    if len(reg):
        show = reg[["train_date", "model_version", "action", "champion_model", "auc_test", "auc_oot"]].copy()
        fig, ax = plt.subplots(figsize=(9, 1.0 + 0.4 * len(show)))
        ax.axis("off")
        tbl = ax.table(cellText=show.values, colLabels=show.columns, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.4)
        for j in range(len(show.columns)):
            c = tbl[0, j]; c.set_facecolor(NAVY); c.set_text_props(color="white", weight="bold")
        ax.set_title("Model bank — version registry", fontsize=12, color=NAVY, weight="bold", pad=12)
        fig.tight_layout(); fig.savefig(f"{ARTIFACTS}/model_registry.png", dpi=140); plt.close(fig)

    print(f"[dashboard] wrote PNGs to {ARTIFACTS}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    main(ap.parse_args().snapshotdate)
