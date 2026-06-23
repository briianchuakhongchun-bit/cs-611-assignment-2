import argparse
import os

import pyspark

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold

DM = "datamart"
DPD_THRESHOLD = 30
MOB_THRESHOLD = 6

BRONZE = {k: os.path.join(DM, "bronze", k) for k in ["lms", "clickstream", "attributes", "financials"]}
SILVER = {k: os.path.join(DM, "silver", k) for k in ["lms", "clickstream", "attributes", "financials"]}
GOLD_LABEL = os.path.join(DM, "gold", "label_store")
GOLD_FEATURE = os.path.join(DM, "gold", "feature_store")


def main(snapshotdate: str):
    print(f"\n--- data pipeline for {snapshotdate} ---")
    spark = (pyspark.sql.SparkSession.builder.appName("data_pipeline").master("local[*]").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    for d in list(BRONZE.values()) + list(SILVER.values()) + [GOLD_LABEL, GOLD_FEATURE]:
        os.makedirs(d, exist_ok=True)

    # Bronze
    bronze.process_bronze_lms(snapshotdate, BRONZE["lms"], spark)
    bronze.process_bronze_clickstream(snapshotdate, BRONZE["clickstream"], spark)
    bronze.process_bronze_attributes(snapshotdate, BRONZE["attributes"], spark)
    bronze.process_bronze_financials(snapshotdate, BRONZE["financials"], spark)

    # Silver
    silver.process_silver_lms(snapshotdate, BRONZE["lms"], SILVER["lms"], spark)
    silver.process_silver_clickstream(snapshotdate, BRONZE["clickstream"], SILVER["clickstream"], spark)
    silver.process_silver_attributes(snapshotdate, BRONZE["attributes"], SILVER["attributes"], spark)
    silver.process_silver_financials(snapshotdate, BRONZE["financials"], SILVER["financials"], spark)

    # Gold
    gold.process_gold_label_store(snapshotdate, SILVER["lms"], GOLD_LABEL, spark,
                                  dpd=DPD_THRESHOLD, mob=MOB_THRESHOLD)
    gold.process_gold_feature_store(snapshotdate, SILVER["clickstream"], SILVER["attributes"],
                                    SILVER["financials"], GOLD_FEATURE, spark)

    spark.stop()
    print(f"--- data pipeline complete for {snapshotdate} ---")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    main(ap.parse_args().snapshotdate)
