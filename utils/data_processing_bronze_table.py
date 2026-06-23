import os
from datetime import datetime

from pyspark.sql.functions import col


# Helper function
def _process_bronze(snapshot_date_str: str, bronze_directory: str, spark, csv_file_path: str, table_prefix: str):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()

    df = (spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col("snapshot_date") == snapshot_date))

    print(f"[bronze:{table_prefix}] {snapshot_date_str} row count: {df.count()}")

    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

    partition_name = f"{table_prefix}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)

    df.toPandas().to_csv(filepath, index=False)
    print(f"[bronze:{table_prefix}] saved to: {filepath}")

    return df


# Map each source to bronze file
def process_bronze_lms(snapshot_date_str, bronze_directory, spark):
    return _process_bronze(snapshot_date_str, bronze_directory, spark, csv_file_path="data/lms_loan_daily.csv", table_prefix="bronze_lms_loan_daily")


def process_bronze_clickstream(snapshot_date_str, bronze_directory, spark):
    return _process_bronze(snapshot_date_str, bronze_directory, spark, csv_file_path="data/feature_clickstream.csv", table_prefix="bronze_feature_clickstream")


def process_bronze_attributes(snapshot_date_str, bronze_directory, spark):
    return _process_bronze(snapshot_date_str, bronze_directory, spark, csv_file_path="data/features_attributes.csv", table_prefix="bronze_features_attributes")


def process_bronze_financials(snapshot_date_str, bronze_directory, spark):
    return _process_bronze(snapshot_date_str, bronze_directory, spark, csv_file_path="data/features_financials.csv", table_prefix="bronze_features_financials")
