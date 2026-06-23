import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType, StringType


# Label store
def process_gold_label_store(snapshot_date_str, silver_directory, gold_label_store_directory, spark, dpd: int = 30, mob: int = 6):
    partition_name = f"silver_lms_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_directory, partition_name)
    df = spark.read.parquet(filepath)
    print(f"[gold:label] loaded {filepath} row count: {df.count()}")

    # Keep only loans observed at the target MOB==6
    df = df.filter(col("mob") == mob)

    # Binary label: 1 if dpd >= threshold at MOB
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))

    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)

    out_name = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(gold_label_store_directory, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print(f"[gold:label] saved to {out_path}")
    return df


# Feature store
def process_gold_feature_store(snapshot_date_str, silver_clickstream_directory, silver_attributes_directory, silver_financials_directory, gold_feature_store_directory, spark):
    # Left joined from clickstream silver table which has the highest-frequency feature source
    suffix = snapshot_date_str.replace("-", "_")

    click_path = os.path.join(silver_clickstream_directory, f"silver_feature_clickstream_{suffix}.parquet")
    attr_path = os.path.join(silver_attributes_directory, f"silver_features_attributes_{suffix}.parquet")
    fin_path = os.path.join(silver_financials_directory, f"silver_features_financials_{suffix}.parquet")

    df_click = spark.read.parquet(click_path) if os.path.exists(click_path) else None
    df_attr = spark.read.parquet(attr_path) if os.path.exists(attr_path) else None
    df_fin = spark.read.parquet(fin_path) if os.path.exists(fin_path) else None

    if df_click is None:
        print(f"[gold:feature] no clickstream silver for {snapshot_date_str} - skipping")
        return None

    # Joining on Customer_ID from customer's attributes and financials
    if df_attr is not None:
        df_attr_j = df_attr.drop("snapshot_date")
    else:
        df_attr_j = None

    if df_fin is not None:
        df_fin_j = df_fin.drop("snapshot_date")
    else:
        df_fin_j = None

    df = df_click
    if df_attr_j is not None:
        df = df.join(df_attr_j, on="Customer_ID", how="left")
    if df_fin_j is not None:
        df = df.join(df_fin_j, on="Customer_ID", how="left")

    # Feature engineering on the joined table
    # Income to debt ratio
    df = df.withColumn("income_to_debt_ratio", F.when((col("Outstanding_Debt").isNotNull()) & (col("Outstanding_Debt") > 0), col("Annual_Income") / col("Outstanding_Debt")))
    # EMI burden = EMI / monthly inhand salary
    df = df.withColumn("emi_burden", F.when((col("Monthly_Inhand_Salary").isNotNull()) & (col("Monthly_Inhand_Salary") > 0), col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")))

    # Replacing 0 and "unknown" into columns with null values
    numeric_fill_cols = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Credit_History_Months",
        "Num_Loan_Types_Listed",
        "Age",
        "income_to_debt_ratio",
        "emi_burden",
    ]
    numeric_fill_cols = [c for c in numeric_fill_cols if c in df.columns]
    df = df.fillna(0, subset=numeric_fill_cols)

    categorical_fill_cols = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
    categorical_fill_cols = [c for c in categorical_fill_cols if c in df.columns]
    df = df.fillna("Unknown", subset=categorical_fill_cols)

    # Remove free-text column Type_of_Loan
    drop_cols = [c for c in ["Type_of_Loan"] if c in df.columns]
    if drop_cols:
        df = df.drop(*drop_cols)

    # Tables to be restricted from entering feature store
    forbidden = {
        "label",
        "loan_id",
        "loan_start_date",
        "due_amt",
        "paid_amt",
        "overdue_amt",
        "balance",
        "dpd",
        "installments_missed",
        "first_missed_date",
        "mob",
    }
    leaked = [c for c in df.columns if c in forbidden]
    assert not leaked, f"LEAKAGE: gold feature store contains forbidden columns: {leaked}"

    if not os.path.exists(gold_feature_store_directory):
        os.makedirs(gold_feature_store_directory)

    out_name = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(gold_feature_store_directory, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print(f"[gold:feature] saved to {out_path} ({df.count()} rows, {len(df.columns)} cols)")
    return df
