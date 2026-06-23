import os
import re
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col, regexp_replace, when
from pyspark.sql.types import (DateType, FloatType, IntegerType, StringType)


# LMS loan daily
def process_silver_lms(snapshot_date_str, bronze_directory, silver_directory, spark):
    partition_name = f"bronze_lms_loan_daily_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver:lms] loaded {filepath} row count: {df.count()}")

    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Getting MOB from installment_num
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # Getting DPD using overdue amount and due amount
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    out_name = f"silver_lms_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(silver_directory, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print(f"[silver:lms] saved to {out_path}")
    return df


# Clickstream features
def process_silver_clickstream(snapshot_date_str, bronze_directory, silver_directory, spark):
    partition_name = (f"bronze_feature_clickstream_{snapshot_date_str.replace('-', '_')}.csv")
    filepath = os.path.join(bronze_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver:clickstream] loaded {filepath} row count: {df.count()}")

    # Enforce schema: 20 numeric features + Customer_ID + snapshot_date
    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", col(f"fe_{i}").cast(IntegerType()))
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # Ensuring that rows with no customer ID is dropped
    df = df.dropna(subset=["Customer_ID"])

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    out_name = (
        f"silver_feature_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    )
    out_path = os.path.join(silver_directory, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print(f"[silver:clickstream] saved to {out_path}")
    return df


# Customer attributes
def process_silver_attributes(snapshot_date_str, bronze_directory, silver_directory, spark):
    partition_name = (f"bronze_features_attributes_{snapshot_date_str.replace('-', '_')}.csv")
    filepath = os.path.join(bronze_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver:attributes] loaded {filepath} row count: {df.count()}")

    # Removing non-digits from Age column and removing nonsensical values i.e. <0 or >100
    df = df.withColumn("Age", regexp_replace(col("Age"), "[^0-9]", ""))
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age", when((col("Age") > 0) & (col("Age") <= 100), col("Age")))

    # Occupation has placeholders to be removed
    df = df.withColumn("Occupation", when(col("Occupation").rlike("^_+$"), None).otherwise(col("Occupation")))

    # Drop direct PII (Name, SSN) as they are not useful for modelling
    keep_cols = ["Customer_ID", "Age", "Occupation", "snapshot_date"]
    df = df.select(*keep_cols)
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    out_name = (f"silver_features_attributes_{snapshot_date_str.replace('-', '_')}.parquet")
    out_path = os.path.join(silver_directory, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print(f"[silver:attributes] saved to {out_path}")
    return df


# Customer financials
# Helper functions
def _strip_underscore_to_float(df, c):
    df = df.withColumn(c, regexp_replace(col(c), "_", ""))
    df = df.withColumn(c, col(c).cast(FloatType()))
    return df

def _credit_history_age_to_months(s):
    if s is None:
        return None
    m = re.match(r"\s*(\d+)\s*Years?\s*and\s*(\d+)\s*Months?\s*", s)
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))


def process_silver_financials(snapshot_date_str, bronze_directory, silver_directory, spark):
    partition_name = (f"bronze_features_financials_{snapshot_date_str.replace('-', '_')}.csv")
    filepath = os.path.join(bronze_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver:financials] loaded {filepath} row count: {df.count()}")

    # Removing parts of columns with dirty entry and cast to float
    dirty_numeric_cols = [
        "Annual_Income",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]
    for c in dirty_numeric_cols:
        df = _strip_underscore_to_float(df, c)

    # Clean columns
    df = df.withColumn("Monthly_Inhand_Salary", col("Monthly_Inhand_Salary").cast(FloatType()))
    df = df.withColumn("Num_Bank_Accounts", col("Num_Bank_Accounts").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Card", col("Num_Credit_Card").cast(IntegerType()))
    df = df.withColumn("Interest_Rate", col("Interest_Rate").cast(IntegerType()))
    df = df.withColumn("Delay_from_due_date", col("Delay_from_due_date").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Inquiries", col("Num_Credit_Inquiries").cast(IntegerType()))
    df = df.withColumn("Credit_Utilization_Ratio", col("Credit_Utilization_Ratio").cast(FloatType()))
    df = df.withColumn("Total_EMI_per_month", col("Total_EMI_per_month").cast(FloatType()))

    # Categorical clean-up (Credit_Mix and Payment_of_Min_Amount)
    df = df.withColumn("Credit_Mix", when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")))
    df = df.withColumn("Payment_of_Min_Amount", when(col("Payment_of_Min_Amount") == "NM", None).otherwise(col("Payment_of_Min_Amount")))

    # Clean Credit_History_Age column
    parse_history_age = F.udf(_credit_history_age_to_months, IntegerType())
    df = df.withColumn("Credit_History_Months", parse_history_age(col("Credit_History_Age")))
    df = df.drop("Credit_History_Age")

    # Counting how many types of loan to keep as a feature instead of the initial free-text with high-cardinality
    df = df.withColumn("Num_Loan_Types_Listed", F.when(col("Type_of_Loan").isNull(), F.lit(0)).otherwise(F.size(F.split(col("Type_of_Loan"), ","))).cast(IntegerType()))

    # Key types
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # Additional guards for fields with known dirty values
    df = df.withColumn("Num_Bank_Accounts", when((col("Num_Bank_Accounts") >= 0) & (col("Num_Bank_Accounts") <= 20), col("Num_Bank_Accounts")))
    df = df.withColumn("Num_Credit_Card", when((col("Num_Credit_Card") >= 0) & (col("Num_Credit_Card") <= 20), col("Num_Credit_Card")))
    df = df.withColumn("Num_of_Loan", when((col("Num_of_Loan") >= 0) & (col("Num_of_Loan") <= 20), col("Num_of_Loan")))
    df = df.withColumn("Interest_Rate", when((col("Interest_Rate") >= 0) & (col("Interest_Rate") <= 100), col("Interest_Rate")))

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    out_name = (
        f"silver_features_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    )
    out_path = os.path.join(silver_directory, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print(f"[silver:financials] saved to {out_path}")
    return df
