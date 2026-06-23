## Helper file for conversion of spark to pandas
import glob
import os
import pyspark


def get_spark(name="ml_pipeline"):
    spark = (pyspark.sql.SparkSession.builder
             .appName(name).master("local[*]").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _read_all(spark, folder):
    files = glob.glob(os.path.join(folder, "*.parquet"))
    if not files:
        return None
    return spark.read.parquet(*files)


def load_label_store(spark, folder="datamart/gold/label_store"):
    sdf = _read_all(spark, folder)
    return None if sdf is None else sdf.toPandas()


def load_feature_store(spark, folder="datamart/gold/feature_store"):
    sdf = _read_all(spark, folder)
    return None if sdf is None else sdf.toPandas()
