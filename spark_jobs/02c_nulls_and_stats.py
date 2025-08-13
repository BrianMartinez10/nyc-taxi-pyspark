# spark_jobs/02c_nulls_and_stats.py
# Null counts per column, quantiles, and quick duplicate check

import os
from pyspark.sql import SparkSession, functions as F

# Path to the cleaned sample CSV (output from previous cleaning step)
PROC_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
SAMPLE_CLEAN = os.path.join(PROC_DIR, "part-00000-9cfd3e9f-d9b5-484e-a55f-3f716238fd46-c000.csv")
# Directory to store EDA outputs
OUT_DIR = os.path.join(PROC_DIR, "eda")

def read_sample(spark):
    """
    Reads the cleaned sample CSV into a Spark DataFrame and casts necessary columns
    to appropriate data types for analysis.
    """
    df = (spark.read.option("header", "true").csv(SAMPLE_CLEAN))
    # Cast columns to correct data types
    casted = (df
        .withColumn("VendorID", F.col("VendorID").cast("int"))
        .withColumn("tpep_pickup_datetime", F.to_timestamp("tpep_pickup_datetime"))
        .withColumn("tpep_dropoff_datetime", F.to_timestamp("tpep_dropoff_datetime"))
        .withColumn("trip_distance", F.col("trip_distance").cast("double"))
        .withColumn("fare_amount", F.col("fare_amount").cast("double"))
        .withColumn("tip_amount", F.col("tip_amount").cast("double"))
        .withColumn("total_amount", F.col("total_amount").cast("double"))
        .withColumn("payment_type", F.col("payment_type").cast("int"))
    )
    return casted

def main():
    # Initialize Spark session
    spark = (SparkSession.builder
        .appName("nyc-taxi-02c-nulls-and-stats")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.driver.memory", "4g")  
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    
    # Read sample dataset
    df = read_sample(spark)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Null counts per column ---
    nulls = [(c, df.filter(F.col(c).isNull()).count()) for c in df.columns]
    nulls_df = spark.createDataFrame(nulls, ["column", "null_count"])
    nulls_df.orderBy(F.desc("null_count")).show(15, truncate=False)
    # Save null counts to CSV
    (nulls_df.coalesce(1)
        .write.mode("overwrite").option("header","true")
        .csv(os.path.join(OUT_DIR, "nulls_per_column_csv")))

    # --- Lightweight quantiles (approximate) ---
    q_cols = ["trip_distance","fare_amount","total_amount","tip_amount"]
    rows = []
    for c in q_cols:
        p = df.approxQuantile(c, [0.05, 0.50, 0.95], 0.01)
        rows.append((c, p[0], p[1], p[2]))
    q_df = spark.createDataFrame(rows, ["column","p05","p50","p95"])
    q_df.show(truncate=False)
    # Save quantiles to CSV
    (q_df.coalesce(1)
        .write.mode("overwrite").option("header","true")
        .csv(os.path.join(OUT_DIR, "quantiles_csv")))

    # --- Duplicate check based on logical key used in 02b_basic_clean.py ---
    key_cols = ["VendorID","tpep_pickup_datetime","tpep_dropoff_datetime","fare_amount","trip_distance"]
    dupe_cnt = (df.groupBy(key_cols).count().filter("count > 1").count())
    print(f">>> Records still duplicated by logical key: {dupe_cnt}")

    spark.stop()

if __name__ == "__main__":
    main()
