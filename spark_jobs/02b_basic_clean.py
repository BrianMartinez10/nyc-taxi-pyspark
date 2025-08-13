# spark_jobs/02b_basic_clean.py
# Basic cleaning and metrics

import os
from pyspark.sql import SparkSession, functions as F, types as T

# Paths to raw input and processed output directories
RAW = r"C:\TaxiProject\data\raw\yellow_tripdata_2015-01.csv"
OUT_DIR = r"C:\TaxiProject\data\processed"

def main():
    # Initialize Spark session
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-02b-basic-clean")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")   # lower value for local execution
        .config("spark.driver.memory", "4g")            # opcional, ajusta si tienes >8GB RAM
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
            .csv(RAW))

    # Select and cast only required columns to correct types
    cols = {
        "VendorID": T.IntegerType(),
        "tpep_pickup_datetime": T.TimestampType(),
        "tpep_dropoff_datetime": T.TimestampType(),
        "trip_distance": T.DoubleType(),
        "fare_amount": T.DoubleType(),
        "tip_amount": T.DoubleType(),
        "total_amount": T.DoubleType(),
        "payment_type": T.IntegerType(),
    }
    df2 = df.select([F.col(c).cast(t).alias(c) for c, t in cols.items()])

    print("\n>>> Schema (cleaned):")
    df2.printSchema()

    raw_rows = df2.count()
    print(f"\n>>> Original rows: {raw_rows:,}")

    # Basic filters: avoid heavy operations (no joins or window functions yet)
    cleaned = (
        df2
        # Ensure pickup/dropoff timestamps are present and valid
        .filter(F.col("tpep_pickup_datetime").isNotNull() & F.col("tpep_dropoff_datetime").isNotNull())
        .filter(F.col("trip_distance").isNotNull() & (F.col("trip_distance") >= 0) & (F.col("trip_distance") <= 200))
        # Ensure trip_distance is within valid bounds
        .filter(F.col("fare_amount").isNotNull() & (F.col("fare_amount") >= 0) & (F.col("fare_amount") <= 1000))
        # Ensure fare_amount is within reasonable range
        .filter(F.col("total_amount").isNotNull() & (F.col("total_amount") >= 0) & (F.col("total_amount") <= 2000))
        # Ensure total_amount is within reasonable range
        .filter(F.col("tpep_dropoff_datetime") >= F.col("tpep_pickup_datetime"))  # no tiempos invertidos
    )

    # Simple deduplication based on logical key
    key_cols = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "fare_amount", "trip_distance"]
    cleaned = cleaned.dropDuplicates(key_cols)

    clean_rows = cleaned.count()
    print(f">>> Clean rows: {clean_rows:,}  (removed: {raw_rows - clean_rows:,})")

    # Quick, low-cost metrics
    cleaned.select(
        F.count("*").alias("rows"),
        F.sum("fare_amount").alias("sum_fare"),
        F.sum("total_amount").alias("sum_total"),
        F.mean("trip_distance").alias("avg_distance"),
        F.expr("percentile_approx(trip_distance, 0.95)").alias("p95_distance"),
    ).show(truncate=False)

    # Top 5 payment types
    (cleaned
        .groupBy("payment_type")
        .count()
        .orderBy(F.desc("count"))
        .show(5, truncate=False))

    # Save a sample to CSV for inspection
    os.makedirs(OUT_DIR, exist_ok=True)
    sample_out = os.path.join(OUT_DIR, "clean_sample_2015_01_csv")
    (cleaned
        .limit(100_000)             # limit to sample size for demo
        .coalesce(1)                # single CSV file (OK for small sample; avoid for full data)
        .write.mode("overwrite")    
        .option("header", "true")
        .csv(sample_out))
    print(f">>> Sample saved at: {sample_out}")

    spark.stop()

if __name__ == "__main__":
    main()
