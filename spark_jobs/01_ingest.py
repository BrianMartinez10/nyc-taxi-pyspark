# spark_jobs/01_ingest.py
# Stage 1: Ingestion & Storage (CSV only, no Parquet)
# Run with:
#   C:\Users\brian\anaconda3\envs\spark311\python.exe spark_jobs\01_ingest.py

import os
from pyspark.sql import SparkSession, Row, functions as F

# Paths
RAW = r"C:\TaxiProject\data\raw\yellow_tripdata_2015-01.csv"
PROCESSED_DIR = r"C:\TaxiProject\data\processed"

def main():
    # Initialize Spark session
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-ingest-csv-only")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n>>> Reading large CSV file: {RAW}")
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
          .csv(RAW))

    print("\n>>> Detected schema:")
    df.printSchema()

    print("\n>>> First 5 rows sample:")
    df.show(5, truncate=False)

    # ------- Requirement 2: RDDs (filter / map / reduce) -------
    rdd = df.rdd

    total_rows  = rdd.count()
    long_trips  = rdd.filter(lambda r: (r.trip_distance or 0.0) > 10.0).count()
    sum_fares   = rdd.map(lambda r: float(r.fare_amount or 0.0)).sum()
    max_tip     = rdd.map(lambda r: float(r.tip_amount or 0.0)).max()
    vendor_cnts = (rdd
                   .map(lambda r: (int(r.VendorID) if r.VendorID is not None else -1, 1))
                   .reduceByKey(lambda a, b: a + b)
                   .collect())

    print("\n>>> Metrics via RDD:")
    print(f"Total rows                : {total_rows:,}")
    print(f"Long trips  (>10 miles)   : {long_trips:,}")
    print(f"Sum fare_amount (USD)    : {sum_fares:,.2f}")
    print(f"Maximum tip_amount (USD)   : {max_tip:,.2f}")
    print(f"VendorID → count         : {vendor_cnts}")

    # ------- Requirement 3: Convert RDD → DataFrame -------
    proj_rdd = rdd.map(lambda r: Row(
        vendor=int(r.VendorID) if r.VendorID is not None else None,
        pickup=r.tpep_pickup_datetime,
        dropoff=r.tpep_dropoff_datetime,
        distance=float(r.trip_distance) if r.trip_distance is not None else None,
        fare=float(r.fare_amount) if r.fare_amount is not None else None,
        tip=float(r.tip_amount) if r.tip_amount is not None else None,
        total=float(r.total_amount) if r.total_amount is not None else None
    ))
    df_proj = spark.createDataFrame(proj_rdd)

    print("\n>>> DataFrame view generated from RDD:")
    df_proj.show(5, truncate=False)

    # Quick SQL query over the DataFrame (for the next stage)
    df_proj.createOrReplaceTempView("trips")
    print("\n>>> Basic SQL check:")
    spark.sql("""
        SELECT
            COUNT(*) AS rows,
            MIN(pickup) AS min_pickup,
            MAX(pickup) AS max_pickup
        FROM trips
    """).show(truncate=False)

    # ------- (Optional) Save a sample in CSV for inspection -------
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    sample_out = os.path.join(PROCESSED_DIR, "sample_trips_2015_01_csv")
    print(f"\n>>> Saving sample (100k rows) to: {sample_out}")
    (df_proj
        .limit(100_000)          # avoid writing millions of rows
        .coalesce(1)             # single CSV file for easy inspection
        .write.mode("overwrite")
        .option("header", "true")
        .csv(sample_out))

    print("\n>>> Stage 1 (CSV-only) completed.\n"
          "    - Large file loading \n"
          "    - RDD filter/map/reduce \n"
          "    - RDD → DataFrame for SQL ")

    spark.stop()

if __name__ == "__main__":
    main()
