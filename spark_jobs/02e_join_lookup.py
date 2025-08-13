# spark_jobs/02e_join_lookup.py
# Simple join: payment_type -> label + aggregations

import os
from pyspark.sql import SparkSession, functions as F, types as T

# --- Paths (use the SAMPLE for testing) ---
INPUT_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"

# Base output path (saved alongside the sample)
OUT_BASE  = INPUT_DIR  # guardamos junto a la muestra

def main():
    # Initialize Spark session
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-02e-join-lookup")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Read ALL CSV files inside the sample folder
    df = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(INPUT_DIR))

    # Normalize expected column names (in case they come with aliases from 02b/02d)
    # Columns we expect: VendorID, tpep_pickup_datetime, tpep_dropoff_datetime,
    # trip_distance, fare_amount, tip_amount, total_amount, payment_type
    cols_needed = ["VendorID","tpep_pickup_datetime","tpep_dropoff_datetime",
                   "trip_distance","fare_amount","tip_amount","total_amount","payment_type"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(">>> Warning: Missing columns in the input:", missing)

    # --- Lookup for payment_type ---
    # Reference: NYC TLC (may vary by year, but this mapping is generally valid)
    # 1=Credit card, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown, 6=Voided trip
    lookup_data = [
        (1, "Credit card"),
        (2, "Cash"),
        (3, "No charge"),
        (4, "Dispute"),
        (5, "Unknown"),
        (6, "Voided trip"),
    ]
    schema = T.StructType([
        T.StructField("payment_type", T.IntegerType(), False),
        T.StructField("payment_label", T.StringType(), False),
    ])
    payment_lookup = spark.createDataFrame(lookup_data, schema)

    # Left join to keep all rows even if payment_type has no match
    dfj = (df
           .withColumn("payment_type", F.col("payment_type").cast("int"))
           .join(F.broadcast(payment_lookup), on="payment_type", how="left"))

    # --- Example aggregations (fulfilling "joins + aggregations" requirement) ---
    print("\n>>> Trips by payment method:")
    by_pay = (dfj.groupBy("payment_label")
                .agg(F.count("*").alias("trips"),
                     F.round(F.avg("fare_amount"), 2).alias("avg_fare"))
                .orderBy(F.desc("trips")))
    by_pay.show(truncate=False)

    # Buckets by distance with payment label (useful for EDA)
    buckets = F.when(F.col("trip_distance") < 1, "<1") \
               .when((F.col("trip_distance") >= 1) & (F.col("trip_distance") < 3), "1-3") \
               .when((F.col("trip_distance") >= 3) & (F.col("trip_distance") < 5), "3-5") \
               .when((F.col("trip_distance") >= 5) & (F.col("trip_distance") < 10), "5-10") \
               .otherwise(">=10")

    by_bucket_pay = (dfj.withColumn("bucket", buckets)
                       .groupBy("payment_label", "bucket")
                       .agg(F.count("*").alias("trips"),
                            F.round(F.avg("fare_amount"), 2).alias("avg_fare"))
                       .orderBy("payment_label", "bucket"))
    print("\n>>> Trips by payment method and distance bucket:")
    by_bucket_pay.show(20, truncate=False)

    # --- Save results as CSV (small folders inside the sample) ---
    out1 = os.path.join(OUT_BASE, "joins", "trips_by_payment")
    out2 = os.path.join(OUT_BASE, "joins", "trips_by_payment_bucket")
    (by_pay.coalesce(1)
          .write.mode("overwrite").option("header", "true").csv(out1))
    (by_bucket_pay.coalesce(1)
          .write.mode("overwrite").option("header", "true").csv(out2))
    print(f"\n>>> Saved:\n  - {out1}\n  - {out2}")

    spark.stop()

if __name__ == "__main__":
    main()
