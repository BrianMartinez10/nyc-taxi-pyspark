import os
from pyspark.sql import SparkSession, functions as F

# Paths
RAW = r"C:\TaxiProject\data\raw\yellow_tripdata_2015-01.csv"
OUT_DIR = r"C:\TaxiProject\data\processed"


def main():
    # Initialize Spark session
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-clean-eda-stable")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")   # fewer shuffle partitions for speed
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # 1) Read raw CSV
    df = (spark.read
          .option("header","true")
          .option("inferSchema","true")
          .option("timestampFormat","yyyy-MM-dd HH:mm:ss")
          .csv(RAW))

    # 2) Create derived columns and safe type casting
    df = (df
        .withColumn("pickup_ts",  F.to_timestamp("tpep_pickup_datetime"))
        .withColumn("dropoff_ts", F.to_timestamp("tpep_dropoff_datetime"))
        .withColumn("trip_minutes",
            F.when(F.col("dropoff_ts") > F.col("pickup_ts"),
                   (F.col("dropoff_ts").cast("long") - F.col("pickup_ts").cast("long"))/60.0))
        .withColumn("fare_amount", F.col("fare_amount").cast("double"))
        .withColumn("trip_distance", F.col("trip_distance").cast("double"))
        .withColumn("tip_amount", F.col("tip_amount").cast("double"))
        .withColumn("total_amount", F.col("total_amount").cast("double"))
        .withColumn("fare_per_mile",
            F.when(F.col("trip_distance") > 0, F.col("fare_amount")/F.col("trip_distance")))
        .withColumn("hour", F.hour("pickup_ts"))
        .withColumn("day",  F.to_date("pickup_ts"))
    )

    # 3) Conservative cleaning rules
    cleaned = (
        df
        # Valid passenger count
        .filter((F.col("passenger_count") >= 1) & (F.col("passenger_count") <= 6))
        # Valid trip distances
        .filter((F.col("trip_distance") > 0) & (F.col("trip_distance") <= 100))
        # Reasonable fare and amount limits
        .filter((F.col("fare_amount") >= 0) & (F.col("fare_amount") <= 500))
        .filter((F.col("total_amount") >= 0) & (F.col("total_amount") <= 1000))
        .filter((F.col("tip_amount") >= 0) & (F.col("tip_amount") <= 200))
        # Trip duration between 1 minute and 4 hours
        .filter((F.col("trip_minutes") >= 1) & (F.col("trip_minutes") <= 240))
        # Valid pickup/dropoff timestamps
        .filter(F.col("pickup_ts").isNotNull() & F.col("dropoff_ts").isNotNull()
                & (F.col("dropoff_ts") >= F.col("pickup_ts")))
        # Approximate NYC latitude/longitude bounds
        .filter(F.col("pickup_latitude").between(40, 42))
        .filter(F.col("dropoff_latitude").between(40, 42))
        .filter(F.col("pickup_longitude").between(-75, -72))
        .filter(F.col("dropoff_longitude").between(-75, -72))
        # Remove duplicates based on key trip identifiers
        .dropDuplicates(["tpep_pickup_datetime","tpep_dropoff_datetime",
                         "pickup_longitude","pickup_latitude",
                         "dropoff_longitude","dropoff_latitude","fare_amount","total_amount"])
    )

    raw_cnt = df.count()
    clean_cnt = cleaned.count()
    print("\n>>> Original vs cleaned counts:", {"raw_rows": raw_cnt, "clean_rows": clean_cnt, "removed": raw_cnt - clean_cnt})

    # 4) Nulls per column (single aggregation to avoid OOM)
    null_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cleaned.columns]
    nulls_row = cleaned.agg(*null_exprs).collect()[0].asDict()
    print("\n>>> Null values per column (top 12):")
    for k, v in sorted(nulls_row.items(), key=lambda x: -x[1])[:12]:
        print(f"{k:25s}: {v:,}")

    # 5) Basic statistics and light quantiles
    print("\n>>> Describe:")
    cleaned.select("trip_distance","fare_amount","tip_amount","total_amount",
                   "trip_minutes","fare_per_mile").describe().show()

    for col in ["trip_distance","fare_amount","tip_amount","trip_minutes","fare_per_mile"]:
        q = cleaned.approxQuantile(col, [0.5, 0.9, 0.99], 0.01)
        print(f"Quantiles {col:14s} -> median:{q[0]:.2f}, p90:{q[1]:.2f}, p99:{q[2]:.2f}")

    # 6) Quick EDA
    print("\n>>> Trips by pickup hour (0-23):")
    cleaned.groupBy("hour").count().orderBy("hour").show(24)

    print("\n>>> Distribution by payment_type:")
    cleaned.groupBy("payment_type").count().orderBy(F.desc("count")).show()

    # 7) Save cleaned sample to CSV (optional)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "clean_sample_2015_01_csv")
    (cleaned
        .select("VendorID","pickup_ts","dropoff_ts","passenger_count","trip_distance",
                "fare_amount","tip_amount","total_amount","trip_minutes","fare_per_mile",
                "hour","day","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude")
        .limit(100_000) # For demo purposes;
        .coalesce(1)    # Single CSV file (OK for sample, avoid for full data)
        .write.mode("overwrite").option("header","true").csv(out_csv))
    print(f"\n>>> Clean sample saved at: {out_csv}")

    spark.stop()

if __name__ == "__main__":
    main()


