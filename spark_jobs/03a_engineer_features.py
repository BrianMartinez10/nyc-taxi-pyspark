# spark_jobs/03a_engineer_features.py
import os
from pyspark.sql import SparkSession, functions as F, types as T

# === Config ===
USE_SAMPLE = True  # continue using the sample dataset
SAMPLE_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
OUT_BASE   = os.path.join(SAMPLE_DIR, "features")

def main():
    # Initialize Spark session
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-03a-engineer-features")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    
    # Load cleaned sample CSV
    df = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(SAMPLE_DIR))

    # --- UDF: weekend flag ---
    @F.udf(T.IntegerType())
    def is_weekend(dow: int) -> int:
        """
        Returns 1 if the day of week is Saturday (7) or Sunday (1) according to Spark's dayofweek():
        1=Sunday, 2=Monday, ..., 7=Saturday. Otherwise returns 0.
        """
        if dow is None:
            return 0
        return 1 if dow in (1, 7) else 0

    # Time-based fields
    df = (df
          .withColumn("pickup_ts",  F.to_timestamp("tpep_pickup_datetime"))
          .withColumn("dropoff_ts", F.to_timestamp("tpep_dropoff_datetime"))
          .withColumn("pickup_hour",  F.hour("pickup_ts"))
          .withColumn("pickup_dow",   F.dayofweek("pickup_ts"))
          .withColumn("is_weekend",   is_weekend(F.col("pickup_dow")))
    )

    # Trip duration (in minutes) and average speed (mph)
    df = (df
          .withColumn("duration_min",
                      (F.col("dropoff_ts").cast("long") - F.col("pickup_ts").cast("long")) / 60.0)
          # Replace invalid or negative durations with null
          .withColumn("duration_min",
                      F.when(F.col("duration_min") <= 0, None).otherwise(F.col("duration_min")))
          .withColumn("avg_speed_mph",
                      F.when(F.col("duration_min").isNull(), None)
                       .otherwise(F.col("trip_distance") / (F.col("duration_min") / 60.0)))
    )

    # Tip rate (propina relativa)
    df = df.withColumn(
        "tip_rate",
        F.when((F.col("fare_amount") > 0) & F.col("tip_amount").isNotNull(),
               F.col("tip_amount") / F.col("fare_amount"))
         .otherwise(0.0)
    )

    # --- Example RDD -> DF: night flag (22:00–05:59) ---
    base_cols = ["VendorID","payment_type","trip_distance","fare_amount","tip_amount","total_amount",
                 "pickup_ts","dropoff_ts","pickup_hour","pickup_dow","is_weekend","duration_min",
                 "avg_speed_mph","tip_rate"]
    # Add night_flag using RDD map
    rdd = (df.select(base_cols)
             .rdd
             .map(lambda r: r + (1 if (r["pickup_hour"] is not None and (r["pickup_hour"] >= 22 or r["pickup_hour"] <= 5)) else 0,))
          )
    # Build new schema with added night_flag
    new_schema = T.StructType(df.select(base_cols).schema.fields + [T.StructField("night_flag", T.IntegerType(), False)])
    df_fe = spark.createDataFrame(rdd, new_schema)

    print("\n>>> Feature view:")
    df_fe.select("trip_distance","duration_min","avg_speed_mph","pickup_hour","pickup_dow","is_weekend","night_flag","tip_rate") \
         .show(10, truncate=False)

    # Save features (small CSV in sample)
    out_path = os.path.join(OUT_BASE, "fe_sample_2015_01_csv")
    (df_fe
        .limit(100_000)     # demo limit; remove for full dataset
        .coalesce(1)        # single CSV for sample
        .write.mode("overwrite").option("header","true").csv(out_path))
    print(f"\n>>> Features saved at: {out_path}")

    spark.stop()

if __name__ == "__main__":
    main()
