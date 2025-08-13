# spark_jobs/02d_sql_aggs.py
# Aggregations with Spark SQL (trips/day, trips/hour, payment distribution, distance buckets)

import os
from pyspark.sql import SparkSession, functions as F

# Path to the cleaned sample CSV (output from previous cleaning step)
PROC_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
SAMPLE_CLEAN = os.path.join(PROC_DIR, "part-00000-9cfd3e9f-d9b5-484e-a55f-3f716238fd46-c000.csv")
# Output directory for KPI results
OUT_DIR = os.path.join(PROC_DIR, "kpis")

def read_sample(spark):
    """
    Reads the cleaned sample CSV into a Spark DataFrame and casts
    necessary columns to appropriate data types for analysis.
    """
    df = (spark.read.option("header","true").csv(SAMPLE_CLEAN))
    return (df
        .withColumn("VendorID", F.col("VendorID").cast("int"))
        .withColumn("tpep_pickup_datetime", F.to_timestamp("tpep_pickup_datetime"))
        .withColumn("tpep_dropoff_datetime", F.to_timestamp("tpep_dropoff_datetime"))
        .withColumn("trip_distance", F.col("trip_distance").cast("double"))
        .withColumn("fare_amount", F.col("fare_amount").cast("double"))
        .withColumn("tip_amount", F.col("tip_amount").cast("double"))
        .withColumn("total_amount", F.col("total_amount").cast("double"))
        .withColumn("payment_type", F.col("payment_type").cast("int"))
    )

def save(df_, name):
    """
    Saves the given DataFrame as a single CSV file (coalesced to 1 partition)
    with a specified name in the KPI output directory.
    """
    (df_.coalesce(1)
        .write.mode("overwrite").option("header","true")
        .csv(os.path.join(OUT_DIR, name)))

def main():
    # Initialize Spark session
    spark = (SparkSession.builder
        .appName("nyc-taxi-02d-sql-aggs")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(OUT_DIR, exist_ok=True)
    # Load cleaned sample
    df = read_sample(spark)
    df.createOrReplaceTempView("trips")

    # Trips per day with average distance and fare
    trips_per_day = spark.sql("""
      SELECT DATE(tpep_pickup_datetime) AS day,
             COUNT(*) AS trips,
             ROUND(AVG(trip_distance),2) AS avg_distance,
             ROUND(AVG(fare_amount),2) AS avg_fare
      FROM trips GROUP BY DATE(tpep_pickup_datetime) ORDER BY day
    """)
    trips_per_day.show(7, truncate=False);  save(trips_per_day, "trips_per_day_csv")
    
    # Trips per hour
    trips_per_hour = spark.sql("""
      SELECT HOUR(tpep_pickup_datetime) AS hour, COUNT(*) AS trips
      FROM trips GROUP BY HOUR(tpep_pickup_datetime) ORDER BY hour
    """)
    trips_per_hour.show(24, truncate=False); save(trips_per_hour, "trips_per_hour_csv")

    # Distribution by payment type
    payment_dist = spark.sql("""
      SELECT payment_type, COUNT(*) AS trips
      FROM trips GROUP BY payment_type ORDER BY trips DESC
    """)
    payment_dist.show(truncate=False);       save(payment_dist, "payment_dist_csv")

    # Distance buckets with trip counts and average fare
    distance_buckets = spark.sql("""
      SELECT
        CASE WHEN trip_distance < 1 THEN '<1'
             WHEN trip_distance < 3 THEN '1-3'
             WHEN trip_distance < 5 THEN '3-5'
             WHEN trip_distance < 10 THEN '5-10'
             ELSE '>=10' END AS bucket,
        COUNT(*) AS trips,
        ROUND(AVG(fare_amount),2) AS avg_fare
      FROM trips GROUP BY
        CASE WHEN trip_distance < 1 THEN '<1'
             WHEN trip_distance < 3 THEN '1-3'
             WHEN trip_distance < 5 THEN '3-5'
             WHEN trip_distance < 10 THEN '5-10'
             ELSE '>=10' END
      ORDER BY trips DESC
    """)
    distance_buckets.show(truncate=False);   save(distance_buckets, "distance_buckets_csv")

    spark.stop()

if __name__ == "__main__":
    main()
