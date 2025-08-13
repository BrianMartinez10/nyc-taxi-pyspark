# spark_jobs/05a_sql_insights.py
import os
from pyspark.sql import SparkSession, functions as F

# === Path to the cleaned dataset (the one you used in 05a that already worked) ===
# Current option (CSV of features generated in 03a; it worked in your run):
SAMPLE_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
IN_FE = os.path.join(SAMPLE_DIR, "features", "fe_sample_2015_01_csv")

def main():
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-05a-sql-insights")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # -- Load (CSV that already worked for you) --
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(IN_FE)
    )
    df.createOrReplaceTempView("trips")

    # 1) Summary by payment type
    by_pay = spark.sql("""
        SELECT
            payment_type,
            COUNT(*) AS trips,
            ROUND(AVG(fare_amount), 2) AS avg_fare,
            ROUND(AVG(total_amount), 2) AS avg_total
        FROM trips
        GROUP BY payment_type
        ORDER BY trips DESC
    """)
    by_pay.show(truncate=False)

    # 2) Trips and average fare by hour of the day
    by_hour = spark.sql("""
        SELECT
            pickup_hour,
            COUNT(*) AS trips,
            ROUND(AVG(fare_amount), 2) AS avg_fare
        FROM trips
        GROUP BY pickup_hour
        ORDER BY pickup_hour
    """)
    by_hour.show(24, truncate=False)

    # 3) Weekend vs. weekday comparison
    weekend = spark.sql("""
        SELECT
            is_weekend,
            COUNT(*) AS trips,
            ROUND(AVG(total_amount), 2) AS avg_total,
            ROUND(AVG(tip_rate), 3) AS avg_tip_rate
        FROM trips
        GROUP BY is_weekend
        ORDER BY is_weekend DESC
    """)
    # <- Fix: specify 'truncate' by name or pass 'n' as int first
    weekend.show(truncate=False)

    # 4) Top 10 combinations of hour × day of the week by average fare
    hour_dow = spark.sql("""
        SELECT
            pickup_dow,
            pickup_hour,
            COUNT(*) AS trips,
            ROUND(AVG(total_amount), 2) AS avg_total
        FROM trips
        GROUP BY pickup_dow, pickup_hour
        HAVING COUNT(*) > 50
        ORDER BY avg_total DESC
        LIMIT 10
    """)
    hour_dow.show(truncate=False)

    # 5) Distance distribution (simple buckets)
    buckets = spark.sql("""
        SELECT
            CASE
                WHEN trip_distance < 1 THEN '<1mi'
                WHEN trip_distance < 2 THEN '1-2mi'
                WHEN trip_distance < 5 THEN '2-5mi'
                WHEN trip_distance < 10 THEN '5-10mi'
                ELSE '>=10mi'
            END AS dist_bucket,
            COUNT(*) AS trips,
            ROUND(AVG(total_amount), 2) AS avg_total
        FROM trips
        GROUP BY
            CASE
                WHEN trip_distance < 1 THEN '<1mi'
                WHEN trip_distance < 2 THEN '1-2mi'
                WHEN trip_distance < 5 THEN '2-5mi'
                WHEN trip_distance < 10 THEN '5-10mi'
                ELSE '>=10mi'
            END
        ORDER BY trips DESC
    """)
    buckets.show(truncate=False)

    # 6) Tip rate by payment type
    tip_by_pay = spark.sql("""
        SELECT
            payment_type,
            COUNT(*) AS trips,
            ROUND(AVG(tip_rate), 4) AS avg_tip_rate
        FROM trips
        GROUP BY payment_type
        ORDER BY avg_tip_rate DESC
    """)
    tip_by_pay.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
