# spark_jobs/05b_plots.py
import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F

import matplotlib.pyplot as plt

# === Paths ===
SAMPLE_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"

IN_FE      = os.path.join(SAMPLE_DIR, "features", "fe_sample_2015_01_csv")  # mismo de 05a
RUN        = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = os.path.join(SAMPLE_DIR, "reports", f"plots_{RUN}")
os.makedirs(OUT_DIR, exist_ok=True)

def save_bar(df_pd, x, y, title, xlabel, ylabel, filename, rotate=0):
    plt.figure(figsize=(9,5))
    plt.bar(df_pd[x], df_pd[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved: {path}")

def main():
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-05b-plots")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Load clean sample from 03a (same as 05a)
    df = (spark.read.option("header","true").option("inferSchema","true").csv(IN_FE))

    # 1) Fare/Total by payment_type
    by_pay = (
        df.groupBy("payment_type")
          .agg(F.count("*").alias("trips"),
               F.avg("fare_amount").alias("avg_fare"),
               F.avg("total_amount").alias("avg_total"))
          .orderBy("payment_type")
    )
    save_bar(
        by_pay.toPandas(), x="payment_type", y="trips",
        title="Trips by payment_type", xlabel="payment_type", ylabel="trips",
        filename="01_trips_por_payment_type.png"
    )

    # 2) Trips and avg_fare by hour
    by_hour = (
        df.groupBy("pickup_hour")
          .agg(F.count("*").alias("trips"), F.avg("fare_amount").alias("avg_fare"))
          .orderBy("pickup_hour")
    )
    save_bar(
        by_hour.toPandas(), x="pickup_hour", y="trips",
        title="Trips by hour", xlabel="hour", ylabel="trips",
        filename="02_trips_by_hour.png"
    )
    save_bar(
        by_hour.toPandas(), x="pickup_hour", y="avg_fare",
        title="Average fare by hour", xlabel="hour", ylabel="avg fare (USD)",
        filename="03_avg_fare_by_hora.png"
    )

    # 3) Average tip rate by payment_type
    by_pay_tip = (
        df.groupBy("payment_type")
          .agg(F.count("*").alias("trips"), F.avg("tip_rate").alias("avg_tip_rate"))
          .orderBy("payment_type")
    )
    pd_tip = by_pay_tip.toPandas()
    pd_tip["avg_tip_rate_pct"] = pd_tip["avg_tip_rate"] * 100.0
    save_bar(
        pd_tip, x="payment_type", y="avg_tip_rate_pct",
        title="Average tip rate by payment_type", xlabel="payment_type",
        ylabel="avg tip rate (%)", filename="04_tip_rate_por_payment_type.png"
    )

    # 4) Distance buckets vs avg_total
    by_dist = (
        df.withColumn(
            "dist_bucket",
            F.when(F.col("trip_distance") < 1, "<1mi")
             .when((F.col("trip_distance") >= 1) & (F.col("trip_distance") < 2), "1-2mi")
             .when((F.col("trip_distance") >= 2) & (F.col("trip_distance") < 5), "2-5mi")
             .when((F.col("trip_distance") >= 5) & (F.col("trip_distance") < 10), "5-10mi")
             .otherwise(">=10mi")
        ).groupBy("dist_bucket")
         .agg(F.count("*").alias("trips"), F.avg("total_amount").alias("avg_total"))
    )

    # Manual order so that the X-axis is logical
    order = ["<1mi", "1-2mi", "2-5mi", "5-10mi", ">=10mi"]
    pd_dist = by_dist.toPandas()
    pd_dist["dist_bucket"] = pd_dist["dist_bucket"].astype("category")
    pd_dist["dist_bucket"] = pd_dist["dist_bucket"].cat.set_categories(order, ordered=True)
    pd_dist = pd_dist.sort_values("dist_bucket")

    save_bar(
        pd_dist, x="dist_bucket", y="avg_total",
        title="Average total by distance bucket", xlabel="distance bucket",
        ylabel="avg total (USD)", filename="05_avg_total_by_dist_bucket.png"
    )
    save_bar(
        pd_dist, x="dist_bucket", y="trips",
        title="Trips by distance bucket", xlabel="distance bucket",
        ylabel="trips", filename="06_trips_by_dist_bucket.png"
    )

    print(f"\nDone. Images saved in: {OUT_DIR}")
    spark.stop()

if __name__ == "__main__":
    main()
