# spark_jobs/05c_make_report.py
# Exporta KPIs tabulares (CSV) para la etapa 5. Si faltan columnas derivadas, las calcula.

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F, types as T

# === Paths ===
SAMPLE_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
IN_FE      = os.path.join(SAMPLE_DIR, "features", "fe_sample_2015_01_csv")  # dataset limpio (03a)
RUN        = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = os.path.join(SAMPLE_DIR, "reports", f"report_{RUN}")

# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def with_if_missing(df, col_name, expr_col):
    """Adds the column if it does not exist."""
    if col_name not in df.columns:
        df = df.withColumn(col_name, expr_col)
    return df

def bucket_distance(mi):
    if mi is None:
        return "<1mi"
    if mi < 1:    return "<1mi"
    if mi < 2:    return "1-2mi"
    if mi < 5:    return "2-5mi"
    if mi < 10:   return "5-10mi"
    return ">=10mi"

bucket_udf = F.udf(bucket_distance, T.StringType())

# ----------------------------------------------------------
# KPIs
# ----------------------------------------------------------
def compute_kpis(df):
    """
    Ensures minimum derived columns and calculates several KPI tables.
    Returns a dict with DataFrames ready for export.
    """

    # 1) Ensure types and basic columns are present
    #    (payment_type, fare_amount, total_amount, tip_amount, trip_distance, pickup_datetime)
    #    Cast to double where applicable
    for c in ["fare_amount", "total_amount", "tip_amount", "trip_distance"]:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))

    # Ensure datetime → pickup_hour, pickup_dow
    # If it comes as a string yyyy-MM-dd HH:mm:ss, infer it
    if "pickup_datetime" in df.columns:
        df = with_if_missing(df, "pickup_hour",
                             F.hour(F.to_timestamp("pickup_datetime")))
        # dayofweek: 1=Sunday in Spark SQL → map to 1..7 (Mon=1,…,Sun=7)
        dow = F.date_format(F.to_timestamp("pickup_datetime"), "u").cast("int")  # 1..7 (1=Lunes)
        df = with_if_missing(df, "pickup_dow", dow)
    else:
        # If there is no datetime, at least avoid failures
        df = with_if_missing(df, "pickup_hour", F.lit(0).cast("int"))
        df = with_if_missing(df, "pickup_dow", F.lit(1).cast("int"))

    # is_weekend (Saturday=6, Sunday=7)
    df = with_if_missing(
        df,
        "is_weekend",
        (F.col("pickup_dow").isin([6, 7])).cast("int")
    )

    # tip_rate (avoid division by 0)
    if ("tip_amount" in df.columns) and ("total_amount" in df.columns):
        df = with_if_missing(
            df,
            "tip_rate",
            F.when(F.col("total_amount") > 0, F.col("tip_amount") / F.col("total_amount")).otherwise(F.lit(0.0))
        )
    else:
        df = with_if_missing(df, "tip_rate", F.lit(0.0).cast("double"))

    # Average fare (for clarity)
    df = with_if_missing(df, "fare_amount", F.lit(0.0).cast("double"))

    # 2) KPI Tables

    # a) By payment type
    kpi_by_payment = (
        df.groupBy("payment_type")
          .agg(
              F.count("*").alias("trips"),
              F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
              F.round(F.avg("total_amount"), 2).alias("avg_total")
          )
          .orderBy(F.col("trips").desc())
    )

    # b) By hour of the day
    kpi_by_hour = (
        df.groupBy("pickup_hour")
          .agg(
              F.count("*").alias("trips"),
              F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
          )
          .orderBy("pickup_hour")
    )

    # c) By distance "bucket"
    df_b = df.withColumn("dist_bucket", bucket_udf(F.col("trip_distance")))
    kpi_by_distance = (
        df_b.groupBy("dist_bucket")
            .agg(
                F.count("*").alias("trips"),
                F.round(F.avg("total_amount"), 2).alias("avg_total")
            )
            .orderBy(F.col("trips").desc())
    )

    # d) Heatmap (dow, hour) with average total
    kpi_heatmap = (
        df.groupBy("pickup_dow", "pickup_hour")
          .agg(
              F.count("*").alias("trips"),
              F.round(F.avg("total_amount"), 2).alias("avg_total")
          )
          .orderBy("pickup_dow", "pickup_hour")
    )

    # e) Tip rate by payment_type (useful for 05b/visual)
    kpi_tip_by_pay = (
        df.groupBy("payment_type")
          .agg(
              F.count("*").alias("trips"),
              F.round(F.avg("tip_rate"), 4).alias("avg_tip_rate")
          )
          .orderBy(F.col("trips").desc())
    )

    return {
        "kpi_by_payment_type": kpi_by_payment,
        "kpi_by_pickup_hour": kpi_by_hour,
        "kpi_by_distance_bucket": kpi_by_distance,
        "kpi_heatmap_dow_hour": kpi_heatmap,
        "kpi_tip_rate_by_payment_type": kpi_tip_by_pay,
    }

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-05c-make-report")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    ensure_dir(OUT_DIR)

    # Load clean dataset
    df = (
        spark.read
             .option("header", "true")
             .option("inferSchema", "true")
             .csv(IN_FE)
    )

    kpis = compute_kpis(df)

    # Export each KPI to CSV (one file per table)
    for name, dfi in kpis.items():
        out = os.path.join(OUT_DIR, f"{name}.csv")
        (
            dfi.coalesce(1)
               .write.mode("overwrite")
               .option("header", "true")
               .csv(out)
        )
        print(f"Saved: {out}")

    print(f"\nDone. Tabular reports saved in: {OUT_DIR}")

    spark.stop()

if __name__ == "__main__":
    main()
