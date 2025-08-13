# spark_jobs/04a_reg_total_lr_PIPE.py
import os, json
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# ===== Paths =====
SAMPLE_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
IN_FE      = os.path.join(SAMPLE_DIR, "features", "fe_sample_2015_01_csv")
OUT_DIR    = os.path.join(SAMPLE_DIR, "models", "reg_total_lr_FINAL")

def main():
    # --- Spark session ---
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-04a-reg-total-lr-PIPE")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # --- Load data ---
    df = (spark.read.option("header", "true").option("inferSchema", "true").csv(IN_FE))

    # Label
    df = df.withColumn("label_total", F.col("total_amount").cast("double"))

    # Keep only valid durations (avoid impossible negative/zero times)
    df = df.filter(F.col("duration_min").isNull() | (F.col("duration_min") > 0))

    # ---- Pipeline definition (ALL preprocessing inside the pipeline) ----
    # Numeric feature list used in training and scoring
    numeric_cols = [
        "trip_distance", "duration_min", "avg_speed_mph",
        "pickup_hour", "pickup_dow", "is_weekend", "night_flag", "tip_rate"
    ]

    # 1) Categorical encoding
    idx = StringIndexer(inputCol="payment_type", outputCol="payment_type_idx", handleInvalid="keep")
    ohe = OneHotEncoder(inputCols=["payment_type_idx"], outputCols=["payment_type_ohe"])

    # 2) Imputation INSIDE the pipeline (median for numeric)
    imputer = Imputer(strategy="median", inputCols=numeric_cols, outputCols=numeric_cols)

    # 3) Assemble + scale
    assembler = VectorAssembler(
        inputCols=numeric_cols + ["payment_type_ohe"],
        outputCol="features_raw",
        handleInvalid="keep",
    )
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

    # 4) Linear Regression
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label_total",
        maxIter=50,
        regParam=0.1,
        elasticNetParam=0.0,
        standardization=False,  # ya escalamos nosotros
    )

    pipe = Pipeline(stages=[idx, ohe, imputer, assembler, scaler, lr])

    # --- Train / Test split ---
    train, test = df.dropna(subset=["label_total"]).randomSplit([0.8, 0.2], seed=42)

    # --- Fit & Predict ---
    model = pipe.fit(train)
    pred  = model.transform(test)

    # --- Metrics ---
    rmse = RegressionEvaluator(labelCol="label_total", predictionCol="prediction", metricName="rmse").evaluate(pred)
    mae  = RegressionEvaluator(labelCol="label_total", predictionCol="prediction", metricName="mae").evaluate(pred)
    r2   = RegressionEvaluator(labelCol="label_total", predictionCol="prediction", metricName="r2").evaluate(pred)
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    # --- Save model & artifacts ---
    os.makedirs(OUT_DIR, exist_ok=True)
    model.write().overwrite().save(os.path.join(OUT_DIR, "model"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f, indent=2)

    (pred.select("prediction", "label_total")
         .limit(10_000)
         .coalesce(1)
         .write.mode("overwrite").option("header", "true")
         .csv(os.path.join(OUT_DIR, "preds_sample_csv")))

    print(f"Model (Pipeline) saved at -> {os.path.join(OUT_DIR, 'model')}")
    spark.stop()

if __name__ == "__main__":
    main()
