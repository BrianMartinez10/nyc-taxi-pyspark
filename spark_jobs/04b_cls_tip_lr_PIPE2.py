# spark_jobs/04b_cls_tip_lr_PIPE2.py
import os
import json
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)

SAMPLE_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
IN_FE      = os.path.join(SAMPLE_DIR, "features", "fe_sample_2015_01_csv")
OUT_DIR    = os.path.join(SAMPLE_DIR, "models", "cls_tip_lr_FINAL")

def main():
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-04b-cls-tip-lr-PIPE2")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.local.dir", r"C:\TaxiProject\tmp")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Load features
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(IN_FE)

    # Label: tip > 0
    if "label_tip" not in df.columns:
        df = df.withColumn("label_tip", (F.col("tip_amount") > 0).cast("int"))

    # Filter out impossible trips (protects avg_speed_mph)
    df = df.filter(F.col("duration_min").isNull() | (F.col("duration_min") > 0))

    # Choose only features that actually exist in the file
    base_numeric = [
        "trip_distance", "duration_min", "avg_speed_mph",
        "pickup_hour", "pickup_dow", "is_weekend", "night_flag",
        "passenger_count", "fare_amount", "tip_rate"
    ]
    numeric_cols = [c for c in base_numeric if c in df.columns]
    if not numeric_cols:
        raise RuntimeError("No numeric feature columns found in the dataset.")

    # Categorical feature (optional)
    has_payment = "payment_type" in df.columns
    stages = []
    if has_payment:
        idx = StringIndexer(inputCol="payment_type", outputCol="payment_type_idx", handleInvalid="keep")
        ohe = OneHotEncoder(inputCols=["payment_type_idx"], outputCols=["payment_type_ohe"])
        stages += [idx, ohe]

    # Impute missing numeric values inside the pipeline
    imputer = Imputer(strategy="median", inputCols=numeric_cols, outputCols=numeric_cols)

    # Assemble and scale features
    input_feats = numeric_cols + (["payment_type_ohe"] if has_payment else [])
    assembler = VectorAssembler(inputCols=input_feats, outputCol="features_raw", handleInvalid="keep")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

    # Logistic Regression model
    lr = LogisticRegression(featuresCol="features", labelCol="label_tip", maxIter=50, regParam=0.1)

    # Full pipeline
    pipe = Pipeline(stages=stages + [imputer, assembler, scaler, lr])

    # Train / Test split
    train, test = df.dropna(subset=["label_tip"]).randomSplit([0.8, 0.2], seed=42)
    model = pipe.fit(train)
    pred  = model.transform(test)

    # Metrics
    auc = BinaryClassificationEvaluator(labelCol="label_tip", rawPredictionCol="rawPrediction",
                                        metricName="areaUnderROC").evaluate(pred)
    acc = MulticlassClassificationEvaluator(labelCol="label_tip", predictionCol="prediction",
                                            metricName="accuracy").evaluate(pred)

    # Save model + metrics
    os.makedirs(OUT_DIR, exist_ok=True)
    model.write().overwrite().save(os.path.join(OUT_DIR, "model"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"auc": float(auc), "accuracy": float(acc)}, f, indent=2)

    print(f"Model saved -> {os.path.join(OUT_DIR, 'model')} | AUC={auc:.3f} ACC={acc:.3f}")
    spark.stop()

if __name__ == "__main__":
    main()
