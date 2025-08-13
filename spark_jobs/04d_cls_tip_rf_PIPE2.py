import os, json
from datetime import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
# Define directories for sample data and output
SAMPLE_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
IN_FE      = os.path.join(SAMPLE_DIR, "features", "fe_sample_2015_01_csv")
#RUN        = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = os.path.join(SAMPLE_DIR, "models", f"cls_tip_rf_FINAL")

def main():
    # Create Spark session
    spark = (SparkSession.builder
             .appName("nyc-taxi-04d-cls-tip-rf-PIPE2")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions","48")
             .config("spark.driver.memory","6g")
             .config("spark.local.dir", r"C:\TaxiProject\tmp")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    # Read features dataset
    df = (spark.read.option("header","true").option("inferSchema","true").csv(IN_FE))
    # Create the binary label column: 1 if tip_amount > 0, else 0
    if "label_tip" not in df.columns:
        df = df.withColumn("label_tip", (F.col("tip_amount") > 0).cast("int"))
    # Columns we want to use as features
    wanted = [
        "trip_distance","duration_min","avg_speed_mph",
        "pickup_hour","pickup_dow","is_weekend","night_flag",
        "passenger_count","fare_amount"
    ]
    numeric_cols = [c for c in wanted if c in df.columns]
    # Filter out rows with invalid or zero duration
    df = df.filter(F.col("duration_min").isNull() | (F.col("duration_min") > 0))
    # Replace missing numeric values with the median
    imputer = Imputer(strategy="median", inputCols=numeric_cols, outputCols=numeric_cols)
    df = imputer.fit(df).transform(df)
    # Fill missing boolean flag values with 0
    for c in ["is_weekend","night_flag"]:
        if c in df.columns:
            df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(0)).otherwise(F.col(c)))
    # Encode payment_type if present
    idx = StringIndexer(inputCol="payment_type", outputCol="payment_type_idx", handleInvalid="keep") \
            if "payment_type" in df.columns else None
    stages = []
    if idx: stages += [idx]
    if idx: stages += [OneHotEncoder(inputCols=["payment_type_idx"], outputCols=["payment_type_ohe"])]
    # Assemble all features into a single vector
    input_feats = numeric_cols + (["payment_type_ohe"] if idx else [])
    assembler = VectorAssembler(inputCols=input_feats, outputCol="features_raw", handleInvalid="keep")
    # Standardize features (mean = 0, std = 1)
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    # Configure the Random Forest Classifier
    rf = RandomForestClassifier(featuresCol="features", labelCol="label_tip",
                                numTrees=80, maxDepth=10, maxBins=32,
                                subsamplingRate=0.7, featureSubsetStrategy="sqrt",
                                cacheNodeIds=True, seed=42)
    # Build the pipeline
    pipe = Pipeline(stages=stages + [assembler, scaler, rf])
    # Split the dataset into training and testing
    train, test = df.dropna(subset=["label_tip"]).randomSplit([0.8,0.2], seed=42)
    # Train the pipeline model
    model = pipe.fit(train)
    # Make predictions
    pred  = model.transform(test)
    # Evaluate the model using AUC and Accuracy
    auc = BinaryClassificationEvaluator(labelCol="label_tip", rawPredictionCol="rawPrediction",
                                        metricName="areaUnderROC").evaluate(pred)
    acc = MulticlassClassificationEvaluator(labelCol="label_tip", predictionCol="prediction",
                                            metricName="accuracy").evaluate(pred)
    # Save the model and metrics
    os.makedirs(OUT_DIR, exist_ok=True)
    model.write().overwrite().save(os.path.join(OUT_DIR, "model"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"auc": auc, "accuracy": acc}, f, indent=2)
    # Print confirmation message
    print(f"Guardado Pipeline (sin fuga) -> {os.path.join(OUT_DIR,'model')} | AUC={auc:.3f} ACC={acc:.3f}")
    spark.stop()

if __name__ == "__main__":
    main()
