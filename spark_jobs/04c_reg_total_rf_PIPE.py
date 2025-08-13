# spark_jobs/04c_reg_total_rf_PIPE.py
import os, json
from datetime import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ========================
# Input & output directories
# ========================
SAMPLE_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
IN_FE      = os.path.join(SAMPLE_DIR, "features", "fe_sample_2015_01_csv")
#RUN        = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = os.path.join(SAMPLE_DIR, "models", f"reg_total_rf_FINAL")

def main():
    # Create Spark session
    spark = (SparkSession.builder.appName("nyc-taxi-04c-reg-total-rf-PIPE")
             .master("local[*]").config("spark.sql.shuffle.partitions", "32").config("spark.driver.memory", "6g").config("spark.local.dir", r"C:\TaxiProject\tmp").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    # Read feature dataset
    df = (spark.read.option("header","true").option("inferSchema","true").csv(IN_FE))
    # Target variable for regression (total_amount)
    df = df.withColumn("label_total", F.col("total_amount").cast("double"))
    # Numeric features to be used in the model
    numeric_cols = ["trip_distance","duration_min","avg_speed_mph","pickup_hour",
                    "pickup_dow","is_weekend","night_flag","tip_rate"]
    
    # Handle missing values
    # Filter out trips with invalid duration (<= 0 minutes)
    df = df.filter(F.col("duration_min").isNull() | (F.col("duration_min") > 0))
    # Replace missing numeric values with median
    imputer = Imputer(strategy="median", inputCols=numeric_cols, outputCols=numeric_cols)
    df = imputer.fit(df).transform(df)
    # Replace null flags with 0
    for c in ["is_weekend","night_flag"]:
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(0)).otherwise(F.col(c)))

    # Categorical feature encoding
    idx = StringIndexer(inputCol="payment_type", outputCol="payment_type_idx", handleInvalid="keep")
    ohe = OneHotEncoder(inputCols=["payment_type_idx"], outputCols=["payment_type_ohe"])
    # Feature assembly & scaling
    assembler = VectorAssembler(inputCols=numeric_cols+["payment_type_ohe"], outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    # Random Forest Regressor setup
    rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="label_total",
    numTrees=60,                 
    maxDepth=8,                  
    maxBins=32,                  
    subsamplingRate=0.7,         
    featureSubsetStrategy="sqrt",
    cacheNodeIds=True,
    seed=42
    )
    # Pipeline: Preprocessing + Model
    pipe = Pipeline(stages=[idx, ohe, assembler, scaler, rf])
    # Train/Test split
    train, test = df.dropna(subset=["label_total"]).randomSplit([0.8,0.2], seed=42)
    # Train model
    model = pipe.fit(train)
    # Predictions
    pred  = model.transform(test)
    # Evaluation metrics
    eval_rmse = RegressionEvaluator(labelCol="label_total", predictionCol="prediction", metricName="rmse")
    eval_mae  = RegressionEvaluator(labelCol="label_total", predictionCol="prediction", metricName="mae")
    eval_r2   = RegressionEvaluator(labelCol="label_total", predictionCol="prediction", metricName="r2")
    rmse, mae, r2 = eval_rmse.evaluate(pred), eval_mae.evaluate(pred), eval_r2.evaluate(pred)
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    # Save model and metrics
    os.makedirs(OUT_DIR, exist_ok=True)
    model.write().overwrite().save(os.path.join(OUT_DIR, "model"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f, indent=2)
    # Save a sample of predictions
    (pred.select("prediction","label_total").limit(10_000)
         .coalesce(1).write.mode("overwrite").option("header","true")
         .csv(os.path.join(OUT_DIR, "preds_sample_csv")))

    print(f"Model (Pipeline) -> {os.path.join(OUT_DIR,'model')}")
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
