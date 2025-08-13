# spark_jobs/03b_vectorize_scale.py
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

#SAMPLE_DIR = r"C:\Users\brian\OneDrive\Escritorio\1_FinalProject_S3_SparkTaxiFare\data\processed\clean_sample_2015_01_csv"
SAMPLE_DIR = r"C:\TaxiProject\data\processed\clean_sample_2015_01_csv"
IN_FE      = os.path.join(SAMPLE_DIR, "features", "fe_sample_2015_01_csv")
OUT_BASE   = os.path.join(SAMPLE_DIR, "features")

def main():
    # Initialize Spark session
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-03b-vectorize-scale")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    # Load engineered features dataset
    df = (spark.read
            .option("header","true")
            .option("inferSchema","true")
            .csv(IN_FE))

    # Categorical column to encode (example: payment_type)
    # payment_type: int (1, 2, 3, 4, ...) -> index -> one-hot encoding
    idx = StringIndexer(inputCol="payment_type", outputCol="payment_type_idx", handleInvalid="keep")
    ohe = OneHotEncoder(inputCols=["payment_type_idx"], outputCols=["payment_type_ohe"])

    # Numerical columns to use in the vector
    numeric_cols = ["trip_distance","duration_min","avg_speed_mph","pickup_hour",
                    "pickup_dow","is_weekend","night_flag","tip_rate"]

    assembler = VectorAssembler(
        inputCols = numeric_cols + ["payment_type_ohe"],
        outputCol = "features_raw",
        handleInvalid="keep"
    )
    
    # Standard scaling to normalize features
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

    # Define label for future regression model (total_amount)
    df = df.withColumn("label_total", F.col("total_amount").cast("double"))

    # (optional) define a label for a future model
    df = df.withColumn("label_total", F.col("total_amount").cast("double"))
    
    # ========================
    # 🔹 ANTI-NaN PATCH
    # ========================
    from pyspark.ml.feature import Imputer

    # 1) Remove trips with non-positive duration (protects avg_speed_mph calculation)
    df = df.filter(F.col("duration_min").isNull() | (F.col("duration_min") > 0))

    # 2) Impute NaN/Null in numeric columns with the median value
    numeric_cols = ["trip_distance","duration_min","avg_speed_mph","pickup_hour",
                    "pickup_dow","is_weekend","night_flag","tip_rate"]

    imputer = Imputer(strategy="median", inputCols=numeric_cols, outputCols=numeric_cols)
    df = imputer.fit(df).transform(df)

    # 3) Ensure binary flags have no nulls (replace with 0)
    for f in ["is_weekend","night_flag"]:
        df = df.withColumn(f, F.when(F.col(f).isNull(), F.lit(0)).otherwise(F.col(f)))
    
    # Build transformation pipeline
    pipe = Pipeline(stages=[idx, ohe, assembler, scaler])
    model = pipe.fit(df)
    out  = model.transform(df)

    print("\n>>> Example of vectorization/scaling:")
    out.select("features_raw","features","label_total").show(5, truncate=False)

    # Save a small vectorized sample
    out_path = os.path.join(OUT_BASE, "vectorized_sample_2015_01")
    (out.select("features","label_total")
        .limit(100_000)     # limit for demo
        .coalesce(1)        # single file for sample
        .write.mode("overwrite").option("header","true").parquet(out_path))
    print(f"\n>>> Vectorized sample saved at: {out_path}")

    spark.stop()

if __name__ == "__main__":
    main()
