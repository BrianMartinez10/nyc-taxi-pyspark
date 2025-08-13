from pyspark.sql import SparkSession, functions as F

# Path
RAW = r"C:\TaxiProject\data\raw\yellow_tripdata_2015-01.csv"

def main():
    # Initialize Spark session
    spark = (SparkSession.builder
             .appName("02a-read-counts")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "64")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    
    # Read raw CSV file
    df = (spark.read
          .option("header","true")
          .option("inferSchema","true")
          .option("timestampFormat","yyyy-MM-dd HH:mm:ss")
          .csv(RAW))
    
    # Convert pickup and dropoff datetime strings to proper timestamp types
    df = (df
          .withColumn("pickup_ts", F.to_timestamp("tpep_pickup_datetime"))
          .withColumn("dropoff_ts", F.to_timestamp("tpep_dropoff_datetime")))

    # Show detected schema
    print("\n>>> Schema:")
    df.printSchema()
    
    # Display first 5 sample rows with key columns
    print("\n>>> First 5 rows:")
    df.select("VendorID","pickup_ts","dropoff_ts","trip_distance","fare_amount").show(5, False)

    # Count total rows in dataset
    total = df.count()
    print(f"\n>>> Total rows: {total:,}")

    spark.stop()

if __name__ == "__main__":
    main()
