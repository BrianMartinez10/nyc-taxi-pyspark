# app.py
import os, json, io, glob, tempfile
import streamlit as st
import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F, types as T

# ========= PATH CONFIG (adjust if you changed folders) =========
BASE_DIR = os.getcwd()  # assuming you run in C:\TaxiProject
SAMPLE_DIR = os.path.join(
    BASE_DIR,
    r"data\processed\clean_sample_2015_01_csv"
)
# Base dataset used in stage 5
IN_FE = os.path.join(SAMPLE_DIR, "features", "fe_sample_2015_01_csv")

REPORTS_DIR   = os.path.join(SAMPLE_DIR, "reports")
PLOTS_PATTERN = os.path.join(REPORTS_DIR, "plots_*", "*.png")
# IMPORTANT: here we search *everything* and then resolve folders with part-*.csv
TABLES_PATTERN = os.path.join(REPORTS_DIR, "report_*", "*")

# === Friendly names for model directories ===
FRIENDLY_MODEL_NAMES = {
    "reg_total_lr_FINAL": "Total Amount — Linear Regression (FINAL)",
    "reg_total_rf_FINAL": "Total Amount — Random Forest (FINAL)",
    "cls_tip_lr_FINAL":   "Tip > 0 — Logistic Regression (FINAL)",
    "cls_tip_rf_FINAL":   "Tip > 0 — Random Forest (FINAL)",
}

def friendly_label_for_path(p: str) -> str:
    if p in ("(none)", "(ninguno)"):
        return "(none)"
    base = os.path.basename(p.rstrip("\\/"))
    return FRIENDLY_MODEL_NAMES.get(base, base)  # fallback: folder name


# ========= SPARK SESSION =========
@st.cache_resource
def get_spark():
    spark = (
        SparkSession.builder
        .appName("nyc-taxi-frontend")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

spark = get_spark()

# ========= HELPERS =========
def load_existing_sample():
    # Read the CSV from 03a (features) and prepare derived columns as in step 05
    sdf = (spark.read
           .option("header", "true")
           .option("inferSchema", "true")
           .csv(IN_FE))
    return enrich_features(sdf)

def enrich_features(sdf):
    # Ensure correct data types and derived columns
    sdf = (sdf
        .withColumn("trip_distance", F.col("trip_distance").cast(DoubleType()))
        .withColumn("fare_amount",  F.col("fare_amount").cast(DoubleType()))
        .withColumn("tip_amount",   F.col("tip_amount").cast(DoubleType()))
        .withColumn("total_amount", F.col("total_amount").cast(DoubleType()))
        .withColumn("payment_type", F.col("payment_type").cast(IntegerType()))
    )

    # Try with two common naming conventions
    datetime_cols = [c for c in sdf.columns if c.lower().startswith("tpep_pickup") or c.lower().startswith("pickup_datetime")]
    dropoff_cols  = [c for c in sdf.columns if c.lower().startswith("tpep_dropoff") or c.lower().startswith("dropoff_datetime")]

    if datetime_cols:
        pick = datetime_cols[0]
        sdf = sdf.withColumn("pickup_ts", F.to_timestamp(F.col(pick)))
    if dropoff_cols:
        drop = dropoff_cols[0]
        sdf = sdf.withColumn("dropoff_ts", F.to_timestamp(F.col(drop)))

    if "pickup_ts" in sdf.columns and "dropoff_ts" in sdf.columns:
        sdf = sdf.withColumn("duration_min",
                             (F.col("dropoff_ts").cast("long") - F.col("pickup_ts").cast("long"))/60.0)
        sdf = sdf.withColumn("pickup_hour", F.hour("pickup_ts"))
        sdf = sdf.withColumn("pickup_dow",  F.date_format("pickup_ts","u").cast("int"))  # 1..7
        sdf = sdf.withColumn("is_weekend", F.when(F.col("pickup_dow").isin(6,7), 1).otherwise(0))
        sdf = sdf.withColumn("night_flag", F.when((F.col("pickup_hour")>=22) | (F.col("pickup_hour")<=5), 1).otherwise(0))
    else:
        # Fallback if timestamps are missing
        sdf = (sdf
               .withColumn("duration_min", F.lit(None).cast(DoubleType()))
               .withColumn("pickup_hour",  F.lit(None).cast(IntegerType()))
               .withColumn("pickup_dow",   F.lit(None).cast(IntegerType()))
               .withColumn("is_weekend",   F.lit(0).cast(IntegerType()))
               .withColumn("night_flag",   F.lit(0).cast(IntegerType()))
              )

    if "tip_amount" in sdf.columns and "fare_amount" in sdf.columns:
        sdf = sdf.withColumn("tip_rate",
                             F.when(F.col("fare_amount")>0, F.col("tip_amount")/F.col("fare_amount"))
                              .otherwise(0.0))

    if "duration_min" in sdf.columns and "trip_distance" in sdf.columns:
        sdf = sdf.withColumn("avg_speed_mph",
                             F.when(F.col("duration_min")>0, F.col("trip_distance")/(F.col("duration_min")/60.0))
                              .otherwise(None).cast(DoubleType()))

    if "tip_rate" in sdf.columns:
        sdf = sdf.withColumn("label_tip", (F.col("tip_rate") > 0).cast("int"))

    return sdf

def load_uploaded_csv(uploaded_file):
    # Save temporarily and load with Spark
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name
    sdf = (spark.read
           .option("header","true")
           .option("inferSchema","true")
           .csv(temp_path))
    return enrich_features(sdf)

def sdf_to_pandas_safe(sdf, n=1000):
    """Convert Spark DF -> pandas with safe types and row limit n."""
    sdf.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    simple_allowed = (T.StringType, T.IntegerType, T.LongType, T.DoubleType,
                      T.FloatType, T.BooleanType, T.DateType, T.TimestampType,
                      T.ShortType, T.ByteType, T.BinaryType, T.NullType)
    cast_exprs = []
    for f in sdf.schema.fields:
        dt = type(f.dataType)
        if dt in (T.ArrayType, T.MapType, T.StructType):
            continue
        col = F.col(f.name)
        if isinstance(f.dataType, T.DecimalType):
            cast_exprs.append(col.cast("double").alias(f.name)); continue
        if dt is T.TimestampType:
            cast_exprs.append(F.date_format(col, "yyyy-MM-dd HH:mm:ss").alias(f.name)); continue
        if isinstance(f.dataType, simple_allowed):
            cast_exprs.append(col.alias(f.name))
    sdf_simple = sdf.select(*cast_exprs).limit(n)
    try:
        return sdf_simple.toPandas()
    except Exception:
        sdf.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        return sdf_simple.toPandas()

def list_model_dirs(prefix, only_pipeline=True):
    base = os.path.join(SAMPLE_DIR, "models")
    if not os.path.isdir(base):
        return []
    cand = [d for d in glob.glob(os.path.join(base, f"{prefix}*")) if os.path.isdir(d)]
    if only_pipeline:
        # Valid PipelineModel → …/model/stages/
        cand = [d for d in cand if os.path.isdir(os.path.join(d, "model", "stages"))]
    # Names include timestamp → sort descending (most recent first)
    return sorted(cand, reverse=True)

def load_model(model_dir):
    try:
        return PipelineModel.load(os.path.join(model_dir, "model"))
    except Exception as e:
        st.error(f"Could not load the model in {model_dir}: {e}")
        return None

def load_metrics(model_dir):
    """Read metrics.json if present inside the model folder."""
    try:
        metrics_path = os.path.join(model_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def csv_download_from_pred(pred_df, columns, filename="predictions.csv", limit=5000):
    """Show a CSV download button for selected columns from a Spark DF."""
    try:
        pdf = sdf_to_pandas_safe(pred_df.select(*columns), n=limit)
        buf = io.StringIO()
        pdf.to_csv(buf, index=False)
        st.download_button("Download predictions (CSV)", buf.getvalue(), file_name=filename, mime="text/csv")
    except Exception:
        pass

# ========= UI =========
st.set_page_config(page_title="NYC Taxi ", layout="wide")
st.title("🚖 NYC Taxi  ")

with st.sidebar:
    st.header("Dataset")
    choice = st.radio("Data source:", ["Project sample", "Upload CSV"])

    if choice == "Project sample":
        st.caption(f"Using bundled sample features")
        sdf = load_existing_sample()
    else:
        up = st.file_uploader("Upload a CSV (with headers)", type=["csv"])
        if up is not None:
            sdf = load_uploaded_csv(up)
            st.success("CSV loaded and enriched.")
        else:
            st.stop()

    st.divider()
    st.header("Models")
    # Find model folders on disk
    reg_dirs = list_model_dirs("reg_total_", only_pipeline=True)
    cls_dirs = list_model_dirs("cls_tip_",   only_pipeline=True)

    # Keep the underlying value as the folder path, but show a friendly label
    reg_opts = ["(none)"] + reg_dirs
    cls_opts = ["(none)"] + cls_dirs

    reg_sel = st.selectbox(
        "Regression Model (total)",
        reg_opts,
        index=1 if len(reg_opts) > 1 else 0,
        format_func=friendly_label_for_path
    )

    cls_sel = st.selectbox(
        "Classification Model (tip > 0)",
        cls_opts,
        index=1 if len(cls_opts) > 1 else 0,
        format_func=friendly_label_for_path
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Data & Summary",
    "🔎 Spark SQL",
    "📊 KPIs & Charts",
    "📈 Reports (CSV/PNG)",
    "🤖 Predictions"
])

# ========== TAB 1: Data & Summary ==========
with tab1:
    st.subheader("Preview")
    st.dataframe(sdf_to_pandas_safe(sdf))

    st.subheader("Quick summary")
    total = sdf.count()
    fares = sdf.select(F.mean("fare_amount").alias("avg_fare"),
                       F.mean("total_amount").alias("avg_total"),
                       F.mean("tip_rate").alias("avg_tip_rate")).collect()[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{total:,}")
    c2.metric("Average fare", f"{fares['avg_fare']:.2f}" if fares['avg_fare'] else "-")
    c3.metric("Average total", f"{fares['avg_total']:.2f}" if fares['avg_total'] else "-")
    c4.metric("Average tip rate", f"{fares['avg_tip_rate']:.3f}" if fares['avg_tip_rate'] else "-")

# ========== TAB 2: Spark SQL ==========
with tab2:
    st.subheader("Query Builder (Spark SQL)")
    view_name = "taxi"
    sdf.createOrReplaceTempView(view_name)

    example_sql = f"SELECT payment_type, COUNT(*) trips, ROUND(AVG(fare_amount),2) avg_fare FROM {view_name} GROUP BY payment_type ORDER BY trips DESC"
    sql_text = st.text_area("Write your SQL query:", value=example_sql, height=160)
    if st.button("Run SQL"):
        try:
            out = spark.sql(sql_text)
            st.dataframe(sdf_to_pandas_safe(out, n=5000))
            csv_buf = io.StringIO()
            out.toPandas().to_csv(csv_buf, index=False)
            st.download_button("Download CSV", csv_buf.getvalue(), file_name="query_result.csv", mime="text/csv")
        except Exception as e:
            st.error(f"SQL error: {e}")

# ========== TAB 3: KPIs & Charts ==========
with tab3:
    st.subheader("KPIs by payment method")
    by_pay = (sdf.groupBy("payment_type")
                .agg(F.count("*").alias("trips"),
                     F.round(F.avg("fare_amount"),2).alias("avg_fare"),
                     F.round(F.avg("total_amount"),2).alias("avg_total"),
                     F.round(F.avg("tip_rate"),3).alias("avg_tip_rate"))
                .orderBy(F.desc("trips")))
    st.dataframe(sdf_to_pandas_safe(by_pay, 1000))

    st.subheader("Average fare by hour")
    by_hour = (sdf.groupBy("pickup_hour")
                .agg(F.count("*").alias("trips"),
                     F.round(F.avg("fare_amount"),2).alias("avg_fare"))
                .orderBy("pickup_hour"))
    df_hour = sdf_to_pandas_safe(by_hour, 1000)
    st.line_chart(df_hour.set_index("pickup_hour")[["avg_fare"]])

    st.subheader("Distance distribution (buckets)")
    dist_bucket = (F.when(F.col("trip_distance")<1, "<1mi")
                     .when(F.col("trip_distance")<2, "1-2mi")
                     .when(F.col("trip_distance")<5, "2-5mi")
                     .when(F.col("trip_distance")<10,"5-10mi")
                     .otherwise(">=10mi"))
    by_dist = (sdf.withColumn("dist_bucket", dist_bucket)
                .groupBy("dist_bucket")
                .agg(F.count("*").alias("trips"),
                     F.round(F.avg("total_amount"),2).alias("avg_total"))
                .orderBy("dist_bucket"))
    st.bar_chart(sdf_to_pandas_safe(by_dist, 1000).set_index("dist_bucket")[["trips"]])

# ========== TAB 4: Reports (CSV/PNG) ==========
with tab4:
    st.subheader("Generated images (05b)")
    pngs = glob.glob(PLOTS_PATTERN)
    if not pngs:
        st.info("No images found. Run 05b_plots.py to generate PNGs.")
    else:
        for p in sorted(pngs):
            st.image(p, caption=os.path.basename(p), use_container_width=True)

    st.subheader("Tables (05c)")
    candidates = glob.glob(TABLES_PATTERN)
    csv_paths = []
    for p in sorted(candidates):
        if os.path.isdir(p):
            parts = sorted(glob.glob(os.path.join(p, "part-*.csv")))
            if parts:
                csv_paths.append(parts[0])   # take the first part
        elif p.lower().endswith(".csv"):
            csv_paths.append(p)

    if not csv_paths:
        st.info("No CSV reports found. Run 05c_make_report.py to generate them.")
    else:
        for cpath in csv_paths:
            nice_name = f"{os.path.basename(os.path.dirname(cpath))}/{os.path.basename(cpath)}" if os.path.dirname(cpath) else os.path.basename(cpath)
            st.markdown(f"**{nice_name}**")
            try:
                pdf = pd.read_csv(cpath)
                st.dataframe(pdf.head(2000))
                with open(cpath, "rb") as f:
                    st.download_button("Download CSV", f.read(),
                                       file_name=os.path.basename(cpath), mime="text/csv")
            except Exception as e:
                st.warning(f"Could not read {cpath}: {e}")

# ========== TAB 5: Predictions  ==========
with tab5:
    st.subheader("Batch predictions with Spark ML models")
    c1, c2 = st.columns([2, 1])
    with c1:
        sample_n = st.slider("Sample size for prediction", 100, 5000, 1000, 100)
    with c2:
        randomize = st.checkbox("Randomize sample", value=False)

    def maybe_randomize(df, seed=42):
        return df.orderBy(F.rand(seed)) if randomize else df

    # ----- Regression: total_amount -----
    st.markdown("### **Regression** Model (total_amount)")
    if reg_sel != "(none)":
        reg_model = load_model(reg_sel)
        if reg_model:
            try:
                st.info("This demo only scores if the pipeline includes the assembler inside.")
                base = sdf.dropna(subset=["total_amount"])
                base = maybe_randomize(base, seed=7)
                sdf_sample = base.limit(sample_n).cache()
                pred = reg_model.transform(sdf_sample)
                show_cols = ["prediction", "total_amount"]
                st.dataframe(sdf_to_pandas_safe(pred.select(*show_cols)))

                # Metrics (RMSE/MAE/R2)
                m = load_metrics(reg_sel)
                if m:
                    mc1, mc2, mc3 = st.columns(3)
                    if "rmse" in m: mc1.metric("RMSE", f"{m['rmse']:.4f}")
                    if "mae"  in m: mc2.metric("MAE",  f"{m['mae']:.4f}")
                    if "r2"   in m: mc3.metric("R²",   f"{m['r2']:.4f}")

                # Download predictions
                csv_download_from_pred(pred, show_cols, filename="predictions_regression.csv", limit=5000)

            except Exception as e:
                st.warning(f"Could not apply the regression model with the current DataFrame: {e}")
    else:
        st.caption("Select a model in the sidebar.")

    st.markdown("---")

    # ----- Classification: tip > 0 -----
    st.markdown("### **Classification** Model (tip > 0)")
    if cls_sel != "(none)":
        cls_model = load_model(cls_sel)
        if cls_model:
            try:
                th_col1, th_col2 = st.columns([2,1])
                with th_col2:
                    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)

                base = maybe_randomize(sdf, seed=11)
                sdf_sample = base.limit(sample_n).cache()
                pred = cls_model.transform(sdf_sample)

                # Extract probability of class 1 and thresholded prediction
                if "probability" in pred.columns:
                    from pyspark.ml.functions import vector_to_array
                    pred = pred.withColumn("proba1", vector_to_array("probability").getItem(1))
                    pred = pred.withColumn("pred_at_threshold", (F.col("proba1") >= F.lit(threshold)).cast("int"))
                    show_cols = ["proba1", "prediction", "pred_at_threshold", "label_tip"] if "label_tip" in pred.columns else ["proba1", "prediction", "pred_at_threshold"]
                else:
                    show_cols = ["prediction", "label_tip"] if "label_tip" in pred.columns else ["prediction"]

                st.dataframe(sdf_to_pandas_safe(pred.select(*show_cols)))

                # Metrics (AUC/Accuracy)
                m = load_metrics(cls_sel)
                if m:
                    mc1, mc2 = st.columns(2)
                    if "auc"      in m: mc1.metric("AUC",      f"{m['auc']:.4f}")
                    if "accuracy" in m: mc2.metric("Accuracy", f"{m['accuracy']:.4f}")

                # Download predictions
                csv_download_from_pred(pred, show_cols, filename="predictions_classification.csv", limit=5000)

            except Exception as e:
                st.warning(f"Could not apply the classification model with the current DataFrame: {e}")
    else:
        st.caption("Select a model in the sidebar.")
