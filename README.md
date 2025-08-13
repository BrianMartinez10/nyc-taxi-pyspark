# NYC Taxi – PySpark + Streamlit

End-to-end analytics and ML pipeline on the NYC Yellow Taxi dataset (January 2015) using **PySpark** for data processing/modeling and **Streamlit** for a simple frontend.

## Project Structure
├─ spark_jobs/ # PySpark scripts by stage
│ ├─ 01_ingest.py
│ ├─ 02a_read_counts.py
│ ├─ 02c_nulls_and_stats.py
│ ├─ 02e_join_lookup.py
│ ├─ 03b_vectorize_scale.py
│ ├─ 04b_cls_tip_lr_PIPE2.py
│ ├─ 04d_cls_tip_rf_PIPE2.py
│ └─ 05b_plots.py
├─ streamlit_app/
│ └─ app5.py # Streamlit UI
├─ data/processed/clean_sample_2015_01_csv/ # small sample + model outputs
├─ reports/plots/ # generated PNG charts
├─ requirements.txt
└─ README.md


## Features
- **Ingestion & Cleaning**: read CSV, basic filters, null checks, quick stats.
- **Feature Engineering**: durations, speed, hour/day, weekend flag, tip rate, etc.
- **SQL/KPIs**: hourly trips, payment distribution, distance buckets, heatmap (DoW × hour).
- **ML (MLlib)**:
  - **Classification**: *tip > 0* (Logistic Regression / Random Forest).
  - **Regression**: predict `total_amount` (Linear Regression / Random Forest).
- **Frontend (Streamlit)**: preview data, run SQL, show KPIs/charts, load models and score.

## Getting Started

### Prerequisites
- Python **3.9+** (3.10/3.11 recommended)
- Java **8+** (required by PySpark on Windows)
- (Windows) Set `JAVA_HOME` environment variable to your JDK directory

### Install
```bash
pip install -r requirements.txt


