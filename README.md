# NYC Taxi – PySpark + Streamlit

End-to-end analytics and ML pipeline on the NYC Yellow Taxi dataset (January 2015) using **PySpark** for data processing/modeling and **Streamlit** for a simple frontend.

## Project Structure
spark_jobs: PySpark scripts by stage
streamlit_app: Streamlit UI
data/processed/ clean_sample_2015_01_csv/: small sample + model outputs
reports/plots:  generated PNG charts
requirements.txt
README.md


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


