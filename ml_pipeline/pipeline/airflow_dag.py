"""
airflow_dag.py
--------------
Defines the full ML pipeline as an Apache Airflow DAG.

WHAT IS AIRFLOW?
  Apache Airflow is a workflow orchestration platform.
  You describe your pipeline as a DAG (Directed Acyclic Graph) –
  a set of tasks with dependencies between them.
  Airflow then schedules, retries, monitors, and alerts on those tasks.

WHY USE AIRFLOW FOR ML?
  ML pipelines have many steps that must run in order:
    data ingestion → feature engineering → training → evaluation → deployment
  If any step fails, you want automatic retries and Slack/email alerts.
  You also want to run this pipeline on a schedule (e.g., retrain weekly).
  Airflow handles all of this out of the box.

HOW TO RUN LOCALLY:
    # 1. Install & initialise Airflow
    export AIRFLOW_HOME=~/airflow
    pip install apache-airflow
    airflow db init
    airflow users create --username admin --role Admin ...

    # 2. Put this file in ~/airflow/dags/

    # 3. Start scheduler + webserver
    airflow scheduler &
    airflow webserver --port 8080

    # 4. Open http://localhost:8080 → enable the dag → trigger it

DAG STRUCTURE (run weekly, every Monday 02:00 UTC):

  [generate_data]
       │
  [feast_materialize]
       │
  [train_model]
       │
  [evaluate_model]
       │
  [promote_or_alert]
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# ── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR = Path(__file__).parents[1]   # ml_pipeline/


# ── Default args (applied to every task unless overridden) ────────────────────
DEFAULT_ARGS = {
    "owner":            "ml-team",
    "depends_on_past":  False,         # don't wait for last week's run to succeed
    "email":            ["ml-alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}


# ── Task functions ─────────────────────────────────────────────────────────────
# Each task is a plain Python function.
# Airflow passes context (run dates, XCom values, …) via **kwargs.

def task_generate_data(**kwargs):
    """Step 1: refresh the raw data CSV (or pull from data warehouse in prod)."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(ML_DIR / "data" / "generate_data.py")],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def task_feast_materialize(**kwargs):
    """
    Step 2: push latest feature values from offline store → online store.
    This ensures the serving API always retrieves fresh features.
    """
    import sys
    sys.path.insert(0, str(ML_DIR))
    from feature_store.materialize import materialize
    materialize()


def task_train_model(**kwargs):
    """
    Step 3: train a new model and log it to MLflow.
    The run_id is pushed to XCom so downstream tasks can reference it.

    XCOM = cross-communication, Airflow's way to pass small values between tasks.
    """
    import sys
    sys.path.insert(0, str(ML_DIR))
    from training.train import train
    run_id, _model, _prep = train()
    # Push the run_id to XCom so the evaluate task can pull it
    kwargs["ti"].xcom_push(key="mlflow_run_id", value=run_id)


def task_evaluate_model(**kwargs):
    """
    Step 4: load the freshly trained model and check quality gates.
    Returns 'promote' or 'alert' to drive the branch downstream.
    """
    import sys
    sys.path.insert(0, str(ML_DIR))
    import mlflow
    import numpy as np
    from training.preprocess import load_data, get_splits

    run_id = kwargs["ti"].xcom_pull(key="mlflow_run_id", task_ids="train_model")
    model  = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    df = load_data()
    _, _, X_test, _, _, y_test, preprocessor = get_splits(df)
    preds     = model.predict(X_test)
    test_rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

    # Push metrics to XCom for display in Airflow logs / downstream tasks
    kwargs["ti"].xcom_push(key="test_rmse", value=test_rmse)

    print(f"Test RMSE: ${test_rmse:.2f}")
    return "promote_model" if test_rmse < 35.0 else "send_alert"


def task_promote_model(**kwargs):
    """Step 5a: transition the new model version to Production in MLflow Registry."""
    import mlflow

    run_id = kwargs["ti"].xcom_pull(key="mlflow_run_id", task_ids="train_model")
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"run_id='{run_id}'")
    for v in versions:
        client.transition_model_version_stage(
            name="hotel-price-predictor",
            version=v.version,
            stage="Production",
        )
        print(f"Promoted model v{v.version} → Production")


def task_send_alert(**kwargs):
    """Step 5b: model didn't pass quality gates – send a Slack/email alert."""
    rmse = kwargs["ti"].xcom_pull(key="test_rmse", task_ids="evaluate_model")
    # In production this would call the Slack API or send an email.
    print(f"ALERT: New model failed quality gate. RMSE=${rmse:.2f} (threshold $35).")
    print("Model NOT promoted. Keeping current Production version.")


# ── DAG definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id="hotel_price_ml_pipeline",
    default_args=DEFAULT_ARGS,
    description="Weekly hotel price model retraining pipeline",
    schedule="0 2 * * 1",         # every Monday at 02:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,                 # don't backfill missed runs
    tags=["ml", "hotel", "pricing"],
) as dag:

    start = EmptyOperator(task_id="start")

    generate_data = PythonOperator(
        task_id="generate_data",
        python_callable=task_generate_data,
    )

    feast_materialize = PythonOperator(
        task_id="feast_materialize",
        python_callable=task_feast_materialize,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        execution_timeout=timedelta(minutes=30),
    )

    # BranchPythonOperator lets the pipeline take different paths
    # based on the return value of the callable.
    evaluate_model = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
    )

    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=task_promote_model,
    )

    send_alert = PythonOperator(
        task_id="send_alert",
        python_callable=task_send_alert,
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    # ── Wire up dependencies ────────────────────────────────────────────────────
    # A >> B means "B depends on A" (run A first)
    (
        start
        >> generate_data
        >> feast_materialize
        >> train_model
        >> evaluate_model
        >> [promote_model, send_alert]
        >> end
    )
