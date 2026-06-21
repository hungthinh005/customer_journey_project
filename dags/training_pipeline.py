"""
Training pipeline DAG (weekly / manual).

Mirrors models/train_all.sh as discrete, retryable Airflow tasks:
feature engineering -> churn models -> retrieval models -> FAISS -> ranking ->
evaluation -> register to MLflow -> load fresh scores into the serving store.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _bash(task_id, command, dag):
    return BashOperator(
        task_id=task_id,
        bash_command=command,
        cwd=PROJECT_DIR,
        dag=dag,
    )


def register_models(**_):
    """Log evaluation metrics + model artifacts to the MLflow registry."""
    import os
    from pathlib import Path

    # Silence the git-not-found warning that MLflow emits at import time.
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

    import mlflow
    import pandas as pd

    from settings import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    project = Path(PROJECT_DIR)
    run_name = f"train-{datetime.utcnow():%Y%m%d-%H%M}"
    print(f"Starting MLflow run: {run_name}")

    metrics_logged = 0
    artifacts_logged = []

    with mlflow.start_run(run_name=run_name):
        # Log metrics from every *_metrics.csv produced by the training scripts.
        for metrics_csv in sorted(project.glob("models/**/*_metrics.csv")):
            try:
                df = pd.read_csv(metrics_csv)
                if df.empty:
                    print(f"  skip {metrics_csv.name}: empty file")
                    continue
                row = df.iloc[0].to_dict()
                prefix = metrics_csv.stem.replace("_metrics", "")
                for k, v in row.items():
                    try:
                        mlflow.log_metric(f"{prefix}.{k}", float(v))
                        metrics_logged += 1
                    except (TypeError, ValueError):
                        pass  # skip non-numeric columns (e.g. "model" name string)
                print(f"  metrics logged: {metrics_csv.name}")
            except Exception as exc:
                print(f"  skip {metrics_csv.name}: {exc}")

        # Log model artifacts directory by directory (skip large .npy embedding files
        # to avoid bloating the MLflow store - they live on the shared volume).
        SKIP_SUFFIXES = {".npy"}
        for artifact_dir in ["models/churn", "models/ranking", "faiss_index"]:
            p = project / artifact_dir
            if not p.exists():
                print(f"  skip artifact dir (not found): {artifact_dir}")
                continue
            try:
                # Filter out large binary files before logging.
                files = [f for f in p.iterdir() if f.is_file() and f.suffix not in SKIP_SUFFIXES]
                if not files:
                    print(f"  skip artifact dir (no eligible files): {artifact_dir}")
                    continue
                mlflow.log_artifacts(str(p), artifact_path=artifact_dir)
                artifacts_logged.append(artifact_dir)
                print(f"  artifacts logged: {artifact_dir}")
            except Exception as exc:
                print(f"  skip artifact dir {artifact_dir}: {exc}")

        mlflow.log_param("metrics_logged", metrics_logged)
        mlflow.log_param("artifacts_logged", ",".join(artifacts_logged))

    print(
        f"[OK] MLflow run complete | metrics={metrics_logged} "
        f"artifacts={artifacts_logged} | {settings.mlflow_tracking_uri}"
    )


with DAG(
    dag_id="training_pipeline",
    description="Weekly retrain of churn + recommendation models",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "ml"],
) as dag:
    features = _bash("feature_engineering", "python features/feature_engineering.py", dag)

    train_churn = _bash(
        "train_churn",
        "python models/churn/train_bgnbd.py && "
        "python models/churn/train_survival.py && "
        "python models/churn/compare_churn.py",
        dag,
    )
    train_retrieval = _bash(
        "train_retrieval",
        "python models/retrieval/train_als.py && "
        "python models/retrieval/train_item2vec.py && "
        "python models/retrieval/train_two_tower.py && "
        "python models/retrieval/compare_retrieval.py",
        dag,
    )
    build_faiss = _bash("build_faiss_index", "python faiss_index/build_index.py", dag)
    train_ranking = _bash("train_ranking", "python models/ranking/train_ranking.py", dag)
    evaluate = _bash(
        "evaluate",
        "python evaluation/evaluate_churn.py && "
        "python evaluation/evaluate_rec.py && "
        "python evaluation/ablation_study.py",
        dag,
    )
    register = PythonOperator(task_id="register_models", python_callable=register_models)
    load_db = _bash("load_to_serving_db", "python db/load_to_db.py", dag)

    features >> train_churn >> train_retrieval >> build_faiss >> train_ranking >> evaluate >> register >> load_db
