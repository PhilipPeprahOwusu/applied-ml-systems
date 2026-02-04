"""Airflow DAG for recommendation system retraining pipeline."""
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

PROJECT_DIR = os.environ.get(
    "RECOMMENDATION_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PYTHON = os.path.join(PROJECT_DIR, "rsenv", "bin", "python")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'recommender_pipeline',
    default_args=default_args,
    description='Retrain recommendation model and refresh cache',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    generate_data = BashOperator(
        task_id='generate_data',
        bash_command=f'{PYTHON} {PROJECT_DIR}/pipelines/generate_data.py',
    )

    feature_engineering = BashOperator(
        task_id='feature_engineering',
        bash_command=f'{PYTHON} {PROJECT_DIR}/pipelines/feature_engineering.py',
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'{PYTHON} {PROJECT_DIR}/pipelines/train_model.py',
    )

    load_to_redis = BashOperator(
        task_id='load_to_redis',
        bash_command=f'cd {PROJECT_DIR} && {PYTHON} -m src.load_to_redis',
    )

    generate_data >> feature_engineering >> train_model >> load_to_redis
