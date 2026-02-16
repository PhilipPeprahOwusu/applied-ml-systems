from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Setting project directory - this is the base path of the project
PROJECT_DIR = "/Users/philipowusu/Development/recommendation_system"

# Setting default rules for all tasks
default_args = {
    'owner':'philip',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

#  now we are defining the DAG workflow
with DAG(
    'recommender_pipeline',
    default_args = default_args,
    description = 'Automated pipeline for Offer Recommendation',
    schedule_interval = timedelta(days=1),
    start_date = datetime(2024,1,1),
    catchup = False,
) as dag:
    
    # Task 1: Generate Data (Using vectorized script)
    task_1 = BashOperator(
        task_id = 'generate_data',
        bash_command = f'cd {PROJECT_DIR} && python pipelines/generate_data.py'
    )

    # Task 2: Feature Engineering (Using standard script)
    task_2 = BashOperator(
        task_id = 'feature_engineering',
        bash_command = f'cd {PROJECT_DIR} && python pipelines/feature_engineering.py'
    )

    # Task 3: Train Model (Using LightGBM Ranking script)
    task_3 = BashOperator(
        task_id = 'train_model_mlflow',
        bash_command = f'cd {PROJECT_DIR} && python pipelines/train_model.py'
    )

    # Task 4: Serving (Load to Redis)
    task_4 = BashOperator(
        task_id='load_to_redis',
        bash_command=f'cd {PROJECT_DIR} && python src/load_to_redis.py'
    )

    task_1 >> task_2 >> task_3 >> task_4