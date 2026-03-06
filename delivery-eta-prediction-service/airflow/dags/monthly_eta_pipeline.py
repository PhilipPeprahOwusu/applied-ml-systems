from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# 1. Default Arguments: The "Rules" for our factory workers
default_args = {
    'owner': 'philip',                   
    'depends_on_past': False,            
    'start_date': datetime(2025, 1, 1),  
    'email_on_failure': False,           
    'retries': 2,                        
    'retry_delay': timedelta(minutes=5), 
}

# 2. Define the DAG (The Blueprint)
with DAG(
    'monthly_nyc_eta_pipeline',          
    default_args=default_args,
    description='Automated pipeline to fetch NYC data, update Feast, and train models',
    schedule_interval='0 0 15 * *',      
    catchup=False,                       
) as dag:
    
    #  TASK 1: Check if the NYC data is ready 
    check_data_task = BashOperator(
        task_id='check_data_availability',
        bash_command='cd /opt/airflow && python pipelines/check_availability.py --month {{ logical_date.strftime("%Y-%m") }}'
    )

    #  TASK 2: Download and Process the Data 
    ingest_data_task = BashOperator(
        task_id='ingest_new_data',
        bash_command='cd /opt/airflow && python pipelines/ingest_data.py --month {{ logical_date.strftime("%Y-%m") }}'
    )

    #  TASK 3: Push new features to Redis 
    update_feast_task = BashOperator(
        task_id='materialize_feast_features',
        bash_command='cd /opt/airflow/feature_repo && feast apply && feast materialize 2024-01-01T00:00:00 2030-01-01T00:00:00'
    )

    # TASK 4: Run the training script on the newly downloaded month data
    train_model = BashOperator(
        task_id='train_new_model',
        bash_command='python /opt/airflow/pipelines/train_model.py --data_path /opt/airflow/data/processed/yellow_{{ logical_date.strftime("%Y-%m") }}.parquet --run_name automated_{{ logical_date.strftime("%Y-%m") }}',
        env={"MLFLOW_TRACKING_URI": "http://mlflow-server:5000"}
    )

    #  THE FACTORY LOGIC (Dependencies) 
    check_data_task >> ingest_data_task >> update_feast_task >> train_model
