from __future__ import annotations

import pendulum
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.models import Variable

# Get the project root path from the Airflow Variable
PROJECT_ROOT = Variable.get("PROJECT_ROOT", default_var="/opt/airflow")

@dag(
    dag_id="data_pipeline_dag",
    schedule=None,
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["fraud_detection", "data_processing"],
)
def data_pipeline_dag():
    """
    ### Data Pipeline DAG

    Orchestrates the data processing part of the fraud detection pipeline.
    It runs the `make process-data` command.
    """
    
    BashOperator(
        task_id="run_data_processing",
        # This command navigates to the project directory and runs make
        bash_command=f"cd {PROJECT_ROOT} && make process-data",
        doc_md="Runs the `make process-data` target to build the curated dataset.",
    )

# This line tells Airflow to create the DAG
data_pipeline_dag()