from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# Governance Rules
INITIAL_ANCHOR = "2024-06-01"   # bootstrap month (just after the first OOT window)
INITIAL_MODEL = "logistic_regression"
REFRESH_MONTHS = "3"
AUC_FLOOR = "0.70"
PSI_RETRAIN = "0.25"
OOT_MONTHS = "3"

PROJECT = "/opt/airflow"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="credit_default_ml_pipeline",
    default_args=default_args,
    description="Serve the champion monthly; retrain on governance rules",
    schedule_interval="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id="start")

    run_data_pipeline = BashOperator(
        task_id="run_data_pipeline",
        bash_command=(f"cd {PROJECT} && python3 scripts/run_data_pipeline.py "
                      f'--snapshotdate "{{{{ ds }}}}"'),
    )
    data_pipeline_completed = EmptyOperator(task_id="data_pipeline_completed")

    # Decoupled from training - runs first
    run_inference = BashOperator(
        task_id="run_inference", priority_weight=50,
        bash_command=(f"cd {PROJECT} && python3 scripts/run_inference.py "
                      f'--snapshotdate "{{{{ ds }}}}"'),
    )
    run_monitoring = BashOperator(
        task_id="run_monitoring", priority_weight=50,
        bash_command=(f"cd {PROJECT} && python3 scripts/run_monitoring.py "
                      f'--snapshotdate "{{{{ ds }}}}"'),
    )
    generate_dashboard = BashOperator(
        task_id="generate_dashboard", priority_weight=50,
        bash_command=(f"cd {PROJECT} && python3 scripts/generate_dashboard.py "
                      f'--snapshotdate "{{{{ ds }}}}" || true'),
    )

    # governance/training branch
    model_governance_train = BashOperator(
        task_id="model_governance_train", priority_weight=1,
        bash_command=(f"cd {PROJECT} && python3 scripts/train_model.py "
                      f'--snapshotdate "{{{{ ds }}}}" '
                      f"--initial_anchor {INITIAL_ANCHOR} "
                      f"--initial_model {INITIAL_MODEL} "
                      f"--refresh_months {REFRESH_MONTHS} "
                      f"--auc_floor {AUC_FLOOR} --psi_retrain {PSI_RETRAIN} "
                      f"--oot_months {OOT_MONTHS}"),
    )

    end = EmptyOperator(task_id="end")

    start >> run_data_pipeline >> data_pipeline_completed
    data_pipeline_completed >> run_inference >> run_monitoring >> generate_dashboard >> end
    data_pipeline_completed >> model_governance_train >> end
