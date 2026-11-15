from pathlib import Path
import importlib.util


def test_airflow_dag_import():
    dag_path = Path(__file__).resolve().parent.parent / "dags" / "recommender_pipeline_dag.py"
    assert dag_path.exists(), "Airflow DAG file must exist"
    
    spec = importlib.util.spec_from_file_location("recommender_pipeline_dag", dag_path)
    dag_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dag_module)
    
    assert hasattr(dag_module, "dag"), "DAG object must be defined"
    assert hasattr(dag_module, "task_validate_data"), "Data validation task must exist"
    assert hasattr(dag_module, "task_evaluate_and_gate"), "Gating task must exist"
