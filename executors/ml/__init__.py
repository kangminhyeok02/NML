"""
executors/ml/__init__.py
------------------------
ML Executor 패키지.

모든 executor는 BaseExecutor를 상속하며 run() 메서드를 통해 실행된다.
직접 사용 시에는 각 executor를 import하거나,
ProcessExecutor를 통해 파이프라인으로 실행한다.

사용 예시:
    from executors.ml.predict_executor import PredictExecutor

    executor = PredictExecutor(config={
        "model_id":   "my_model_v1",
        "input_path": "mart/input.parquet",
        "output_id":  "batch_2024_01",
        "job_id":     "job_001",
    })
    result = executor.run()
"""

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

__all__ = [
    "BaseExecutor",
    "ExecutorException",
    "ExecutorStatus",
]
