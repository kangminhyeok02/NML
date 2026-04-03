"""
process_executor.py
-------------------
전체 ML 파이프라인을 순차적으로 제어하는 오케스트레이션 실행기.

개별 executor를 단계별로 호출하고 각 단계의 결과를 다음 단계로 전달한다.
파이프라인 진행 상황을 잡 상태 파일에 기록하며, 특정 단계 실패 시
중단/계속/스킵 정책에 따라 동작한다.

파이프라인 예시:
  mart → data_analysis → python_model → scorecard → predict → stg → report → export

실행 순서:
  1. 파이프라인 설정 파싱
  2. 단계별 executor 인스턴스 생성
  3. 순차 실행 및 단계 결과 수집
  4. 전체 파이프라인 결과 요약
"""

import importlib
import logging
from typing import Any, Optional

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)

# executor 모듈 레지스트리 (module_path, class_name)
EXECUTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "mart":             ("executors.ml.mart_executor",          "MartExecutor"),
    "data_analysis":    ("executors.ml.data_analysis_executor", "DataAnalysisExecutor"),
    "python_model":     ("executors.ml.python_model_executor",  "PythonModelExecutor"),
    "h2o_model":        ("executors.ml.h2o_model_executor",     "H2OModelExecutor"),
    "r_model":          ("executors.ml.r_model_executor",       "RModelExecutor"),
    "automl":           ("executors.ml.automl_executor",        "AutoMLExecutor"),
    "scorecard":        ("executors.ml.scorecard_executor",     "ScorecardExecutor"),
    "predict":          ("executors.ml.predict_executor",       "PredictExecutor"),
    "pretrained":       ("executors.ml.pretrained_executor",    "PretrainedExecutor"),
    "report":           ("executors.ml.report_executor",        "ReportExecutor"),
    "export":           ("executors.ml.export_executor",        "ExportExecutor"),
    "rulesearch":       ("executors.ml.rulesearch_executor",    "RuleSearchExecutor"),
    "stg":              ("executors.ml.stg_executor",           "StrategyExecutor"),
    "rl":               ("executors.ml.rl_executor",            "RLExecutor"),
}


class ProcessExecutor(BaseExecutor):
    """
    ML 파이프라인 오케스트레이션 executor.

    config 필수 키
    --------------
    pipeline : list  실행할 단계 목록
      각 항목:
        - name     : str   단계 이름 (고유)
        - executor : str   executor 유형 (EXECUTOR_REGISTRY 키)
        - config   : dict  해당 executor의 config
        - on_error : str   "stop" | "skip" | "continue" (기본: "stop")
        - input_from: str  이전 단계 결과를 이 단계 config에 병합할 필드명 (선택)

    config 선택 키
    --------------
    stop_on_first_failure : bool  첫 번째 실패 시 중단 (기본 True)
    """

    def execute(self) -> dict:
        pipeline = self.config.get("pipeline", [])
        if not pipeline:
            raise ExecutorException("pipeline이 비어 있습니다.")

        stop_on_fail = self.config.get("stop_on_first_failure", True)
        step_results: list[dict] = []
        context: dict[str, Any] = {}   # 단계 간 결과 공유

        total_steps = len(pipeline)

        for step_idx, step in enumerate(pipeline):
            step_name     = step["name"]
            executor_type = step["executor"]
            step_config   = dict(step.get("config", {}))

            # 이전 단계 결과 주입
            input_from = step.get("input_from")
            if input_from and input_from in context:
                step_config.update(context[input_from])

            # job_id / service_id 전파
            step_config.setdefault("job_id",     f"{self.job_id}__{step_name}")
            step_config.setdefault("service_id", self.service_id)
            step_config.setdefault("project_id", self.project_id)

            progress_start = int(step_idx / total_steps * 90)
            self._update_job_status(
                ExecutorStatus.RUNNING,
                progress=float(progress_start),
                message=f"실행 중: [{step_idx+1}/{total_steps}] {step_name}",
            )
            logger.info("[Pipeline] step %d/%d: %s (%s)", step_idx + 1, total_steps, step_name, executor_type)

            # executor 인스턴스 생성 및 실행
            try:
                executor = self._build_executor(executor_type, step_config)
                step_result = executor.run()
            except Exception as exc:
                step_result = {
                    "status":  ExecutorStatus.FAILED,
                    "job_id":  step_config["job_id"],
                    "result":  {},
                    "message": str(exc),
                }

            step_results.append({"step": step_name, **step_result})

            # 결과를 context에 저장
            context[step_name] = step_result.get("result", {})

            # 실패 처리
            if step_result["status"] == ExecutorStatus.FAILED:
                on_error = step.get("on_error", "stop")
                logger.warning("[Pipeline] step failed: %s  policy=%s", step_name, on_error)
                if on_error == "stop" and stop_on_fail:
                    break
                elif on_error == "skip":
                    continue

        # 전체 결과 요약
        failed_steps  = [r["step"] for r in step_results if r["status"] == ExecutorStatus.FAILED]
        overall_status = ExecutorStatus.FAILED if failed_steps else ExecutorStatus.COMPLETED

        return {
            "status":        overall_status,
            "result": {
                "pipeline_name": self.config.get("pipeline_name", "unnamed"),
                "total_steps":   total_steps,
                "executed":      len(step_results),
                "failed_steps":  failed_steps,
                "step_results":  step_results,
            },
            "message": (
                f"파이프라인 완료  {len(step_results)}/{total_steps}단계"
                + (f"  실패={failed_steps}" if failed_steps else "")
            ),
        }

    # ------------------------------------------------------------------

    def _build_executor(self, executor_type: str, config: dict) -> BaseExecutor:
        if executor_type not in EXECUTOR_REGISTRY:
            raise ExecutorException(f"등록되지 않은 executor: {executor_type}  등록 목록: {list(EXECUTOR_REGISTRY)}")

        module_path, class_name = EXECUTOR_REGISTRY[executor_type]
        module = importlib.import_module(module_path)
        cls    = getattr(module, class_name)
        return cls(config=config, db_session=self.db_session, file_root_dir=str(self.file_root))
