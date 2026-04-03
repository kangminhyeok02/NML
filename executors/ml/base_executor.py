"""
base_executor.py
----------------
모든 ML Executor의 공통 인터페이스를 정의하는 추상 기반 클래스.

모든 executor는 BaseExecutor를 상속받아 execute() 메서드를 구현해야 한다.
공통 기능(로깅, 잡 상태 관리, 데이터 로드, 결과 저장)은 이 클래스에서 제공한다.
"""

import os
import json
import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


logger = logging.getLogger(__name__)


class ExecutorStatus:
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"


class ExecutorException(Exception):
    """Executor 실행 중 발생하는 내부 예외."""
    pass


class BaseExecutor(ABC):
    """
    모든 ML Executor의 추상 기반 클래스.

    서브클래스는 반드시 execute()를 구현해야 한다.
    공통 유틸(잡 상태 기록, 데이터 로드, 결과 저장)은 이 클래스에서 상속받아 사용한다.

    Parameters
    ----------
    config : dict
        executor 실행에 필요한 파라미터 딕셔너리.
        필수 키: job_id, service_id, project_id
    db_session : optional
        SQLAlchemy 또는 커스텀 DB 세션 객체.
    file_root_dir : str, optional
        파일 서버 루트 경로. 기본값은 FILE_ROOT_DIR 환경변수.
    """

    def __init__(
        self,
        config: dict,
        db_session=None,
        file_root_dir: Optional[str] = None,
    ):
        self.config       = config
        self.db_session   = db_session
        self.file_root    = Path(file_root_dir or os.getenv("FILE_ROOT_DIR", "/data"))
        self.job_id       = config.get("job_id", "unknown")
        self.service_id   = config.get("service_id", "unknown")
        self.project_id   = config.get("project_id", "unknown")
        self.started_at   = None
        self.finished_at  = None
        self._status      = ExecutorStatus.PENDING

        self._setup_logger()

    # ------------------------------------------------------------------
    # 추상 메서드 — 서브클래스에서 반드시 구현
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self) -> dict:
        """
        실제 ML 작업을 수행하고 결과 딕셔너리를 반환한다.

        Returns
        -------
        dict
            최소한 다음 키를 포함해야 한다.
            - status   : "COMPLETED" | "FAILED"
            - job_id   : str
            - result   : dict  (executor별 결과)
            - message  : str   (요약 메시지)
        """

    # ------------------------------------------------------------------
    # 공통 실행 래퍼
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        execute()를 감싸는 공통 실행 메서드.
        시작/종료 시각 기록, 예외 처리, 잡 상태 업데이트를 담당한다.
        """
        self.started_at = datetime.now()
        self._update_job_status(ExecutorStatus.RUNNING)
        logger.info("[%s] executor started  job_id=%s", self.__class__.__name__, self.job_id)

        try:
            result = self.execute()
            self.finished_at = datetime.now()
            result.setdefault("job_id", self.job_id)
            result.setdefault("elapsed_sec", self._elapsed())
            self._update_job_status(ExecutorStatus.COMPLETED, result=result)
            logger.info(
                "[%s] executor completed  job_id=%s  elapsed=%.1fs",
                self.__class__.__name__, self.job_id, self._elapsed(),
            )
            return result

        except ExecutorException as exc:
            return self._handle_failure(str(exc))
        except Exception:
            return self._handle_failure(traceback.format_exc())

    # ------------------------------------------------------------------
    # 잡 상태 관리
    # ------------------------------------------------------------------

    def _update_job_status(
        self,
        status: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[dict] = None,
    ) -> None:
        """잡 상태 파일을 갱신한다."""
        self._status = status
        job_dir = self.file_root / "jobs"
        job_dir.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "job_id":     self.job_id,
            "status":     status,
            "updated_at": datetime.now().isoformat(),
        }
        if progress is not None:
            payload["progress"] = round(progress, 2)
        if message:
            payload["message"] = message
        if result:
            payload["result"] = result

        with open(job_dir / f"{self.job_id}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 데이터 로드 / 저장 헬퍼
    # ------------------------------------------------------------------

    def _load_dataframe(self, relative_path: str) -> pd.DataFrame:
        """
        FILE_ROOT_DIR 기준 상대 경로의 parquet 또는 csv 파일을 로드한다.
        """
        full_path = self.file_root / relative_path
        if not full_path.exists():
            raise ExecutorException(f"데이터 파일을 찾을 수 없습니다: {full_path}")

        suffix = full_path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(full_path)
        elif suffix in (".csv", ".txt"):
            return pd.read_csv(full_path)
        else:
            raise ExecutorException(f"지원하지 않는 파일 형식입니다: {suffix}")

    def _save_dataframe(self, df: pd.DataFrame, relative_path: str) -> str:
        """
        DataFrame을 FILE_ROOT_DIR 기준 상대 경로에 parquet으로 저장한다.
        저장된 절대 경로를 반환한다.
        """
        full_path = self.file_root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(full_path, index=False)
        logger.debug("saved dataframe → %s  shape=%s", full_path, df.shape)
        return str(full_path)

    def _save_json(self, data: dict, relative_path: str) -> str:
        """dict를 JSON 파일로 저장하고 절대 경로를 반환한다."""
        full_path = self.file_root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(full_path)

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def _handle_failure(self, error_msg: str) -> dict:
        self.finished_at = datetime.now()
        logger.error("[%s] executor failed  job_id=%s\n%s", self.__class__.__name__, self.job_id, error_msg)
        self._update_job_status(ExecutorStatus.FAILED, message=error_msg)
        return {
            "status":      ExecutorStatus.FAILED,
            "job_id":      self.job_id,
            "result":      {},
            "message":     error_msg,
            "elapsed_sec": self._elapsed(),
        }

    def _setup_logger(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
