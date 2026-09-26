"""Run provenance: git commit + MLflow logging (optional; runs fine without MLflow)."""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from api.config import ROOT

REPORTS_DIR = Path(os.environ.get("PS_REPORTS_DIR", str(ROOT / "eval" / "reports")))


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout
        return sha + ("-dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


@contextmanager
def mlflow_run(experiment: str, run_name: str, enabled: bool = True):
    """Yields an object with log_params / log_metrics / log_artifact / run_id,
    or a no-op stand-in when MLflow is disabled or not installed."""
    if not enabled:
        yield _NoRun()
        return
    try:
        import mlflow
    except ImportError:
        yield _NoRun()
        return
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        yield _Run(mlflow, run.info.run_id, run.info.experiment_id)


class _NoRun:
    run_id: str | None = None
    experiment_id: str | None = None

    def log_params(self, p: dict[str, Any]) -> None: ...
    def log_metrics(self, m: dict[str, float]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def set_tags(self, t: dict[str, str]) -> None: ...


class _Run(_NoRun):
    def __init__(self, mlflow, run_id: str, experiment_id: str):
        self._m = mlflow
        self.run_id = run_id
        self.experiment_id = experiment_id

    def log_params(self, p: dict[str, Any]) -> None:
        self._m.log_params({k: str(v)[:500] for k, v in p.items()})

    def log_metrics(self, m: dict[str, float]) -> None:
        clean = {_safe(k): float(v) for k, v in m.items() if v == v}  # drop NaN
        self._m.log_metrics(clean)

    def log_artifact(self, path: Path) -> None:
        self._m.log_artifact(str(path))

    def set_tags(self, t: dict[str, str]) -> None:
        self._m.set_tags(t)


def _safe(k: str) -> str:
    # MLflow metric names allow alphanumerics, _ - . / and space; '@' is not allowed
    return k.replace("@", "_at_")
