"""Storage management for experiments, ground truth, and run results."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config import EXPERIMENTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    step_id: int
    prev_step_id: int
    prev_step_type: str  # e.g., "mappings", "classifications", etc.
    field_name: str
    match_keys: list[str]
    truth_job_ids: list[str]
    # Mode: "manual" (existing test job) or "auto_split" (create jobs from truth data)
    mode: str = "manual"
    # For manual mode
    test_job_id: str | None = None
    # For auto_split mode
    train_job_id: str | None = None  # Created job for iteration (75%)
    eval_job_id: str | None = None  # Created job for final evaluation (25%)
    train_ratio: float = 0.75
    random_seed: int = 42
    project_id: int | None = None
    model_version: int | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        # Handle backward compatibility
        if "mode" not in data:
            data["mode"] = "manual"
        return cls(**data)

    @property
    def iteration_job_id(self) -> str:
        """Get the job ID to use for iteration (test_job_id for manual, train_job_id for auto_split)."""
        if self.mode == "auto_split":
            return self.train_job_id
        return self.test_job_id


@dataclass
class GroundTruthRow:
    """A single row of ground truth data."""

    match_key: dict[str, str]
    source_data: dict[str, Any]
    categories: list[list[str]]  # List of category paths (multi-category support)
    status: str
    source_job: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruthRow":
        return cls(**data)


@dataclass
class GroundTruth:
    """Ground truth data for an experiment."""

    extracted_at: str
    source_jobs: list[str]
    field_name: str
    rows: list[GroundTruthRow]
    # Mapping from category codes to human-readable labels
    # e.g., {"csbe00001": "Root Category", "csbe01076": "Specific Product Type"}
    code_to_label: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "extracted_at": self.extracted_at,
            "source_jobs": self.source_jobs,
            "field_name": self.field_name,
            "rows": [row.to_dict() for row in self.rows],
            "code_to_label": self.code_to_label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":
        rows = [GroundTruthRow.from_dict(r) for r in data.get("rows", [])]
        return cls(
            extracted_at=data["extracted_at"],
            source_jobs=data["source_jobs"],
            field_name=data["field_name"],
            rows=rows,
            code_to_label=data.get("code_to_label", {}),
        )


@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation subset."""

    accuracy: float
    coverage: float
    total_rows: int
    correct_rows: int
    covered_rows: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationMetrics":
        return cls(**data)


@dataclass
class RunResult:
    """Result of a single evaluation run."""

    run_id: str
    timestamp: str
    prompt: str
    metrics: dict[str, EvaluationMetrics]  # 'all', 'automated', 'to_check'
    errors_sample: list[dict]
    anthropic_analysis: str | None = None
    suggested_prompt: str | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "errors_sample": self.errors_sample,
            "anthropic_analysis": self.anthropic_analysis,
            "suggested_prompt": self.suggested_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunResult":
        metrics = {
            k: EvaluationMetrics.from_dict(v)
            for k, v in data.get("metrics", {}).items()
        }
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            prompt=data["prompt"],
            metrics=metrics,
            errors_sample=data.get("errors_sample", []),
            anthropic_analysis=data.get("anthropic_analysis"),
            suggested_prompt=data.get("suggested_prompt"),
        )


class ExperimentStorage:
    """Manages storage for a single experiment."""

    def __init__(self, experiment_name: str, base_dir: Path = EXPERIMENTS_DIR):
        self.name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = base_dir / experiment_name
        self.runs_dir = self.experiment_dir / "runs"

    @property
    def config_path(self) -> Path:
        return self.experiment_dir / "config.json"

    @property
    def ground_truth_path(self) -> Path:
        return self.experiment_dir / "ground_truth.json"

    def exists(self) -> bool:
        """Check if experiment exists."""
        return self.experiment_dir.exists()

    def create(self, config: ExperimentConfig) -> None:
        """Create a new experiment."""
        if self.exists():
            raise ValueError(f"Experiment '{self.name}' already exists")

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

        self._save_json(self.config_path, config.to_dict())
        logger.info(f"Created experiment '{self.name}' at {self.experiment_dir}")

    def load_config(self) -> ExperimentConfig:
        """Load experiment configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {self.config_path}")
        data = self._load_json(self.config_path)
        return ExperimentConfig.from_dict(data)

    def save_ground_truth(self, ground_truth: GroundTruth) -> None:
        """Save ground truth data."""
        self._save_json(self.ground_truth_path, ground_truth.to_dict())
        logger.info(f"Saved ground truth with {len(ground_truth.rows)} rows")

    def load_ground_truth(self) -> GroundTruth:
        """Load ground truth data."""
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {self.ground_truth_path}")
        data = self._load_json(self.ground_truth_path)
        return GroundTruth.from_dict(data)

    def has_ground_truth(self) -> bool:
        """Check if ground truth exists."""
        return self.ground_truth_path.exists()

    def save_run(self, result: RunResult) -> Path:
        """Save a run result."""
        run_dir = self.runs_dir / result.run_id
        run_dir.mkdir(exist_ok=True)

        # Save prompt
        prompt_path = run_dir / "prompt.txt"
        prompt_path.write_text(result.prompt)

        # Save metrics
        metrics_path = run_dir / "metrics.json"
        self._save_json(metrics_path, result.to_dict())

        logger.info(f"Saved run {result.run_id} to {run_dir}")
        return run_dir

    def list_runs(self) -> list[str]:
        """List all run IDs for this experiment."""
        if not self.runs_dir.exists():
            return []
        return sorted(
            [d.name for d in self.runs_dir.iterdir() if d.is_dir()], reverse=True
        )

    def load_run(self, run_id: str) -> RunResult:
        """Load a specific run result."""
        metrics_path = self.runs_dir / run_id / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        data = self._load_json(metrics_path)
        return RunResult.from_dict(data)

    def get_latest_run(self) -> RunResult | None:
        """Get the most recent run result."""
        runs = self.list_runs()
        if not runs:
            return None
        return self.load_run(runs[0])

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        """Save data to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_json(path: Path) -> dict:
        """Load data from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def list_experiments(base_dir: Path = EXPERIMENTS_DIR) -> list[str]:
    """List all available experiments."""
    if not base_dir.exists():
        return []
    return sorted(
        [
            d.name
            for d in base_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]
    )


def generate_run_id() -> str:
    """Generate a unique run ID."""
    import uuid

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"
