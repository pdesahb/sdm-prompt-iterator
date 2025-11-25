"""Evaluator for comparing classification predictions against ground truth."""

import logging
from dataclasses import dataclass
from typing import Any

from config import CATEGORY_SUFFIX, CONFIDENCE_COLUMN, STATUS_AUTOMATED, STATUS_TO_CHECK

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluation for a subset of data."""

    accuracy: float
    coverage: float  # Recouvrement
    total_rows: int
    correct_rows: int
    covered_rows: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "total_rows": self.total_rows,
            "correct_rows": self.correct_rows,
            "covered_rows": self.covered_rows,
        }


@dataclass
class ComparisonRow:
    """A single comparison between ground truth and prediction."""

    match_key: dict[str, str]
    source_data: dict[str, Any]
    truth_categories: list[list[str]]
    pred_categories: list[list[str]]
    status: str  # Status from prediction (automated, to_check, etc.)
    is_exact_match: bool
    has_overlap: bool


class Evaluator:
    """Evaluates classification predictions against ground truth."""

    def __init__(self, match_keys: list[str], field_name: str):
        """
        Initialize evaluator.

        Args:
            match_keys: Column names used to match rows between datasets
            field_name: The classification field name (e.g., 'category')
        """
        self.match_keys = match_keys
        self.field_name = field_name
        self.category_column = f"{field_name}{CATEGORY_SUFFIX}"

    def compare(
        self,
        ground_truth_rows: list[dict],
        prediction_rows: list[dict],
    ) -> dict[str, EvaluationResult]:
        """
        Compare predictions against ground truth.

        Returns metrics for:
        - 'all': All matched rows
        - 'automated': Rows with automated status
        - 'to_check': Rows with to_check status
        """
        # Build lookup for ground truth by match key
        truth_lookup = self._build_lookup(ground_truth_rows, is_ground_truth=True)
        logger.info(f"Ground truth lookup: {len(truth_lookup)} unique keys")

        # Compare each prediction against ground truth
        comparisons: list[ComparisonRow] = []
        unmatched_predictions = 0

        for pred_row in prediction_rows:
            key = self._build_key(pred_row)
            if key not in truth_lookup:
                unmatched_predictions += 1
                continue

            truth_row = truth_lookup[key]
            truth_cats = truth_row.get("categories", [])
            pred_cats = self._extract_categories(pred_row)
            status = pred_row.get(CONFIDENCE_COLUMN, "unknown")

            comparison = ComparisonRow(
                match_key=dict(zip(self.match_keys, key)),
                source_data={k: pred_row.get(k) for k in self.match_keys},
                truth_categories=truth_cats,
                pred_categories=pred_cats,
                status=status,
                is_exact_match=self._exact_match(truth_cats, pred_cats),
                has_overlap=self._has_overlap(truth_cats, pred_cats),
            )
            comparisons.append(comparison)

        if unmatched_predictions > 0:
            logger.warning(
                f"{unmatched_predictions} predictions could not be matched to ground truth"
            )

        logger.info(f"Compared {len(comparisons)} rows")

        # Calculate metrics for different subsets
        return {
            "all": self._calculate_metrics(comparisons),
            "automated": self._calculate_metrics(
                [c for c in comparisons if c.status == STATUS_AUTOMATED]
            ),
            "to_check": self._calculate_metrics(
                [c for c in comparisons if c.status == STATUS_TO_CHECK]
            ),
        }

    def get_errors(
        self,
        ground_truth_rows: list[dict],
        prediction_rows: list[dict],
        *,
        max_errors: int = 50,
    ) -> list[dict]:
        """Get a sample of misclassified rows for analysis."""
        truth_lookup = self._build_lookup(ground_truth_rows, is_ground_truth=True)
        errors = []

        for pred_row in prediction_rows:
            key = self._build_key(pred_row)
            if key not in truth_lookup:
                continue

            truth_row = truth_lookup[key]
            truth_cats = truth_row.get("categories", [])
            pred_cats = self._extract_categories(pred_row)

            if not self._exact_match(truth_cats, pred_cats):
                errors.append(
                    {
                        "match_key": dict(zip(self.match_keys, key)),
                        "source_data": {
                            k: pred_row.get(k)
                            for k in self._get_source_columns(pred_row)
                        },
                        "truth_categories": truth_cats,
                        "pred_categories": pred_cats,
                        "status": pred_row.get(CONFIDENCE_COLUMN, "unknown"),
                        "has_partial_overlap": self._has_overlap(truth_cats, pred_cats),
                    }
                )

                if len(errors) >= max_errors:
                    break

        return errors

    def _build_key(self, row: dict) -> tuple:
        """Build a matching key from a row."""
        return tuple(str(row.get(k, "")).strip().lower() for k in self.match_keys)

    def _build_lookup(
        self, rows: list[dict], *, is_ground_truth: bool = False
    ) -> dict[tuple, dict]:
        """Build a lookup dictionary from rows."""
        lookup = {}
        duplicates = 0

        for row in rows:
            key = self._build_key(row)
            if key in lookup:
                duplicates += 1
                # Keep the latest one (in case of duplicates)
            lookup[key] = row

        if duplicates > 0:
            source = "ground truth" if is_ground_truth else "predictions"
            logger.warning(f"Found {duplicates} duplicate keys in {source}")

        return lookup

    def _extract_categories(self, row: dict) -> list[list[str]]:
        """Extract categories from a prediction row."""
        value = row.get(self.category_column)

        if value is None:
            return []

        # Handle different formats
        if isinstance(value, list):
            # Could be single category path or list of paths
            if not value:
                return []
            if isinstance(value[0], list):
                # Already list of paths
                return value
            else:
                # Single path
                return [value]
        elif isinstance(value, str):
            # Single category as string (possibly separated by /)
            if "/" in value:
                return [[p.strip() for p in value.split("/")]]
            return [[value]]

        return []

    def _get_source_columns(self, row: dict) -> list[str]:
        """Get source column names (non-internal columns)."""
        internal_prefixes = ("general_UNiFAi", "_UNiFAi")
        return [
            k
            for k in row.keys()
            if not any(k.startswith(p) or k.endswith(p) for p in internal_prefixes)
        ]

    def _exact_match(
        self, truth_cats: list[list[str]], pred_cats: list[list[str]]
    ) -> bool:
        """Check if predictions exactly match ground truth (ignoring order)."""
        if not truth_cats and not pred_cats:
            return True
        if not truth_cats or not pred_cats:
            return False

        truth_set = set(tuple(c) for c in truth_cats)
        pred_set = set(tuple(c) for c in pred_cats)
        return truth_set == pred_set

    def _has_overlap(
        self, truth_cats: list[list[str]], pred_cats: list[list[str]]
    ) -> bool:
        """Check if there's at least one category in common."""
        if not truth_cats or not pred_cats:
            return False

        truth_set = set(tuple(c) for c in truth_cats)
        pred_set = set(tuple(c) for c in pred_cats)
        return bool(truth_set & pred_set)

    def _calculate_metrics(self, comparisons: list[ComparisonRow]) -> EvaluationResult:
        """Calculate metrics for a list of comparisons."""
        if not comparisons:
            return EvaluationResult(
                accuracy=0.0,
                coverage=0.0,
                total_rows=0,
                correct_rows=0,
                covered_rows=0,
            )

        total = len(comparisons)
        correct = sum(1 for c in comparisons if c.is_exact_match)
        covered = sum(1 for c in comparisons if c.has_overlap)

        return EvaluationResult(
            accuracy=correct / total if total > 0 else 0.0,
            coverage=covered / total if total > 0 else 0.0,
            total_rows=total,
            correct_rows=correct,
            covered_rows=covered,
        )
