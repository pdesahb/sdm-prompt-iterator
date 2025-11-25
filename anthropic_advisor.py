"""Anthropic AI integration for prompt analysis and improvement suggestions."""

import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """You are an expert at analyzing classification systems and improving prompts.
Your task is to analyze classification errors and suggest improvements to the classification prompt.

IMPORTANT: The AI classifier only sees category LABELS (human-readable names), NOT codes.
When suggesting specific category rules in prompts, always use the LABEL names shown in the errors.
Never reference internal codes like "csbe00001" - use the readable labels like "Cosmétiques" instead.

When analyzing errors:
1. Look for patterns in misclassifications
2. Identify ambiguous cases
3. Note systematic errors (e.g., always choosing parent category instead of child)
4. Consider edge cases that the current prompt doesn't handle well

Be concise and actionable in your analysis."""

ANALYSIS_USER_PROMPT = """Analyze these classification errors and identify patterns.

## Current Prompt
{current_prompt}

## Metrics
- Accuracy (exact match): {accuracy:.1%}
- Coverage (partial overlap): {coverage:.1%}
- Total rows evaluated: {total_rows}

## Sample of Misclassified Items
{errors_formatted}

Please provide:
1. **Key patterns** in the errors (2-3 main observations)
2. **Root causes** for the misclassifications
3. **Specific suggestions** for improving the prompt
"""

PROMPT_GENERATION_SYSTEM = """You are an expert prompt engineer specializing in classification tasks.
Your task is to improve a classification prompt based on error analysis.

IMPORTANT: The AI classifier only sees category LABELS (human-readable names), NOT internal codes.
When writing rules that reference specific categories, always use the LABEL names (e.g., "Cosmétiques", "Soins visage").
Never use codes like "csbe00001" - the AI cannot understand them.

Guidelines:
- Be specific and unambiguous
- Include examples when helpful
- Address the specific error patterns identified
- Keep the prompt concise but comprehensive
- Preserve any working aspects of the original prompt
- Use category LABELS, not codes, when referencing specific categories"""

PROMPT_GENERATION_USER = """Based on the following analysis, generate an improved classification prompt.

## Current Prompt
{current_prompt}

## Error Analysis
{error_analysis}

## Current Metrics
- Accuracy: {accuracy:.1%}
- Coverage: {coverage:.1%}

Generate an improved prompt that addresses the identified issues.
Return ONLY the new prompt text, nothing else."""


class AnthropicAdvisor:
    """Uses Anthropic to analyze errors and suggest prompt improvements."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        code_to_label: dict[str, str] | None = None,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.code_to_label = code_to_label or {}

    def analyze_errors(
        self,
        errors: list[dict],
        current_prompt: str,
        metrics: dict[str, Any],
    ) -> str:
        """Analyze classification errors and return insights."""
        errors_formatted = self._format_errors(errors)

        all_metrics = metrics.get("all", {})
        accuracy = all_metrics.get("accuracy", 0)
        coverage = all_metrics.get("coverage", 0)
        total_rows = all_metrics.get("total_rows", 0)

        user_message = ANALYSIS_USER_PROMPT.format(
            current_prompt=current_prompt,
            accuracy=accuracy,
            coverage=coverage,
            total_rows=total_rows,
            errors_formatted=errors_formatted,
        )

        logger.info("Sending error analysis request to Anthropic...")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=ANALYSIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        analysis = response.content[0].text
        logger.info("Received error analysis from Anthropic")
        return analysis

    def suggest_prompt(
        self,
        current_prompt: str,
        error_analysis: str,
        metrics: dict[str, Any],
    ) -> str:
        """Generate an improved prompt based on error analysis."""
        all_metrics = metrics.get("all", {})
        accuracy = all_metrics.get("accuracy", 0)
        coverage = all_metrics.get("coverage", 0)

        user_message = PROMPT_GENERATION_USER.format(
            current_prompt=current_prompt,
            error_analysis=error_analysis,
            accuracy=accuracy,
            coverage=coverage,
        )

        logger.info("Requesting prompt suggestion from Anthropic...")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=PROMPT_GENERATION_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        suggested_prompt = response.content[0].text.strip()
        logger.info("Received prompt suggestion from Anthropic")
        return suggested_prompt

    def iterate(
        self,
        current_prompt: str,
        errors: list[dict],
        metrics: dict[str, Any],
    ) -> dict[str, str]:
        """Complete iteration: analyze errors and suggest new prompt."""
        analysis = self.analyze_errors(errors, current_prompt, metrics)
        suggested_prompt = self.suggest_prompt(current_prompt, analysis, metrics)

        return {
            "analysis": analysis,
            "suggested_prompt": suggested_prompt,
        }

    def _translate_category_path(self, path: list[str]) -> str:
        """Translate a category path from codes to labels."""
        if not path:
            return "(none)"
        translated = []
        for code in path:
            label = self.code_to_label.get(code, code)
            translated.append(label)
        return " > ".join(translated)

    def _format_errors(self, errors: list[dict], max_errors: int = 20) -> str:
        """Format errors for inclusion in prompt."""
        lines = []
        for i, error in enumerate(errors[:max_errors], 1):
            source_data = error.get("source_data", {})
            source_str = ", ".join(
                f"{k}: {v}" for k, v in list(source_data.items())[:5]
            )

            truth = error.get("truth_categories", [])
            pred = error.get("pred_categories", [])
            overlap = "✓ partial" if error.get("has_partial_overlap") else "✗ none"

            # Translate codes to labels for display
            truth_str = (
                " | ".join(self._translate_category_path(c) for c in truth)
                if truth
                else "(none)"
            )
            pred_str = (
                " | ".join(self._translate_category_path(c) for c in pred)
                if pred
                else "(none)"
            )

            lines.append(
                f"""### Error {i}
**Source**: {source_str}
**Expected**: {truth_str}
**Got**: {pred_str}
**Overlap**: {overlap}
"""
            )

        if len(errors) > max_errors:
            lines.append(f"\n... and {len(errors) - max_errors} more errors")

        return "\n".join(lines)
