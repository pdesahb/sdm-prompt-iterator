#!/usr/bin/env python3
"""CLI for SDM Prompt Iterator - Automate classification prompt optimization."""

import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
from dotenv import load_dotenv

# Load .env file from script directory
load_dotenv(Path(__file__).parent / ".env")
from rich.console import Console
from rich.table import Table

from anthropic_advisor import AnthropicAdvisor
from config import (
    CATEGORY_SUFFIX,
    CONFIDENCE_COLUMN,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAIN_RATIO,
    GROUND_TRUTH_STATUSES,
    SDM_BASE_URL,
)
from evaluator import Evaluator
from sdm_client import SDMClient
from storage import (
    ExperimentConfig,
    ExperimentStorage,
    GroundTruth,
    GroundTruthRow,
    RunResult,
    generate_run_id,
    list_experiments,
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


# ===== VERBOSITY HELPERS =====


def log_progress(msg: str, verbosity: int, min_level: int = 1) -> None:
    """Print message if verbosity >= min_level."""
    if verbosity >= min_level:
        console.print(msg)


def log_detail(msg: str, verbosity: int) -> None:
    """Print detailed message (level 2 only)."""
    if verbosity >= 2:
        console.print(msg)


# ===== CLIENT HELPER =====


def get_client(email: str, password: str) -> SDMClient:
    """Create and authenticate SDM client."""
    client = SDMClient(base_url=SDM_BASE_URL, email=email, password=password)
    client.authenticate()
    return client


def check_job_at_classification_step(
    client: SDMClient, job_id: str, step_id: int, verbosity: int
) -> bool:
    """Check if job is at or past the classification step.

    Returns True if the job is ready, False if not (with user-friendly error message).
    """
    step_info = client.is_job_at_or_past_step(job_id, step_id)

    if not step_info["is_at_or_past"]:
        console.print(
            f"\n[red]Error: Job {job_id} is not at the classification step yet.[/red]"
        )
        console.print(
            f"  Current step: [yellow]{step_info['current_step_name']}[/yellow] "
            f"(step {step_info['current_step_number']})"
        )
        console.print(
            f"  Required step: Classification (step {step_info['target_step_number']})"
        )
        console.print(f"  Job status: {step_info['status']}")
        console.print(
            "\n[yellow]Please advance the job to the classification step in SDM before running this command.[/yellow]"
        )
        return False

    log_detail(
        f"[dim]Job {job_id} is at step: {step_info['current_step_name']}[/dim]",
        verbosity,
    )
    return True


@click.group()
def cli():
    """SDM Prompt Iterator - Automate classification prompt optimization."""
    pass


# ===== CREATE EXPERIMENT =====


@cli.command("create")
@click.option("--experiment", help="Experiment name")
@click.option("--email", envvar="SDM_USER", help="SDM email (or SDM_USER env var)")
@click.option(
    "--password", envvar="SDM_PASSWORD", help="SDM password (or SDM_PASSWORD env var)"
)
@click.option("--step-id", type=int, help="Classification step ID")
@click.option(
    "--prev-step-id", type=int, help="Previous step ID (auto-detected if not provided)"
)
@click.option("--field", help="Classification field name (e.g., 'category')")
@click.option("--match-keys", help="Comma-separated column names for row matching")
@click.option("--truth-jobs", help="Comma-separated job IDs containing ground truth")
@click.option(
    "--mode",
    type=click.Choice(["auto-split", "manual"]),
    help="Experiment mode: auto-split creates train/eval jobs, manual uses existing test job",
)
@click.option("--train-job", help="Train job ID (manual mode)")
@click.option("--eval-job", help="Eval job ID (manual mode, optional)")
@click.option(
    "--train-ratio", type=float, help="Train ratio for auto-split (default: 0.75)"
)
@click.option("--seed", type=int, help="Random seed for auto-split (default: 42)")
@click.option(
    "-v",
    "--verbosity",
    default=1,
    type=click.IntRange(0, 2),
    help="Output verbosity: 0=errors, 1=progress (default), 2=detailed",
)
def create(
    experiment: str | None,
    email: str | None,
    password: str | None,
    step_id: int | None,
    prev_step_id: int | None,
    field: str | None,
    match_keys: str | None,
    truth_jobs: str | None,
    mode: str | None,
    train_job: str | None,
    eval_job: str | None,
    train_ratio: float | None,
    seed: int | None,
    verbosity: int,
):
    """Create a new experiment (interactive if options not provided)."""
    # Prompt for required fields if not provided
    if not experiment:
        experiment = click.prompt("Experiment name")
    if not step_id:
        step_id = click.prompt("Classification step ID", type=int)
    if not field:
        field = click.prompt("Classification field name (e.g., 'category')")
    if not match_keys:
        match_keys = click.prompt("Columns to match rows (comma-separated)")
    if not truth_jobs:
        truth_jobs = click.prompt("Job IDs containing ground truth (comma-separated)")

    # Mode selection
    if not mode:
        mode = click.prompt(
            "Mode",
            type=click.Choice(["auto-split", "manual"]),
            default="auto-split",
        )

    # Mode-specific options
    if mode == "auto-split":
        if train_ratio is None:
            train_ratio = click.prompt(
                "Train ratio", default=DEFAULT_TRAIN_RATIO, type=float
            )
        if seed is None:
            seed = click.prompt("Random seed", default=DEFAULT_RANDOM_SEED, type=int)
    else:  # manual
        if not train_job:
            train_job = click.prompt("Train job ID")
        if not eval_job:
            eval_job = click.prompt(
                "Eval job ID (optional, press Enter to skip)",
                default="",
                show_default=False,
            )
            if not eval_job:
                eval_job = None

    # Check if experiment already exists
    storage = ExperimentStorage(experiment)
    if storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' already exists[/red]")
        sys.exit(1)

    # Parse arguments
    match_keys_list = [k.strip() for k in match_keys.split(",")]
    truth_jobs_list = [j.strip() for j in truth_jobs.split(",")]

    # Prompt for credentials if not provided
    if not email:
        email = click.prompt("SDM email")
    if not password:
        password = click.prompt("SDM password", hide_input=True)

    client = get_client(email, password)

    if mode == "auto-split":
        # ===== AUTO-SPLIT MODE =====
        # Get project info from first truth job
        log_progress("[blue]Getting project info from truth jobs...[/blue]", verbosity)
        project_info = client.get_project_info_from_job(truth_jobs_list[0])
        project_id = project_info["project_id"]
        model_version = project_info["model_version"]
        log_detail(
            f"  Project ID: {project_id}, Model version: {model_version}", verbosity
        )

        # Auto-detect previous step if not provided
        prev_step_type = None
        if prev_step_id is None:
            log_progress("[yellow]Auto-detecting previous step...[/yellow]", verbosity)
            prev_step_info = client.get_previous_step_info(truth_jobs_list[0], step_id)

            if prev_step_info is None:
                console.print("[red]Could not auto-detect previous step.[/red]")
                prev_step_id = click.prompt(
                    "Please enter the previous step ID", type=int
                )
                prev_step_type = click.prompt(
                    "Please enter the previous step type (e.g., 'mappings')", type=str
                )
            else:
                prev_step_id = prev_step_info["id"]
                prev_step_type = prev_step_info["type"]
                log_progress(
                    f"[green]Detected previous step: {prev_step_info['name']} (ID: {prev_step_id})[/green]",
                    verbosity,
                )
                log_detail(f"[dim]Step type: {prev_step_type}[/dim]", verbosity)
        else:
            prev_step_info = client.get_previous_step_info(truth_jobs_list[0], step_id)
            prev_step_type = prev_step_info["type"] if prev_step_info else "mappings"

        # Download and concatenate source data from all truth jobs
        log_progress(
            "[blue]Downloading source data from truth jobs...[/blue]", verbosity
        )
        all_source_rows = []
        for job_id in truth_jobs_list:
            log_detail(f"  Downloading from job {job_id}...", verbosity)
            rows = client.download_job_source_data(job_id)
            all_source_rows.extend(rows)
            log_detail(f"    Got {len(rows)} rows", verbosity)

        log_progress(
            f"[green]Total source rows: {len(all_source_rows)}[/green]", verbosity
        )

        # Shuffle and split
        random.seed(seed)
        random.shuffle(all_source_rows)

        split_idx = int(len(all_source_rows) * train_ratio)
        train_rows = all_source_rows[:split_idx]
        eval_rows = all_source_rows[split_idx:]

        log_progress(
            f"  Train set: {len(train_rows)} rows ({train_ratio:.0%})", verbosity
        )
        log_progress(
            f"  Eval set: {len(eval_rows)} rows ({1 - train_ratio:.0%})", verbosity
        )

        # Upload files and create jobs
        log_progress("[blue]Creating train job...[/blue]", verbosity)
        train_file_id = client.upload_file_json(train_rows)
        train_job_id = client.create_job(
            name=f"{experiment}_train",
            project_id=project_id,
            file_id=train_file_id,
            model_version=model_version,
        )
        log_progress(f"  Train job created: {train_job_id}", verbosity)

        log_progress("[blue]Creating eval job...[/blue]", verbosity)
        eval_file_id = client.upload_file_json(eval_rows)
        eval_job_id = client.create_job(
            name=f"{experiment}_eval",
            project_id=project_id,
            file_id=eval_file_id,
            model_version=model_version,
        )
        log_progress(f"  Eval job created: {eval_job_id}", verbosity)

        # Wait for jobs to complete initial processing
        log_progress(
            "[blue]Waiting for jobs to complete initial run...[/blue]", verbosity
        )
        client.wait_for_completion(train_job_id)
        client.wait_for_completion(eval_job_id)

        # Create experiment config
        config = ExperimentConfig(
            name=experiment,
            step_id=step_id,
            prev_step_id=prev_step_id,
            prev_step_type=prev_step_type,
            field_name=field,
            match_keys=match_keys_list,
            truth_job_ids=truth_jobs_list,
            mode="auto_split",
            train_job_id=train_job_id,
            eval_job_id=eval_job_id,
            train_ratio=train_ratio,
            random_seed=seed,
            project_id=project_id,
            model_version=model_version,
        )

        storage.create(config)
        log_progress(
            f"\n[green]Created experiment '{experiment}' (auto-split mode)[/green]",
            verbosity,
        )
        log_progress(f"  Step ID: {step_id}", verbosity)
        log_detail(f"  Previous Step: {prev_step_id} ({prev_step_type})", verbosity)
        log_detail(f"  Field: {field}", verbosity)
        log_progress(f"  Train job (for iteration): {train_job_id}", verbosity)
        log_progress(f"  Eval job (for final eval): {eval_job_id}", verbosity)
        log_detail(
            f"  Split ratio: {train_ratio:.0%} train / {1 - train_ratio:.0%} eval",
            verbosity,
        )

    else:
        # ===== MANUAL MODE =====
        # Auto-detect previous step if not provided
        prev_step_type = None
        if prev_step_id is None:
            log_progress("[yellow]Auto-detecting previous step...[/yellow]", verbosity)
            prev_step_info = client.get_previous_step_info(train_job, step_id)

            if prev_step_info is None:
                console.print("[red]Could not auto-detect previous step.[/red]")
                prev_step_id = click.prompt(
                    "Please enter the previous step ID", type=int
                )
                prev_step_type = click.prompt(
                    "Please enter the previous step type (e.g., 'mappings')", type=str
                )
            else:
                prev_step_id = prev_step_info["id"]
                prev_step_type = prev_step_info["type"]
                log_progress(
                    f"[green]Detected previous step: {prev_step_info['name']} (ID: {prev_step_id})[/green]",
                    verbosity,
                )
                log_detail(f"[dim]Step type: {prev_step_type}[/dim]", verbosity)
        else:
            prev_step_info = client.get_previous_step_info(train_job, step_id)
            prev_step_type = prev_step_info["type"] if prev_step_info else "mappings"

        # Create experiment config
        config = ExperimentConfig(
            name=experiment,
            step_id=step_id,
            prev_step_id=prev_step_id,
            prev_step_type=prev_step_type,
            field_name=field,
            match_keys=match_keys_list,
            truth_job_ids=truth_jobs_list,
            mode="manual",
            train_job_id=train_job,
            eval_job_id=eval_job,
        )

        storage.create(config)
        log_progress(
            f"\n[green]Created experiment '{experiment}' (manual mode)[/green]",
            verbosity,
        )
        log_progress(f"  Step ID: {step_id}", verbosity)
        log_detail(f"  Previous Step: {prev_step_id} ({prev_step_type})", verbosity)
        log_detail(f"  Field: {field}", verbosity)
        log_progress(f"  Train job: {train_job}", verbosity)
        if eval_job:
            log_progress(f"  Eval job: {eval_job}", verbosity)
        log_detail(f"  Truth jobs: {truth_jobs_list}", verbosity)

    # Extract ground truth from truth jobs
    log_progress("\n[blue]Extracting ground truth from truth jobs...[/blue]", verbosity)
    _extract_ground_truth(storage, config, client, verbosity)

    log_progress(
        "\n[yellow]Next step: Run 'iterate' or 'evaluate' to start optimizing[/yellow]",
        verbosity,
    )


# ===== EXTRACT GROUND TRUTH =====


def _extract_ground_truth(
    storage: ExperimentStorage,
    config: ExperimentConfig,
    client: SDMClient,
    verbosity: int,
) -> None:
    """Extract ground truth from validation jobs and save to storage."""
    all_rows: list[GroundTruthRow] = []
    all_code_to_label: dict[str, str] = {}
    category_column = f"{config.field_name}{CATEGORY_SUFFIX}"

    for job_id in config.truth_job_ids:
        log_progress(f"[blue]Extracting from job {job_id}...[/blue]", verbosity)

        # Get classification results with labels, filtered by validated/corrected status
        results, labels = client.get_classification_results(
            action_id=config.step_id,
            job_id=job_id,
            status_filter=GROUND_TRUTH_STATUSES,
            include_labels=True,
        )

        log_progress(f"  Found {len(results)} validated rows", verbosity)

        # Build code-to-label mapping from this job
        job_code_to_label = _build_code_to_label_map(results, labels, category_column)
        all_code_to_label.update(job_code_to_label)
        log_detail(f"  Extracted {len(job_code_to_label)} category labels", verbosity)

        for row in results:
            # Build match key
            match_key = {k: str(row.get(k, "")).strip() for k in config.match_keys}

            # Extract categories (handle multiple)
            categories = _extract_categories(row.get(category_column))

            # Extract source data (non-internal columns)
            source_data = {
                k: v
                for k, v in row.items()
                if not k.startswith("general_UNiFAi") and not k.endswith("_UNiFAi")
            }

            gt_row = GroundTruthRow(
                match_key=match_key,
                source_data=source_data,
                categories=categories,
                status=row.get(CONFIDENCE_COLUMN, "unknown"),
                source_job=job_id,
            )
            all_rows.append(gt_row)

    # Deduplicate by match key (keep latest)
    unique_rows = _deduplicate_rows(all_rows, config.match_keys)

    ground_truth = GroundTruth(
        extracted_at=datetime.now(UTC).isoformat(),
        source_jobs=config.truth_job_ids,
        field_name=config.field_name,
        rows=unique_rows,
        code_to_label=all_code_to_label,
    )

    storage.save_ground_truth(ground_truth)
    log_progress(
        f"[green]Saved ground truth with {len(unique_rows)} unique rows[/green]",
        verbosity,
    )
    log_detail(
        f"[green]Saved {len(all_code_to_label)} category code-to-label mappings[/green]",
        verbosity,
    )


@cli.command("extract-truth")
@click.option("--experiment", required=True, help="Experiment name")
@click.option(
    "--email", envvar="SDM_USER", required=True, help="SDM email (or SDM_USER env var)"
)
@click.option(
    "--password",
    envvar="SDM_PASSWORD",
    required=True,
    help="SDM password (or SDM_PASSWORD env var)",
)
@click.option(
    "-v",
    "--verbosity",
    default=1,
    type=click.IntRange(0, 2),
    help="Output verbosity: 0=errors, 1=progress (default), 2=detailed",
)
def extract_truth(experiment: str, email: str, password: str, verbosity: int):
    """Re-extract ground truth from validation jobs (already done during create)."""
    storage = ExperimentStorage(experiment)

    if not storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' not found[/red]")
        sys.exit(1)

    config = storage.load_config()
    client = get_client(email, password)

    _extract_ground_truth(storage, config, client, verbosity)


# ===== EVALUATE =====


@cli.command()
@click.option("--experiment", required=True, help="Experiment name")
@click.option(
    "--email", envvar="SDM_USER", required=True, help="SDM email (or SDM_USER env var)"
)
@click.option(
    "--password",
    envvar="SDM_PASSWORD",
    required=True,
    help="SDM password (or SDM_PASSWORD env var)",
)
@click.option("--skip-rerun", is_flag=True, help="Skip job re-run, use current results")
@click.option(
    "--use-eval-job",
    is_flag=True,
    help="For auto-split mode: use eval job instead of train job",
)
@click.option(
    "-v",
    "--verbosity",
    default=1,
    type=click.IntRange(0, 2),
    help="Output verbosity: 0=errors, 1=progress (default), 2=detailed",
)
def evaluate(
    experiment: str,
    email: str,
    password: str,
    skip_rerun: bool,
    use_eval_job: bool,
    verbosity: int,
):
    """Evaluate current prompt against ground truth."""
    storage = ExperimentStorage(experiment)

    if not storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' not found[/red]")
        sys.exit(1)

    if not storage.has_ground_truth():
        console.print(
            "[red]Error: Ground truth not found. Run 'extract-truth' first.[/red]"
        )
        sys.exit(1)

    config = storage.load_config()
    ground_truth = storage.load_ground_truth()
    client = get_client(email, password)

    # Determine which job to use
    if config.mode == "auto_split" and use_eval_job:
        job_id = config.eval_job_id
        log_progress(
            "[cyan]Using EVAL job for evaluation (holdout set)[/cyan]", verbosity
        )
    else:
        job_id = config.iteration_job_id
        if config.mode == "auto_split":
            log_progress(
                "[cyan]Using TRAIN job for evaluation (use --use-eval-job for final eval)[/cyan]",
                verbosity,
            )

    # Check if job is at the classification step (only if not re-running)
    if skip_rerun and not check_job_at_classification_step(
        client, job_id, config.step_id, verbosity
    ):
        sys.exit(1)

    # Get current prompt
    fields_config = client.get_fields_config(config.step_id)
    current_prompt = (
        fields_config.get("fields_config", {})
        .get(config.field_name, {})
        .get("custom_prompt", "")
    )

    if not skip_rerun:
        # Return to previous step and re-run classification
        log_progress(
            f"[blue]Returning job to step {config.prev_step_id}...[/blue]", verbosity
        )
        log_detail(f"[dim]Job ID: {job_id}[/dim]", verbosity)
        client.return_to_step(job_id, config.prev_step_id, reset=True)

        # Wait for previous step to be ready for validation
        log_progress(
            f"[blue]Waiting for step {config.prev_step_id} to be ready...[/blue]",
            verbosity,
        )
        client.wait_for_completion(job_id, target_step_id=config.prev_step_id)

        # Validate the previous step to move forward to classification
        log_progress(
            f"[blue]Validating step {config.prev_step_id}...[/blue]", verbosity
        )
        log_detail(f"[dim]Step type: {config.prev_step_type}[/dim]", verbosity)
        client.validate_step(job_id, config.prev_step_id, config.prev_step_type)

        log_progress("[blue]Waiting for job completion...[/blue]", verbosity)
        success = client.wait_for_completion(job_id)

        if not success:
            console.print("[red]Job failed or timed out[/red]")
            sys.exit(1)

    # Get prediction results
    log_progress("[blue]Fetching classification results...[/blue]", verbosity)
    predictions = client.get_classification_results(
        action_id=config.step_id,
        job_id=job_id,
    )

    # Evaluate
    evaluator = Evaluator(match_keys=config.match_keys, field_name=config.field_name)

    # Convert ground truth rows to dict format for evaluator
    gt_rows = [
        {"categories": r.categories, **r.match_key, **r.source_data}
        for r in ground_truth.rows
    ]

    metrics = evaluator.compare(gt_rows, predictions)
    errors = evaluator.get_errors(gt_rows, predictions)

    # Display results
    eval_type = "EVAL (holdout)" if use_eval_job else "TRAIN"
    _display_metrics(metrics, current_prompt, eval_type=eval_type, verbosity=verbosity)

    # Save run
    run_id = generate_run_id()
    run_result = RunResult(
        run_id=run_id,
        timestamp=datetime.now(UTC).isoformat(),
        prompt=current_prompt,
        metrics={k: v for k, v in metrics.items()},
        errors_sample=errors[:50],
    )
    storage.save_run(run_result)

    log_progress(f"\n[green]Run saved: {run_id}[/green]", verbosity)


# ===== ITERATE =====


@cli.command()
@click.option("--experiment", required=True, help="Experiment name")
@click.option(
    "--email", envvar="SDM_USER", required=True, help="SDM email (or SDM_USER env var)"
)
@click.option(
    "--password",
    envvar="SDM_PASSWORD",
    required=True,
    help="SDM password (or SDM_PASSWORD env var)",
)
@click.option(
    "--anthropic-api-key",
    envvar="ANTHROPIC_API_KEY",
    required=True,
    help="Anthropic API key (or ANTHROPIC_API_KEY env var)",
)
@click.option("--iterations", default=3, help="Number of iterations")
@click.option(
    "--auto-apply",
    is_flag=True,
    help="Auto-apply suggested prompts without confirmation",
)
@click.option(
    "-v",
    "--verbosity",
    default=1,
    type=click.IntRange(0, 2),
    help="Output verbosity: 0=errors, 1=progress (default), 2=detailed",
)
def iterate(
    experiment: str,
    email: str,
    password: str,
    anthropic_api_key: str,
    iterations: int,
    auto_apply: bool,
    verbosity: int,
):
    """Iterate on prompt with Anthropic's suggestions (uses train job in auto-split mode)."""
    storage = ExperimentStorage(experiment)

    if not storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' not found[/red]")
        sys.exit(1)

    if not storage.has_ground_truth():
        console.print(
            "[red]Error: Ground truth not found. Run 'extract-truth' first.[/red]"
        )
        sys.exit(1)

    config = storage.load_config()
    ground_truth = storage.load_ground_truth()
    client = get_client(email, password)
    advisor = AnthropicAdvisor(
        api_key=anthropic_api_key, code_to_label=ground_truth.code_to_label
    )

    # Use train job for iteration
    job_id = config.iteration_job_id

    # Check if job is at the classification step
    if not check_job_at_classification_step(client, job_id, config.step_id, verbosity):
        sys.exit(1)

    if config.mode == "auto_split":
        log_progress(
            f"[cyan]Auto-split mode: iterating on TRAIN job ({job_id})[/cyan]",
            verbosity,
        )
        log_progress(
            "[cyan]After iteration, run 'evaluate --use-eval-job' for final evaluation on holdout set[/cyan]\n",
            verbosity,
        )

    for i in range(iterations):
        log_progress(
            f"\n[bold cyan]═══ Iteration {i + 1}/{iterations} ═══[/bold cyan]",
            verbosity,
        )

        # 1. Get current prompt and classification results
        fields_config = client.get_fields_config(config.step_id)
        current_prompt = (
            fields_config.get("fields_config", {})
            .get(config.field_name, {})
            .get("custom_prompt", "")
        )

        log_progress("[blue]Fetching classification results...[/blue]", verbosity)
        predictions = client.get_classification_results(
            action_id=config.step_id,
            job_id=job_id,
        )

        # 2. Evaluate current results
        evaluator = Evaluator(
            match_keys=config.match_keys, field_name=config.field_name
        )
        gt_rows = [
            {"categories": r.categories, **r.match_key, **r.source_data}
            for r in ground_truth.rows
        ]

        metrics = evaluator.compare(gt_rows, predictions)
        errors = evaluator.get_errors(gt_rows, predictions)

        _display_metrics(metrics, current_prompt, verbosity=verbosity)

        # 3. Get Anthropic's analysis and suggestion
        log_progress("\n[blue]Analyzing errors with Anthropic...[/blue]", verbosity)
        metrics_dict = {k: v.to_dict() for k, v in metrics.items()}
        result = advisor.iterate(current_prompt, errors, metrics_dict)

        # Level 1: show summary
        log_progress("\n[bold]Error Analysis:[/bold]", verbosity)
        if verbosity >= 2:
            console.print(result["analysis"])
        elif verbosity >= 1:
            # Show first 200 chars of analysis
            analysis_preview = (
                result["analysis"][:200] + "..."
                if len(result["analysis"]) > 200
                else result["analysis"]
            )
            console.print(analysis_preview)

        # Level 2: show full suggested prompt
        log_progress("\n[bold]Suggested Prompt:[/bold]", verbosity)
        if verbosity >= 2:
            console.print("─" * 40)
            console.print(result["suggested_prompt"])
            console.print("─" * 40)
        elif verbosity >= 1:
            prompt_preview = (
                result["suggested_prompt"][:100] + "..."
                if len(result["suggested_prompt"]) > 100
                else result["suggested_prompt"]
            )
            console.print(f"[italic]{prompt_preview}[/italic]")

        # Save run
        run_id = generate_run_id()
        run_result = RunResult(
            run_id=run_id,
            timestamp=datetime.now(UTC).isoformat(),
            prompt=current_prompt,
            metrics=metrics,
            errors_sample=errors[:50],
            anthropic_analysis=result["analysis"],
            suggested_prompt=result["suggested_prompt"],
        )
        storage.save_run(run_result)

        # 4. Apply suggestion and re-run classification?
        if i < iterations - 1:  # Don't re-run on last iteration
            if auto_apply or click.confirm(
                "\nApply suggested prompt and re-run?", default=True
            ):
                # Update prompt
                log_progress("[blue]Updating prompt...[/blue]", verbosity)
                client.update_prompt(
                    config.step_id, config.field_name, result["suggested_prompt"]
                )
                log_progress("[green]Prompt updated[/green]", verbosity)

                # Return to previous step
                log_progress(
                    f"[blue]Returning to step {config.prev_step_id} to re-run classification...[/blue]",
                    verbosity,
                )
                client.return_to_step(job_id, config.prev_step_id, reset=True)

                # Wait for previous step to be ready for validation
                log_progress(
                    f"[blue]Waiting for step {config.prev_step_id} to be ready...[/blue]",
                    verbosity,
                )
                client.wait_for_completion(job_id, target_step_id=config.prev_step_id)

                # Validate the previous step to move forward to classification
                log_progress(
                    f"[blue]Validating step {config.prev_step_id}...[/blue]", verbosity
                )
                log_detail(f"[dim]Step type: {config.prev_step_type}[/dim]", verbosity)
                client.validate_step(job_id, config.prev_step_id, config.prev_step_type)

                # Wait for classification to complete
                log_progress(
                    "[blue]Waiting for job to reach classification step...[/blue]",
                    verbosity,
                )
                log_detail(f"[dim]Target step ID: {config.step_id}[/dim]", verbosity)
                success = client.wait_for_completion(
                    job_id, target_step_id=config.step_id
                )

                if not success:
                    console.print("[red]Job failed or timed out[/red]")
                    break
            else:
                log_progress("[yellow]Stopping iteration[/yellow]", verbosity)
                break

    log_progress("\n[green]Iteration complete![/green]", verbosity)

    if config.mode == "auto_split":
        log_progress(
            "\n[yellow]Tip: Run 'evaluate --use-eval-job' to evaluate on the holdout set[/yellow]",
            verbosity,
        )


# ===== FINAL EVAL (auto-split shortcut) =====


@cli.command("final-eval")
@click.option("--experiment", required=True, help="Experiment name")
@click.option(
    "--email", envvar="SDM_USER", required=True, help="SDM email (or SDM_USER env var)"
)
@click.option(
    "--password",
    envvar="SDM_PASSWORD",
    required=True,
    help="SDM password (or SDM_PASSWORD env var)",
)
def final_eval(experiment: str, email: str, password: str):
    """Run final evaluation on holdout set (requires eval job)."""
    storage = ExperimentStorage(experiment)

    if not storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' not found[/red]")
        sys.exit(1)

    config = storage.load_config()

    if not config.eval_job_id:
        console.print("[red]Error: No eval job configured for this experiment[/red]")
        console.print(
            "[yellow]Tip: Use --eval-job when creating the experiment to enable final evaluation[/yellow]"
        )
        sys.exit(1)

    console.print("[bold cyan]═══ Final Evaluation on Holdout Set ═══[/bold cyan]")

    # Call evaluate with --use-eval-job
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        evaluate,
        [
            "--experiment",
            experiment,
            "--email",
            email,
            "--password",
            password,
            "--use-eval-job",
        ],
        catch_exceptions=False,
    )
    console.print(result.output)


# ===== RESTORE =====


@cli.command()
@click.option("--experiment", required=True, help="Experiment name")
@click.option("--run-id", required=True, help="Run ID to restore prompt from")
@click.option(
    "--email", envvar="SDM_USER", required=True, help="SDM email (or SDM_USER env var)"
)
@click.option(
    "--password",
    envvar="SDM_PASSWORD",
    required=True,
    help="SDM password (or SDM_PASSWORD env var)",
)
@click.option(
    "-v",
    "--verbosity",
    default=1,
    type=click.IntRange(0, 2),
    help="Output verbosity: 0=errors, 1=progress (default), 2=detailed",
)
def restore(experiment: str, run_id: str, email: str, password: str, verbosity: int):
    """Restore a prompt from a previous run and re-evaluate."""
    storage = ExperimentStorage(experiment)

    if not storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' not found[/red]")
        sys.exit(1)

    if not storage.has_ground_truth():
        console.print(
            "[red]Error: Ground truth not found. Run 'extract-truth' first.[/red]"
        )
        sys.exit(1)

    # Load the run to restore
    try:
        old_run = storage.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Error: Run '{run_id}' not found[/red]")
        sys.exit(1)

    config = storage.load_config()
    ground_truth = storage.load_ground_truth()

    # Display prompt preview and original metrics (always show for confirmation)
    prompt_preview = (
        old_run.prompt[:200] + "..." if len(old_run.prompt) > 200 else old_run.prompt
    )
    console.print(f"\n[bold]Restoring prompt from run:[/bold] {run_id}")
    console.print(f"\n[bold]Prompt preview:[/bold]\n{prompt_preview}")

    if verbosity >= 2:
        console.print("\n[bold]Full Prompt:[/bold]")
        console.print("─" * 40)
        console.print(old_run.prompt)
        console.print("─" * 40)

    if old_run.metrics:
        all_metrics = old_run.metrics.get("all")
        if all_metrics:
            console.print(
                f"\n[bold]Original metrics:[/bold] Accuracy: {all_metrics.accuracy:.1%}, Coverage: {all_metrics.coverage:.1%}"
            )

    if not click.confirm(
        "\nRestore this prompt and re-run classification?", default=True
    ):
        console.print("[yellow]Cancelled[/yellow]")
        return

    client = get_client(email, password)
    job_id = config.iteration_job_id

    # Apply the restored prompt to SDM
    log_progress("\n[blue]Updating prompt in SDM...[/blue]", verbosity)
    client.update_prompt(config.step_id, config.field_name, old_run.prompt)
    log_progress("[green]Prompt restored[/green]", verbosity)

    # Re-run the job (same pattern as evaluate command)
    log_progress(
        f"[blue]Returning job to step {config.prev_step_id}...[/blue]", verbosity
    )
    log_detail(f"[dim]Job ID: {job_id}[/dim]", verbosity)
    client.return_to_step(job_id, config.prev_step_id, reset=True)

    log_progress(
        f"[blue]Waiting for step {config.prev_step_id} to be ready...[/blue]", verbosity
    )
    client.wait_for_completion(job_id, target_step_id=config.prev_step_id)

    log_progress(f"[blue]Validating step {config.prev_step_id}...[/blue]", verbosity)
    log_detail(f"[dim]Step type: {config.prev_step_type}[/dim]", verbosity)
    client.validate_step(job_id, config.prev_step_id, config.prev_step_type)

    log_progress("[blue]Waiting for classification to complete...[/blue]", verbosity)
    success = client.wait_for_completion(job_id)

    if not success:
        console.print("[red]Job failed or timed out[/red]")
        sys.exit(1)

    # Get and display new results
    log_progress("[blue]Fetching classification results...[/blue]", verbosity)
    predictions = client.get_classification_results(
        action_id=config.step_id,
        job_id=job_id,
    )

    evaluator = Evaluator(match_keys=config.match_keys, field_name=config.field_name)
    gt_rows = [
        {"categories": r.categories, **r.match_key, **r.source_data}
        for r in ground_truth.rows
    ]

    metrics = evaluator.compare(gt_rows, predictions)
    errors = evaluator.get_errors(gt_rows, predictions)

    _display_metrics(
        metrics,
        old_run.prompt,
        eval_type=f"RESTORED from {run_id}",
        verbosity=verbosity,
    )

    # Save as new run
    new_run_id = generate_run_id()
    run_result = RunResult(
        run_id=new_run_id,
        timestamp=datetime.now(UTC).isoformat(),
        prompt=old_run.prompt,
        metrics={k: v for k, v in metrics.items()},
        errors_sample=errors[:50],
    )
    storage.save_run(run_result)

    log_progress(f"\n[green]Run saved: {new_run_id}[/green]", verbosity)
    log_progress(f"[green]Prompt restored from: {run_id}[/green]", verbosity)


# ===== HISTORY & LIST =====


@cli.command()
@click.option("--experiment", required=True, help="Experiment name")
def history(experiment: str):
    """Show run history for an experiment."""
    storage = ExperimentStorage(experiment)

    if not storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' not found[/red]")
        sys.exit(1)

    runs = storage.list_runs()

    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return

    table = Table(title=f"Run History: {experiment}")
    table.add_column("Run ID", style="cyan")
    table.add_column("Accuracy (all)", justify="right")
    table.add_column("Coverage (all)", justify="right")
    table.add_column("Accuracy (auto)", justify="right")
    table.add_column("Has Anthropic", justify="center")
    table.add_column("Prompt Preview", max_width=40)

    for run_id in runs:
        run = storage.load_run(run_id)
        all_metrics = run.metrics.get("all")
        auto_metrics = run.metrics.get("automated")

        # Create prompt preview (first 40 chars, single line)
        prompt_preview = run.prompt.replace("\n", " ")[:40] if run.prompt else ""
        if len(run.prompt) > 40:
            prompt_preview += "..."

        table.add_row(
            run_id,
            f"{all_metrics.accuracy:.1%}" if all_metrics else "-",
            f"{all_metrics.coverage:.1%}" if all_metrics else "-",
            f"{auto_metrics.accuracy:.1%}" if auto_metrics else "-",
            "✓" if run.anthropic_analysis else "",
            prompt_preview,
        )

    console.print(table)


@cli.command("list")
def list_cmd():
    """List all experiments."""
    experiments = list_experiments()

    if not experiments:
        console.print("[yellow]No experiments found[/yellow]")
        return

    table = Table(title="Experiments")
    table.add_column("Name", style="cyan")
    table.add_column("Mode")
    table.add_column("Field")
    table.add_column("Step ID", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Has GT", justify="center")

    for exp_name in experiments:
        storage = ExperimentStorage(exp_name)
        config = storage.load_config()
        runs = storage.list_runs()

        table.add_row(
            exp_name,
            config.mode,
            config.field_name,
            str(config.step_id),
            str(len(runs)),
            "✓" if storage.has_ground_truth() else "",
        )

    console.print(table)


# ===== HELPER FUNCTIONS =====


def _extract_categories(value) -> list[list[str]]:
    """Extract categories from various formats."""
    if value is None:
        return []

    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], list):
            return value
        else:
            return [value]
    elif isinstance(value, str):
        if "/" in value:
            return [[p.strip() for p in value.split("/")]]
        return [[value]]

    return []


def _deduplicate_rows(
    rows: list[GroundTruthRow], match_keys: list[str]
) -> list[GroundTruthRow]:
    """Deduplicate rows by match key, keeping the last occurrence."""
    seen: dict[tuple, GroundTruthRow] = {}
    for row in rows:
        key = tuple(row.match_key.get(k, "").lower() for k in match_keys)
        seen[key] = row
    return list(seen.values())


def _build_code_to_label_map(
    results: list[dict], labels: list[dict], category_column: str
) -> dict[str, str]:
    """Build a mapping from category codes to human-readable labels.

    The API returns:
    - results[i][category_column]: codes like [['categories_BE', 'csbe00001', 'csbe01076']]
    - labels[i][category_column]: "Cat/Root/Specific | Cat/Root/Other" (multi-path with " | " separator)

    This builds a dict mapping each code segment to its label segment.
    """
    code_to_label: dict[str, str] = {}

    for result, label_row in zip(results, labels):
        code_value = result.get(category_column)
        label_value = label_row.get(category_column)

        if not code_value or not label_value:
            continue

        # Handle code formats: list of lists, list, or string
        if isinstance(code_value, list):
            if code_value and isinstance(code_value[0], list):
                # [[code1, code2, ...], [code3, code4, ...]] - multiple paths
                code_paths = code_value
            else:
                # [code1, code2, ...] - single path
                code_paths = [code_value]
        elif isinstance(code_value, str):
            # "code1///code2///code3" - split by separator
            code_paths = [[c.strip() for c in code_value.split("///")]]
        else:
            continue

        # Labels can be multi-path with " | " separator
        if isinstance(label_value, str):
            # Split by " | " for multiple paths, then by "/" for path segments
            label_paths = [lbl.split("/") for lbl in label_value.split(" | ")]
        elif isinstance(label_value, dict):
            # If it's a dict with locales, get the first value
            label_str = next(iter(label_value.values()), "")
            label_paths = [lbl.split("/") for lbl in label_str.split(" | ")]
        else:
            continue

        # Map each code to its corresponding label for each path
        for codes, labels_list in zip(code_paths, label_paths):
            for code, label in zip(codes, labels_list):
                code = code.strip() if isinstance(code, str) else code
                label = label.strip() if isinstance(label, str) else label
                if code and label and code not in code_to_label:
                    code_to_label[code] = label

    return code_to_label


def _display_metrics(
    metrics: dict, prompt: str, eval_type: str = "", verbosity: int = 1
):
    """Display metrics in a formatted table."""
    # Level 1: truncated prompt preview
    if verbosity >= 1:
        prompt_display = f"{prompt[:100]}..." if len(prompt) > 100 else prompt
        console.print(f"\n[bold]Current Prompt:[/bold] {prompt_display}")

    # Level 2: full prompt
    if verbosity >= 2:
        console.print("\n[bold]Full Prompt:[/bold]")
        console.print("─" * 40)
        console.print(prompt)
        console.print("─" * 40)

    # Always show metrics table (even at level 0)
    title = f"Evaluation Metrics ({eval_type})" if eval_type else "Evaluation Metrics"
    table = Table(title=title)
    table.add_column("Subset", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Covered", justify="right")

    for subset_name in ["all", "automated", "to_check"]:
        m = metrics.get(subset_name)
        if m:
            table.add_row(
                subset_name,
                f"{m.accuracy:.1%}",
                f"{m.coverage:.1%}",
                str(m.total_rows),
                str(m.correct_rows),
                str(m.covered_rows),
            )

    console.print(table)


if __name__ == "__main__":
    cli()
