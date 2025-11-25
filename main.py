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
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


def get_client(email: str, password: str) -> SDMClient:
    """Create and authenticate SDM client."""
    client = SDMClient(base_url=SDM_BASE_URL, email=email, password=password)
    client.authenticate()
    return client


@click.group()
def cli():
    """SDM Prompt Iterator - Automate classification prompt optimization."""
    pass


# ===== MANUAL MODE INIT =====


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
@click.option("--step-id", required=True, type=int, help="Classification step ID")
@click.option(
    "--prev-step-id", type=int, help="Previous step ID (auto-detected if not provided)"
)
@click.option(
    "--field", required=True, help="Classification field name (e.g., 'category')"
)
@click.option(
    "--match-keys", required=True, help="Comma-separated column names for row matching"
)
@click.option("--test-job", required=True, help="Job ID for testing prompt changes")
@click.option(
    "--truth-jobs",
    required=True,
    help="Comma-separated job IDs containing ground truth",
)
def init(
    experiment: str,
    email: str,
    password: str,
    step_id: int,
    prev_step_id: int | None,
    field: str,
    match_keys: str,
    test_job: str,
    truth_jobs: str,
):
    """Initialize a new experiment (manual mode with existing test job)."""
    storage = ExperimentStorage(experiment)

    if storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' already exists[/red]")
        sys.exit(1)

    # Parse arguments
    match_keys_list = [k.strip() for k in match_keys.split(",")]
    truth_jobs_list = [j.strip() for j in truth_jobs.split(",")]

    # Auto-detect previous step if not provided
    client = get_client(email, password)
    prev_step_type = None
    if prev_step_id is None:
        console.print("[yellow]Auto-detecting previous step...[/yellow]")
        prev_step_info = client.get_previous_step_info(test_job, step_id)

        if prev_step_info is None:
            console.print("[red]Could not auto-detect previous step.[/red]")
            prev_step_id = click.prompt("Please enter the previous step ID", type=int)
            prev_step_type = click.prompt(
                "Please enter the previous step type (e.g., 'mappings')", type=str
            )
        else:
            prev_step_id = prev_step_info["id"]
            prev_step_type = prev_step_info["type"]
            console.print(
                f"[green]Detected previous step: {prev_step_info['name']} (ID: {prev_step_id}, type: {prev_step_type})[/green]"
            )
    else:
        # If prev_step_id provided manually, we need to get the type
        prev_step_info = client.get_previous_step_info(test_job, step_id)
        prev_step_type = prev_step_info["type"] if prev_step_info else "mappings"

    # Create experiment
    config = ExperimentConfig(
        name=experiment,
        step_id=step_id,
        prev_step_id=prev_step_id,
        prev_step_type=prev_step_type,
        field_name=field,
        match_keys=match_keys_list,
        truth_job_ids=truth_jobs_list,
        mode="manual",
        test_job_id=test_job,
    )

    storage.create(config)
    console.print(f"[green]Created experiment '{experiment}' (manual mode)[/green]")
    console.print(f"  Step ID: {step_id}")
    console.print(f"  Previous Step: {prev_step_id} ({prev_step_type})")
    console.print(f"  Field: {field}")
    console.print(f"  Match keys: {match_keys_list}")
    console.print(f"  Test job: {test_job}")
    console.print(f"  Truth jobs: {truth_jobs_list}")
    console.print(
        "\n[yellow]Next step: Run 'extract-truth' to build ground truth[/yellow]"
    )


# ===== AUTO-SPLIT MODE INIT =====


@cli.command("init-auto")
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
@click.option("--step-id", required=True, type=int, help="Classification step ID")
@click.option(
    "--prev-step-id", type=int, help="Previous step ID (auto-detected if not provided)"
)
@click.option(
    "--field", required=True, help="Classification field name (e.g., 'category')"
)
@click.option(
    "--match-keys", required=True, help="Comma-separated column names for row matching"
)
@click.option(
    "--truth-jobs",
    required=True,
    help="Comma-separated job IDs containing ground truth",
)
@click.option(
    "--train-ratio",
    default=DEFAULT_TRAIN_RATIO,
    help="Ratio of data for training (default: 0.75)",
)
@click.option(
    "--seed",
    default=DEFAULT_RANDOM_SEED,
    type=int,
    help="Random seed for split (default: 42)",
)
def init_auto(
    experiment: str,
    email: str,
    password: str,
    step_id: int,
    prev_step_id: int | None,
    field: str,
    match_keys: str,
    truth_jobs: str,
    train_ratio: float,
    seed: int,
):
    """Initialize experiment with auto-split mode (creates train/eval jobs from truth data)."""
    storage = ExperimentStorage(experiment)

    if storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' already exists[/red]")
        sys.exit(1)

    # Parse arguments
    match_keys_list = [k.strip() for k in match_keys.split(",")]
    truth_jobs_list = [j.strip() for j in truth_jobs.split(",")]

    client = get_client(email, password)

    # Get project info from first truth job
    console.print("[blue]Getting project info from truth jobs...[/blue]")
    project_info = client.get_project_info_from_job(truth_jobs_list[0])
    project_id = project_info["project_id"]
    model_version = project_info["model_version"]
    console.print(f"  Project ID: {project_id}, Model version: {model_version}")

    # Auto-detect previous step if not provided
    prev_step_type = None
    if prev_step_id is None:
        console.print("[yellow]Auto-detecting previous step...[/yellow]")
        prev_step_info = client.get_previous_step_info(truth_jobs_list[0], step_id)

        if prev_step_info is None:
            console.print("[red]Could not auto-detect previous step.[/red]")
            prev_step_id = click.prompt("Please enter the previous step ID", type=int)
            prev_step_type = click.prompt(
                "Please enter the previous step type (e.g., 'mappings')", type=str
            )
        else:
            prev_step_id = prev_step_info["id"]
            prev_step_type = prev_step_info["type"]
            console.print(
                f"[green]Detected previous step: {prev_step_info['name']} (ID: {prev_step_id}, type: {prev_step_type})[/green]"
            )
    else:
        # If prev_step_id provided manually, we need to get the type
        prev_step_info = client.get_previous_step_info(truth_jobs_list[0], step_id)
        prev_step_type = prev_step_info["type"] if prev_step_info else "mappings"

    # Download and concatenate source data from all truth jobs
    console.print("[blue]Downloading source data from truth jobs...[/blue]")
    all_source_rows = []
    for job_id in truth_jobs_list:
        console.print(f"  Downloading from job {job_id}...")
        rows = client.download_job_source_data(job_id)
        all_source_rows.extend(rows)
        console.print(f"    Got {len(rows)} rows")

    console.print(f"[green]Total source rows: {len(all_source_rows)}[/green]")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_source_rows)

    split_idx = int(len(all_source_rows) * train_ratio)
    train_rows = all_source_rows[:split_idx]
    eval_rows = all_source_rows[split_idx:]

    console.print(f"  Train set: {len(train_rows)} rows ({train_ratio:.0%})")
    console.print(f"  Eval set: {len(eval_rows)} rows ({1 - train_ratio:.0%})")

    # Upload files and create jobs
    console.print("[blue]Creating train job...[/blue]")
    train_file_id = client.upload_file_json(train_rows)
    train_job_id = client.create_job(
        name=f"{experiment}_train",
        project_id=project_id,
        file_id=train_file_id,
        model_version=model_version,
    )
    console.print(f"  Train job created: {train_job_id}")

    console.print("[blue]Creating eval job...[/blue]")
    eval_file_id = client.upload_file_json(eval_rows)
    eval_job_id = client.create_job(
        name=f"{experiment}_eval",
        project_id=project_id,
        file_id=eval_file_id,
        model_version=model_version,
    )
    console.print(f"  Eval job created: {eval_job_id}")

    # Wait for jobs to complete initial processing
    console.print("[blue]Waiting for jobs to complete initial run...[/blue]")
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
    console.print(
        f"\n[green]Created experiment '{experiment}' (auto-split mode)[/green]"
    )
    console.print(f"  Step ID: {step_id}")
    console.print(f"  Previous Step: {prev_step_id} ({prev_step_type})")
    console.print(f"  Field: {field}")
    console.print(f"  Train job (for iteration): {train_job_id}")
    console.print(f"  Eval job (for final eval): {eval_job_id}")
    console.print(
        f"  Split ratio: {train_ratio:.0%} train / {1 - train_ratio:.0%} eval"
    )
    console.print(
        "\n[yellow]Next step: Run 'extract-truth' to build ground truth from the jobs[/yellow]"
    )


# ===== EXTRACT GROUND TRUTH =====


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
def extract_truth(experiment: str, email: str, password: str):
    """Extract ground truth from validation jobs."""
    storage = ExperimentStorage(experiment)

    if not storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' not found[/red]")
        sys.exit(1)

    config = storage.load_config()
    client = get_client(email, password)

    all_rows: list[GroundTruthRow] = []
    all_code_to_label: dict[str, str] = {}
    category_column = f"{config.field_name}{CATEGORY_SUFFIX}"

    for job_id in config.truth_job_ids:
        console.print(f"[blue]Extracting from job {job_id}...[/blue]")

        # Get classification results with labels, filtered by validated/corrected status
        results, labels = client.get_classification_results(
            action_id=config.step_id,
            job_id=job_id,
            status_filter=GROUND_TRUTH_STATUSES,
            include_labels=True,
        )

        console.print(f"  Found {len(results)} validated rows")

        # Build code-to-label mapping from this job
        job_code_to_label = _build_code_to_label_map(results, labels, category_column)
        all_code_to_label.update(job_code_to_label)
        console.print(f"  Extracted {len(job_code_to_label)} category labels")

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
    console.print(
        f"[green]Saved ground truth with {len(unique_rows)} unique rows[/green]"
    )
    console.print(
        f"[green]Saved {len(all_code_to_label)} category code-to-label mappings[/green]"
    )


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
def evaluate(
    experiment: str, email: str, password: str, skip_rerun: bool, use_eval_job: bool
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
        console.print("[cyan]Using EVAL job for evaluation (holdout set)[/cyan]")
    else:
        job_id = config.iteration_job_id
        if config.mode == "auto_split":
            console.print(
                "[cyan]Using TRAIN job for evaluation (use --use-eval-job for final eval)[/cyan]"
            )

    # Get current prompt
    fields_config = client.get_fields_config(config.step_id)
    current_prompt = (
        fields_config.get("fields_config", {})
        .get(config.field_name, {})
        .get("custom_prompt", "")
    )

    if not skip_rerun:
        # Return to previous step and re-run classification
        console.print(
            f"[blue]Returning job {job_id} to step {config.prev_step_id}...[/blue]"
        )
        client.return_to_step(job_id, config.prev_step_id, reset=True)

        # Wait for previous step to be ready for validation
        console.print(
            f"[blue]Waiting for step {config.prev_step_id} to be ready...[/blue]"
        )
        client.wait_for_completion(job_id, target_step_id=config.prev_step_id)

        # Validate the previous step to move forward to classification
        console.print(
            f"[blue]Validating step {config.prev_step_id} ({config.prev_step_type})...[/blue]"
        )
        client.validate_step(job_id, config.prev_step_id, config.prev_step_type)

        console.print("[blue]Waiting for job completion...[/blue]")
        success = client.wait_for_completion(job_id)

        if not success:
            console.print("[red]Job failed or timed out[/red]")
            sys.exit(1)

    # Get prediction results
    console.print("[blue]Fetching classification results...[/blue]")
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
    _display_metrics(metrics, current_prompt, eval_type=eval_type)

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

    console.print(f"\n[green]Run saved: {run_id}[/green]")


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
def iterate(
    experiment: str,
    email: str,
    password: str,
    anthropic_api_key: str,
    iterations: int,
    auto_apply: bool,
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

    if config.mode == "auto_split":
        console.print(
            f"[cyan]Auto-split mode: iterating on TRAIN job ({job_id})[/cyan]"
        )
        console.print(
            "[cyan]After iteration, run 'evaluate --use-eval-job' for final evaluation on holdout set[/cyan]\n"
        )

    for i in range(iterations):
        console.print(
            f"\n[bold cyan]═══ Iteration {i + 1}/{iterations} ═══[/bold cyan]"
        )

        # 1. Get current prompt and classification results
        fields_config = client.get_fields_config(config.step_id)
        current_prompt = (
            fields_config.get("fields_config", {})
            .get(config.field_name, {})
            .get("custom_prompt", "")
        )

        console.print("[blue]Fetching classification results...[/blue]")
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

        _display_metrics(metrics, current_prompt)

        # 3. Get Anthropic's analysis and suggestion
        console.print("\n[blue]Analyzing errors with Anthropic...[/blue]")
        metrics_dict = {k: v.to_dict() for k, v in metrics.items()}
        result = advisor.iterate(current_prompt, errors, metrics_dict)

        console.print("\n[bold]Error Analysis:[/bold]")
        console.print(result["analysis"])

        console.print("\n[bold]Suggested Prompt:[/bold]")
        console.print(f"[italic]{result['suggested_prompt']}[/italic]")

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
                console.print("[blue]Updating prompt...[/blue]")
                client.update_prompt(
                    config.step_id, config.field_name, result["suggested_prompt"]
                )
                console.print("[green]Prompt updated[/green]")

                # Return to previous step
                console.print(
                    f"[blue]Returning to step {config.prev_step_id} to re-run classification...[/blue]"
                )
                client.return_to_step(job_id, config.prev_step_id, reset=True)

                # Wait for previous step to be ready for validation
                console.print(
                    f"[blue]Waiting for step {config.prev_step_id} to be ready...[/blue]"
                )
                client.wait_for_completion(job_id, target_step_id=config.prev_step_id)

                # Validate the previous step to move forward to classification
                console.print(
                    f"[blue]Validating step {config.prev_step_id} ({config.prev_step_type})...[/blue]"
                )
                client.validate_step(job_id, config.prev_step_id, config.prev_step_type)

                # Wait for classification to complete
                console.print(
                    f"[blue]Waiting for job to reach classification step {config.step_id}...[/blue]"
                )
                success = client.wait_for_completion(
                    job_id, target_step_id=config.step_id
                )

                if not success:
                    console.print("[red]Job failed or timed out[/red]")
                    break
            else:
                console.print("[yellow]Stopping iteration[/yellow]")
                break

    console.print("\n[green]Iteration complete![/green]")

    if config.mode == "auto_split":
        console.print(
            "\n[yellow]Tip: Run 'evaluate --use-eval-job' to evaluate on the holdout set[/yellow]"
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
    """Run final evaluation on holdout set (auto-split mode only)."""
    storage = ExperimentStorage(experiment)

    if not storage.exists():
        console.print(f"[red]Error: Experiment '{experiment}' not found[/red]")
        sys.exit(1)

    config = storage.load_config()

    if config.mode != "auto_split":
        console.print(
            "[red]Error: final-eval is only for auto-split mode experiments[/red]"
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

    for run_id in runs:
        run = storage.load_run(run_id)
        all_metrics = run.metrics.get("all")
        auto_metrics = run.metrics.get("automated")

        table.add_row(
            run_id,
            f"{all_metrics.accuracy:.1%}" if all_metrics else "-",
            f"{all_metrics.coverage:.1%}" if all_metrics else "-",
            f"{auto_metrics.accuracy:.1%}" if auto_metrics else "-",
            "✓" if run.anthropic_analysis else "",
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


def _display_metrics(metrics: dict, prompt: str, eval_type: str = ""):
    """Display metrics in a formatted table."""
    prompt_display = f"{prompt[:100]}..." if len(prompt) > 100 else prompt
    console.print(f"\n[bold]Current Prompt:[/bold] {prompt_display}")

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
