# SDM Prompt Iterator

CLI tool to automate iteration and optimization of SDM categorization prompts.

## Features

- **Ground truth extraction** from human-validated jobs
- **Automatic prompt evaluation** with metrics (accuracy, coverage)
- **Anthropic-assisted iteration** to improve prompts
- **Auto-split mode** to prevent overfitting (train/eval separation)
- **Run history** to track performance evolution
- **Prompt restoration** to rollback to previous versions

## Installation

```bash
cd ~/Programs/akeneo/tools/sdm-prompt-iterator
pip install -r requirements.txt
```

## Configuration

Create a `.env` file at the project root:

```bash
SDM_USER=user@akeneo.com
SDM_PASSWORD=your_password
ANTHROPIC_API_KEY=sk-ant-xxx
```

Note: make sure that your user is in the correct organization to be able ot fetch & iterate on your prompts.


These variables can also be passed as CLI arguments if needed.

## Quick Start

```bash
# Interactive mode - prompts for all required fields
python main.py create

# Or provide arguments directly
python main.py create \
  --experiment "category_v2" \
  --step-id 123 \
  --field category \
  --match-keys "product_name,brand" \
  --truth-jobs job1,job2,job3
```

## Usage Modes

### Auto-Split Mode (Recommended)

Automatically creates train/eval jobs from ground truth data to prevent overfitting.

```bash
# 1. Create experiment with automatic split (75% train, 25% eval)
python main.py create \
  --experiment "category_v2" \
  --step-id 123 \
  --field category \
  --match-keys "product_name,brand" \
  --truth-jobs job1,job2,job3 \
  --mode auto-split \
  --train-ratio 0.75 \
  --seed 42

# 2. Extract ground truth
python main.py extract-truth --experiment category_v2

# 3. Iterate on the train job
python main.py iterate --experiment category_v2 --iterations 5

# 4. Final evaluation on the holdout set
python main.py final-eval --experiment category_v2
```

### Manual Mode

Uses existing job(s) to iterate on the prompt.

```bash
# 1. Create the experiment with existing train/eval jobs
python main.py create \
  --experiment "category_optimization" \
  --step-id 123 \
  --field category \
  --match-keys "product_name,brand" \
  --truth-jobs job1,job2,job3 \
  --mode manual \
  --train-job abc-123-def \
  --eval-job xyz-789-ghi  # optional

# 2. Extract ground truth
python main.py extract-truth --experiment category_optimization

# 3. Iterate with Anthropic
python main.py iterate --experiment category_optimization --iterations 3

# 4. Final evaluation on the eval job (if provided)
python main.py final-eval --experiment category_optimization
```

## Commands

### `create`
Creates a new experiment. Prompts interactively for missing required fields.

| Option | Description |
|--------|-------------|
| `--experiment` | Experiment name |
| `--email` | SDM email (or `SDM_USER` env var) |
| `--password` | SDM password (or `SDM_PASSWORD` env var) |
| `--step-id` | Classification step ID |
| `--prev-step-id` | Previous step ID (auto-detected if omitted) |
| `--field` | Classification field name (e.g., `category`) |
| `--match-keys` | Columns to match rows (e.g., `product_name,brand`) |
| `--truth-jobs` | IDs of jobs containing ground truth (comma-separated) |
| `--mode` | Experiment mode: `auto-split` (default) or `manual` |
| `--train-job` | Train job ID (manual mode) |
| `--eval-job` | Eval job ID (manual mode, optional) |
| `--train-ratio` | Ratio for the train set (auto-split mode, default: 0.75) |
| `--seed` | Random seed for the split (auto-split mode, default: 42) |
| `-v`, `--verbosity` | Output verbosity: 0=errors, 1=progress (default), 2=detailed |

**Interactive mode**: Run `python main.py create` without arguments to be prompted for each required field.

### `extract-truth`
Extracts ground truth from validated jobs.

| Option | Description |
|--------|-------------|
| `--experiment` | Experiment name |
| `--email` | SDM email (or `SDM_USER` env var) |
| `--password` | SDM password (or `SDM_PASSWORD` env var) |
| `-v`, `--verbosity` | Output verbosity: 0=errors, 1=progress (default), 2=detailed |

### `evaluate`
Evaluates the current prompt against ground truth.

| Option | Description |
|--------|-------------|
| `--experiment` | Experiment name |
| `--email` | SDM email (or `SDM_USER` env var) |
| `--password` | SDM password (or `SDM_PASSWORD` env var) |
| `--skip-rerun` | Do not rerun the job, use current results |
| `--use-eval-job` | Use the evaluation job (auto-split mode) |
| `-v`, `--verbosity` | Output verbosity: 0=errors, 1=progress (default), 2=detailed |

### `iterate`
Iterates on the prompt with Anthropic's help.

| Option | Description |
|--------|-------------|
| `--experiment` | Experiment name |
| `--email` | SDM email (or `SDM_USER` env var) |
| `--password` | SDM password (or `SDM_PASSWORD` env var) |
| `--anthropic-api-key` | Anthropic API key (or `ANTHROPIC_API_KEY` env var) |
| `--iterations` | Number of iterations (default: 3) |
| `--auto-apply` | Automatically apply suggestions |
| `-v`, `--verbosity` | Output verbosity: 0=errors, 1=progress (default), 2=detailed |

### `restore`
Restores a prompt from a previous run and re-evaluates.

| Option | Description |
|--------|-------------|
| `--experiment` | Experiment name |
| `--run-id` | Run ID to restore prompt from |
| `--email` | SDM email (or `SDM_USER` env var) |
| `--password` | SDM password (or `SDM_PASSWORD` env var) |
| `-v`, `--verbosity` | Output verbosity: 0=errors, 1=progress (default), 2=detailed |

```bash
# List runs to find the one to restore
python main.py history --experiment my-experiment

# Restore a specific run's prompt
python main.py restore --experiment my-experiment --run-id 20251125_232541_176a142a
```

### `final-eval`
Final evaluation on the holdout set (requires eval job).

| Option | Description |
|--------|-------------|
| `--experiment` | Experiment name |
| `--email` | SDM email (or `SDM_USER` env var) |
| `--password` | SDM password (or `SDM_PASSWORD` env var) |

### `history`
Displays the run history of an experiment with prompt previews.

| Option | Description |
|--------|-------------|
| `--experiment` | Experiment name |

```bash
python main.py history --experiment <name>
```

### `list`
Lists all experiments.

```bash
python main.py list
```

## Verbosity Levels

Long-running commands support a `-v` / `--verbosity` option with three levels:

| Level | Description | Output |
|-------|-------------|--------|
| 0 | Silent | Errors only + final metrics table |
| 1 | Normal (default) | Progress messages + stats + truncated prompt preview |
| 2 | Detailed | Full prompts, full Anthropic analysis, API call details |

```bash
# Silent mode - just show final result
python main.py evaluate --experiment my-exp -v 0

# Normal mode (default)
python main.py iterate --experiment my-exp

# Detailed mode - show everything
python main.py iterate --experiment my-exp -v 2
```

## Metrics

The script calculates two metrics on three data sets:

### Metrics
- **Accuracy**: % of exact matches (all categories identical)
- **Coverage**: % of rows with at least one common category

### Data Sets
- **all**: All rows
- **automated**: Automatically classified rows (high confidence)
- **to_check**: Rows marked "to check" (low confidence)

## File Structure

```
sdm-prompt-iterator/
├── main.py              # Main CLI
├── sdm_client.py        # SDM API client
├── evaluator.py         # Metrics calculation
├── anthropic_advisor.py # Anthropic integration
├── storage.py           # Experiment management
├── config.py            # Configuration
├── requirements.txt     # Dependencies
└── experiments/         # Experiment data
    └── {experiment_name}/
        ├── config.json       # Configuration
        ├── ground_truth.json # Ground truth
        └── runs/             # Run history
            └── {run_id}/
                ├── prompt.txt
                └── metrics.json
```

## Ground Truth Format

```json
{
  "extracted_at": "2024-01-15T10:00:00Z",
  "source_jobs": ["job1", "job2"],
  "field_name": "category",
  "rows": [
    {
      "match_key": {"product_name": "Widget A", "brand": "Acme"},
      "source_data": {"product_name": "Widget A", "brand": "Acme", "description": "..."},
      "categories": [
        ["Electronics", "Gadgets"],
        ["Home", "Tools"]
      ],
      "status": "validated",
      "source_job": "job1"
    }
  ]
}
```

## Recommended Workflow

1. **Collect validated jobs**: Have multiple jobs where a human has validated/corrected the classifications
2. **Use auto-split mode**: Prevents overfitting by separating train and eval
3. **Iterate 3-5 times**: Let Anthropic analyze errors and suggest improvements
4. **Evaluate on holdout**: Verify that improvements generalize well
5. **Keep history**: Compare performance between runs

## Notes

- The SDM production URL is used by default: `https://sdm.akeneo.cloud`
- Job polling waits up to 10 minutes with 10-second intervals
- Credentials are read from environment variables (`SDM_USER`, `SDM_PASSWORD`, `ANTHROPIC_API_KEY`) or the `.env` file
- The Anthropic API key is optional (evaluation-only mode is possible)
- The `--prev-step-id` field is auto-detected if not provided
