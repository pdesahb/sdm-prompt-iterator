"""Configuration and constants for SDM Prompt Iterator."""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# SDM API Configuration
SDM_BASE_URL = "https://sdm.akeneo.cloud"

# API Endpoints
ENDPOINTS = {
    "auth": "/api/auth/token/",
    "job_details": "/api/jobs/{job_id}/",
    "return_to_step": "/api/jobs/{job_id}/return-to-step/",
    "classification_results": "/api/beta/classifications/{action_id}/{job_id}/",
    "fields_config": "/api/beta/classifications/{step_id}/fields-config/",
    "project_details": "/api/v1/projects/{project_id}/",
    "file_upload": "/api/v1/files/",
    "file_download": "/api/v1/files/{file_id}/download_parsed/",
    "job_create": "/api/v1/jobs/",
    "job_data": "/api/v1/jobs/{job_id}/data/",
    "validate_step": "/api/v1/{step_type}/{step_id}/{job_id}/",
}

# Classification status values
STATUS_VALIDATED = "validated"
STATUS_CORRECTED = "corrected"
STATUS_AUTOMATED = "automated"
STATUS_TO_CHECK = "to_check"
STATUS_TO_COMPLETE = "to_complete"

# Statuses considered as "ground truth" (human validated)
GROUND_TRUTH_STATUSES = [STATUS_VALIDATED, STATUS_CORRECTED]

# SDM internal column names
CONFIDENCE_COLUMN = "general_UNiFAi__confidence"
ID_COLUMN = "general_UNiFAi__id"
GLOBAL_ID_COLUMN = "general_UNiFAi__global_id"
CATEGORY_SUFFIX = "_UNiFAi"

# Polling configuration
DEFAULT_POLL_INTERVAL = 10  # seconds
DEFAULT_TIMEOUT = 600  # 10 minutes

# Pagination
DEFAULT_PAGE_SIZE = 500

# Auto-split mode
DEFAULT_TRAIN_RATIO = 0.75  # 75% for iteration, 25% for evaluation
DEFAULT_RANDOM_SEED = 42
