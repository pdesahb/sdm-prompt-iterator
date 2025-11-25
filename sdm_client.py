"""SDM API Client for classification prompt iteration."""

import io
import logging
import time
from typing import Any

import requests

from config import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    ENDPOINTS,
    SDM_BASE_URL,
)

logger = logging.getLogger(__name__)


class SDMClientError(Exception):
    """Base exception for SDM client errors."""

    pass


class SDMAuthError(SDMClientError):
    """Authentication error."""

    pass


class SDMAPIError(SDMClientError):
    """API request error."""

    def __init__(
        self, message: str, status_code: int | None = None, response: dict | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class SDMClient:
    """Client for interacting with SDM API."""

    def __init__(
        self,
        base_url: str = SDM_BASE_URL,
        email: str | None = None,
        password: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.token: str | None = None
        self.session = requests.Session()

    def authenticate(self) -> str:
        """Authenticate with SDM API and return token."""
        if not self.email or not self.password:
            raise SDMAuthError("Email and password are required for authentication")

        url = f"{self.base_url}{ENDPOINTS['auth']}"
        payload = {
            "email": self.email,
            "password": self.password,
            "token_name": "sdm-prompt-iterator",
        }

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            self.token = data.get("token") or data.get("key")
            if not self.token:
                raise SDMAuthError("No token returned from authentication")
            self.session.headers["Authorization"] = f"Token {self.token}"
            logger.info("Successfully authenticated with SDM API")
            return self.token
        except requests.exceptions.HTTPError as e:
            raise SDMAuthError(f"Authentication failed: {e}")

    def _ensure_authenticated(self) -> None:
        """Ensure client is authenticated."""
        if not self.token:
            self.authenticate()

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict | None = None,
        json_data: dict | None = None,
        **kwargs,
    ) -> dict | list | None:
        """Make an authenticated API request."""
        self._ensure_authenticated()
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method, url, params=params, json=json_data, **kwargs
            )
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return None
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_data = None
            try:
                error_data = e.response.json()
            except (ValueError, AttributeError):
                pass
            raise SDMAPIError(
                f"API request failed: {e}",
                status_code=e.response.status_code if e.response else None,
                response=error_data,
            )

    def get_job_details(self, job_id: str) -> dict:
        """Get job details including current step and project info."""
        endpoint = ENDPOINTS["job_details"].format(job_id=job_id)
        result = self._request("GET", endpoint)
        return result

    def get_project_details(self, project_id: int) -> dict:
        """Get project details including steps."""
        endpoint = ENDPOINTS["project_details"].format(project_id=project_id)
        return self._request("GET", endpoint)

    def get_project_steps(self, project_id: int) -> list[dict]:
        """Get all steps for a project."""
        project = self.get_project_details(project_id)
        return project.get("steps", [])

    def get_previous_step_info(self, job_id: str, current_step_id: int) -> dict | None:
        """Auto-detect the previous step ID and type for a given step.

        Returns dict with 'id' and 'type' keys, or None if not found.
        """
        job_details = self.get_job_details(job_id)
        project_field = job_details.get("project")

        # Handle both formats: project can be an int or an object with id
        if isinstance(project_field, dict):
            project_id = project_field.get("id")
        else:
            project_id = project_field

        if not project_id:
            logger.warning("Could not determine project_id from job details")
            return None

        steps = self.get_project_steps(project_id)

        # Find current step and its predecessor
        current_step = None
        for step in steps:
            if step.get("id") == current_step_id:
                current_step = step
                break

        if not current_step:
            logger.warning(f"Could not find step {current_step_id} in project steps")
            return None

        # Try to find previous step by number
        current_number = current_step.get("number", 0)
        prev_step = None
        for step in steps:
            step_number = step.get("number", -1)
            if step_number < current_number:
                if prev_step is None or step_number > prev_step.get("number", -1):
                    prev_step = step

        if prev_step:
            step_type = prev_step.get("type", "")
            return {
                "id": prev_step.get("id"),
                "type": step_type,
                "name": prev_step.get("name", ""),
            }

        logger.warning(f"Could not find previous step for step {current_step_id}")
        return None

    def get_previous_step_id(self, job_id: str, current_step_id: int) -> int | None:
        """Auto-detect the previous step ID for a given step (legacy wrapper)."""
        info = self.get_previous_step_info(job_id, current_step_id)
        return info.get("id") if info else None

    def get_classification_results(
        self,
        action_id: int,
        job_id: str,
        *,
        status_filter: list[str] | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        include_labels: bool = False,
    ) -> list[dict] | tuple[list[dict], list[dict]]:
        """Get all classification results for a job with pagination.

        Args:
            action_id: The classification step ID
            job_id: The job ID
            status_filter: Optional filter by confidence status
            page_size: Number of results per page
            include_labels: If True, return (results, labels) tuple

        Returns:
            If include_labels=False: List of result dicts
            If include_labels=True: Tuple of (results, labels) lists
        """
        endpoint = ENDPOINTS["classification_results"].format(
            action_id=action_id, job_id=job_id
        )
        all_results = []
        all_labels = []
        page = 1

        while True:
            params: dict[str, Any] = {"page": page, "page_size": page_size}

            if status_filter:
                # SDM uses filters as JSON array in query params
                import json

                params["filters"] = json.dumps(
                    [
                        {
                            "column": "general_UNiFAi__confidence",
                            "rule": "isin",
                            "value": status_filter,
                        }
                    ]
                )

            response = self._request("GET", endpoint, params=params)

            if not response:
                break

            results = response.get("results", [])
            all_results.extend(results)

            if include_labels:
                labels = response.get("labels", [])
                all_labels.extend(labels)

            # Check if there are more pages
            total_count = response.get("count", 0)
            if len(all_results) >= total_count or not results:
                break

            page += 1
            logger.info(
                f"Fetched page {page}, total rows: {len(all_results)}/{total_count}"
            )

        logger.info(f"Retrieved {len(all_results)} classification results")

        if include_labels:
            return all_results, all_labels
        return all_results

    def get_classification_stats(self, action_id: int, job_id: str) -> dict:
        """Get classification stats for a job."""
        endpoint = ENDPOINTS["classification_results"].format(
            action_id=action_id, job_id=job_id
        )
        response = self._request("GET", endpoint, params={"page_size": 1})
        return response.get("stats", {}) if response else {}

    def get_fields_config(self, step_id: int) -> dict:
        """Get classification field configuration for a step."""
        endpoint = ENDPOINTS["fields_config"].format(step_id=step_id)
        result = self._request("GET", endpoint)
        return result

    def update_prompt(self, step_id: int, field_name: str, new_prompt: str) -> None:
        """Update the custom prompt for a classification field."""
        endpoint = ENDPOINTS["fields_config"].format(step_id=step_id)

        # Get current config first
        current_config = self.get_fields_config(step_id)
        fields_config = current_config.get("fields_config", {})

        # Update the prompt for the specified field
        if field_name not in fields_config:
            fields_config[field_name] = {}
        fields_config[field_name]["custom_prompt"] = new_prompt

        payload = {"fields_config": fields_config}

        self._request("PATCH", endpoint, json_data=payload)
        logger.info(f"Updated prompt for field '{field_name}' on step {step_id}")

    def return_to_step(self, job_id: str, step_id: int, *, reset: bool = True) -> None:
        """Return a job to a step (can be current or previous step)."""
        endpoint = ENDPOINTS["return_to_step"].format(job_id=job_id)
        payload = {"step_id": step_id, "reset_step": reset}

        self._request("POST", endpoint, json_data=payload)
        logger.info(f"Returned job {job_id} to step {step_id} (reset={reset})")

    def validate_step(self, job_id: str, step_id: int, step_type: str) -> None:
        """Validate a step to move the job forward."""
        endpoint = ENDPOINTS["validate_step"].format(
            step_type=step_type, step_id=step_id, job_id=job_id
        )

        self._request("POST", endpoint)
        logger.info(f"Validated step {step_id} ({step_type}) for job {job_id}")

    def get_job_status(self, job_id: str) -> str:
        """Get the current status of a job."""
        job_details = self.get_job_details(job_id)
        return job_details.get("status", "unknown")

    def wait_for_completion(
        self,
        job_id: str,
        *,
        target_step_id: int | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> bool:
        """Wait for a job to be ready for evaluation.

        In SDM:
        - RUNNING: Job is being processed
        - PENDING: Job is waiting for user input (ready for evaluation)
        - DONE: Job is fully completed
        - ERROR: Job failed

        Args:
            job_id: The job ID to wait for
            target_step_id: If specified, wait until job reaches this step AND is pending/done
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds
        """
        start_time = time.time()
        last_status = None
        last_step_id = None

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Timeout waiting for job {job_id} after {timeout}s")
                return False

            job_details = self.get_job_details(job_id)
            status = job_details.get("status", "unknown")
            current_step_id = job_details.get("step", {}).get("id")
            current_step_name = job_details.get("step", {}).get("name", "unknown")

            if status != last_status or current_step_id != last_step_id:
                logger.info(
                    f"Job {job_id} status: {status}, step: {current_step_name} ({current_step_id})"
                )
                last_status = status
                last_step_id = current_step_id

            status_upper = status.upper() if status else ""

            # Check if we've reached target step (if specified)
            at_target_step = target_step_id is None or current_step_id == target_step_id

            if status_upper in ("DONE", "PENDING") and at_target_step:
                logger.info(
                    f"Job {job_id} ready (status: {status}, step: {current_step_name})"
                )
                return True
            elif status_upper == "ERROR":
                logger.error(f"Job {job_id} failed with error")
                return False
            else:
                time.sleep(poll_interval)

        return False

    # ===== File and Job Creation Methods =====

    def get_job_file_id(self, job_id: str) -> int | None:
        """Get the file ID associated with a job."""
        job_details = self.get_job_details(job_id)
        file_info = job_details.get("file", {})
        return file_info.get("id")

    def download_file_data(self, file_id: int) -> list[dict]:
        """Download parsed file data as list of dicts."""
        endpoint = ENDPOINTS["file_download"].format(file_id=file_id)
        self._ensure_authenticated()

        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url)
        response.raise_for_status()

        # Parse CSV content
        import csv

        content = response.content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(content), delimiter=";")
        rows = list(reader)
        logger.info(f"Downloaded {len(rows)} rows from file {file_id}")
        return rows

    def download_job_source_data(self, job_id: str) -> list[dict]:
        """Download the source data for a job."""
        file_id = self.get_job_file_id(job_id)
        if not file_id:
            raise SDMAPIError(f"Could not get file ID for job {job_id}")
        return self.download_file_data(file_id)

    def upload_file_json(self, data: list[dict], filename: str = "upload.csv") -> int:
        """Upload data as a new file via JSON endpoint. Returns file ID."""
        endpoint = ENDPOINTS["file_upload"]

        payload = {"data": data}
        result = self._request("POST", endpoint, json_data=payload)

        file_id = result.get("id")
        status = result.get("status")
        logger.info(f"Uploaded file {file_id} with status {status}")

        # Wait for file to be ready
        if status == "parsing":
            file_id = self._wait_for_file_ready(file_id)

        return file_id

    def _wait_for_file_ready(
        self, file_id: int, timeout: int = 120, poll_interval: int = 2
    ) -> int:
        """Wait for a file to be ready after upload."""
        start_time = time.time()
        endpoint = f"{ENDPOINTS['file_upload']}{file_id}/"

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise SDMAPIError(f"Timeout waiting for file {file_id} to be ready")

            result = self._request("GET", endpoint)
            status = result.get("status")

            if status == "ready":
                logger.info(f"File {file_id} is ready")
                return file_id
            elif status == "error":
                error = result.get("parsing_error", "Unknown error")
                raise SDMAPIError(f"File parsing failed: {error}")
            else:
                time.sleep(poll_interval)

    def create_job(
        self, name: str, project_id: int, file_id: int, model_version: int
    ) -> str:
        """Create a new job from an uploaded file. Returns job ID."""
        endpoint = ENDPOINTS["job_create"]

        payload = {
            "name": name,
            "project": project_id,
            "file_id": file_id,
            "model_version": model_version,
        }

        result = self._request("POST", endpoint, json_data=payload)
        job_id = result.get("id")
        logger.info(f"Created job {job_id} with name '{name}'")
        return job_id

    def get_project_info_from_job(self, job_id: str) -> dict:
        """Get project ID and model version from an existing job."""
        job_details = self.get_job_details(job_id)
        project_field = job_details.get("project")

        # Handle both formats: project can be an int or an object with id
        if isinstance(project_field, dict):
            project_id = project_field.get("id")
        else:
            project_id = project_field

        return {
            "project_id": project_id,
            "model_version": job_details.get("model_version"),
        }
