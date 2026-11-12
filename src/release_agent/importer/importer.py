import json
import logging
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from pydantic import ValidationError

from release_agent.importer.mapping import (
    has_fingerprint,
    has_source_marker,
    jira_content_fingerprint,
    map_comment,
    map_ticket_to_jira,
    source_fingerprint,
)
from release_agent.importer.schema import DatasetComment, DatasetFile, DatasetTicket
from release_agent.jira_client import (
    JiraClientError,
    JiraNetworkError,
    JiraNotFoundError,
)
from release_agent.models import (
    ImportCommentRecord,
    ImportRecord,
    ImportResult,
    ImportVerificationReport,
    JiraImportBaseline,
    ReleaseManifest,
    TicketVerificationRecord,
)

logger = logging.getLogger(__name__)
BASELINE_PROPERTY_KEY = "tyn.import.baseline"


class ImportJiraClient(Protocol):
    def find_imported_issues(self, project_key: str, label: str) -> list[dict]: ...
    def find_project_issues(self, project_key: str) -> list[dict]: ...
    def create_issue(self, fields: dict[str, object]) -> str: ...
    def update_issue(
        self,
        issue_key: str,
        fields: dict[str, object],
        update: dict[str, object] | None = None,
    ) -> None: ...
    def get_issue(self, issue_key: str) -> dict: ...
    def get_issue_property(self, issue_key: str, property_key: str) -> dict | None: ...
    def set_issue_property(
        self, issue_key: str, property_key: str, value: dict[str, object]
    ) -> None: ...
    def get_comments(self, issue_key: str) -> list[dict]: ...
    def add_comment(
        self, issue_key: str, body: dict[str, object]
    ) -> dict[str, object] | None: ...


class DatasetImporter:
    def __init__(
        self,
        jira_client: ImportJiraClient | None,
        project_key: str,
        import_label: str,
    ) -> None:
        self.jira_client = jira_client
        self.project_key = project_key
        self.import_label = import_label

    @staticmethod
    def load_dataset(path: Path) -> DatasetFile:
        try:
            with path.open(encoding="utf-8") as dataset_file:
                return DatasetFile.model_validate(json.load(dataset_file))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Unable to load valid dataset at {path}: {exc}") from exc

    def validate_local(
        self, path: Path, manifest_path: Path | None = None
    ) -> dict[str, Any]:
        started_at = perf_counter()
        errors: list[str] = []
        duplicate_ids: dict[str, list[str]] = {
            "ticket_source_ids": [],
            "original_issue_keys": [],
            "comment_source_ids": [],
        }
        ticket_count = 0
        comment_count = 0
        manifest_references_valid: bool | None = None
        unresolved_issue_keys: list[str] = []
        dataset_project_key: str | None = None

        try:
            with path.open(encoding="utf-8") as dataset_file:
                raw_dataset = json.load(dataset_file)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Unable to read dataset: {exc}")
            raw_dataset = None

        if isinstance(raw_dataset, dict):
            raw_tickets = raw_dataset.get("tickets")
            if isinstance(raw_tickets, list):
                ticket_count = len(raw_tickets)
                source_ids: list[str] = []
                issue_keys: list[str] = []
                comment_ids: list[str] = []
                for raw_ticket in raw_tickets:
                    if not isinstance(raw_ticket, dict):
                        continue
                    source_id = raw_ticket.get("source_id")
                    issue_key = raw_ticket.get("issue_key")
                    if isinstance(source_id, str):
                        source_ids.append(source_id)
                    if isinstance(issue_key, str):
                        issue_keys.append(issue_key)
                    raw_comments = raw_ticket.get("comments")
                    if isinstance(raw_comments, list):
                        comment_count += len(raw_comments)
                        comment_ids.extend(
                            comment["source_id"]
                            for comment in raw_comments
                            if isinstance(comment, dict)
                            and isinstance(comment.get("source_id"), str)
                        )
                duplicate_ids = {
                    "ticket_source_ids": self._duplicates(source_ids),
                    "original_issue_keys": self._duplicates(issue_keys),
                    "comment_source_ids": self._duplicates(comment_ids),
                }

        try:
            dataset = DatasetFile.model_validate(raw_dataset)
        except ValidationError as exc:
            errors.extend(
                f"Dataset validation error at {'.'.join(map(str, item['loc']))}: "
                f"{item['msg']}"
                for item in exc.errors(include_input=False)
            )
            dataset = None
        if dataset is not None:
            dataset_project_key = dataset.project_key

        if manifest_path is not None:
            try:
                with manifest_path.open(encoding="utf-8") as manifest_file:
                    raw_manifest = json.load(manifest_file)
                manifest = ReleaseManifest.model_validate(raw_manifest)
            except (OSError, json.JSONDecodeError) as exc:
                manifest_references_valid = False
                errors.append(f"Unable to read release manifest: {exc}")
            except ValidationError as exc:
                manifest_references_valid = False
                errors.extend(
                    f"Manifest validation error at {'.'.join(map(str, item['loc']))}: "
                    f"{item['msg']}"
                    for item in exc.errors(include_input=False)
                )
            else:
                if isinstance(raw_dataset, dict):
                    raw_tickets = raw_dataset.get("tickets")
                    input_issue_keys = {
                        ticket.get("issue_key")
                        for ticket in raw_tickets
                        if isinstance(ticket, dict)
                        and isinstance(ticket.get("issue_key"), str)
                    } if isinstance(raw_tickets, list) else set()
                else:
                    input_issue_keys = set()
                unresolved_issue_keys = sorted(
                    {
                        issue_key
                        for requirement in manifest.requirements
                        for issue_key in requirement.evidence_issue_keys
                        if issue_key not in input_issue_keys
                    }
                )
                manifest_references_valid = not unresolved_issue_keys
                if unresolved_issue_keys:
                    errors.append(
                        "Manifest references issue keys absent from the dataset: "
                        + ", ".join(unresolved_issue_keys)
                    )

        if any(duplicate_ids.values()):
            errors.append("Dataset contains duplicate source IDs or original issue keys.")

        return {
            "execution_mode": "local_validation",
            "project_key": self.project_key,
            "dataset_project_key": dataset_project_key,
            "import_label": self.import_label,
            "ticket_count": ticket_count,
            "comment_count": comment_count,
            "duplicate_ids": duplicate_ids,
            "manifest_references_valid": manifest_references_valid,
            "unresolved_manifest_issue_keys": unresolved_issue_keys,
            "jira_requests": 0,
            "jira_writes": 0,
            "status": "VALIDATION_PASSED" if not errors else "VALIDATION_FAILED",
            "errors": errors,
            "duration_seconds": round(perf_counter() - started_at, 6),
        }

    def jira_dry_run(
        self, path: Path, manifest_path: Path | None = None
    ) -> dict[str, Any]:
        started = perf_counter()
        local_validation = self.validate_local(path, manifest_path)
        try:
            dataset = self.load_dataset(path)
        except ValueError as exc:
            report: dict[str, Any] = {
                "execution_mode": "jira_aware_dry_run",
                "source_project_key": local_validation.get("dataset_project_key"),
                "destination_project_key": self.project_key,
                "import_label": self.import_label,
                "release": None,
                "local_validation": local_validation,
                "project_scan": {
                    "complete": False,
                    "issues_scanned": 0,
                    "unrelated_issues": 0,
                    "malformed_imported_issues": [],
                },
                "duplicate_source_ids": {},
                "duplicate_comment_markers": [],
                "ticket_counts": {
                    "PLANNED_CREATE": 0,
                    "UNCHANGED": 0,
                    "PLANNED_SAFE_UPDATE": 0,
                    "CONFLICT": 0,
                    "BLOCKED": int(local_validation.get("ticket_count", 0)),
                    "UNVERIFIED": 0,
                },
                "total_tickets": int(local_validation.get("ticket_count", 0)),
                "total_source_comments": int(local_validation.get("comment_count", 0)),
                "comment_counts": {
                    "already_present": 0,
                    "missing": 0,
                    "planned_additions": 0,
                    "conflicts": 0,
                    "blocked": int(local_validation.get("comment_count", 0)),
                    "unverified": 0,
                },
                "tickets": [],
                "api_errors": [],
                "blockers": [str(exc)],
                "overall_result": "BLOCKED",
                "verification_performed": False,
                "writes_performed": 0,
                "duration_seconds": 0.0,
            }
            self._finalize_jira_dry_run(report, started)
            return report

        manifest: ReleaseManifest | None = None
        if manifest_path is not None:
            try:
                with manifest_path.open(encoding="utf-8") as manifest_file:
                    manifest = ReleaseManifest.model_validate(json.load(manifest_file))
            except (OSError, json.JSONDecodeError, ValidationError):
                manifest = None

        report: dict[str, Any] = {
            "execution_mode": "jira_aware_dry_run",
            "source_project_key": dataset.project_key,
            "destination_project_key": self.project_key,
            "import_label": self.import_label,
            "release": (
                {
                    "release_id": manifest.release_id,
                    "release_name": manifest.release_name,
                    "version": manifest.version,
                    "target_date": (
                        manifest.target_date.isoformat()
                        if manifest.target_date
                        else None
                    ),
                }
                if manifest
                else None
            ),
            "local_validation": local_validation,
            "project_scan": {
                "complete": False,
                "issues_scanned": 0,
                "unrelated_issues": 0,
                "malformed_imported_issues": [],
            },
            "duplicate_source_ids": {},
            "ticket_counts": {
                "PLANNED_CREATE": 0,
                "UNCHANGED": 0,
                "PLANNED_SAFE_UPDATE": 0,
                "CONFLICT": 0,
                "BLOCKED": 0,
                "UNVERIFIED": 0,
            },
            "total_tickets": len(dataset.tickets),
            "total_source_comments": sum(len(ticket.comments) for ticket in dataset.tickets),
            "comment_counts": {
                "already_present": 0,
                "missing": 0,
                "planned_additions": 0,
                "conflicts": 0,
                "blocked": 0,
                "unverified": 0,
            },
            "tickets": [],
            "api_errors": [],
            "blockers": [],
            "overall_result": "UNVERIFIED",
            "verification_performed": False,
            "writes_performed": 0,
            "duration_seconds": 0.0,
        }

        if local_validation["status"] != "VALIDATION_PASSED":
            report["overall_result"] = "BLOCKED"
            report["blockers"] = local_validation["errors"]
            report["tickets"] = [
                self._jira_dry_run_ticket(
                    ticket,
                    "BLOCKED",
                    "Local validation failed; Jira discovery was not attempted.",
                )
                for ticket in dataset.tickets
            ]
            self._finalize_jira_dry_run(report, started)
            return report

        client = self._required_jira_client()
        try:
            inventory = client.find_project_issues(self.project_key)
            self._validate_issue_inventory(inventory)
        except (JiraClientError, TypeError, ValueError, AttributeError) as exc:
            report["api_errors"] = [
                {
                    "operation": "project_issue_search",
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                }
            ]
            report["tickets"] = [
                self._jira_dry_run_ticket(
                    ticket,
                    "UNVERIFIED",
                    "Project-wide Jira discovery did not complete.",
                )
                for ticket in dataset.tickets
            ]
            for record in report["tickets"]:
                record["comments"] = [
                    self._jira_dry_run_comment(
                        comment,
                        "UNVERIFIED",
                        "Project-wide discovery did not complete.",
                    )
                    for comment in next(
                        ticket.comments
                        for ticket in dataset.tickets
                        if ticket.source_id == record["source_id"]
                    )
                ]
            report["ticket_counts"]["UNVERIFIED"] = len(dataset.tickets)
            report["comment_counts"]["unverified"] = report["total_source_comments"]
            self._finalize_jira_dry_run(report, started)
            return report

        by_source_id: dict[str, list[dict]] = {}
        malformed_imported_issues: list[dict[str, str]] = []
        imported_label_orphans: set[str] = set()
        for issue in inventory:
            fields = issue["fields"]
            issue_key = str(issue["key"])
            description = fields.get("description")
            source_ids = self._source_ids_from_description(description)
            labels = fields.get("labels") or []
            if not isinstance(labels, list) or any(
                not isinstance(label, str) for label in labels
            ):
                malformed_imported_issues.append(
                    {"issue_key": issue_key, "reason": "malformed_labels"}
                )
                imported_label_orphans.add(issue_key)
                labels = []
            if self._has_malformed_source_marker(description):
                malformed_imported_issues.append(
                    {"issue_key": issue_key, "reason": "malformed_source_marker"}
                )
                imported_label_orphans.add(issue_key)
            if self.import_label in labels and not source_ids:
                malformed_imported_issues.append(
                    {"issue_key": issue_key, "reason": "import_label_without_source_marker"}
                )
                imported_label_orphans.add(issue_key)
            for source_id in source_ids:
                by_source_id.setdefault(source_id, []).append(issue)
            if len(source_ids) != len(set(source_ids)):
                malformed_imported_issues.append(
                    {"issue_key": issue_key, "reason": "duplicate_source_marker_in_issue"}
                )

        duplicate_source_ids = {
            source_id: sorted(str(issue["key"]) for issue in issues)
            for source_id, issues in by_source_id.items()
            if len(issues) > 1
        }
        source_marked_issues = {
            str(issue["key"])
            for issues in by_source_id.values()
            for issue in issues
        }
        report["project_scan"] = {
            "complete": True,
            "issues_scanned": len(inventory),
            "unrelated_issues": sum(
                1
                for issue in inventory
                if str(issue["key"]) not in source_marked_issues
                and str(issue["key"]) not in imported_label_orphans
            ),
            "malformed_imported_issues": sorted(
                malformed_imported_issues, key=lambda item: (item["issue_key"], item["reason"])
            ),
        }
        report["duplicate_source_ids"] = duplicate_source_ids

        ticket_records: list[dict[str, Any]] = []
        api_errors: list[dict[str, str]] = []
        expected_comment_ids = {
            comment.source_id
            for ticket in dataset.tickets
            for comment in ticket.comments
        }
        comment_marker_occurrences: dict[str, list[str]] = {}
        for ticket in dataset.tickets:
            matches = by_source_id.get(ticket.source_id, [])
            if len(matches) > 1:
                record = self._jira_dry_run_ticket(
                    ticket,
                    "CONFLICT",
                    "Multiple destination issues contain this source ID.",
                    matching_issue_keys=sorted(str(issue["key"]) for issue in matches),
                    baseline_state="NOT_CHECKED",
                )
                record["comments"] = [
                    self._jira_dry_run_comment(
                        comment,
                        "BLOCKED",
                        "Duplicate Jira source matches must be resolved before comment reconciliation.",
                    )
                    for comment in ticket.comments
                ]
                ticket_records.append(record)
                continue
            if not matches:
                if malformed_imported_issues:
                    record = self._jira_dry_run_ticket(
                        ticket,
                        "BLOCKED",
                        "An imported-label issue has no source marker; safe duplicate detection is incomplete.",
                        baseline_state="NOT_APPLICABLE",
                    )
                    record["comments"] = [
                        self._jira_dry_run_comment(
                            comment,
                            "BLOCKED",
                            "Issue creation is blocked by malformed imported Jira records.",
                        )
                        for comment in ticket.comments
                    ]
                else:
                    record = self._jira_dry_run_ticket(
                        ticket,
                        "PLANNED_CREATE",
                        "No destination issue with this source ID was found in the complete project scan.",
                        baseline_state="NOT_APPLICABLE",
                    )
                    record["comments"] = [
                        self._jira_dry_run_comment(
                            comment,
                            "WOULD_ADD",
                            "Comment will be added only if the issue is created successfully.",
                        )
                        for comment in ticket.comments
                    ]
                ticket_records.append(record)
                continue

            inventory_issue = matches[0]
            issue_key = str(inventory_issue["key"])
            try:
                issue = client.get_issue(issue_key)
                fields = issue.get("fields")
                if not isinstance(fields, dict):
                    raise TypeError("Jira issue response has no valid fields object.")
            except (JiraClientError, TypeError, ValueError, AttributeError) as exc:
                api_errors.append(
                    {
                        "operation": "issue_details",
                        "issue_key": issue_key,
                        "error_type": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
                record = self._jira_dry_run_ticket(
                    ticket,
                    "UNVERIFIED",
                    "Existing issue details could not be verified.",
                    issue_key=issue_key,
                    baseline_state="UNVERIFIED",
                )
                record["comments"] = [
                    self._jira_dry_run_comment(
                        comment,
                        "UNVERIFIED",
                        "Issue details are unavailable; comment state cannot be safely reconciled.",
                    )
                    for comment in ticket.comments
                ]
                ticket_records.append(record)
                continue

            baseline: JiraImportBaseline | None = None
            baseline_state = "MISSING"
            try:
                property_response = client.get_issue_property(
                    issue_key, BASELINE_PROPERTY_KEY
                )
                if property_response is not None:
                    baseline = JiraImportBaseline.model_validate(
                        property_response.get("value")
                    )
                    if baseline.source_id != ticket.source_id:
                        raise ValueError("Baseline source ID does not match the ticket.")
                    baseline_state = "VALID"
            except JiraNotFoundError:
                baseline_state = "MISSING"
            except JiraClientError as exc:
                baseline_state = "UNVERIFIABLE"
                api_errors.append(
                    {
                        "operation": "baseline_read",
                        "issue_key": issue_key,
                        "error_type": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
            except (ValidationError, TypeError, ValueError, AttributeError):
                baseline_state = "MALFORMED"

            record_status: str
            reason: str
            source_comparison = "UNAVAILABLE"
            content_comparison = "UNAVAILABLE"
            current_content_fingerprint: str | None = None
            incoming_source_hash = source_fingerprint(ticket)
            if baseline_state == "UNVERIFIABLE":
                record_status = "UNVERIFIED"
                reason = "The existing issue baseline could not be read."
            elif baseline_state in {"MISSING", "MALFORMED"}:
                record_status = "BLOCKED"
                reason = (
                    "Existing issue has no versioned import baseline."
                    if baseline_state == "MISSING"
                    else "Existing issue baseline is malformed or incompatible."
                )
            else:
                assert baseline is not None
                if self.import_label not in baseline.managed_labels:
                    baseline_state = "MALFORMED"
                    record_status = "BLOCKED"
                    reason = "The versioned baseline does not manage the required import label."
                else:
                    record_status = ""
                    reason = ""
                if baseline_state == "MALFORMED":
                    source_comparison = "UNAVAILABLE"
                    content_comparison = "UNAVAILABLE"
                else:
                    source_comparison = (
                        "MATCH"
                        if baseline.source_fingerprint == incoming_source_hash
                        else "CHANGED"
                    )
                    current_labels = fields.get("labels") or []
                    labels_valid = isinstance(current_labels, list) and all(
                        isinstance(label, str) for label in current_labels
                    )
                    if not labels_valid:
                        current_labels = []
                    marker_matches_baseline = has_fingerprint(
                        fields.get("description"), baseline.source_fingerprint
                    )
                    managed_labels_present = set(baseline.managed_labels).issubset(
                        set(current_labels)
                    )
                    current_content_fingerprint = jira_content_fingerprint(
                        fields, baseline.managed_labels
                    )
                    content_matches_baseline = (
                        current_content_fingerprint
                        == baseline.jira_content_fingerprint
                        and marker_matches_baseline
                        and managed_labels_present
                    )
                    content_comparison = (
                        "MATCH" if content_matches_baseline else "MISMATCH"
                    )
                    if not content_matches_baseline:
                        record_status = "CONFLICT"
                        reason = "Current source-managed Jira content differs from its stored baseline."
                    elif source_comparison == "MATCH":
                        record_status = "UNCHANGED"
                        reason = "Incoming source fingerprint and Jira content match the stored baseline."
                    else:
                        record_status = "PLANNED_SAFE_UPDATE"
                        reason = "Source changed and current Jira content still matches the stored baseline."
                    if not labels_valid:
                        record_status = "UNVERIFIED"
                        reason = "Jira returned malformed labels; content cannot be safely reconciled."

            comments: list[dict[str, object]] = []
            comments_read = True
            try:
                jira_comments = self._read_comments(issue_key)
            except (JiraClientError, TypeError, ValueError, AttributeError) as exc:
                comments_read = False
                api_errors.append(
                    {
                        "operation": "comment_read",
                        "issue_key": issue_key,
                        "error_type": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
                jira_comments = []
            if comments_read:
                for jira_comment in jira_comments:
                    for comment_source_id in self._comment_marker_ids(
                        jira_comment.get("body")
                    ):
                        if comment_source_id in expected_comment_ids:
                            comment_marker_occurrences.setdefault(
                                comment_source_id, []
                            ).append(issue_key)
            for source_comment in ticket.comments:
                if not comments_read:
                    comment_record = self._jira_dry_run_comment(
                        source_comment,
                        "UNVERIFIED",
                        "All Jira comment pages could not be retrieved.",
                    )
                else:
                    matches_for_comment = self._comment_matches(
                        jira_comments, source_comment.source_id
                    )
                    if len(matches_for_comment) > 1:
                        comment_record = self._jira_dry_run_comment(
                            source_comment,
                            "CONFLICT",
                            "Duplicate imported comment markers were found.",
                            jira_comment_ids=[
                                self._jira_comment_id(item) or ""
                                for item in matches_for_comment
                            ],
                        )
                    elif matches_for_comment:
                        existing_comment = matches_for_comment[0]
                        actual_text = self._normalized_comment_text(
                            existing_comment.get("body")
                        )
                        expected_text = self._normalized_comment_text(
                            map_comment(source_comment)
                        )
                        if (
                            not self._is_valid_adf(existing_comment.get("body"))
                            or actual_text != expected_text
                        ):
                            comment_record = self._jira_dry_run_comment(
                                source_comment,
                                "CONFLICT",
                                "Imported comment marker exists but its content differs from source.",
                                jira_comment_ids=[
                                    self._jira_comment_id(existing_comment) or ""
                                ],
                            )
                        else:
                            comment_record = self._jira_dry_run_comment(
                                source_comment,
                                "ALREADY_PRESENT",
                                "Exactly one matching imported comment exists.",
                                jira_comment_ids=[
                                    self._jira_comment_id(existing_comment) or ""
                                ],
                            )
                    elif record_status in {"UNCHANGED", "PLANNED_SAFE_UPDATE"}:
                        comment_record = self._jira_dry_run_comment(
                            source_comment,
                            "WOULD_ADD",
                            "Source comment is missing and could be added after safe issue reconciliation.",
                        )
                    elif record_status == "CONFLICT":
                        comment_record = self._jira_dry_run_comment(
                            source_comment,
                            "BLOCKED",
                            "Issue has a manual-content conflict; no comments would be added.",
                        )
                    else:
                        comment_record = self._jira_dry_run_comment(
                            source_comment,
                            "BLOCKED",
                            "Issue reconciliation is blocked; no comments would be added.",
                        )
                comments.append(comment_record)

            if not comments_read:
                record_status = "UNVERIFIED"
                reason = "Jira comments could not be completely retrieved."
            record = self._jira_dry_run_ticket(
                ticket,
                record_status,
                reason,
                issue_key=issue_key,
                baseline_state=baseline_state,
                source_comparison=source_comparison,
                incoming_source_fingerprint=incoming_source_hash,
                stored_source_fingerprint=(
                    baseline.source_fingerprint if baseline else None
                ),
                content_comparison=content_comparison,
                stored_jira_content_fingerprint=(
                    baseline.jira_content_fingerprint if baseline else None
                ),
                current_jira_content_fingerprint=current_content_fingerprint,
                comments=comments,
            )
            ticket_records.append(record)

        duplicate_comment_markers = {
            comment_id: sorted(issue_keys)
            for comment_id, issue_keys in comment_marker_occurrences.items()
            if len(issue_keys) > 1
        }
        if duplicate_comment_markers:
            report["duplicate_comment_markers"] = duplicate_comment_markers
            for record in ticket_records:
                duplicate_ids = {
                    item["source_comment_id"]
                    for item in record.get("comments", [])
                    if item["source_comment_id"] in duplicate_comment_markers
                }
                if not duplicate_ids:
                    continue
                if record["status"] in {"UNCHANGED", "PLANNED_SAFE_UPDATE"}:
                    record["status"] = "CONFLICT"
                    record["reason"] = (
                        "One or more imported comment markers are duplicated across "
                        "destination issues."
                    )
                for comment in record["comments"]:
                    if comment["source_comment_id"] in duplicate_ids:
                        comment["status"] = "CONFLICT"
                        comment["reason"] = (
                            "Imported comment marker appears on multiple Jira issues."
                        )

        status_counts = {
            status: sum(record["status"] == status for record in ticket_records)
            for status in report["ticket_counts"]
        }
        comment_records = [
            comment
            for record in ticket_records
            for comment in record.get("comments", [])
        ]
        comment_counts = {
            "already_present": sum(
                item["status"] == "ALREADY_PRESENT" for item in comment_records
            ),
            "missing": sum(
                item["status"] in {"WOULD_ADD", "PLANNED_ADD"}
                for item in comment_records
            ),
            "planned_additions": sum(
                item["status"] in {"WOULD_ADD", "PLANNED_ADD"}
                for item in comment_records
            ),
            "conflicts": sum(item["status"] == "CONFLICT" for item in comment_records),
            "blocked": sum(item["status"] == "BLOCKED" for item in comment_records),
            "unverified": sum(
                item["status"] == "UNVERIFIED" for item in comment_records
            ),
        }
        report["tickets"] = ticket_records
        report["ticket_counts"] = status_counts
        report["comment_counts"] = comment_counts
        report["api_errors"] = api_errors
        if api_errors:
            report["overall_result"] = "UNVERIFIED"
        elif (
            duplicate_source_ids
            or report["project_scan"]["malformed_imported_issues"]
            or any(status_counts[status] for status in ("CONFLICT", "BLOCKED"))
        ):
            report["overall_result"] = "REQUIRES_REVIEW"
        elif status_counts["UNVERIFIED"]:
            report["overall_result"] = "UNVERIFIED"
        else:
            report["overall_result"] = "SAFE_TO_PROCEED"
        self._finalize_jira_dry_run(report, started)
        return report

    @staticmethod
    def _jira_dry_run_ticket(
        ticket: DatasetTicket,
        status: str,
        reason: str,
        *,
        issue_key: str | None = None,
        matching_issue_keys: list[str] | None = None,
        baseline_state: str = "NOT_CHECKED",
        source_comparison: str = "UNAVAILABLE",
        incoming_source_fingerprint: str | None = None,
        stored_source_fingerprint: str | None = None,
        content_comparison: str = "UNAVAILABLE",
        stored_jira_content_fingerprint: str | None = None,
        current_jira_content_fingerprint: str | None = None,
        comments: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "source_id": ticket.source_id,
            "original_issue_key": ticket.issue_key,
            "destination_issue_key": issue_key,
            "matching_issue_keys": matching_issue_keys or (
                [issue_key] if issue_key else []
            ),
            "status": status,
            "reason": reason,
            "baseline_state": baseline_state,
            "source_fingerprint": {
                "comparison": source_comparison,
                "incoming": incoming_source_fingerprint or source_fingerprint(ticket),
                "stored": stored_source_fingerprint,
            },
            "jira_content_fingerprint": {
                "comparison": content_comparison,
                "current": current_jira_content_fingerprint,
                "stored_baseline": stored_jira_content_fingerprint,
            },
            "comments": comments or [],
        }

    @staticmethod
    def _jira_dry_run_comment(
        comment: DatasetComment,
        status: str,
        reason: str,
        *,
        jira_comment_ids: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "source_comment_id": comment.source_id,
            "status": status,
            "reason": reason,
            "jira_comment_ids": jira_comment_ids or [],
        }

    @classmethod
    def _source_ids_from_description(cls, description: object) -> list[str]:
        source_ids: list[str] = []
        for line in cls._adf_text(description).splitlines():
            candidate = line.strip()
            if candidate.startswith("Source ID:"):
                source_id = candidate[len("Source ID:") :].strip()
                if source_id:
                    source_ids.append(source_id)
        return source_ids

    @classmethod
    def _has_malformed_source_marker(cls, description: object) -> bool:
        return any(
            line.strip().startswith("Source ID:")
            and not line.strip()[len("Source ID:") :].strip()
            for line in cls._adf_text(description).splitlines()
        )

    @staticmethod
    def _finalize_jira_dry_run(report: dict[str, object], started: float) -> None:
        report["duration_seconds"] = round(perf_counter() - started, 6)
        report["generated_at"] = datetime.now(UTC).isoformat()

    @staticmethod
    def _duplicates(values: list[str]) -> list[str]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for value in values:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        return sorted(duplicates)

    def run(self, path: Path, dry_run: bool = False) -> ImportResult:
        started_at = perf_counter()
        import_started_at = datetime.now(UTC)
        dataset = self.load_dataset(path)
        if not dry_run and self.jira_client is None:
            raise RuntimeError("A configured Jira client is required for a live import")

        existing_by_source: dict[str, dict] = {}
        if not dry_run and self.jira_client is not None:
            for issue in self.jira_client.find_imported_issues(
                self.project_key, self.import_label
            ):
                fields = issue.get("fields") or {}
                description = fields.get("description") or ""
                key = issue.get("key")
                if not key:
                    continue
                for ticket in dataset.tickets:
                    if has_source_marker(description, ticket.source_id):
                        existing_by_source[ticket.source_id] = issue
                        break

        records: list[ImportRecord] = []
        created = updated = unchanged = 0
        for ticket in dataset.tickets:
            existing = existing_by_source.get(ticket.source_id)
            existing_key = existing.get("key") if existing else None
            if dry_run:
                action = "would_update" if existing_key else "would_create"
                records.append(
                    ImportRecord(
                        source_id=ticket.source_id,
                        issue_key=existing_key,
                        action=action,
                        decision=action.upper(),
                        reason="Legacy local-only dry-run; Jira was not inspected.",
                    )
                )
                continue
            if self.jira_client is None:
                raise RuntimeError("Jira client became unavailable during import")
            fields = map_ticket_to_jira(ticket, self.project_key, self.import_label)
            if existing_key:
                record = self._reconcile_existing(
                    ticket,
                    str(existing_key),
                    fields,
                )
                if record.decision in {"UPDATED", "UNCHANGED"}:
                    comment_records = self._import_comments(str(existing_key), ticket)
                    record = record.model_copy(
                        update={
                            "comments_added": sum(
                                item.decision == "CREATED"
                                for item in comment_records
                            ),
                            "comment_records": comment_records,
                        }
                    )
                else:
                    record = record.model_copy(
                        update={
                            "comment_records": self._blocked_comment_records(
                                ticket,
                                str(existing_key),
                                "Ticket reconciliation was blocked; no comments were read or written.",
                            )
                        }
                    )
            else:
                record = self._create_with_baseline(ticket, fields)
                if record.decision == "CREATED":
                    comment_records = self._import_comments(str(record.issue_key), ticket)
                    record = record.model_copy(
                        update={
                            "comments_added": sum(
                                item.decision == "CREATED"
                                for item in comment_records
                            ),
                            "comment_records": comment_records,
                        }
                    )
                else:
                    record = record.model_copy(
                        update={
                            "comment_records": self._blocked_comment_records(
                                ticket,
                                record.issue_key,
                                "Issue creation or verification failed; no comments were read or written.",
                            )
                        }
                    )

            if record.action == "created":
                created += 1
            elif record.action == "updated":
                updated += 1
            elif record.action == "unchanged":
                unchanged += 1
            records.append(record)
            logger.info(
                "Processed source %s as %s (%s)",
                ticket.source_id,
                record.issue_key,
                record.decision,
            )

        if dry_run:
            created = sum(item.action == "would_create" for item in records)
            updated = sum(item.action == "would_update" for item in records)
        failed_decisions = {
            "FAILED",
            "VERIFICATION_FAILED",
            "RECONCILIATION_FAILED",
        }
        if any(record.decision in failed_decisions for record in records):
            status = "PARTIAL_FAILURE" if created or updated or unchanged else "FAILED"
        elif any(
            record.decision in {"CONFLICT", "BASELINE_REQUIRED", "BASELINE_INVALID"}
            for record in records
        ):
            status = "REQUIRES_REVIEW"
        else:
            status = "PLAN_ONLY" if dry_run else "UNVERIFIED"
        all_comment_records = [
            comment_record
            for record in records
            for comment_record in record.comment_records
        ]
        comment_status = self._comment_status(all_comment_records)
        if comment_status == "VERIFICATION_FAILED":
            status = "VERIFICATION_FAILED"
        elif comment_status == "AMBIGUOUS_OUTCOME":
            status = "AMBIGUOUS_OUTCOME"
        elif comment_status == "FAILED":
            status = "FAILED_COMMENT_RECONCILIATION"

        verification: ImportVerificationReport | None = None
        if not dry_run:
            verification, records = self._verify_import(dataset, records)
            all_comment_records = [
                comment_record
                for record in records
                for comment_record in record.comment_records
            ]
            comment_status = self._comment_status(all_comment_records)
            status = self._final_import_status(records, verification)

        completed_at = datetime.now(UTC)
        return ImportResult(
            dry_run=dry_run,
            label=self.import_label,
            total_tickets=len(dataset.tickets),
            created=created,
            updated=updated,
            unchanged=unchanged,
            records=records,
            project_key=self.project_key,
            status=status,
            duration_seconds=round(perf_counter() - started_at, 6),
            comment_status=comment_status,
            total_comments=sum(len(ticket.comments) for ticket in dataset.tickets),
            comments_created=sum(
                item.decision == "CREATED" for item in all_comment_records
            ),
            comments_already_present=sum(
                item.decision == "ALREADY_EXISTS" for item in all_comment_records
            ),
            comments_conflicted=sum(
                item.decision == "CONFLICT" for item in all_comment_records
            ),
            comments_failed=sum(
                item.decision in {"FAILED", "AMBIGUOUS_OUTCOME"}
                or item.verification_status == "VERIFICATION_FAILED"
                for item in all_comment_records
            ),
            execution_mode="local_dry_run" if dry_run else "live_import",
            started_at=import_started_at,
            completed_at=completed_at,
            verification=verification,
            verified_tickets=verification.verified_tickets if verification else 0,
            missing_tickets=len(verification.missing_tickets) if verification else 0,
            duplicate_source_ids=(
                verification.duplicate_source_ids if verification else {}
            ),
            verified_comments=verification.verified_comments if verification else 0,
            missing_comments=len(verification.missing_comments) if verification else 0,
            duplicate_comment_markers=(
                verification.duplicate_comment_markers if verification else []
            ),
        )

    def _verify_import(
        self, dataset: DatasetFile, records: list[ImportRecord]
    ) -> tuple[ImportVerificationReport, list[ImportRecord]]:
        client = self._required_jira_client()
        verified_at = datetime.now(UTC)
        expected_comments = sum(len(ticket.comments) for ticket in dataset.tickets)
        try:
            project_issues = client.find_project_issues(self.project_key)
            self._validate_issue_inventory(project_issues)
        except (JiraClientError, TypeError, ValueError, AttributeError) as exc:
            report = ImportVerificationReport(
                status="INCOMPLETE",
                verified_at=verified_at,
                expected_tickets=len(dataset.tickets),
                expected_comments=expected_comments,
                missing_tickets=[ticket.source_id for ticket in dataset.tickets],
                incomplete_retrievals=[f"Project-wide issue retrieval failed: {exc}"],
                error="Project-wide verification inventory is incomplete.",
            )
            return report, records

        source_issues: dict[str, list[dict]] = {}
        for issue in project_issues:
            fields = issue.get("fields")
            if not isinstance(fields, dict):
                raise TypeError("Jira issue response has no fields object.")
            source_id = self._source_id_from_description(fields.get("description"))
            if source_id:
                source_issues.setdefault(source_id, []).append(issue)
        duplicate_source_ids = {
            source_id: sorted(str(issue["key"]) for issue in issues)
            for source_id, issues in source_issues.items()
            if len(issues) > 1
        }

        comment_reads: dict[str, list[dict] | None] = {}
        incomplete_retrievals: list[str] = []
        for issue in project_issues:
            issue_key = str(issue["key"])
            try:
                comment_reads[issue_key] = self._read_comments(issue_key)
            except (JiraClientError, TypeError) as exc:
                comment_reads[issue_key] = None
                incomplete_retrievals.append(
                    f"Comment retrieval failed for {issue_key}: {exc}"
                )

        marker_locations: dict[str, list[tuple[str, dict]]] = {}
        for issue_key, comments in comment_reads.items():
            if comments is None:
                continue
            for comment in comments:
                for source_comment_id in self._comment_marker_ids(comment.get("body")):
                    marker_locations.setdefault(source_comment_id, []).append(
                        (issue_key, comment)
                    )
        duplicate_comment_markers = sorted(
            source_comment_id
            for source_comment_id, locations in marker_locations.items()
            if len(locations) > 1
        )

        record_by_source = {record.source_id: record for record in records}
        ticket_checks: list[TicketVerificationRecord] = []
        verified_tickets = 0
        missing_tickets: list[str] = []
        conflicted_tickets: list[str] = []
        failed_tickets: list[str] = []
        verified_comments = 0
        missing_comments: list[str] = []
        failed_comments: list[str] = []
        all_comment_checks: list[ImportCommentRecord] = []
        updated_records: list[ImportRecord] = []

        for ticket in dataset.tickets:
            matches = source_issues.get(ticket.source_id, [])
            record = record_by_source.get(ticket.source_id)
            issue_key = str(matches[0]["key"]) if len(matches) == 1 else None
            failed_checks: list[str] = []
            explanations: list[str] = []
            baseline_status = "NOT_CHECKED"
            source_fingerprint_status = "NOT_CHECKED"
            content_fingerprint_status = "NOT_CHECKED"
            current_fingerprint: str | None = None

            if not matches:
                missing_tickets.append(ticket.source_id)
                failed_checks.append("source_issue_missing")
                explanations.append("No project issue contains the expected source ID.")
            elif len(matches) > 1:
                conflicted_tickets.append(ticket.source_id)
                failed_checks.append("duplicate_source_id")
                explanations.append(
                    "Multiple project issues contain this source ID: "
                    + ", ".join(sorted(str(issue["key"]) for issue in matches))
                )
            else:
                issue = matches[0]
                issue_key = str(issue["key"])
                fields = issue["fields"]
                mapped_fields = map_ticket_to_jira(
                    ticket, self.project_key, self.import_label
                )
                expected_labels = self._mapped_labels(mapped_fields)
                jira_labels = fields.get("labels") or []
                if not isinstance(jira_labels, list) or any(
                    not isinstance(label, str) for label in jira_labels
                ):
                    failed_checks.append("malformed_labels")
                    explanations.append("Jira labels are not a list of strings.")
                    jira_labels = []
                if self.import_label not in jira_labels:
                    failed_checks.append("import_label_missing")
                    explanations.append("The expected import label is absent.")

                description_text = self._adf_text(fields.get("description"))
                provenance = self._source_provenance(description_text)
                if not has_source_marker(fields.get("description"), ticket.source_id):
                    failed_checks.append("source_marker_mismatch")
                    explanations.append("The Jira description source marker is missing or incorrect.")
                if not provenance or provenance.get("source_id") != ticket.source_id:
                    failed_checks.append("source_provenance_id_mismatch")
                    explanations.append("Source ID in provenance metadata is missing or incorrect.")
                if not provenance or provenance.get("original_issue_key") != ticket.issue_key:
                    failed_checks.append("original_issue_key_mismatch")
                    explanations.append("Original issue key provenance is missing or incorrect.")

                expected_source_hash = source_fingerprint(ticket)
                if has_fingerprint(fields.get("description"), expected_source_hash):
                    source_fingerprint_status = "PASSED"
                else:
                    source_fingerprint_status = "FAILED"
                    failed_checks.append("source_fingerprint_mismatch")
                    explanations.append(
                        "Description source fingerprint does not match the incoming record."
                    )

                try:
                    property_response = client.get_issue_property(
                        issue_key, BASELINE_PROPERTY_KEY
                    )
                    baseline = (
                        JiraImportBaseline.model_validate(property_response.get("value"))
                        if property_response is not None
                        else None
                    )
                except JiraNotFoundError:
                    baseline = None
                except JiraClientError as exc:
                    baseline = None
                    failed_checks.append("baseline_retrieval_failed")
                    explanations.append(f"Baseline could not be verified: {exc}")
                    incomplete_retrievals.append(
                        f"Baseline retrieval failed for {issue_key}: {exc}"
                    )
                except (ValidationError, AttributeError) as exc:
                    baseline = None
                    failed_checks.append("baseline_invalid")
                    explanations.append(f"Baseline response was malformed: {exc}")

                if baseline is None:
                    if not any(
                        check in failed_checks
                        for check in (
                            "baseline_retrieval_failed",
                            "baseline_invalid",
                        )
                    ):
                        failed_checks.append("baseline_missing_or_invalid")
                        explanations.append(
                            "A valid versioned Jira import baseline is required."
                        )
                    baseline_status = "FAILED"
                else:
                    baseline_valid = (
                        baseline.source_id == ticket.source_id
                        and baseline.source_fingerprint == expected_source_hash
                        and baseline.managed_labels == expected_labels
                    )
                    baseline_status = "PASSED" if baseline_valid else "FAILED"
                    if not baseline_valid:
                        failed_checks.append("baseline_source_or_schema_mismatch")
                        explanations.append(
                            "Baseline source ID, source fingerprint, or managed labels do not match."
                        )
                    current_managed_labels = sorted(
                        set(jira_labels) & set(baseline.managed_labels)
                    )
                    current_fingerprint = jira_content_fingerprint(
                        fields, current_managed_labels
                    )
                    expected_fingerprint = jira_content_fingerprint(
                        mapped_fields, expected_labels
                    )
                    content_matches = (
                        current_fingerprint == baseline.jira_content_fingerprint
                        and current_fingerprint == expected_fingerprint
                    )
                    content_fingerprint_status = (
                        "PASSED" if content_matches else "FAILED"
                    )
                    if not content_matches:
                        failed_checks.append("jira_content_fingerprint_mismatch")
                        explanations.append(
                            "Current normalized Jira content differs from expected content or baseline."
                        )

                if record and record.decision in {
                    "CONFLICT",
                    "BASELINE_REQUIRED",
                    "BASELINE_INVALID",
                    "FAILED",
                    "VERIFICATION_FAILED",
                    "RECONCILIATION_FAILED",
                }:
                    failed_checks.append("import_operation_not_successful")
                    explanations.append(
                        f"Import decision was {record.decision}: {record.reason or 'no reason recorded'}"
                    )

            ticket_status = "VERIFIED" if not failed_checks else "FAILED"
            if ticket_status == "VERIFIED":
                verified_tickets += 1
            elif ticket.source_id not in missing_tickets and ticket.source_id not in conflicted_tickets:
                failed_tickets.append(ticket.source_id)
            ticket_checks.append(
                TicketVerificationRecord(
                    source_id=ticket.source_id,
                    original_issue_key=ticket.issue_key,
                    issue_key=issue_key,
                    import_decision=record.decision if record else None,
                    verification_status=ticket_status,
                    failed_checks=failed_checks,
                    explanations=explanations,
                    baseline_status=baseline_status,
                    source_fingerprint_status=source_fingerprint_status,
                    jira_content_fingerprint_status=content_fingerprint_status,
                )
            )

            source_comment_checks = self._verify_ticket_comments(
                ticket,
                issue_key,
                marker_locations,
                comment_reads,
                comment_inventory_complete=all(
                    comments is not None for comments in comment_reads.values()
                ),
            )
            for comment_check in source_comment_checks:
                if comment_check.verification_status == "VERIFIED":
                    verified_comments += 1
                elif comment_check.decision == "MISSING":
                    missing_comments.append(
                        f"{ticket.source_id}:{comment_check.source_comment_id}"
                    )
                else:
                    failed_comments.append(
                        f"{ticket.source_id}:{comment_check.source_comment_id}"
                    )
            all_comment_checks.extend(source_comment_checks)

            if record:
                prior_comment_records = {
                    item.source_comment_id: item for item in record.comment_records
                }
                merged_comments: list[ImportCommentRecord] = []
                for verified in source_comment_checks:
                    prior = prior_comment_records.get(verified.source_comment_id)
                    if prior:
                        verified = verified.model_copy(
                            update={
                                "decision": (
                                    prior.decision
                                    if prior.decision in {"CREATED", "ALREADY_EXISTS"}
                                    and verified.verification_status == "VERIFIED"
                                    else verified.decision
                                ),
                                "created_jira_comment_id": (
                                    prior.created_jira_comment_id
                                    or verified.created_jira_comment_id
                                ),
                                "creation_attempt_count": prior.creation_attempt_count,
                                "timeout_recovery_outcome": prior.timeout_recovery_outcome,
                                "reason": verified.reason,
                            }
                        )
                    merged_comments.append(verified)
                updated_records.append(
                    record.model_copy(
                        update={
                            "issue_key": issue_key or record.issue_key,
                            "comment_records": merged_comments,
                            "verification_outcome": (
                                "PASSED" if ticket_status == "VERIFIED" else "FAILED"
                            ),
                        }
                    )
                )
        report_status = "INCOMPLETE" if incomplete_retrievals else (
            "PASSED"
            if (
                verified_tickets == len(dataset.tickets)
                and verified_comments == expected_comments
                and not duplicate_source_ids
                and not duplicate_comment_markers
            )
            else "FAILED"
        )
        if duplicate_source_ids:
            for source_id in duplicate_source_ids:
                if source_id not in conflicted_tickets:
                    conflicted_tickets.append(source_id)
        report = ImportVerificationReport(
            status=report_status,
            verified_at=verified_at,
            expected_tickets=len(dataset.tickets),
            verified_tickets=verified_tickets,
            missing_tickets=sorted(set(missing_tickets)),
            duplicate_source_ids=duplicate_source_ids,
            conflicted_tickets=sorted(set(conflicted_tickets)),
            failed_tickets=sorted(set(failed_tickets)),
            expected_comments=expected_comments,
            verified_comments=verified_comments,
            missing_comments=sorted(set(missing_comments)),
            duplicate_comment_markers=duplicate_comment_markers,
            failed_comments=sorted(set(failed_comments)),
            incomplete_retrievals=incomplete_retrievals,
            checks=ticket_checks,
            comment_checks=all_comment_checks,
            error=(
                "Project-wide verification completed with failed checks."
                if report_status == "FAILED"
                else None
            ),
        )
        return report, updated_records

    @staticmethod
    def _validate_issue_inventory(issues: object) -> None:
        if not isinstance(issues, list):
            raise TypeError("Project-wide issue retrieval returned a non-list response.")
        for issue in issues:
            if (
                not isinstance(issue, dict)
                or not isinstance(issue.get("key"), str)
                or not issue["key"]
                or not isinstance(issue.get("fields"), dict)
            ):
                raise TypeError("Project-wide issue retrieval returned a malformed issue.")

    @classmethod
    def _source_id_from_description(cls, description: object) -> str | None:
        text = cls._adf_text(description)
        marker_prefix = "Source ID:"
        for line in text.splitlines():
            candidate = line.strip()
            if candidate.startswith(marker_prefix):
                source_id = candidate[len(marker_prefix) :].strip()
                return source_id or None
        return None

    @staticmethod
    def _source_provenance(description_text: str) -> dict[str, object] | None:
        decoder = json.JSONDecoder()
        provenance = None
        for index, character in enumerate(description_text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(description_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "source_id" in value:
                provenance = value
        return provenance

    @classmethod
    def _comment_marker_ids(cls, body: object) -> list[str]:
        return re.findall(
            r"(?<!\S)\[Imported comment ID:\s*([^\]]+?)\s*\](?=\s|$)",
            cls._normalized_comment_text(body),
        )

    @staticmethod
    def _is_valid_adf(body: object) -> bool:
        if (
            not isinstance(body, dict)
            or body.get("type") != "doc"
            or body.get("version") != 1
            or not isinstance(body.get("content"), list)
        ):
            return False

        def valid_node(node: object) -> bool:
            if not isinstance(node, dict) or not isinstance(node.get("type"), str):
                return False
            if node["type"] == "text" and not isinstance(node.get("text"), str):
                return False
            children = node.get("content")
            return children is None or (
                isinstance(children, list) and all(valid_node(child) for child in children)
            )

        return all(valid_node(node) for node in body["content"])

    def _verify_ticket_comments(
        self,
        ticket: DatasetTicket,
        issue_key: str | None,
        marker_locations: dict[str, list[tuple[str, dict]]],
        comment_reads: dict[str, list[dict] | None],
        *,
        comment_inventory_complete: bool,
    ) -> list[ImportCommentRecord]:
        checks: list[ImportCommentRecord] = []
        for source_comment in ticket.comments:
            source_id = source_comment.source_id
            locations = marker_locations.get(source_id, [])
            expected_key = issue_key
            correct_locations = [
                (key, comment) for key, comment in locations if key == expected_key
            ]
            failed_checks: list[str] = []
            provenance_checks: dict[str, bool] = {}
            verification_status = "VERIFICATION_FAILED"
            if expected_key is None:
                decision = "MISSING"
                reason = "The source issue was not uniquely verified."
                failed_checks.append("source_issue_not_uniquely_verified")
                matching_comment = None
            elif comment_reads.get(expected_key) is None:
                decision = "FAILED"
                reason = "Comments for the expected Jira issue could not be retrieved."
                failed_checks.append("comment_retrieval_failed")
                matching_comment = None
                verification_status = "NOT_VERIFIED"
            elif not comment_inventory_complete:
                decision = "FAILED"
                reason = "Project-wide comment inventory is incomplete."
                failed_checks.append("project_comment_inventory_incomplete")
                matching_comment = None
                verification_status = "NOT_VERIFIED"
            elif len(locations) > 1:
                decision = "CONFLICT"
                reason = "The source comment marker appears multiple times project-wide."
                failed_checks.append("duplicate_comment_marker")
                matching_comment = correct_locations[0][1] if correct_locations else None
            elif not locations:
                decision = "MISSING"
                reason = "No Jira comment contains the expected source comment marker."
                failed_checks.append("comment_missing")
                matching_comment = None
            elif not correct_locations:
                decision = "CONFLICT"
                reason = (
                    "The imported comment marker exists, but it belongs to a different Jira issue."
                )
                failed_checks.append("wrong_issue_association")
                matching_comment = locations[0][1]
            else:
                matching_comment = correct_locations[0][1]
                body = matching_comment.get("body")
                if not self._is_valid_adf(body):
                    failed_checks.append("invalid_adf")
                actual_text = self._normalized_comment_text(body)
                expected_text = self._normalized_comment_text(map_comment(source_comment))
                provenance_checks = {
                    "source_marker": f"[Imported comment ID: {source_id}]" in actual_text,
                    "author": source_comment.author in actual_text,
                    "timestamp": source_comment.created.isoformat() in actual_text,
                    "body": source_comment.body in actual_text,
                    "links": all(str(link) in actual_text for link in source_comment.links),
                    "exact_mapped_content": actual_text == expected_text,
                    "correct_issue": matching_comment is not None,
                }
                failed_checks.extend(
                    check for check, passed in provenance_checks.items() if not passed
                )
                if failed_checks:
                    decision = "CONFLICT"
                    reason = "Imported comment provenance or content did not match its source."
                else:
                    decision = "VERIFIED"
                    reason = "Exactly one matching ADF comment and all source metadata verified."

            comment_id = self._jira_comment_id(matching_comment)
            checks.append(
                ImportCommentRecord(
                    source_comment_id=source_id,
                    original_issue_key=ticket.issue_key,
                    issue_key=expected_key,
                    decision=decision,
                    reason=reason,
                    existing_jira_comment_id=comment_id,
                    created_jira_comment_id=comment_id,
                    verification_status=(
                        "VERIFIED"
                        if decision == "VERIFIED"
                        else verification_status
                    ),
                    provenance_checks=provenance_checks,
                    failed_checks=failed_checks,
                    error_details=(
                        "Imported comment marker removed entirely cannot be identified "
                        "as an imported comment."
                        if decision == "MISSING"
                        else None
                    ),
                )
            )
        return checks

    @staticmethod
    def _final_import_status(
        records: list[ImportRecord], verification: ImportVerificationReport
    ) -> str:
        if any(
            record.decision == "AMBIGUOUS_OUTCOME"
            or any(
                comment.decision == "AMBIGUOUS_OUTCOME"
                for comment in record.comment_records
            )
            for record in records
        ):
            return "AMBIGUOUS_OUTCOME"
        if any(
            comment.verification_status == "VERIFICATION_FAILED"
            for record in records
            for comment in record.comment_records
        ):
            return "VERIFICATION_FAILED"
        if verification.status == "FAILED":
            return "VERIFICATION_FAILED"
        if verification.status == "INCOMPLETE":
            return "UNVERIFIED"

        operation_failures = {
            "FAILED",
            "RECONCILIATION_FAILED",
            "BASELINE_REQUIRED",
            "BASELINE_INVALID",
            "CONFLICT",
        }
        failed_operations = any(record.decision in operation_failures for record in records)
        if failed_operations:
            has_verified_work = verification.verified_tickets > 0 or verification.verified_comments > 0
            return "PARTIAL_SUCCESS" if has_verified_work else "FAILED"
        if (
            verification.verified_tickets == verification.expected_tickets
            and verification.verified_comments == verification.expected_comments
        ):
            return "COMPLETE_SUCCESS"
        has_verified_work = verification.verified_tickets > 0 or verification.verified_comments > 0
        return "PARTIAL_SUCCESS" if has_verified_work else "FAILED"

    def _create_with_baseline(
        self, ticket: DatasetTicket, mapped_fields: dict[str, object]
    ) -> ImportRecord:
        client = self._required_jira_client()
        source_hash = source_fingerprint(ticket)
        try:
            issue_key = client.create_issue(mapped_fields)
        except JiraClientError as exc:
            return self._record(
                ticket,
                None,
                "FAILED",
                f"Issue creation failed: {exc}",
                incoming_source_fingerprint=source_hash,
                write_outcome="FAILED",
            )

        managed_labels = self._mapped_labels(mapped_fields)
        expected_hash = jira_content_fingerprint(mapped_fields, managed_labels)
        try:
            created_issue = client.get_issue(issue_key)
            created_fields = created_issue.get("fields")
            if not isinstance(created_fields, dict):
                raise TypeError("Created issue response has no fields object.")
            actual_hash = jira_content_fingerprint(created_fields, managed_labels)
            actual_labels = set(created_fields.get("labels") or [])
            if actual_hash != expected_hash or not set(managed_labels).issubset(actual_labels):
                return self._record(
                    ticket,
                    issue_key,
                    "VERIFICATION_FAILED",
                    "Created Jira content does not match the mapped source fields.",
                    incoming_source_fingerprint=source_hash,
                    current_jira_content_fingerprint=actual_hash,
                    write_outcome="CREATED",
                    verification_outcome="FAILED",
                    baseline_persistence_outcome="NOT_ATTEMPTED",
                )
        except (JiraClientError, ValueError, TypeError) as exc:
            return self._record(
                ticket,
                issue_key,
                "VERIFICATION_FAILED",
                f"Created issue could not be verified: {exc}",
                incoming_source_fingerprint=source_hash,
                write_outcome="CREATED",
                verification_outcome="FAILED",
                baseline_persistence_outcome="NOT_ATTEMPTED",
            )

        baseline = self._new_baseline(ticket, actual_hash, managed_labels)
        persistence = self._persist_baseline(issue_key, baseline)
        if not persistence:
            return self._record(
                ticket,
                issue_key,
                "FAILED",
                "Issue was created and verified, but its baseline could not be confirmed.",
                incoming_source_fingerprint=source_hash,
                current_jira_content_fingerprint=actual_hash,
                write_outcome="CREATED",
                verification_outcome="PASSED",
                baseline_persistence_outcome="FAILED_OR_UNRESOLVED",
            )
        return self._record(
            ticket,
            issue_key,
            "CREATED",
            "Issue created, read back, verified, and baseline persisted.",
            incoming_source_fingerprint=source_hash,
            current_jira_content_fingerprint=actual_hash,
            write_outcome="CREATED",
            verification_outcome="PASSED",
            baseline_persistence_outcome="PERSISTED",
        )

    def _reconcile_existing(
        self,
        ticket: DatasetTicket,
        issue_key: str,
        mapped_fields: dict[str, object],
    ) -> ImportRecord:
        client = self._required_jira_client()
        source_hash = source_fingerprint(ticket)
        try:
            issue = client.get_issue(issue_key)
            property_response = client.get_issue_property(issue_key, BASELINE_PROPERTY_KEY)
        except JiraClientError as exc:
            return self._record(
                ticket,
                issue_key,
                "RECONCILIATION_FAILED",
                f"Unable to read issue or import baseline: {exc}",
                incoming_source_fingerprint=source_hash,
            )
        if property_response is None:
            return self._record(
                ticket,
                issue_key,
                "BASELINE_REQUIRED",
                "No prior verified import baseline exists; update was blocked.",
                incoming_source_fingerprint=source_hash,
            )
        try:
            baseline = JiraImportBaseline.model_validate(property_response.get("value"))
        except (ValidationError, AttributeError) as exc:
            return self._record(
                ticket,
                issue_key,
                "BASELINE_INVALID",
                f"Stored import baseline is malformed or unsupported: {exc}",
                incoming_source_fingerprint=source_hash,
            )
        if baseline.source_id != ticket.source_id:
            return self._record(
                ticket,
                issue_key,
                "BASELINE_INVALID",
                "Stored baseline source ID does not match the dataset ticket.",
                incoming_source_fingerprint=source_hash,
                previous_source_fingerprint=baseline.source_fingerprint,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
            )

        current_fields = issue.get("fields") if isinstance(issue, dict) else None
        if not isinstance(current_fields, dict):
            return self._record(
                ticket,
                issue_key,
                "RECONCILIATION_FAILED",
                "Jira issue response has no valid fields object.",
                incoming_source_fingerprint=source_hash,
                previous_source_fingerprint=baseline.source_fingerprint,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
            )
        if not has_fingerprint(
            current_fields.get("description"), baseline.source_fingerprint
        ):
            return self._record(
                ticket,
                issue_key,
                "BASELINE_INVALID",
                "Stored source fingerprint does not match the fingerprint marker in Jira.",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
            )
        current_hash = jira_content_fingerprint(current_fields, baseline.managed_labels)
        current_labels = set(current_fields.get("labels") or [])
        if (
            current_hash != baseline.jira_content_fingerprint
            or not set(baseline.managed_labels).issubset(current_labels)
        ):
            return self._record(
                ticket,
                issue_key,
                "CONFLICT",
                "Current source-managed Jira content differs from the stored baseline.",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                current_jira_content_fingerprint=current_hash,
                verification_outcome="MANUAL_EDIT_DETECTED",
            )

        if baseline.source_fingerprint == source_hash:
            return self._record(
                ticket,
                issue_key,
                "UNCHANGED",
                "Incoming source fingerprint and Jira content both match the baseline.",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                current_jira_content_fingerprint=current_hash,
                verification_outcome="PASSED",
            )

        try:
            latest_property = client.get_issue_property(issue_key, BASELINE_PROPERTY_KEY)
            latest_issue = client.get_issue(issue_key)
        except JiraClientError as exc:
            return self._record(
                ticket,
                issue_key,
                "RECONCILIATION_FAILED",
                f"Unable to recheck Jira before update: {exc}",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                current_jira_content_fingerprint=current_hash,
            )
        if latest_property is None:
            return self._record(
                ticket,
                issue_key,
                "BASELINE_REQUIRED",
                "Import baseline disappeared during pre-update reconciliation.",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                current_jira_content_fingerprint=current_hash,
            )
        try:
            latest_baseline = JiraImportBaseline.model_validate(
                latest_property.get("value") if latest_property else None
            )
        except (ValidationError, AttributeError) as exc:
            return self._record(
                ticket,
                issue_key,
                "BASELINE_INVALID",
                f"Baseline changed or became invalid before update: {exc}",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                current_jira_content_fingerprint=current_hash,
            )
        latest_fields = latest_issue.get("fields") if isinstance(latest_issue, dict) else None
        latest_hash = (
            jira_content_fingerprint(latest_fields, baseline.managed_labels)
            if isinstance(latest_fields, dict)
            else None
        )
        if latest_baseline != baseline or latest_hash != baseline.jira_content_fingerprint:
            return self._record(
                ticket,
                issue_key,
                "CONFLICT",
                "Jira issue or baseline changed during pre-update reconciliation.",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                current_jira_content_fingerprint=latest_hash,
                verification_outcome="PRE_UPDATE_RECHECK_FAILED",
            )

        incoming_managed_labels = set(self._mapped_labels(mapped_fields))
        next_managed_labels = sorted(incoming_managed_labels)
        update_fields = {
            name: mapped_fields[name]
            for name in ("summary", "description", "issuetype")
            if name in mapped_fields
        }
        labels_to_add = sorted(incoming_managed_labels - set(baseline.managed_labels))
        labels_to_remove = sorted(set(baseline.managed_labels) - incoming_managed_labels)
        label_operations = [
            *({"remove": label} for label in labels_to_remove),
            *({"add": label} for label in labels_to_add),
        ]
        label_update: dict[str, object] | None = (
            {"labels": label_operations} if label_operations else None
        )
        expected_hash = jira_content_fingerprint(update_fields, next_managed_labels)
        try:
            client.update_issue(issue_key, update_fields, update=label_update)
            write_outcome = "SUCCEEDED"
        except JiraNetworkError as exc:
            write_outcome = "AMBIGUOUS"
            try:
                ambiguous_property = client.get_issue_property(
                    issue_key, BASELINE_PROPERTY_KEY
                )
                ambiguous_baseline = JiraImportBaseline.model_validate(
                    ambiguous_property.get("value") if ambiguous_property else None
                )
                ambiguous_issue = client.get_issue(issue_key)
                ambiguous_fields = ambiguous_issue.get("fields")
                if not isinstance(ambiguous_fields, dict):
                    raise TypeError("Jira issue response has no fields object.")
                ambiguous_hash = jira_content_fingerprint(
                    ambiguous_fields, next_managed_labels
                )
                ambiguous_labels = set(ambiguous_fields.get("labels") or [])
            except (JiraClientError, ValidationError, AttributeError, TypeError) as read_exc:
                return self._record(
                    ticket,
                    issue_key,
                    "FAILED",
                    f"Update timed out ({exc}); outcome could not be resolved: {read_exc}",
                    previous_source_fingerprint=baseline.source_fingerprint,
                    incoming_source_fingerprint=source_hash,
                    stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                    current_jira_content_fingerprint=None,
                    write_outcome="AMBIGUOUS_UNRESOLVED",
                )
            if ambiguous_baseline != baseline:
                return self._record(
                    ticket,
                    issue_key,
                    "FAILED",
                    f"Update timed out ({exc}); the baseline changed and outcome is unresolved.",
                    previous_source_fingerprint=baseline.source_fingerprint,
                    incoming_source_fingerprint=source_hash,
                    stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                    write_outcome="AMBIGUOUS_UNRESOLVED",
                )
            if (
                ambiguous_hash != expected_hash
                or not set(next_managed_labels).issubset(ambiguous_labels)
            ):
                return self._record(
                    ticket,
                    issue_key,
                    "FAILED",
                    f"Update timed out ({exc}); Jira content does not confirm the intended update.",
                    previous_source_fingerprint=baseline.source_fingerprint,
                    incoming_source_fingerprint=source_hash,
                    stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                    current_jira_content_fingerprint=ambiguous_hash,
                    write_outcome="AMBIGUOUS_UNRESOLVED",
                    verification_outcome="FAILED",
                )
        except JiraClientError as exc:
            return self._record(
                ticket,
                issue_key,
                "FAILED",
                f"Issue update failed: {exc}",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                current_jira_content_fingerprint=current_hash,
                write_outcome="FAILED",
            )

        try:
            updated_issue = client.get_issue(issue_key)
            updated_fields = updated_issue.get("fields")
            if not isinstance(updated_fields, dict):
                raise TypeError("Updated issue response has no fields object.")
            verified_hash = jira_content_fingerprint(updated_fields, next_managed_labels)
            verified_labels = set(updated_fields.get("labels") or [])
            if (
                verified_hash != expected_hash
                or not set(next_managed_labels).issubset(verified_labels)
            ):
                return self._record(
                    ticket,
                    issue_key,
                    "VERIFICATION_FAILED",
                    "Updated Jira content does not match the intended source-managed content.",
                    previous_source_fingerprint=baseline.source_fingerprint,
                    incoming_source_fingerprint=source_hash,
                    stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                    current_jira_content_fingerprint=verified_hash,
                    write_outcome=write_outcome,
                    verification_outcome="FAILED",
                    baseline_persistence_outcome="NOT_ATTEMPTED",
                )
        except (JiraClientError, ValueError, TypeError) as exc:
            return self._record(
                ticket,
                issue_key,
                "VERIFICATION_FAILED",
                f"Updated issue could not be verified: {exc}",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                write_outcome=write_outcome,
                verification_outcome="FAILED",
                baseline_persistence_outcome="NOT_ATTEMPTED",
            )

        next_baseline = self._new_baseline(ticket, verified_hash, next_managed_labels)
        if not self._persist_baseline(issue_key, next_baseline):
            return self._record(
                ticket,
                issue_key,
                "FAILED",
                "Issue update was verified, but the new baseline could not be confirmed.",
                previous_source_fingerprint=baseline.source_fingerprint,
                incoming_source_fingerprint=source_hash,
                stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
                current_jira_content_fingerprint=verified_hash,
                write_outcome=write_outcome,
                verification_outcome="PASSED",
                baseline_persistence_outcome="FAILED_OR_UNRESOLVED",
            )
        return self._record(
            ticket,
            issue_key,
            "UPDATED",
            "Source changed; Jira content matched the prior baseline, and the update and new baseline were verified.",
            previous_source_fingerprint=baseline.source_fingerprint,
            incoming_source_fingerprint=source_hash,
            stored_jira_baseline_fingerprint=baseline.jira_content_fingerprint,
            current_jira_content_fingerprint=verified_hash,
            write_outcome=write_outcome,
            verification_outcome="PASSED",
            baseline_persistence_outcome="PERSISTED",
        )

    @staticmethod
    def _new_baseline(
        ticket: DatasetTicket,
        content_fingerprint: str,
        managed_labels: list[str],
    ) -> JiraImportBaseline:
        return JiraImportBaseline(
            source_id=ticket.source_id,
            source_fingerprint=source_fingerprint(ticket),
            jira_content_fingerprint=content_fingerprint,
            last_imported_at=datetime.now(UTC),
            managed_labels=managed_labels,
        )

    def _persist_baseline(
        self, issue_key: str, baseline: JiraImportBaseline
    ) -> bool:
        client = self._required_jira_client()
        expected = baseline.model_dump(mode="json")
        try:
            client.set_issue_property(issue_key, BASELINE_PROPERTY_KEY, expected)
        except JiraClientError:
            pass
        try:
            response = client.get_issue_property(issue_key, BASELINE_PROPERTY_KEY)
            actual = JiraImportBaseline.model_validate(
                response.get("value") if response else None
            )
        except (JiraClientError, ValidationError, AttributeError):
            return False
        return actual == baseline

    def _required_jira_client(self) -> ImportJiraClient:
        if self.jira_client is None:
            raise RuntimeError("A configured Jira client is required for baseline reconciliation.")
        return self.jira_client

    @staticmethod
    def _mapped_labels(mapped_fields: dict[str, object]) -> list[str]:
        labels = mapped_fields.get("labels")
        if not isinstance(labels, list) or any(
            not isinstance(label, str) for label in labels
        ):
            raise TypeError("Mapped Jira labels must be a list of strings.")
        return sorted(set(labels))

    @staticmethod
    def _record(
        ticket: DatasetTicket,
        issue_key: str | None,
        decision: str,
        reason: str,
        *,
        previous_source_fingerprint: str | None = None,
        incoming_source_fingerprint: str | None = None,
        stored_jira_baseline_fingerprint: str | None = None,
        current_jira_content_fingerprint: str | None = None,
        write_outcome: str | None = None,
        verification_outcome: str | None = None,
        baseline_persistence_outcome: str | None = None,
    ) -> ImportRecord:
        action_by_decision = {
            "CREATED": "created",
            "UPDATED": "updated",
            "UNCHANGED": "unchanged",
        }
        return ImportRecord(
            source_id=ticket.source_id,
            issue_key=issue_key,
            action=action_by_decision.get(decision, decision.lower()),
            previous_source_fingerprint=previous_source_fingerprint,
            incoming_source_fingerprint=incoming_source_fingerprint,
            stored_jira_baseline_fingerprint=stored_jira_baseline_fingerprint,
            current_jira_content_fingerprint=current_jira_content_fingerprint,
            decision=decision,
            reason=reason,
            write_outcome=write_outcome,
            verification_outcome=verification_outcome,
            baseline_persistence_outcome=baseline_persistence_outcome,
        )

    @staticmethod
    def _adf_text(value: object) -> str:
        block_types = {
            "blockquote",
            "codeBlock",
            "heading",
            "listItem",
            "orderedList",
            "paragraph",
            "panel",
            "bulletList",
        }
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(DatasetImporter._adf_text(item) for item in value)
        if not isinstance(value, dict):
            return ""
        node_type = value.get("type")
        if node_type == "text":
            text = value.get("text")
            return text if isinstance(text, str) else ""
        if node_type == "hardBreak":
            return "\n"
        content = value.get("content")
        if isinstance(content, (dict, list)):
            text = DatasetImporter._adf_text(content)
            return f"{text}\n" if node_type in block_types else text
        return ""

    @classmethod
    def _normalized_comment_text(cls, body: object) -> str:
        text = cls._adf_text(body).replace("\r\n", "\n").replace("\r", "\n")
        return unicodedata.normalize("NFC", text).rstrip("\n")

    @classmethod
    def _comment_matches(
        cls, comments: list[dict], source_comment_id: str
    ) -> list[dict]:
        marker = f"[Imported comment ID: {source_comment_id}]"
        pattern = re.compile(rf"(?<!\S){re.escape(marker)}(?=\s|$)")
        return [
            comment
            for comment in comments
            if pattern.search(cls._normalized_comment_text(comment.get("body")))
        ]

    @staticmethod
    def _jira_comment_id(comment: dict | None) -> str | None:
        if comment is None:
            return None
        comment_id = comment.get("id")
        return str(comment_id) if isinstance(comment_id, (str, int)) else None

    def _read_comments(self, issue_key: str) -> list[dict]:
        client = self._required_jira_client()
        comments = client.get_comments(issue_key)
        if not isinstance(comments, list):
            raise TypeError("Jira comment retrieval returned a non-list response.")
        for comment in comments:
            if (
                not isinstance(comment, dict)
                or not isinstance(comment.get("id"), (str, int))
                or "body" not in comment
            ):
                raise TypeError("Jira returned a malformed comment record.")
        return comments

    @staticmethod
    def _comment_status(records: list[ImportCommentRecord]) -> str:
        if not records:
            return "NOT_APPLICABLE"
        if any(record.verification_status == "VERIFICATION_FAILED" for record in records):
            return "VERIFICATION_FAILED"
        if any(record.decision == "AMBIGUOUS_OUTCOME" for record in records):
            return "AMBIGUOUS_OUTCOME"
        failures = [
            record
            for record in records
            if record.decision in {"FAILED", "CONFLICT"}
        ]
        if failures:
            if any(record.verification_status == "VERIFIED" for record in records):
                return "PARTIAL_SUCCESS"
            return "FAILED_COMMENT_RECONCILIATION"
        if all(record.verification_status == "VERIFIED" for record in records):
            return "COMPLETE_SUCCESS"
        return "FAILED_COMMENT_RECONCILIATION"

    @staticmethod
    def _comment_record(
        ticket: DatasetTicket,
        issue_key: str | None,
        comment_source_id: str,
        decision: str,
        reason: str,
        *,
        existing_comment_id: str | None = None,
        created_comment_id: str | None = None,
        attempts: int = 0,
        timeout_recovery: str | None = None,
        verification_status: str = "NOT_VERIFIED",
        error_details: str | None = None,
    ) -> ImportCommentRecord:
        return ImportCommentRecord(
            source_comment_id=comment_source_id,
            original_issue_key=ticket.issue_key,
            issue_key=issue_key,
            decision=decision,
            reason=reason,
            existing_jira_comment_id=existing_comment_id,
            created_jira_comment_id=created_comment_id,
            creation_attempt_count=attempts,
            timeout_recovery_outcome=timeout_recovery,
            verification_status=verification_status,
            error_details=error_details,
        )

    def _blocked_comment_records(
        self, ticket: DatasetTicket, issue_key: str | None, reason: str
    ) -> list[ImportCommentRecord]:
        if not ticket.comments:
            return []
        decision = "CONFLICT" if issue_key else "FAILED"
        return [
            self._comment_record(
                ticket,
                issue_key,
                comment.source_id,
                decision,
                reason,
                verification_status="NOT_VERIFIED",
            )
            for comment in ticket.comments
        ]

    def _import_comments(
        self, issue_key: str, ticket: DatasetTicket
    ) -> list[ImportCommentRecord]:
        if not ticket.comments:
            return []
        try:
            current_comments = self._read_comments(issue_key)
        except (JiraClientError, TypeError) as exc:
            return [
                self._comment_record(
                    ticket,
                    issue_key,
                    comment.source_id,
                    "FAILED",
                    "Existing Jira comments could not be inspected; no creation was attempted.",
                    error_details=str(exc),
                )
                for comment in ticket.comments
            ]

        records: list[ImportCommentRecord] = []
        expected_texts = {
            comment.source_id: self._normalized_comment_text(map_comment(comment))
            for comment in ticket.comments
        }
        for source_comment in ticket.comments:
            source_id = source_comment.source_id
            matches = self._comment_matches(current_comments, source_id)
            if len(matches) > 1:
                records.append(
                    self._comment_record(
                        ticket,
                        issue_key,
                        source_id,
                        "CONFLICT",
                        "Multiple Jira comments contain this imported source marker.",
                        verification_status="VERIFICATION_FAILED",
                    )
                )
                continue
            if matches:
                existing = matches[0]
                if self._normalized_comment_text(existing.get("body")) != expected_texts[source_id]:
                    records.append(
                        self._comment_record(
                            ticket,
                            issue_key,
                            source_id,
                            "CONFLICT",
                            "Imported comment content differs from its source; existing comment was preserved.",
                            existing_comment_id=self._jira_comment_id(existing),
                            verification_status="VERIFICATION_FAILED",
                        )
                    )
                else:
                    records.append(
                        self._comment_record(
                            ticket,
                            issue_key,
                            source_id,
                            "ALREADY_EXISTS",
                            "Exactly one matching imported comment exists.",
                            existing_comment_id=self._jira_comment_id(existing),
                            verification_status="VERIFIED",
                        )
                    )
                continue

            attempts = 0
            timeout_recovery: str | None = None
            created_comment_id: str | None = None
            decision = "CREATED"
            reason = "Comment creation request succeeded; awaiting readback verification."
            try:
                attempts += 1
                response = self._required_jira_client().add_comment(
                    issue_key, map_comment(source_comment)
                )
                created_comment_id = self._jira_comment_id(response)
            except JiraNetworkError as exc:
                try:
                    reread = self._read_comments(issue_key)
                except (JiraClientError, TypeError) as read_exc:
                    records.append(
                        self._comment_record(
                            ticket,
                            issue_key,
                            source_id,
                            "AMBIGUOUS_OUTCOME",
                            "Comment creation timed out and comment reread failed; no retry was attempted.",
                            attempts=attempts,
                            timeout_recovery="REREAD_FAILED",
                            verification_status="VERIFICATION_FAILED",
                            error_details=f"{exc}; reread failed: {read_exc}",
                        )
                    )
                    continue
                recovered_matches = self._comment_matches(reread, source_id)
                if len(recovered_matches) > 1:
                    records.append(
                        self._comment_record(
                            ticket,
                            issue_key,
                            source_id,
                            "CONFLICT",
                            "Multiple comments appeared after an ambiguous create.",
                            attempts=attempts,
                            timeout_recovery="DUPLICATE_MARKERS_FOUND",
                            verification_status="VERIFICATION_FAILED",
                            error_details=str(exc),
                        )
                    )
                    continue
                if recovered_matches:
                    recovered = recovered_matches[0]
                    if self._normalized_comment_text(recovered.get("body")) != expected_texts[source_id]:
                        records.append(
                            self._comment_record(
                                ticket,
                                issue_key,
                                source_id,
                                "CONFLICT",
                                "A comment with this marker appeared but its content differs.",
                                existing_comment_id=self._jira_comment_id(recovered),
                                attempts=attempts,
                                timeout_recovery="MISMATCHING_MARKER_FOUND",
                                verification_status="VERIFICATION_FAILED",
                                error_details=str(exc),
                            )
                        )
                        continue
                    created_comment_id = self._jira_comment_id(recovered)
                    timeout_recovery = "FOUND_AFTER_TIMEOUT"
                    reason = "Comment creation timed out; exactly one matching comment was found on reread."
                else:
                    timeout_recovery = "ABSENT_ON_REREAD_RETRYING_ONCE"
                    try:
                        attempts += 1
                        response = self._required_jira_client().add_comment(
                            issue_key, map_comment(source_comment)
                        )
                        created_comment_id = self._jira_comment_id(response)
                        reason = "First attempt timed out with no matching marker; one bounded retry succeeded."
                    except JiraNetworkError as retry_exc:
                        try:
                            retry_reread = self._read_comments(issue_key)
                        except (JiraClientError, TypeError) as read_exc:
                            records.append(
                                self._comment_record(
                                    ticket,
                                    issue_key,
                                    source_id,
                                    "AMBIGUOUS_OUTCOME",
                                    "Retry timed out and the confirming reread failed; no further retry was attempted.",
                                    attempts=attempts,
                                    timeout_recovery="RETRY_REREAD_FAILED",
                                    verification_status="VERIFICATION_FAILED",
                                    error_details=f"{retry_exc}; reread failed: {read_exc}",
                                )
                            )
                            continue
                        retry_matches = self._comment_matches(retry_reread, source_id)
                        if len(retry_matches) == 1 and self._normalized_comment_text(
                            retry_matches[0].get("body")
                        ) == expected_texts[source_id]:
                            created_comment_id = self._jira_comment_id(retry_matches[0])
                            timeout_recovery = "FOUND_AFTER_RETRY_TIMEOUT"
                            reason = "Retry timed out; exactly one matching comment was found."
                        elif len(retry_matches) > 1:
                            records.append(
                                self._comment_record(
                                    ticket,
                                    issue_key,
                                    source_id,
                                    "CONFLICT",
                                    "Multiple comments appeared after the bounded retry.",
                                    attempts=attempts,
                                    timeout_recovery="DUPLICATE_MARKERS_FOUND",
                                    verification_status="VERIFICATION_FAILED",
                                    error_details=str(retry_exc),
                                )
                            )
                            continue
                        elif retry_matches:
                            records.append(
                                self._comment_record(
                                    ticket,
                                    issue_key,
                                    source_id,
                                    "CONFLICT",
                                    "A comment with this marker appeared after retry but content differs.",
                                    existing_comment_id=self._jira_comment_id(retry_matches[0]),
                                    attempts=attempts,
                                    timeout_recovery="MISMATCHING_MARKER_FOUND",
                                    verification_status="VERIFICATION_FAILED",
                                    error_details=str(retry_exc),
                                )
                            )
                            continue
                        else:
                            records.append(
                                self._comment_record(
                                    ticket,
                                    issue_key,
                                    source_id,
                                    "AMBIGUOUS_OUTCOME",
                                    "Two bounded attempts timed out and no matching comment was found.",
                                    attempts=attempts,
                                    timeout_recovery="ABSENT_AFTER_BOUNDED_RETRY",
                                    verification_status="VERIFICATION_FAILED",
                                    error_details=str(retry_exc),
                                )
                            )
                            continue
                    except JiraClientError as retry_exc:
                        records.append(
                            self._comment_record(
                                ticket,
                                issue_key,
                                source_id,
                                "FAILED",
                                "Bounded retry returned a definitive Jira error.",
                                attempts=attempts,
                                timeout_recovery="RETRY_FAILED",
                                error_details=str(retry_exc),
                            )
                        )
                        continue
            except JiraClientError as exc:
                records.append(
                    self._comment_record(
                        ticket,
                        issue_key,
                        source_id,
                        "FAILED",
                        "Comment creation returned a definitive Jira error.",
                        attempts=attempts,
                        error_details=str(exc),
                    )
                )
                continue

            records.append(
                self._comment_record(
                    ticket,
                    issue_key,
                    source_id,
                    decision,
                    reason,
                    created_comment_id=created_comment_id,
                    attempts=attempts,
                    timeout_recovery=timeout_recovery,
                    verification_status="PENDING",
                )
            )

        if any(record.verification_status == "PENDING" for record in records):
            try:
                final_comments = self._read_comments(issue_key)
            except (JiraClientError, TypeError) as exc:
                return [
                    record.model_copy(
                        update={
                            "verification_status": "VERIFICATION_FAILED",
                            "error_details": (
                                f"{record.error_details}; " if record.error_details else ""
                            )
                            + f"Post-creation comment verification failed: {exc}",
                        }
                    )
                    if record.verification_status == "PENDING"
                    else record
                    for record in records
                ]

            verified_records: list[ImportCommentRecord] = []
            for record in records:
                if record.verification_status != "PENDING":
                    verified_records.append(record)
                    continue
                matches = self._comment_matches(final_comments, record.source_comment_id)
                if len(matches) > 1:
                    verified_records.append(
                        record.model_copy(
                            update={
                                "decision": "CONFLICT",
                                "reason": "Post-creation verification found duplicate source markers.",
                                "verification_status": "VERIFICATION_FAILED",
                            }
                        )
                    )
                elif not matches:
                    verified_records.append(
                        record.model_copy(
                            update={
                                "decision": "VERIFICATION_FAILED",
                                "reason": "Expected imported comment is missing after creation.",
                                "verification_status": "VERIFICATION_FAILED",
                            }
                        )
                    )
                elif self._normalized_comment_text(matches[0].get("body")) != expected_texts[
                    record.source_comment_id
                ]:
                    verified_records.append(
                        record.model_copy(
                            update={
                                "decision": "CONFLICT",
                                "reason": "Post-creation comment content differs from source.",
                                "existing_jira_comment_id": self._jira_comment_id(matches[0]),
                                "verification_status": "VERIFICATION_FAILED",
                            }
                        )
                    )
                else:
                    comment_id = self._jira_comment_id(matches[0])
                    verified_records.append(
                        record.model_copy(
                            update={
                                "created_jira_comment_id": record.created_jira_comment_id
                                or comment_id,
                                "verification_status": "VERIFIED",
                            }
                        )
                    )
            records = verified_records
        return records
