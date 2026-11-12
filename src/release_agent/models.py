from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

CallbackDecision = Literal["PASS", "FAIL", "INSUFFICIENT_EVIDENCE"]


class OverallDecision(StrEnum):
    READY = "READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"


class RequirementDecision(StrEnum):
    SATISFIED = "SATISFIED"
    PARTIAL = "PARTIAL"
    NOT_SATISFIED = "NOT_SATISFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ReleaseRequirement(BaseModel):
    id: str
    title: str
    description: str
    mandatory: bool = True
    evidence_issue_keys: list[str] = Field(default_factory=list)


class ReleaseManifest(BaseModel):
    release_id: str
    release_name: str
    version: str
    target_date: datetime | None = None
    requirements: list[ReleaseRequirement] = Field(min_length=1)


class EvidenceComment(BaseModel):
    id: str
    author: str
    created: datetime
    body: str
    links: list[HttpUrl] = Field(default_factory=list)


class EvidenceIssue(BaseModel):
    key: str
    source_id: str
    summary: str
    description: str = ""
    status: str
    issue_type: str = "Task"
    assignee: str | None = None
    reporter: str | None = None
    created: datetime | None = None
    updated: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    links: list[HttpUrl] = Field(default_factory=list)
    comments: list[EvidenceComment] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    requirement_id: str
    issues: list[EvidenceIssue] = Field(default_factory=list)
    collection_errors: list[str] = Field(default_factory=list)
    collected_at: datetime


class RequirementAssessment(BaseModel):
    requirement_id: str
    title: str
    decision: RequirementDecision
    rationale: str
    evidence_issue_keys: list[str] = Field(default_factory=list)
    evidence_summary: str
    confidence: float = Field(ge=0, le=1)
    mandatory: bool = True


class TraceEvent(BaseModel):
    sequence: int
    timestamp: datetime
    requirement_id: str
    stage: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class AssessmentReport(BaseModel):
    release_id: str
    release_name: str
    version: str
    overall_decision: OverallDecision
    summary: str
    assessed_at: datetime
    requirements: list[RequirementAssessment]
    trace: list[TraceEvent]
    model: str


class ImportRecord(BaseModel):
    source_id: str
    issue_key: str | None = None
    action: str
    comments_added: int = 0
    previous_source_fingerprint: str | None = None
    incoming_source_fingerprint: str | None = None
    stored_jira_baseline_fingerprint: str | None = None
    current_jira_content_fingerprint: str | None = None
    decision: str | None = None
    reason: str | None = None
    write_outcome: str | None = None
    verification_outcome: str | None = None
    baseline_persistence_outcome: str | None = None
    comment_records: list["ImportCommentRecord"] = Field(default_factory=list)


class ImportCommentRecord(BaseModel):
    source_comment_id: str
    original_issue_key: str
    issue_key: str | None = None
    decision: str
    reason: str
    existing_jira_comment_id: str | None = None
    created_jira_comment_id: str | None = None
    creation_attempt_count: int = 0
    timeout_recovery_outcome: str | None = None
    verification_status: str
    error_details: str | None = None
    provenance_checks: dict[str, bool] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)


class TicketVerificationRecord(BaseModel):
    source_id: str
    original_issue_key: str
    issue_key: str | None = None
    import_decision: str | None = None
    verification_status: str
    failed_checks: list[str] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)
    baseline_status: str = "NOT_CHECKED"
    source_fingerprint_status: str = "NOT_CHECKED"
    jira_content_fingerprint_status: str = "NOT_CHECKED"


class ImportVerificationReport(BaseModel):
    status: str
    verified_at: datetime
    expected_tickets: int
    verified_tickets: int = 0
    missing_tickets: list[str] = Field(default_factory=list)
    duplicate_source_ids: dict[str, list[str]] = Field(default_factory=dict)
    conflicted_tickets: list[str] = Field(default_factory=list)
    failed_tickets: list[str] = Field(default_factory=list)
    expected_comments: int
    verified_comments: int = 0
    missing_comments: list[str] = Field(default_factory=list)
    duplicate_comment_markers: list[str] = Field(default_factory=list)
    failed_comments: list[str] = Field(default_factory=list)
    incomplete_retrievals: list[str] = Field(default_factory=list)
    checks: list[TicketVerificationRecord] = Field(default_factory=list)
    comment_checks: list[ImportCommentRecord] = Field(default_factory=list)
    error: str | None = None


class ImportResult(BaseModel):
    dry_run: bool
    label: str
    total_tickets: int
    created: int
    updated: int
    unchanged: int
    records: list[ImportRecord]
    project_key: str | None = None
    status: str = "UNVERIFIED"
    duration_seconds: float = 0
    comment_status: str = "NOT_APPLICABLE"
    total_comments: int = 0
    comments_created: int = 0
    comments_already_present: int = 0
    comments_conflicted: int = 0
    comments_failed: int = 0
    execution_mode: str = "live_import"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    verification: ImportVerificationReport | None = None
    verified_tickets: int = 0
    missing_tickets: int = 0
    duplicate_source_ids: dict[str, list[str]] = Field(default_factory=dict)
    verified_comments: int = 0
    missing_comments: int = 0
    duplicate_comment_markers: list[str] = Field(default_factory=list)


class JiraImportBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source_id: str = Field(min_length=1)
    source_fingerprint: str = Field(min_length=1)
    jira_content_fingerprint: str = Field(min_length=1)
    last_imported_at: datetime
    managed_labels: list[str]

    @field_validator("last_imported_at")
    @classmethod
    def validate_timestamp_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("baseline timestamp must include a timezone")
        return value

    @field_validator("managed_labels")
    @classmethod
    def normalize_managed_labels(cls, value: list[str]) -> list[str]:
        return sorted(set(value))


class AgentCallbackRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentCallbackResponse(BaseModel):
    answer: str
    decision: CallbackDecision | None = None
