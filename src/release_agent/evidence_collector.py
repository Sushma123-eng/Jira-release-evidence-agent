import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from release_agent.config import get_settings
from release_agent.importer.schema import DatasetFile
from release_agent.jira_client import JiraClient, JiraClientError
from release_agent.models import (
    EvidenceBundle,
    EvidenceComment,
    EvidenceIssue,
    ReleaseRequirement,
)
from release_agent.utils import utc_now

logger = logging.getLogger(__name__)


class EvidenceCollector:
    def __init__(
        self,
        dataset_path: Path | None = None,
        jira_client: JiraClient | None = None,
        import_label: str | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.jira_client = jira_client
        self.import_label = import_label or get_settings().import_label
        self._dataset: DatasetFile | None = None
        self._jira_source_key_map: dict[str, str] | None = None

    def collect(self, requirement: ReleaseRequirement) -> EvidenceBundle:
        issues: list[EvidenceIssue] = []
        errors: list[str] = []
        if self.jira_client is not None:
            for issue_key in requirement.evidence_issue_keys:
                try:
                    issues.append(self._from_jira(self._resolve_jira_key(issue_key)))
                except (JiraClientError, ValueError, ValidationError, TypeError) as exc:
                    logger.exception("Failed to collect Jira issue %s", issue_key)
                    errors.append(f"{issue_key}: {exc}")
        elif self.dataset_path is not None:
            try:
                dataset = self._load_dataset()
                by_key = {ticket.issue_key: ticket for ticket in dataset.tickets}
                for issue_key in requirement.evidence_issue_keys:
                    ticket = by_key.get(issue_key)
                    if ticket is None:
                        errors.append(f"{issue_key}: not found in local dataset")
                        continue
                    issues.append(
                        EvidenceIssue(
                            key=ticket.issue_key,
                            source_id=ticket.source_id,
                            summary=ticket.summary,
                            description=ticket.description,
                            status=ticket.status,
                            issue_type=ticket.issue_type,
                            assignee=ticket.assignee,
                            reporter=ticket.reporter,
                            created=ticket.created,
                            updated=ticket.updated,
                            labels=ticket.labels,
                            links=ticket.links,
                            comments=[
                                EvidenceComment(
                                    id=comment.source_id,
                                    author=comment.author,
                                    created=comment.created,
                                    body=comment.body,
                                    links=comment.links,
                                )
                                for comment in ticket.comments
                            ],
                        )
                    )
            except (OSError, ValueError, ValidationError) as exc:
                logger.exception("Failed to load local evidence dataset")
                errors.append(f"Dataset: {exc}")
        else:
            errors.append("No Jira client or local dataset is available")
        return EvidenceBundle(
            requirement_id=requirement.id,
            issues=issues,
            collection_errors=errors,
            collected_at=utc_now(),
        )

    def _resolve_jira_key(self, source_issue_key: str) -> str:
        if self.dataset_path is None or self.jira_client is None:
            return source_issue_key
        dataset = self._load_dataset()
        source_ticket = next(
            (
                ticket
                for ticket in dataset.tickets
                if ticket.issue_key == source_issue_key
            ),
            None,
        )
        if source_ticket is None:
            return source_issue_key
        if self._jira_source_key_map is None:
            self._jira_source_key_map = (
                self.jira_client.find_imported_issue_keys_by_source_id(
                    get_settings().jira_project_key, self.import_label
                )
            )
        destination_issue_key = self._jira_source_key_map.get(source_ticket.source_id)
        if destination_issue_key is None:
            raise ValueError(
                f"No imported Jira issue was found in destination project "
                f"{get_settings().jira_project_key!r} for source issue "
                f"{source_issue_key!r}."
            )
        return destination_issue_key

    def _load_dataset(self) -> DatasetFile:
        if self._dataset is None:
            if self.dataset_path is None:
                raise ValueError("A local dataset path was not supplied")
            try:
                with self.dataset_path.open(encoding="utf-8") as file:
                    self._dataset = DatasetFile.model_validate(json.load(file))
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Unable to load evidence dataset: {exc}") from exc
        return self._dataset

    def _from_jira(self, issue_key: str) -> EvidenceIssue:
        if self.jira_client is None:
            raise RuntimeError("Jira client is not available")
        issue = self.jira_client.get_issue(issue_key)
        fields = issue.get("fields") or {}
        description = fields.get("description") or ""
        if isinstance(description, dict):
            description = self._adf_text(description)
        comments = [
            self._jira_comment(item)
            for item in self.jira_client.get_issue_comments(issue_key)
        ]
        return EvidenceIssue(
            key=issue.get("key", issue_key),
            source_id=self._source_id(str(description), issue_key),
            summary=fields.get("summary") or "",
            description=str(description),
            status=(fields.get("status") or {}).get("name", "Unknown"),
            issue_type=(fields.get("issuetype") or {}).get("name", "Task"),
            assignee=self._display_name(fields.get("assignee")),
            reporter=self._display_name(fields.get("reporter")),
            created=fields.get("created"),
            updated=fields.get("updated"),
            labels=fields.get("labels") or [],
            links=[],
            comments=comments,
        )

    @staticmethod
    def _display_name(person: dict[str, Any] | None) -> str | None:
        if not person:
            return None
        return person.get("displayName") or person.get("name") or person.get("accountId")

    @staticmethod
    def _source_id(description: str, fallback: str) -> str:
        for line in description.splitlines():
            if line.startswith("Source ID:"):
                return line.partition(":")[2].strip()
        return fallback

    @staticmethod
    def _adf_text(value: dict[str, Any]) -> str:
        fragments: list[str] = []
        def visit(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "text" and isinstance(node.get("text"), str):
                    fragments.append(node["text"])
                for child in node.values():
                    if isinstance(child, (dict, list)):
                        visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)
        visit(value)
        return "\n".join(fragments)

    @staticmethod
    def _jira_comment(comment: dict[str, Any]) -> EvidenceComment:
        author = comment.get("author") or {}
        body = comment.get("body") or ""
        if isinstance(body, dict):
            body = EvidenceCollector._adf_text(body)
        created = comment.get("created")
        if not created:
            created = datetime.now().astimezone()
        return EvidenceComment(
            id=str(comment.get("id", "unknown")),
            author=EvidenceCollector._display_name(author) or "Unknown",
            created=created,
            body=str(body),
        )
