import hashlib
import json
import re
import unicodedata
from typing import Any

from release_agent.importer.schema import DatasetComment, DatasetTicket

JIRA_ISSUE_TYPE_MAPPING = {
    "Bug": "Task",
    "Approval": "Task",
    "Story": "Task",
    "Test": "Task",
}


def source_marker(source_id: str) -> str:
    return f"Source ID: {source_id}"


def source_fingerprint(ticket: DatasetTicket) -> str:
    """Hash the validated source model, preserving existing Jira marker compatibility."""
    serialized = ticket.model_dump_json()
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_adf(value: Any) -> Any:
    if isinstance(value, str):
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        return unicodedata.normalize("NFC", normalized)
    if isinstance(value, list):
        return [_normalize_adf(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_adf(item)
            for key, item in sorted(value.items())
        }
    return value


def jira_content_fingerprint(
    fields: dict[str, Any], managed_labels: list[str] | set[str]
) -> str:
    issue_type = fields.get("issuetype")
    if isinstance(issue_type, dict):
        issue_type_name = issue_type.get("name")
    else:
        issue_type_name = issue_type

    content = {
        "summary": unicodedata.normalize("NFC", str(fields.get("summary", ""))),
        "description": _normalize_adf(fields.get("description")),
        "issuetype": issue_type_name,
        "labels": sorted(set(managed_labels)),
    }
    serialized = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def fingerprint_marker(fingerprint: str) -> str:
    return f"Import fingerprint: {fingerprint}"


def _contains_marker(description: object, marker: str) -> bool:
    def text_content(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return " ".join(text_content(child) for child in value.values())
        if isinstance(value, list):
            return " ".join(text_content(child) for child in value)
        return ""

    return re.search(rf"(?<!\S){re.escape(marker)}(?=\s|$)", text_content(description)) is not None


def has_source_marker(description: object, source_id: str) -> bool:
    return _contains_marker(description, source_marker(source_id))


def has_fingerprint(description: object, fingerprint: str) -> bool:
    return _contains_marker(description, fingerprint_marker(fingerprint))


def comment_marker(comment: DatasetComment) -> str:
    return f"[Imported comment ID: {comment.source_id}]"


def has_comment_marker(body: object, source_id: str) -> bool:
    return _contains_marker(body, f"[Imported comment ID: {source_id}]")


def _text_to_adf(text: str) -> dict[str, object]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    content: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if line:
            content.append({"type": "text", "text": line})
        if index < len(lines) - 1:
            content.append({"type": "hardBreak"})
    return {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": content}],
    }


def map_ticket_to_jira(
    ticket: DatasetTicket, project_key: str, import_label: str
) -> dict[str, object]:
    destination_issue_type = JIRA_ISSUE_TYPE_MAPPING.get(
        ticket.issue_type, ticket.issue_type
    )
    provenance = {
        "source_id": ticket.source_id,
        "original_issue_key": ticket.issue_key,
        "assignee": ticket.assignee,
        "reporter": ticket.reporter,
        "status": ticket.status,
        "created": ticket.created.isoformat(),
        "updated": ticket.updated.isoformat(),
        "links": [str(link) for link in ticket.links],
    }
    if destination_issue_type != ticket.issue_type:
        provenance["source_issue_type"] = ticket.issue_type
    description = (
        f"{ticket.description}\n\n--- Imported source metadata ---\n"
        f"{source_marker(ticket.source_id)}\n"
        f"{fingerprint_marker(source_fingerprint(ticket))}\n"
        f"{json.dumps(provenance, ensure_ascii=False, sort_keys=True)}"
    )
    labels = sorted(set(ticket.labels + [import_label]))
    return {
        "project": {"key": project_key},
        "summary": ticket.summary,
        "description": _text_to_adf(description),
        "issuetype": {"name": destination_issue_type},
        "labels": labels,
    }


def map_comment(comment: DatasetComment) -> dict[str, object]:
    metadata = {
        "source_comment_id": comment.source_id,
        "fictional_author": comment.author,
        "original_created": comment.created.isoformat(),
        "links": [str(link) for link in comment.links],
    }
    text = (
        f"{comment_marker(comment)}\n"
        f"[Original author: {comment.author}; created: {comment.created.isoformat()}]\n"
        f"{comment.body}"
        + (
            f"\nOriginal links: {', '.join(str(link) for link in comment.links)}"
            if comment.links
            else ""
        )
        + f"\n[Import metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}]"
    )
    return _text_to_adf(text)
