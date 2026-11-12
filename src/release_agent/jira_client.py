import json
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlsplit

import requests
from atlassian import Jira

from release_agent.config import Settings

DEFAULT_PAGE_SIZE = 100
MAX_PAGES = 10_000
ISSUE_FIELDS = ["summary", "description", "labels", "status", "issuetype"]
ADF_BLOCK_TYPES = {
    "blockquote",
    "codeBlock",
    "heading",
    "listItem",
    "orderedList",
    "paragraph",
    "panel",
    "bulletList",
}


class JiraClientError(RuntimeError):
    """Base error for safe, user-facing Jira client failures."""


class JiraAuthenticationError(JiraClientError):
    """Jira rejected the configured credentials."""


class JiraPermissionError(JiraClientError):
    """The configured Jira account lacks permission for the operation."""


class JiraNotFoundError(JiraClientError):
    """The requested Jira resource does not exist or is inaccessible."""


class JiraRateLimitError(JiraClientError):
    """Jira rejected the request due to rate limiting."""


class JiraServiceError(JiraClientError):
    """Jira returned a server-side failure."""


class JiraQueryError(JiraClientError):
    """Jira rejected an issue search query."""


class JiraPayloadError(JiraClientError):
    """A local or Jira validation error occurred for a write payload."""


class JiraNetworkError(JiraClientError):
    """A Jira request failed because of a timeout or network error."""


class JiraPaginationError(JiraClientError):
    """Jira returned invalid or incomplete pagination metadata."""


class JiraResponseError(JiraClientError):
    """Jira returned a response that does not match the expected structure."""


class DuplicateSourceIdError(JiraClientError):
    """More than one Jira issue has the same imported source ID."""


class JiraClient:
    def __init__(self, settings: Settings, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        if not settings.jira_configured:
            raise JiraAuthenticationError(
                "Jira is not configured. Set JIRA_URL, JIRA_USERNAME, and JIRA_API_TOKEN."
            )
        if page_size < 1:
            raise ValueError("Jira page size must be a positive integer.")

        base_url = (settings.jira_url or "").strip().rstrip("/")
        parsed_url = urlsplit(base_url)
        if (
            parsed_url.scheme.lower() != "https"
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise ValueError("JIRA_URL must be an HTTPS URL.")
        if not settings.jira_username or not settings.jira_username.strip():
            raise JiraAuthenticationError("JIRA_USERNAME must not be empty.")
        if not settings.jira_api_token or not settings.jira_api_token.strip():
            raise JiraAuthenticationError("JIRA_API_TOKEN must not be empty.")

        self._project_key = settings.jira_project_key.strip()
        if not self._project_key:
            raise ValueError("JIRA_PROJECT_KEY must not be empty.")
        self._page_size = page_size
        self._jira = Jira(
            url=base_url,
            username=settings.jira_username,
            password=settings.jira_api_token,
            api_version="3",
            cloud=True,
            timeout=settings.jira_timeout_seconds,
            retry_with_header=False,
        )

    def find_imported_issues(self, project_key: str, label: str) -> list[dict[str, Any]]:
        if project_key.strip().casefold() != self._project_key.casefold():
            raise ValueError("Issue search project must match configured JIRA_PROJECT_KEY.")
        if not label.strip():
            raise ValueError("Import label must not be empty.")
        project = self._escape_jql(project_key)
        escaped_label = self._escape_jql(label)
        query = f'project = "{project}" AND labels = "{escaped_label}" ORDER BY key'
        results: list[dict[str, Any]] = []
        token: str | None = None
        seen_tokens: set[str] = set()

        page_count = 0
        while True:
            page_count += 1
            if page_count > MAX_PAGES:
                raise JiraPaginationError("Jira issue search exceeded the page safety limit.")
            page = self._execute(
                "issue search",
                lambda token=token: self._jira.enhanced_jql(
                    query,
                    fields=ISSUE_FIELDS,
                    nextPageToken=token,
                    limit=self._page_size,
                ),
                query=True,
            )
            if not isinstance(page, dict):
                raise JiraResponseError("Jira issue search returned an invalid response.")
            issues = page.get("issues")
            if not isinstance(issues, list):
                raise JiraResponseError("Jira issue search response has no valid issues list.")
            for issue in issues:
                if (
                    not isinstance(issue, dict)
                    or not isinstance(issue.get("key"), str)
                    or not issue["key"]
                    or not isinstance(issue.get("fields"), dict)
                ):
                    raise JiraResponseError(
                        "Jira issue search returned an issue without its key or fields."
                    )
            results.extend(issues)

            is_last = page.get("isLast")
            if is_last is not None and not isinstance(is_last, bool):
                raise JiraPaginationError("Jira issue search returned invalid isLast metadata.")
            next_token = page.get("nextPageToken")
            if next_token is not None and (not isinstance(next_token, str) or not next_token):
                raise JiraPaginationError("Jira issue search returned an invalid page token.")
            if is_last is True:
                return results
            if next_token is None:
                if is_last is False:
                    raise JiraPaginationError(
                        "Jira issue search marked a page incomplete without a next-page token."
                    )
                return results
            if not issues:
                raise JiraPaginationError(
                    "Jira issue search returned an empty page with a next-page token."
                )
            if next_token in seen_tokens:
                raise JiraPaginationError("Jira issue search repeated a page token.")
            seen_tokens.add(next_token)
            token = next_token

    def find_project_issues(self, project_key: str) -> list[dict[str, Any]]:
        if project_key.strip().casefold() != self._project_key.casefold():
            raise ValueError("Issue search project must match configured JIRA_PROJECT_KEY.")
        project = self._escape_jql(project_key)
        query = f'project = "{project}" ORDER BY key'
        results: list[dict[str, Any]] = []
        token: str | None = None
        seen_tokens: set[str] = set()

        for _ in range(MAX_PAGES):
            page = self._execute(
                "project issue search",
                lambda token=token: self._jira.enhanced_jql(
                    query,
                    fields=ISSUE_FIELDS,
                    nextPageToken=token,
                    limit=self._page_size,
                ),
                query=True,
            )
            if not isinstance(page, dict):
                raise JiraResponseError("Jira project search returned an invalid response.")
            issues = page.get("issues")
            if not isinstance(issues, list):
                raise JiraResponseError("Jira project search has no valid issues list.")
            for issue in issues:
                if (
                    not isinstance(issue, dict)
                    or not isinstance(issue.get("key"), str)
                    or not issue["key"]
                    or not isinstance(issue.get("fields"), dict)
                ):
                    raise JiraResponseError(
                        "Jira project search returned an issue without its key or fields."
                    )
            results.extend(issues)

            is_last = page.get("isLast")
            if is_last is not None and not isinstance(is_last, bool):
                raise JiraPaginationError("Jira project search returned invalid isLast metadata.")
            next_token = page.get("nextPageToken")
            if next_token is not None and (
                not isinstance(next_token, str) or not next_token
            ):
                raise JiraPaginationError("Jira project search returned an invalid page token.")
            if is_last is True:
                return results
            if next_token is None:
                if is_last is False:
                    raise JiraPaginationError(
                        "Jira project search marked a page incomplete without a next-page token."
                    )
                return results
            if not issues:
                raise JiraPaginationError(
                    "Jira project search returned an empty page with a next-page token."
                )
            if next_token in seen_tokens:
                raise JiraPaginationError("Jira project search repeated a page token.")
            seen_tokens.add(next_token)
            token = next_token

        raise JiraPaginationError("Jira project search exceeded the page safety limit.")

    def find_imported_issue_keys_by_source_id(
        self, project_key: str, label: str
    ) -> dict[str, str]:
        source_keys: dict[str, str] = {}
        for issue in self.find_imported_issues(project_key, label):
            fields = issue.get("fields") or {}
            source_id = self._source_id(fields.get("description"))
            if not source_id:
                continue
            key = issue["key"]
            if source_id in source_keys:
                raise DuplicateSourceIdError(
                    "Multiple Jira issues contain the same imported source ID."
                )
            source_keys[source_id] = str(key)
        return source_keys

    def create_issue(self, fields: dict[str, object]) -> str:
        self._validate_issue_fields(fields)
        result = self._execute(
            "issue creation",
            lambda: self._jira.issue_create(fields),
            payload=True,
        )
        if not isinstance(result, dict) or not isinstance(result.get("key"), str):
            raise JiraResponseError("Jira issue creation returned no issue key.")
        return result["key"]

    def update_issue(
        self,
        issue_key: str,
        fields: dict[str, object],
        update: dict[str, object] | None = None,
    ) -> None:
        if not issue_key.strip():
            raise ValueError("Jira issue key must not be empty.")
        self._validate_issue_fields(fields, creation=False)
        if update is not None and not self._valid_issue_update(update):
            raise JiraPayloadError("Jira issue update operations are invalid.")
        self._execute(
            "issue update",
            lambda: self._jira.issue_update(issue_key, fields, update=update),
            payload=True,
        )

    def get_comments(self, issue_key: str) -> list[dict[str, Any]]:
        if not issue_key.strip():
            raise ValueError("Jira issue key must not be empty.")
        url = f"{self._jira.resource_url('issue')}/{issue_key}/comment"
        comments: list[dict[str, Any]] = []
        start_at = 0
        observed_total: int | None = None

        page_count = 0
        while True:
            page_count += 1
            if page_count > MAX_PAGES:
                raise JiraPaginationError("Jira comment retrieval exceeded the page safety limit.")
            page = self._execute(
                "comment retrieval",
                lambda start_at=start_at: self._jira.get(
                    url,
                    params={"startAt": start_at, "maxResults": self._page_size},
                ),
            )
            if not isinstance(page, dict):
                raise JiraResponseError("Jira comment retrieval returned an invalid response.")
            page_comments = page.get("comments")
            if not isinstance(page_comments, list):
                raise JiraResponseError(
                    "Jira comment response has no valid comments list."
                )
            for comment in page_comments:
                if (
                    not isinstance(comment, dict)
                    or not isinstance(comment.get("id"), (str, int))
                    or "body" not in comment
                ):
                    raise JiraResponseError(
                        "Jira comment response contains a comment without an ID or body."
                    )

            page_start = page.get("startAt")
            if page_start is not None and page_start != start_at:
                raise JiraPaginationError("Jira comment response has an unexpected startAt.")
            total = page.get("total")
            if total is not None:
                if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                    raise JiraPaginationError("Jira comment response has invalid total metadata.")
                if observed_total is not None and total != observed_total:
                    raise JiraPaginationError("Jira comment total changed during pagination.")
                observed_total = total

            comments.extend(page_comments)
            start_at += len(page_comments)
            if observed_total is not None:
                if start_at >= observed_total:
                    return comments
                if not page_comments:
                    raise JiraPaginationError(
                        "Jira returned an empty comment page before the reported total."
                    )
                continue

            is_last = page.get("isLast")
            if is_last is not None and not isinstance(is_last, bool):
                raise JiraPaginationError("Jira comment response has invalid isLast metadata.")
            if is_last is True:
                return comments
            if is_last is False:
                if not page_comments:
                    raise JiraPaginationError(
                        "Jira returned an empty comment page marked as incomplete."
                    )
                continue
            if len(page_comments) < self._page_size:
                return comments

    def add_comment(self, issue_key: str, body: dict[str, object]) -> dict[str, Any] | None:
        if not issue_key.strip():
            raise ValueError("Jira issue key must not be empty.")
        self._validate_adf(body, "comment body")
        url = f"{self._jira.resource_url('issue')}/{issue_key}/comment"
        result = self._execute(
            "comment creation",
            lambda: self._jira.post(url, data={"body": body}),
            payload=True,
        )
        if result is None:
            return None
        if not isinstance(result, dict):
            raise JiraResponseError("Jira comment creation returned an invalid response.")
        return result

    def get_issue(self, issue_key: str) -> dict[str, Any]:
        if not issue_key.strip():
            raise ValueError("Jira issue key must not be empty.")
        result = self._execute(
            "issue retrieval",
            lambda: self._jira.issue(
                issue_key,
                fields="summary,description,status,issuetype,labels,assignee,reporter,created,updated",
            ),
        )
        if not isinstance(result, dict) or not isinstance(result.get("fields"), dict):
            raise JiraResponseError("Jira issue retrieval returned an invalid response.")
        return result

    def get_issue_property(self, issue_key: str, property_key: str) -> dict[str, Any] | None:
        if not issue_key.strip() or not property_key.strip():
            raise ValueError("Jira issue key and property key must not be empty.")
        url = (
            f"{self._jira.resource_url('issue')}/{quote(issue_key, safe='')}"
            f"/properties/{quote(property_key, safe='')}"
        )
        try:
            result = self._execute(
                "issue property retrieval",
                lambda: self._jira.get(url),
            )
        except JiraNotFoundError:
            return None
        if result is None:
            return None
        if (
            not isinstance(result, dict)
            or result.get("key") != property_key
            or "value" not in result
        ):
            raise JiraResponseError("Jira issue property response is invalid.")
        return result

    def set_issue_property(
        self, issue_key: str, property_key: str, value: dict[str, Any]
    ) -> None:
        if not issue_key.strip() or not property_key.strip():
            raise ValueError("Jira issue key and property key must not be empty.")
        try:
            serialized = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise JiraPayloadError("Jira issue property must be valid JSON.") from exc
        if not serialized or len(serialized.encode("utf-8")) > 32_768:
            raise JiraPayloadError(
                "Jira issue property must be non-empty and no larger than 32768 bytes."
            )
        url = (
            f"{self._jira.resource_url('issue')}/{quote(issue_key, safe='')}"
            f"/properties/{quote(property_key, safe='')}"
        )
        self._execute(
            "issue property update",
            lambda: self._jira.put(url, data=value),
            payload=True,
        )

    def get_issue_comments(self, issue_key: str) -> list[dict[str, Any]]:
        return self.get_comments(issue_key)

    @classmethod
    def _description_text(cls, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            node_type = value.get("type")
            if node_type == "text":
                text = value.get("text")
                return text if isinstance(text, str) else ""
            if node_type == "hardBreak":
                return "\n"
            if "content" in value:
                text = cls._description_text(value["content"])
                return f"{text}\n" if node_type in ADF_BLOCK_TYPES else text
            attrs = value.get("attrs")
            if node_type in {"mention", "emoji"} and isinstance(attrs, dict):
                return str(attrs.get("text", ""))
            return ""
        if isinstance(value, list):
            return "".join(cls._description_text(item) for item in value)
        return ""

    @classmethod
    def _source_id(cls, description: Any) -> str | None:
        text = cls._description_text(description)
        match = re.search(r"(?m)^\s*Source ID:\s*([^\s\r\n]+)\s*$", text)
        return match.group(1) if match else None

    @staticmethod
    def _escape_jql(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _validate_adf(value: object, field_name: str) -> None:
        if (
            not isinstance(value, dict)
            or value.get("type") != "doc"
            or value.get("version") != 1
            or not isinstance(value.get("content"), list)
        ):
            raise JiraPayloadError(f"{field_name} must be a Jira Cloud ADF document.")

        def validate_node(node: object) -> bool:
            if not isinstance(node, dict) or not isinstance(node.get("type"), str):
                return False
            if node["type"] == "text" and not isinstance(node.get("text"), str):
                return False
            if "content" in node:
                children = node["content"]
                if not isinstance(children, list) or not all(
                    validate_node(child) for child in children
                ):
                    return False
            return True

        if not all(validate_node(node) for node in value["content"]):
            raise JiraPayloadError(f"{field_name} contains an invalid ADF node.")

    @classmethod
    def _validate_issue_fields(
        cls, fields: dict[str, object], *, creation: bool = True
    ) -> None:
        if not isinstance(fields, dict):
            raise JiraPayloadError("Jira issue fields must be an object.")
        if creation:
            for name in ("project", "summary", "issuetype"):
                if name not in fields:
                    raise JiraPayloadError(f"Jira issue creation requires '{name}'.")
        if "description" in fields and fields["description"] is not None:
            cls._validate_adf(fields["description"], "issue description")

    @staticmethod
    def _valid_issue_update(update: dict[str, object]) -> bool:
        if not isinstance(update, dict) or set(update) != {"labels"}:
            return False
        operations = update["labels"]
        if not isinstance(operations, list):
            return False
        for operation in operations:
            if (
                not isinstance(operation, dict)
                or len(operation) != 1
                or next(iter(operation), None) not in {"add", "remove"}
                or not isinstance(next(iter(operation.values()), None), str)
            ):
                return False
        return True

    @staticmethod
    def _http_error_type(
        status_code: int, *, query: bool, payload: bool
    ) -> type[JiraClientError]:
        if status_code == 401:
            return JiraAuthenticationError
        if status_code == 403:
            return JiraPermissionError
        if status_code == 404:
            return JiraNotFoundError
        if status_code == 429:
            return JiraRateLimitError
        if status_code >= 500:
            return JiraServiceError
        if status_code == 400 and query:
            return JiraQueryError
        if status_code == 400 and payload:
            return JiraPayloadError
        return JiraClientError

    @classmethod
    def _execute(
        cls,
        operation: str,
        request: Callable[[], Any],
        *,
        query: bool = False,
        payload: bool = False,
    ) -> Any:
        try:
            return request()
        except requests.Timeout as exc:
            raise JiraNetworkError(f"Jira {operation} timed out.") from exc
        except requests.ConnectionError as exc:
            raise JiraNetworkError(f"Jira {operation} failed due to a network error.") from exc
        except requests.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else 0
            error_type = cls._http_error_type(
                status_code, query=query, payload=payload
            )
            raise error_type(
                f"Jira {operation} failed with HTTP {status_code}."
            ) from exc
        except requests.RequestException as exc:
            raise JiraClientError(f"Jira {operation} request failed.") from exc
        except JiraClientError:
            raise
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int):
                error_type = cls._http_error_type(
                    status_code, query=query, payload=payload
                )
                raise error_type(
                    f"Jira {operation} failed with HTTP {status_code}."
                ) from exc
            raise JiraClientError(f"Jira {operation} failed.") from exc
