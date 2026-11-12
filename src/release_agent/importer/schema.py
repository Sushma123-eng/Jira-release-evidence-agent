from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


def _require_non_empty(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty")
    return value


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value


class DatasetComment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    author: str
    created: datetime
    body: str
    links: list[HttpUrl] = Field(default_factory=list)

    @field_validator("source_id", "author", "body")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("created")
    @classmethod
    def validate_created_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class DatasetTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    issue_key: str
    summary: str
    description: str
    issue_type: str = "Task"
    status: str = "To Do"
    assignee: str | None = None
    reporter: str | None = None
    created: datetime
    updated: datetime
    labels: list[str] = Field(default_factory=list)
    links: list[HttpUrl] = Field(default_factory=list)
    comments: list[DatasetComment] = Field(default_factory=list)

    @field_validator(
        "source_id", "issue_key", "summary", "description", "issue_type", "status"
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("created", "updated")
    @classmethod
    def validate_timestamp_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class DatasetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_key: str
    tickets: list[DatasetTicket] = Field(min_length=1)

    @field_validator("project_key")
    @classmethod
    def validate_project_key(cls, value: str) -> str:
        return _require_non_empty(value)

    @model_validator(mode="after")
    def validate_unique_source_ids(self) -> "DatasetFile":
        source_ids = [ticket.source_id for ticket in self.tickets]
        issue_keys = [ticket.issue_key for ticket in self.tickets]
        comment_ids = [
            comment.source_id
            for ticket in self.tickets
            for comment in ticket.comments
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Ticket source_id values must be unique")
        if len(issue_keys) != len(set(issue_keys)):
            raise ValueError("Ticket issue_key values must be unique")
        if len(comment_ids) != len(set(comment_ids)):
            raise ValueError("Comment source_id values must be unique")
        return self
