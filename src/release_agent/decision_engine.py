import logging
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from release_agent.config import Settings
from release_agent.models import (
    EvidenceBundle,
    OverallDecision,
    ReleaseManifest,
    ReleaseRequirement,
    RequirementAssessment,
    RequirementDecision,
)
from release_agent.utils import estimate_tokens, load_prompt, truncate_to_tokens

logger = logging.getLogger(__name__)


class LLMDecision(BaseModel):
    decision: RequirementDecision
    rationale: str = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class DecisionEngine:
    def __init__(self, settings: Settings, llm: Any | None = None) -> None:
        self.settings = settings
        if llm is not None:
            self.llm = llm
        else:
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is required to run an assessment")
            self.llm = ChatGroq(
                model=settings.groq_model,
                api_key=SecretStr(settings.groq_api_key),
                temperature=settings.groq_temperature,
                max_retries=settings.groq_max_retries,
            )
        self.parser = PydanticOutputParser(pydantic_object=LLMDecision)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("decision_review.txt")),
                ("human", load_prompt("decision_request.txt")),
            ]
        )
        self.chain = self.prompt | self.llm | self.parser

    def assess_requirement(
        self, requirement: ReleaseRequirement, evidence: EvidenceBundle
    ) -> RequirementAssessment:
        evidence_text = self._format_evidence(evidence)
        estimated = estimate_tokens(evidence_text)
        bounded_evidence = truncate_to_tokens(evidence_text, 10_000)
        output = self._invoke(
            {
                "requirement": requirement.model_dump_json(indent=2),
                "evidence": bounded_evidence,
                "token_count": estimated,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )
        return RequirementAssessment(
            requirement_id=requirement.id,
            title=requirement.title,
            decision=output.decision,
            rationale=output.rationale,
            evidence_issue_keys=[issue.key for issue in evidence.issues],
            evidence_summary=output.evidence_summary,
            confidence=output.confidence,
            mandatory=requirement.mandatory,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _invoke(self, values: dict[str, Any]) -> LLMDecision:
        return self.chain.invoke(values)

    @staticmethod
    def _format_evidence(evidence: EvidenceBundle) -> str:
        sections = [
            f"Collected at: {evidence.collected_at.isoformat()}",
            f"Collection errors: {evidence.collection_errors or 'none'}",
        ]
        if not evidence.issues:
            sections.append("No Jira issues were successfully collected.")
        for issue in evidence.issues:
            sections.append(
                f"\n[{issue.key}] {issue.summary}\n"
                f"Source ID: {issue.source_id}; status: {issue.status}; "
                f"type: {issue.issue_type}; updated: {issue.updated}\n"
                f"Description: {issue.description}\n"
                f"Links: {', '.join(str(link) for link in issue.links) or 'none'}"
            )
            for comment in issue.comments:
                sections.append(
                    f"Comment {comment.id} by {comment.author} at "
                    f"{comment.created.isoformat()}: {comment.body}"
                    + (
                        f" (links: {', '.join(str(link) for link in comment.links)})"
                        if comment.links
                        else ""
                    )
                )
        return "\n".join(sections)


def overall_decision(
    manifest: ReleaseManifest, assessments: list[RequirementAssessment]
) -> OverallDecision:
    by_id = {assessment.requirement_id: assessment for assessment in assessments}
    mandatory = [item for item in manifest.requirements if item.mandatory]
    mandatory_assessments = [by_id[item.id] for item in mandatory if item.id in by_id]
    if len(mandatory_assessments) != len(mandatory):
        return OverallDecision.NEEDS_REVIEW
    if any(item.decision == RequirementDecision.NOT_SATISFIED for item in mandatory_assessments):
        return OverallDecision.BLOCKED
    if all(item.decision == RequirementDecision.SATISFIED for item in mandatory_assessments):
        return OverallDecision.READY
    return OverallDecision.NEEDS_REVIEW
