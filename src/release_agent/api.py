import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from pydantic import BaseModel, SecretStr
from starlette.concurrency import run_in_threadpool

from release_agent.agent import NoToolArguments, ToolCallingAgent
from release_agent.config import PROJECT_ROOT, Settings, get_settings
from release_agent.decision_engine import DecisionEngine
from release_agent.evidence_collector import EvidenceCollector
from release_agent.importer.importer import DatasetImporter
from release_agent.jira_client import JiraClient
from release_agent.models import (
    AgentCallbackRequest,
    AgentCallbackResponse,
    AssessmentReport,
    CallbackDecision,
    EvidenceBundle,
    OverallDecision,
    ReleaseManifest,
    ReleaseRequirement,
    RequirementAssessment,
    RequirementDecision,
    TraceEvent,
)
from release_agent.report_generator import AssessmentRunner
from release_agent.utils import load_prompt, utc_now, write_json_atomic

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Jira Release Evidence Agent",
    description="Route natural-language release questions to evidence tools.",
    version="1.0.0",
)


class RequirementToolInput(BaseModel):
    requirement_id: str


def _decision_text(decision: RequirementDecision | str) -> str:
    value = decision.value if isinstance(decision, RequirementDecision) else decision
    return {
        "SATISFIED": "PASS",
        "NOT_SATISFIED": "FAIL",
        "PARTIAL": "INSUFFICIENT_EVIDENCE",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
    }.get(value, value)


def _overall_response_decision(decision: OverallDecision) -> CallbackDecision:
    decisions: dict[OverallDecision, CallbackDecision] = {
        OverallDecision.READY: "PASS",
        OverallDecision.BLOCKED: "FAIL",
        OverallDecision.NEEDS_REVIEW: "INSUFFICIENT_EVIDENCE",
    }
    return decisions[decision]


def _decision_phrase(decision: str) -> str:
    return {
        "PASS": "passes",
        "FAIL": "fails",
        "INSUFFICIENT_EVIDENCE": "does not have enough evidence for a decision",
    }.get(decision, decision.lower())


def _fit_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(".,;:") + "."


def _assessment_summary(
    report: AssessmentReport | None,
    assessments: list[RequirementAssessment],
    evidence: list[EvidenceBundle],
    decision: str,
) -> str:
    if report is not None:
        release_info = f"{report.release_name} ({report.release_id}, version {report.version})"
        items = report.requirements
    else:
        release_info = "Requirement assessment"
        items = assessments

    evidence_by_id = {bundle.requirement_id: bundle for bundle in evidence}
    issue_count = sum(len(bundle.issues) for bundle in evidence)
    comment_count = sum(
        len(issue.comments) for bundle in evidence for issue in bundle.issues
    )
    error_messages = [
        _fit_words(" ".join(error.split()), 18)
        for bundle in evidence
        for error in bundle.collection_errors
    ]

    def compose(detail_limit: int) -> str:
        parts = [
            f"{release_info} {_decision_phrase(decision)}.",
            (
                f"The assessment reviewed {len(items)} requirement(s), "
                f"{issue_count} Jira issue(s), and {comment_count} comment(s)."
            ),
        ]
        for item in items:
            bundle = evidence_by_id.get(item.requirement_id)
            keys = (
                ", ".join(issue.key for issue in bundle.issues)
                if bundle and bundle.issues
                else ", ".join(item.evidence_issue_keys) or "no Jira issues retrieved"
            )
            finding = " ".join(item.evidence_summary.split()[:detail_limit])
            explanation = " ".join(item.rationale.split()[:detail_limit])
            parts.append(
                f"{item.requirement_id}, {item.title}: "
                f"{_decision_phrase(_decision_text(item.decision))}. "
                f"Jira records: {keys}. "
                f"Evidence: {finding} Explanation: {explanation}"
            )
        if error_messages:
            parts.append(
                "Evidence retrieval reported: "
                + "; ".join(error_messages)
                + ". These errors may have left relevant records unavailable."
            )
        else:
            parts.append("No Jira evidence retrieval errors were reported.")
        if decision == "PASS":
            parts.append(
                "The cited records support the stated acceptance conditions and "
                "were considered for relevance to this release. The decision is "
                "based on evidence, not Jira status alone. External links were "
                "not independently verified. This summary is decision support; "
                "the release owner should confirm that the records remain current "
                "and apply to the intended build, service, and environment."
            )
        else:
            parts.append(
                "The available evidence does not establish every required "
                "condition. Review the cited records, resolve the gaps or retrieval "
                "errors described above, and provide current evidence before "
                "reconsidering release readiness. A Jira status or planned activity "
                "does not by itself prove completion."
            )
        return " ".join(parts)

    summary = compose(12)
    if len(summary.split()) > 150:
        summary = compose(5)
    if len(summary.split()) > 150:
        summary = _fit_words(summary, 150)
    if len(summary.split()) < 100:
        summary = (
            summary
            + " Confirm that each cited item addresses the acceptance criteria "
            "for the correct release and that its content is complete, current, "
            "and attributable. Where evidence is missing, obtain a specific "
            "record or confirmation rather than relying on assumptions."
        )
    if len(summary.split()) < 100:
        summary += (
            " Record the owner, completion date, and source for each follow-up. "
            "Then rerun the assessment against the updated Jira evidence."
        )
    if decision == "PASS":
        summary = _fit_words(summary, 150)
        filler = (
            "Each requirement was reviewed against its stated purpose. The cited "
            "Jira records explain the basis for the result, while status labels "
            "and unverified external references were not treated as proof. "
            "The release owner remains responsible for confirming applicability "
            "to the final release candidate. Any material change should trigger "
            "a fresh evidence review before approval."
        )
        if len(summary.split()) < 150:
            summary = _fit_words(f"{summary} {filler}", 150)
    elif len(summary.split()) > 150:
        summary = _fit_words(summary, 150)
    if not 100 <= len(summary.split()) <= 150:
        raise ValueError("Assessment summary must contain between 100 and 150 words.")
    if decision == "PASS" and len(summary.split()) != 150:
        raise ValueError("A passing assessment summary must contain exactly 150 words.")
    return summary


def _resolve_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Configured paths must stay inside the project directory.") from exc
    return resolved


def _make_llm(settings: Settings) -> ChatGroq:
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="Agent operations require GROQ_API_KEY.",
        )
    return ChatGroq(
        model=settings.groq_model,
        api_key=SecretStr(settings.groq_api_key),
        temperature=settings.groq_temperature,
        max_retries=settings.groq_max_retries,
    )


@app.get("/health")
def health() -> dict[str, str | bool]:
    settings = get_settings()
    return {
        "status": "ok",
        "jira_configured": settings.jira_configured,
        "llm_configured": bool(settings.groq_api_key),
    }


@app.post("/agent_callback", response_model=AgentCallbackResponse)
async def agent_callback(
    request: AgentCallbackRequest,
) -> AgentCallbackResponse:
    settings = get_settings()
    try:
        return await run_in_threadpool(_execute_callback, request, settings)
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        logger.exception("Agent callback input failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Agent callback failed")
        raise HTTPException(
            status_code=502,
            detail=f"Agent callback failed: {exc}",
        ) from exc


def _execute_callback(
    request: AgentCallbackRequest,
    settings: Settings,
) -> AgentCallbackResponse:
    llm = _make_llm(settings)
    manifest_path = _resolve_path(settings.resolved_manifest_path)
    dataset_path = _resolve_path(settings.resolved_dataset_path)
    try:
        manifest = ReleaseManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load release manifest: {exc}") from exc

    jira_client = JiraClient(settings) if settings.jira_configured else None
    collector = EvidenceCollector(
        dataset_path=dataset_path,
        jira_client=jira_client,
        import_label=settings.import_label,
    )
    decision_engine = DecisionEngine(settings, llm=llm)
    runner = AssessmentRunner(
        settings=settings,
        collector=collector,
        decision_engine=decision_engine,
    )
    agent = ToolCallingAgent(llm=llm)
    requirement_by_id = {item.id: item for item in manifest.requirements}
    evidence_by_id: dict[str, EvidenceBundle] = {}
    assessments_by_id: dict[str, RequirementAssessment] = {}
    state: dict[str, Any] = {}
    trace: list[TraceEvent] = []
    tool_descriptions = json.loads(load_prompt("tool_descriptions.json"))

    def record(
        stage: str,
        message: str,
        requirement_id: str = "QUERY",
        **details: Any,
    ) -> None:
        trace.append(
            TraceEvent(
                sequence=len(trace) + 1,
                timestamp=utc_now(),
                requirement_id=requirement_id,
                stage=stage,
                message=message,
                details=details,
            )
        )
        write_json_atomic(
            settings.resolved_output_dir / "latest_investigation_trace.json",
            [event.model_dump(mode="json") for event in trace],
        )

    def requirement_for(requirement_id: str) -> ReleaseRequirement:
        requirement = requirement_by_id.get(requirement_id)
        if requirement is None:
            raise ValueError(
                f"Unknown requirement {requirement_id!r}. Use an ID from the release manifest."
            )
        return requirement

    def collect_for(requirement_id: str) -> EvidenceBundle:
        requirement = requirement_for(requirement_id)
        bundle = collector.collect(requirement)
        evidence_by_id[requirement_id] = bundle
        record(
            "evidence_collected",
            f"Collected {len(bundle.issues)} Jira issue(s)",
            requirement_id=requirement_id,
            issue_keys=[issue.key for issue in bundle.issues],
            errors=bundle.collection_errors,
        )
        return bundle

    def get_evidence(requirement_id: str) -> str:
        return collect_for(requirement_id).model_dump_json()

    def assess_one(requirement_id: str) -> str:
        requirement = requirement_for(requirement_id)
        bundle = evidence_by_id.get(requirement_id) or collect_for(requirement_id)
        assessment = decision_engine.assess_requirement(requirement, bundle)
        assessments_by_id[requirement_id] = assessment
        return json.dumps(
            {
                "evidence": bundle.model_dump(mode="json"),
                "assessment": assessment.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )

    def assess_all() -> str:
        def processor(
            requirement: ReleaseRequirement,
            runner_record: Any,
        ) -> tuple[EvidenceBundle, RequirementAssessment]:
            runner_record(
                "tool_processing_started",
                "Sequential evidence and decision tools started",
            )
            bundle = collector.collect(requirement)
            evidence_by_id[requirement.id] = bundle
            assessment = decision_engine.assess_requirement(requirement, bundle)
            assessments_by_id[requirement.id] = assessment
            return bundle, assessment

        report = runner.assess(manifest, requirement_processor=processor)
        for event in report.trace:
            record(
                event.stage,
                event.message,
                requirement_id=event.requirement_id,
                **event.details,
            )
        state["report"] = report
        return report.model_dump_json()

    def import_dataset_preview() -> str:
        dataset = DatasetImporter.load_dataset(dataset_path)
        importer = DatasetImporter(
            jira_client=None,
            project_key=dataset.project_key,
            import_label=settings.import_label,
        )
        result = importer.run(dataset_path, dry_run=True)
        return result.model_dump_json()

    def list_requirements() -> str:
        return json.dumps(
            [
                {
                    "id": item.id,
                    "title": item.title,
                    "description": item.description,
                    "mandatory": item.mandatory,
                    "evidence_issue_keys": item.evidence_issue_keys,
                }
                for item in manifest.requirements
            ],
            ensure_ascii=False,
        )

    tools = [
        StructuredTool.from_function(
            func=list_requirements,
            name="list_release_requirements",
            description=tool_descriptions["list_release_requirements"],
            args_schema=NoToolArguments,
        ),
        StructuredTool.from_function(
            func=get_evidence,
            name="get_evidence",
            description=tool_descriptions["get_evidence"],
            args_schema=RequirementToolInput,
        ),
        StructuredTool.from_function(
            func=assess_one,
            name="assess_requirement",
            description=tool_descriptions["assess_requirement"],
            args_schema=RequirementToolInput,
        ),
        StructuredTool.from_function(
            func=assess_all,
            name="assess_release",
            description=tool_descriptions["assess_release"],
            args_schema=NoToolArguments,
        ),
        StructuredTool.from_function(
            func=import_dataset_preview,
            name="import_dataset",
            description=tool_descriptions["import_dataset"],
            args_schema=NoToolArguments,
        ),
    ]

    requirement_catalog = json.dumps(
        [
            {
                "id": requirement.id,
                "title": requirement.title,
                "mandatory": requirement.mandatory,
            }
            for requirement in manifest.requirements
        ],
        ensure_ascii=False,
    )
    answer = agent.route_query(
        request.query,
        tools,
        record,
        requirement_catalog=requirement_catalog,
    )
    report = state.get("report")
    response_decision: CallbackDecision | None = None
    if isinstance(report, AssessmentReport):
        response_decision = _overall_response_decision(
            report.overall_decision
        )
        items = report.requirements
    elif assessments_by_id:
        assessed = list(assessments_by_id.values())
        decisions = {item.decision for item in assessed}
        response_decision = (
            "FAIL"
            if RequirementDecision.NOT_SATISFIED in decisions
            else "INSUFFICIENT_EVIDENCE"
            if decisions.intersection(
                {
                    RequirementDecision.PARTIAL,
                    RequirementDecision.INSUFFICIENT_EVIDENCE,
                }
            )
            else "PASS"
        )
        items = assessed
    else:
        items = []
    if response_decision is not None:
        answer = _assessment_summary(
            report if isinstance(report, AssessmentReport) else None,
            items,
            list(evidence_by_id.values()),
            response_decision,
        )
    return AgentCallbackResponse(
        answer=answer,
        decision=response_decision,
    )
