import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from release_agent.config import Settings
from release_agent.decision_engine import DecisionEngine, overall_decision
from release_agent.evidence_collector import EvidenceCollector
from release_agent.models import (
    AssessmentReport,
    EvidenceBundle,
    ReleaseManifest,
    ReleaseRequirement,
    RequirementAssessment,
    TraceEvent,
)
from release_agent.utils import utc_now, write_json_atomic

logger = logging.getLogger(__name__)
RequirementProcessor = Callable[
    [ReleaseRequirement, Callable[..., None]],
    tuple[EvidenceBundle, RequirementAssessment],
]


class AssessmentRunner:
    def __init__(
        self,
        settings: Settings,
        collector: EvidenceCollector,
        decision_engine: DecisionEngine,
    ) -> None:
        self.settings = settings
        self.collector = collector
        self.decision_engine = decision_engine

    def assess(
        self,
        manifest: ReleaseManifest,
        requirement_processor: RequirementProcessor | None = None,
    ) -> AssessmentReport:
        trace: list[TraceEvent] = []
        assessments: list[RequirementAssessment] = []

        def record(requirement_id: str, stage: str, message: str, **details: object) -> None:
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
                self.settings.resolved_output_dir / "latest_investigation_trace.json",
                [event.model_dump(mode="json") for event in trace],
            )

        for position, requirement in enumerate(manifest.requirements, start=1):
            logger.info(
                "Assessing requirement %s (%d/%d)",
                requirement.id,
                position,
                len(manifest.requirements),
            )
            record(
                requirement.id,
                "started",
                "Sequential assessment started",
                position=position,
                total=len(manifest.requirements),
                mandatory=requirement.mandatory,
            )
            try:
                if requirement_processor is None:
                    evidence = self.collector.collect(requirement)
                    record(
                        requirement.id,
                        "evidence_collected",
                        f"Collected {len(evidence.issues)} Jira issue(s)",
                        issue_keys=[issue.key for issue in evidence.issues],
                        errors=evidence.collection_errors,
                    )
                    assessment = self.decision_engine.assess_requirement(
                        requirement, evidence
                    )
                else:
                    evidence, assessment = requirement_processor(
                        requirement,
                        lambda stage, message, requirement_id=requirement.id, **details: record(
                            requirement_id, stage, message, **details
                        ),
                    )
                if evidence.requirement_id != requirement.id:
                    raise ValueError("Evidence processor returned the wrong requirement")
                if assessment.requirement_id != requirement.id:
                    raise ValueError("Requirement processor returned the wrong assessment")
            except Exception as exc:
                record(
                    requirement.id,
                    "requirement_processing_failed",
                    "Sequential requirement processing failed before a result was recorded",
                    error_type=type(exc).__name__,
                )
                raise
            assessments.append(assessment)
            record(
                requirement.id,
                "decision_recorded",
                f"Requirement decision: {assessment.decision.value}",
                confidence=assessment.confidence,
                evidence_issue_keys=assessment.evidence_issue_keys,
            )

        decision = overall_decision(manifest, assessments)
        summary = self._summary(decision.value, assessments)
        report = AssessmentReport(
            release_id=manifest.release_id,
            release_name=manifest.release_name,
            version=manifest.version,
            overall_decision=decision,
            summary=summary,
            assessed_at=datetime.now(UTC),
            requirements=assessments,
            trace=trace,
            model=self.settings.groq_model,
        )
        self._save(report)
        return report

    def _save(self, report: AssessmentReport) -> Path:
        path = self.settings.resolved_output_dir / "latest_assessment.json"
        write_json_atomic(path, report.model_dump(mode="json"))
        logger.info("Saved assessment report and trace to %s", path)
        return path

    @staticmethod
    def _summary(decision: str, assessments: list[RequirementAssessment]) -> str:
        unresolved = [
            f"{item.requirement_id} ({item.decision.value})"
            for item in assessments
            if item.mandatory and item.decision.value != "SATISFIED"
        ]
        if decision == "READY":
            return "All mandatory release requirements are supported by sufficient evidence."
        if decision == "BLOCKED":
            return "At least one mandatory release requirement is explicitly not satisfied: " + ", ".join(unresolved)
        return "Mandatory release requirements need additional evidence or review: " + ", ".join(unresolved)
