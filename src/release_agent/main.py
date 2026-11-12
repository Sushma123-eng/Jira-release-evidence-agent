import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from release_agent.config import get_settings
from release_agent.decision_engine import DecisionEngine
from release_agent.evidence_collector import EvidenceCollector
from release_agent.importer.importer import DatasetImporter
from release_agent.jira_client import JiraClient
from release_agent.models import ReleaseManifest
from release_agent.report_generator import AssessmentRunner
from release_agent.utils import configure_logging

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="release-agent",
        description="Import Jira evidence data or assess release readiness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser("import", help="Import the sample dataset into Jira")
    importer.add_argument("--dataset", type=Path, help="Dataset JSON path")
    dry_run_options = importer.add_mutually_exclusive_group()
    dry_run_options.add_argument(
        "--dry-run",
        action="store_true",
        help="Create a local-only plan without connecting to Jira",
    )
    dry_run_options.add_argument(
        "--jira-dry-run",
        action="store_true",
        help="Inspect Jira read-only and report a reconciliation plan",
    )

    assessor = subparsers.add_parser("assess", help="Assess each release requirement in order")
    assessor.add_argument("--manifest", type=Path, help="Release manifest JSON path")
    assessor.add_argument("--dataset", type=Path, help="Local evidence dataset path")
    assessor.add_argument(
        "--jira",
        action="store_true",
        help="Read evidence from configured Jira rather than the local dataset",
    )
    return parser


def _path_or_default(value: Path | None, fallback: Path) -> Path:
    path = value or fallback
    return path if path.is_absolute() else get_settings().resolve_path(path)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args()

    if args.command == "import":
        dataset_path = _path_or_default(args.dataset, settings.resolved_dataset_path)
        if args.jira_dry_run:
            manifest_path = settings.resolved_manifest_path
            importer = DatasetImporter(
                jira_client=None,
                project_key=settings.jira_project_key,
                import_label=settings.import_label,
            )
            local_validation = importer.validate_local(dataset_path, manifest_path)
            if local_validation["status"] != "VALIDATION_PASSED":
                result = importer.jira_dry_run(dataset_path, manifest_path)
            else:
                importer.jira_client = JiraClient(settings)
                result = importer.jira_dry_run(dataset_path, manifest_path)
            summary = Table(title="Jira-aware dry-run summary")
            summary.add_column("Result")
            summary.add_column("Count", justify="right")
            summary.add_row("Overall", str(result["overall_result"]))
            summary.add_row("Issues scanned", str(result["project_scan"]["issues_scanned"]))
            for status, count in result["ticket_counts"].items():
                summary.add_row(status, str(count))
            summary.add_row(
                "Planned comment additions",
                str(result["comment_counts"]["planned_additions"]),
            )
            summary.add_row("API errors", str(len(result["api_errors"])))
            Console(stderr=True).print(summary)
            console.print_json(data=result)
            return

        jira_client = None
        if not args.dry_run:
            jira_client = JiraClient(settings)
        result = DatasetImporter(
            jira_client=jira_client,
            project_key=settings.jira_project_key,
            import_label=settings.import_label,
        ).run(dataset_path, args.dry_run)
        console.print_json(data=result.model_dump(mode="json"))
        return

    if not settings.groq_api_key:
        raise RuntimeError("Set GROQ_API_KEY in .env before running an assessment.")
    jira_client = JiraClient(settings) if args.jira else None
    dataset_path = _path_or_default(args.dataset, settings.resolved_dataset_path)
    manifest_path = _path_or_default(args.manifest, settings.resolved_manifest_path)
    manifest = ReleaseManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    report = AssessmentRunner(
        settings,
        EvidenceCollector(
            dataset_path=dataset_path,
            jira_client=jira_client,
            import_label=settings.import_label,
        ),
        DecisionEngine(settings),
    ).assess(manifest)

    table = Table(title=f"{report.release_name} - {report.overall_decision.value}")
    table.add_column("Requirement")
    table.add_column("Decision")
    table.add_column("Confidence", justify="right")
    for requirement in report.requirements:
        table.add_row(
            f"{requirement.requirement_id}: {requirement.title}",
            requirement.decision.value,
            f"{requirement.confidence:.0%}",
        )
    console.print(table)
    console.print(report.summary)
    console.print(f"Report and investigation trace: {settings.resolved_output_dir / 'latest_assessment.json'}")


if __name__ == "__main__":
    main()
