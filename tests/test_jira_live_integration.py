import os

import pytest

from release_agent.config import get_settings
from release_agent.importer.importer import DatasetImporter
from release_agent.jira_client import JiraClient


@pytest.mark.jira_integration
@pytest.mark.skipif(
    os.getenv("RUN_JIRA_INTEGRATION") != "1",
    reason="Set RUN_JIRA_INTEGRATION=1 to access the configured live Jira project.",
)
def test_sample_dataset_reconciles_against_live_jira_without_writes() -> None:
    settings = get_settings()
    jira = JiraClient(settings)
    importer = DatasetImporter(
        jira_client=jira,
        project_key=settings.jira_project_key,
        import_label=settings.import_label,
    )

    report = importer.jira_dry_run(
        settings.resolved_dataset_path,
        settings.resolved_manifest_path,
    )
    print(
        "Live Jira reconciliation:",
        f"result={report['overall_result']};",
        f"tickets={report['ticket_counts']};",
        f"comments={report['comment_counts']};",
        f"api_errors={len(report['api_errors'])};",
        f"writes={report['writes_performed']}",
    )

    assert report["local_validation"]["status"] == "VALIDATION_PASSED"
    assert report["total_tickets"] == 13
    assert report["total_source_comments"] == 41
    assert report["project_scan"]["complete"] is True
    assert report["api_errors"] == []
    assert report["writes_performed"] == 0
    assert report["overall_result"] == "SAFE_TO_PROCEED", (
        "Jira reconciliation needs review; inspect ticket_counts, comment_counts, "
        "and the full report from `python -m release_agent.main import --jira-dry-run`."
    )
