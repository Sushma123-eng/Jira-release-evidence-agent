# Jira Release Evidence Agent

A production-minded, explicitly sequential release-gate assistant. It imports a controlled evidence dataset into Jira, then reviews **one mandatory release requirement at a time**: collect linked issue and comment evidence, ask Groq to classify that requirement, record the decision and trace, and continue to the next requirement. It does not use LangGraph, CrewAI, AutoGen, or LlamaIndex.

The included 13-ticket, 41-comment fictional dataset demonstrates both supportive evidence and unresolved release risks. It is safe to preview without Jira credentials. Live assessments require a Groq API key; Jira import and Jira-backed assessment additionally require Jira Cloud credentials.

## Requirements

- Python 3.11 or newer
- Groq API key for assessments (`openai/gpt-oss-120b`)
- Jira Cloud site, account email, and API token for write/read operations against Jira

## Setup

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`. Set `GROQ_API_KEY` for assessments. For a live import or Jira-backed run, also set `JIRA_URL`, `JIRA_USERNAME`, `JIRA_API_TOKEN`, and the target `JIRA_PROJECT_KEY`. Use a Jira API token, not an account password. Do not commit `.env`.

`JIRA_PROJECT_KEY` is the Jira **destination** project. The dataset's root `project_key` identifies the **source** project and may differ. For example, the included source data identifies project `TYN`; setting `JIRA_PROJECT_KEY=SCRUM` imports its tickets into `SCRUM` while retaining original `TYN-...` issue keys, source IDs, manifest references, and provenance. Jira-backed assessment resolves those original manifest keys to their imported Jira issue keys in the configured destination project.

When a source issue type is unavailable in the destination project's create metadata, the importer uses an explicit mapping: `Bug`, `Approval`, `Story`, and `Test` map to Jira `Task`. The original source type is retained as `source_issue_type` in provenance for mapped records. Source fingerprints continue to represent the unchanged source ticket, while Jira content fingerprints represent the mapped destination fields. This mapping does not change source IDs or manifest references.

The project also supports editable installation:

```powershell
pip install -e .
```

## CLI

Preview importing the complete sample dataset (no network call):

```powershell
python -m release_agent.main import --dry-run
```

Import into the configured Jira project:

```powershell
python -m release_agent.main import
```

### Read-only live Jira integration test

The opt-in integration test uses the sample dataset and credentials in `.env`, without starting FastAPI or calling Groq. It scans the configured Jira project and reconciles the sample tickets and comments in read-only mode. It fails if the scan is incomplete, Jira returns API errors, or the data needs review. No Jira issues, comments, or properties are created or changed.

Run it from the project directory:

```powershell
pip install -e ".[test]"
$env:RUN_JIRA_INTEGRATION = "1"
python -m pytest -m jira_integration -v -s
Remove-Item Env:RUN_JIRA_INTEGRATION
```

The scan enumerates all issues in the configured project to detect duplicate source IDs; issue details and comments are read only for sample dataset matches. Run `python -m release_agent.main import --jira-dry-run` to see the full reconciliation report. This test is intentionally opt-in because it accesses a live Jira project.

The importer applies `tyn-release-assessment-v2`. Source IDs are embedded in issue descriptions and imported comment IDs in comment bodies. Re-imports find source IDs and comment markers before creating anything; edits update mapped issue content and previously unseen comments are appended. Jira does not allow ordinary API clients to author comments as fictional users or backdate them, so the original author, timestamp, and links are retained as explicit source metadata in the imported content.

Assess the sample snapshot sequentially:

```powershell
python -m release_agent.main assess
```

Read evidence directly from the Jira issue keys named in the release manifest:

```powershell
python -m release_agent.main assess --jira
```

Use alternate input files with `--manifest` and `--dataset`. Assessments are saved to `outputs/latest_assessment.json`, including the full investigation trace. The overall status is:

- `READY`: every mandatory requirement is `SATISFIED`.
- `BLOCKED`: at least one mandatory requirement is explicitly `NOT_SATISFIED`.
- `NEEDS_REVIEW`: evidence is partial/insufficient, or an assessment is missing.

## REST API

Start the service from the project directory:

```powershell
uvicorn release_agent.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Interactive API documentation is at `http://127.0.0.1:8000/docs`.

### `GET /health`

Returns service status and whether Jira and Groq credentials are configured. It never returns credential values.

### `POST /agent_callback`

Send a natural-language query only. The agent routes it to the relevant registered tool, calls Jira/local evidence tools as needed, and returns a concise decision and human-readable answer. Investigation traces are saved separately rather than included in the response body.

```json
{"query": "What release requirements are mandatory?"}
```

Examples:

```json
{"query": "What Jira evidence do we have for the security review?"}
```

```json
{"query": "Is the release ready? Assess every mandatory requirement."}
```

```json
{"query": "Preview importing the sample release dataset into Jira."}
```

The import tool is deliberately dry-run only. For a broad readiness question, the agent invokes the sequential assessment tool: evidence is collected and each requirement is assessed in manifest order. The response body contains only `decision` and `answer`. Assessment decisions are normalized to `PASS`, `FAIL`, or `INSUFFICIENT_EVIDENCE`; a passing answer is exactly 150 words, while a failing or insufficient-evidence answer is 100–150 words and explains known evidence gaps or retrieval errors in plain language. Non-assessment questions return a natural-language answer and a null decision. Investigation traces and full assessment reports are still saved to the configured output directory.

## Architecture

1. **Configuration and models** — Pydantic Settings loads environment configuration; Pydantic v2 validates manifests, evidence, import requests, reports, and trace records.
2. **Importer** — validates the input dataset, maps source metadata into Jira-compatible fields, identifies previously imported source records, and deduplicates comments. Dry-run generates a plan without connecting to Jira.
3. **Evidence collector** — collects only issue keys explicitly associated with the current requirement, either from the local snapshot or Jira.
4. **Tool-using agent** — `/agent_callback` accepts only a natural-language `query`. LangChain `ChatGroq` selects among `list_release_requirements`, `get_evidence`, `assess_requirement`, `assess_release`, and `import_dataset` tools. Requirement IDs are validated against the manifest; broad assessments collect evidence and assess requirements sequentially in manifest order. The import tool is dry-run only.
5. **Aggregation and report** — derives the release decision from mandatory requirement results and atomically persists the report with trace stages for assessment start, evidence collection, and decision recording.
6. **Interfaces** — CLI and FastAPI share the same importer and assessment runner; the API exposes only `/agent_callback` and `/health`.

Jira evidence is untrusted input. The reviewer prompt treats ticket content as evidence, not instructions. Assessments are decision support: verify Jira permissions, project configuration, and release policy before using them as an operational gate.

## Project layout

```text
src/release_agent/
  agent.py                 Constrained LangChain tool caller
  api.py                   FastAPI endpoints
  config.py                Environment settings and paths
  decision_engine.py       LangChain/Groq decisions and aggregation
  evidence_collector.py    Local/Jira evidence adapters
  jira_client.py           Jira Cloud client wrapper
  main.py                  CLI
  models.py                API, report, and evidence models
  report_generator.py      Sequential workflow and persisted report
  importer/                Dataset validation, mapping, and idempotent import
prompts/                   Agent, tool-routing, and evidence-review prompts
inputs/                    Fictional Jira data and release requirements
outputs/                   Generated assessment reports
```

Agent instructions, decision rules, request templates, and tool descriptions live in `prompts/`, are packaged with the project, and are loaded at runtime.
