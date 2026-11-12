import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tiktoken

from release_agent.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
PROMPT_DIR = PROJECT_ROOT / "prompts"


def utc_now() -> datetime:
    return datetime.now(UTC)


def estimate_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens]) + "\n[Evidence truncated to fit context.]"


def load_prompt(name: str) -> str:
    if Path(name).name != name or Path(name).suffix not in {".txt", ".json"}:
        raise ValueError(f"Invalid prompt filename: {name!r}")
    path = PROMPT_DIR / name
    if not path.is_file():
        path = Path(sys.prefix) / "prompts" / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.exception("Could not load prompt file %s", path)
        raise


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except OSError:
        logger.exception("Could not save JSON report to %s", path)
        raise


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
