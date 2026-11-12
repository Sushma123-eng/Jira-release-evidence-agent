import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from release_agent.utils import load_prompt

logger = logging.getLogger(__name__)
TraceRecorder = Callable[..., None]


class NoToolArguments(BaseModel):
    pass


class ToolCallingAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def route_query(
        self,
        query: str,
        tools: list[StructuredTool],
        record: TraceRecorder,
        requirement_catalog: str = "",
        max_tool_calls: int = 12,
    ) -> str:
        if not tools:
            raise ValueError("At least one query tool must be registered")
        allowed_tools = {tool.name: tool for tool in tools}
        messages: list[Any] = [
            SystemMessage(
                content=load_prompt("agent_router.txt").replace(
                    "{requirement_catalog}", requirement_catalog
                )
            ),
            HumanMessage(content=query),
        ]
        tool_call_count = 0

        while True:
            response = self.llm.bind_tools(tools).invoke(messages)
            tool_calls = getattr(response, "tool_calls", [])
            if not tool_calls:
                answer = self._message_text(getattr(response, "content", ""))
                if not answer.strip():
                    raise RuntimeError("The agent returned an empty answer")
                record(
                    "agent_answer",
                    "Agent returned its evidence-grounded answer",
                    tool_calls=tool_call_count,
                )
                return answer

            if tool_call_count + len(tool_calls) > max_tool_calls:
                raise RuntimeError(
                    f"Agent exceeded the limit of {max_tool_calls} tool calls"
                )
            messages.append(response)
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool = allowed_tools.get(tool_name)
                if tool is None:
                    raise ValueError(f"Agent requested unknown tool {tool_name!r}")
                tool_call_count += 1
                record(
                    "agent_tool_started",
                    f"Agent requested {tool_name}",
                    tool_name=tool_name,
                    tool_call_id=tool_call.get("id"),
                )
                try:
                    result = tool.invoke(tool_call.get("args") or {})
                except Exception:
                    logger.exception("Agent tool %s failed", tool_name)
                    record(
                        "agent_tool_failed",
                        f"Agent tool {tool_name} failed",
                        tool_name=tool_name,
                        tool_call_id=tool_call.get("id"),
                    )
                    raise
                result_text = result if isinstance(result, str) else str(result)
                messages.append(
                    ToolMessage(
                        content=result_text,
                        tool_call_id=tool_call["id"],
                    )
                )
                record(
                    "agent_tool_completed",
                    f"Agent tool {tool_name} completed",
                    tool_name=tool_name,
                    tool_call_id=tool_call.get("id"),
                )

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        return str(content)
