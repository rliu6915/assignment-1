"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        system_information = {
            "machine": self.env.machine,
            "release": self.env.release,
            "system": self.env.system,
            "version": self.env.version,
        }
        self.system_prompt = (
            "You are a software engineering agent working in a terminal.\n"
            "Investigate the repository, reproduce the reported issue, make a "
            "focused fix, and verify it with the project's tests.\n"
            "Commands run in a non-interactive shell; prefer small outputs.\n"
            f"<system_information>\n"
            f"{json.dumps(system_information, indent=2)}\n"
            f"</system_information>"
        )
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nSkills are available. Call `invoke_skill` with a skill's "
                "name to read its instructions, then follow them.\n"
                f"<skills>\n{catalog}\n</skills>"
            )

        self.task_prompt = (
            "Fix the following issue in the repository at the current working "
            "directory.\n\n"
            f"{task}"
        )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        observations: list[dict[str, str]] = []
        for call in tool_calls:
            call_id = call.get("id", "")
            function = call.get("function") or {}
            name = function.get("name", "")
            raw_arguments = function.get("arguments") or ""

            arguments, error = _parse_arguments(raw_arguments)
            if error is not None:
                content = error
            elif name == "execute":
                content = self._execute(arguments)
            elif name == "send_message":
                content = self._send_message(arguments)
            elif name == "invoke_skill":
                content = self._invoke_skill(arguments)
            else:
                content = f"Error: unknown tool '{name}'."

            observations.append(
                {"role": "tool", "tool_call_id": call_id, "content": content}
            )

        return observations

    def _execute(self, arguments: dict[str, Any]) -> str:
        command = arguments.get("command")
        if not isinstance(command, (str, list)):
            return "Error: execute needs a 'command' string or argv list."

        overrides = {
            key: arguments[key]
            for key in ("shell", "cwd", "timeout", "env")
            if key in arguments
        }
        try:
            result = self.env.execute(command, **overrides)
        except Exception as exc:
            return f"Error: execute failed ({type(exc).__name__}: {exc})."
        return format_tool_output(result)

    def _send_message(self, arguments: dict[str, Any]) -> str:
        summary = arguments.get("summary")
        if not isinstance(summary, str):
            return "Error: send_message needs a 'summary' string."
        self.finished = True
        return f"Message delivered: {summary}"

    def _invoke_skill(self, arguments: dict[str, Any]) -> str:
        name = arguments.get("name")
        if not isinstance(name, str):
            return "Error: invoke_skill needs a 'name' string."
        skill = self.skills.get(name)
        if skill is None:
            return f"Error: no skill named '{name}'."
        return skill["content"]


def _parse_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    """Parse one tool call's arguments, reporting problems as observations."""

    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError:
        return {}, "Error: tool arguments were not valid JSON."
    if not isinstance(arguments, dict):
        return {}, "Error: tool arguments must be a JSON object."
    return arguments, None
