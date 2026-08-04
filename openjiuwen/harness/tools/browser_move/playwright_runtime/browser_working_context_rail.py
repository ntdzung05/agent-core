# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lifecycle rail for model- and tool-authored browser working memory."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import ValidationError

from openjiuwen.core.foundation.llm import UserMessage
from openjiuwen.core.single_agent.rail.base import (
    AgentCallbackContext,
    AgentRail,
    InvokeInputs,
    ModelCallInputs,
    ToolCallInputs,
)

from .browser_working_context import (
    BROWSER_CONTEXT_UPDATE_CLOSE_TAG,
    BROWSER_CONTEXT_UPDATE_OPEN_TAG,
    BROWSER_TOOL_MEMORY_METADATA_KEY,
    BrowserWorkingContextConfig,
    BrowserWorkingContextStore,
    BrowserWorkingMemory,
    BrowserTaskItem,
)
from .browser_logging import browser_agent_log_warning


_CONTEXT_UPDATE_RE = re.compile(
    rf"{re.escape(BROWSER_CONTEXT_UPDATE_OPEN_TAG)}\s*(.*?)\s*"
    rf"{re.escape(BROWSER_CONTEXT_UPDATE_CLOSE_TAG)}",
    re.DOTALL | re.IGNORECASE,
)
_REQUEST_STARTED_EXTRA_KEY = "_browser_working_context_request_started"
_REQUIRED_MEMORY_FIELDS = frozenset(
    {
        "task_list",
        "errors",
        "failures",
        "blockers",
        "key_facts",
        "important_information",
    }
)
_FALLBACK_TASK = {
    "en": "Continue the active browser request.",
    "cn": "继续处理当前浏览器请求。",
}
_EPHEMERAL_USER_MESSAGE_NAMES = frozenset(
    {
        "browser_working_context",
        "current_browser_state",
    }
)


class BrowserWorkingContextRail(AgentRail):
    """Commit one model update plus all tool retention at ReAct step boundaries."""

    priority = 45

    def __init__(self, config: Optional[BrowserWorkingContextConfig] = None) -> None:
        super().__init__()
        self.config = config or BrowserWorkingContextConfig()
        self._store = BrowserWorkingContextStore(self.config)

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        # DeepAgent owns the outer invoke callback but creates a separate inner
        # ReAct callback context. Let before_model_call handle that topology so
        # an explicitly supplied Session is not counted twice.
        if getattr(ctx.agent, "react_agent", None) is not None:
            return
        inputs = ctx.inputs
        query = inputs.query if isinstance(inputs, InvokeInputs) else ""
        if ctx.session is None:
            return
        self._store.begin_request(ctx.session, query)
        ctx.extra[_REQUEST_STARTED_EXTRA_KEY] = True

    async def before_model_call(self, ctx: AgentCallbackContext) -> None:
        """Begin the request where the inner ReAct Session actually exists.

        DeepAgent routes ``before_invoke`` to its outer lifecycle. Subagents
        invoked by conversation id have no Session there; the restorable
        Session is created immediately before the inner model loop. This
        bridged callback is therefore the authoritative reinvocation boundary.
        """

        if ctx.extra.get(_REQUEST_STARTED_EXTRA_KEY):
            return
        ctx.extra[_REQUEST_STARTED_EXTRA_KEY] = True
        if ctx.session is None or ctx.extra.get("_resume_continuation"):
            return
        query = self._latest_user_request(ctx)
        if query:
            self._store.begin_request(ctx.session, query)

    async def after_model_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ModelCallInputs):
            return
        response = inputs.response
        if response is None:
            return

        cleaned_content, payloads = self._extract_and_strip_updates(getattr(response, "content", ""))
        response.content = cleaned_content

        memory = None
        update_error = None
        if len(payloads) != 1:
            update_error = (
                "Model omitted the required browser context update."
                if not payloads
                else "Model emitted more than one browser context update."
            )
        else:
            memory, update_error = self._parse_memory(payloads[0])

        if memory is None:
            memory = self._enforce_context_update(ctx, update_error)

        self._store.stage_model_step(
            ctx.session,
            memory=memory,
            model_update_error=update_error,
            tool_calls=getattr(response, "tool_calls", None) or [],
        )

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs) or inputs.tool_msg is None:
            return
        tool_call_id = str(getattr(inputs.tool_call, "id", None) or getattr(inputs.tool_msg, "tool_call_id", "") or "")
        memory = self._store.build_tool_retention(
            tool_name=inputs.tool_name,
            tool_call_id=tool_call_id,
            tool_result=inputs.tool_result,
        )
        if not memory.has_prompt_content():
            return
        metadata = dict(getattr(inputs.tool_msg, "metadata", {}) or {})
        metadata[BROWSER_TOOL_MEMORY_METADATA_KEY] = memory.model_dump(
            mode="json",
            exclude_none=True,
        )
        inputs.tool_msg.metadata = metadata

    async def after_react_iteration(self, ctx: AgentCallbackContext) -> None:
        if ctx.context is None:
            return
        committed = self._store.commit_pending_from_messages(
            ctx.session,
            ctx.context.get_messages(),
        )
        if not committed:
            browser_agent_log_warning(
                "[BrowserWorkingContextRail] step boundary reached before all staged tool messages were available"
            )

    @staticmethod
    def _parse_memory(
        payload: str,
    ) -> tuple[Optional[BrowserWorkingMemory], Optional[str]]:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None, "Model emitted invalid JSON in the browser context update."
        if not isinstance(parsed, dict):
            return None, "Model browser context update must be a JSON object."
        missing_fields = sorted(_REQUIRED_MEMORY_FIELDS.difference(parsed))
        if missing_fields:
            return None, (
                "Model browser context update omitted required fields: "
                f"{', '.join(missing_fields)}."
            )
        try:
            memory = BrowserWorkingMemory.model_validate(parsed)
        except ValidationError as exc:
            invalid_fields = sorted(
                {".".join(str(part) for part in error["loc"]) for error in exc.errors(include_input=False)}
            )
            suffix = f" Invalid fields: {', '.join(invalid_fields)}." if invalid_fields else ""
            return None, f"Model browser context update failed validation.{suffix}"
        if not memory.task_list:
            return None, "Model browser context update must contain at least one task."
        return memory, None

    def _enforce_context_update(
        self,
        ctx: AgentCallbackContext,
        update_error: Optional[str],
    ) -> BrowserWorkingMemory:
        """Synthesize a complete carry-forward update after invalid model output."""

        state = self._store.load(ctx.session)
        memory = self._store.sanitize_memory(state.current)
        if not memory.task_list:
            task = state.active_request or self._latest_user_request(ctx)
            if not task:
                task = _FALLBACK_TASK[self.config.language]
            memory = self._store.sanitize_memory(
                memory.model_copy(
                    update={
                        "task_list": [
                            BrowserTaskItem(task=task, status="pending"),
                        ]
                    }
                )
            )

        browser_agent_log_warning(
            "[BrowserWorkingContextRail] %s Rail synthesized a complete carry-forward context update.",
            update_error or "Model emitted an invalid browser context update.",
        )
        return memory

    @classmethod
    def _extract_and_strip_updates(
        cls,
        content: Any,
    ) -> tuple[Any, list[str]]:
        if isinstance(content, str):
            payloads = [match.group(1).strip() for match in _CONTEXT_UPDATE_RE.finditer(content)]
            return _CONTEXT_UPDATE_RE.sub("", content).strip(), payloads

        if not isinstance(content, list):
            return content, []

        payloads: list[str] = []
        cleaned_parts: list[Any] = []
        for part in content:
            if isinstance(part, str):
                cleaned, part_payloads = cls._extract_and_strip_updates(part)
                payloads.extend(part_payloads)
                if cleaned:
                    cleaned_parts.append(cleaned)
                continue
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                cleaned, part_payloads = cls._extract_and_strip_updates(part["text"])
                payloads.extend(part_payloads)
                if cleaned:
                    cleaned_parts.append({**part, "text": cleaned})
                continue
            cleaned_parts.append(part)
        return cleaned_parts, payloads

    @classmethod
    def _latest_user_request(cls, ctx: AgentCallbackContext) -> str:
        inputs = ctx.inputs
        messages = inputs.messages if isinstance(inputs, ModelCallInputs) else []
        for message in reversed(messages):
            if not isinstance(message, UserMessage):
                continue
            if message.name in _EPHEMERAL_USER_MESSAGE_NAMES:
                continue
            metadata = getattr(message, "metadata", {}) or {}
            if metadata.get("browser_working_context") or metadata.get("browser_state_context"):
                continue
            text = cls._message_content_to_text(message.content)
            if text:
                return text
        return ""

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return str(content or "").strip()
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()


__all__ = ["BrowserWorkingContextRail"]
