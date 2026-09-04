# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lifecycle rail for model- and tool-authored browser working memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import TypeAdapter, ValidationError

from openjiuwen.core.runner.callback.errors import AbortError
from openjiuwen.core.single_agent.rail.base import (
    AgentCallbackContext,
    AgentRail,
    InvokeInputs,
    ModelCallInputs,
    ToolCallInputs,
)

from .browser_logging import browser_agent_log_warning
from .browser_working_context import (
    BROWSER_TOOL_MEMORY_METADATA_KEY,
    BROWSER_WORKING_MEMORY_RECORD_BEGIN,
    BROWSER_WORKING_MEMORY_RECORD_END,
    BrowserActionAssessment,
    BrowserPlannedAction,
    BrowserWorkingContextConfig,
    BrowserWorkingContextStore,
    BrowserWorkingMemory,
    latest_browser_user_request,
)

_WORKING_MEMORY_RECORD_RE = re.compile(
    rf"{re.escape(BROWSER_WORKING_MEMORY_RECORD_BEGIN)}\s*(.*?)\s*"
    rf"{re.escape(BROWSER_WORKING_MEMORY_RECORD_END)}",
    re.DOTALL | re.IGNORECASE,
)
_REQUEST_STARTED_EXTRA_KEY = "_browser_working_context_request_started"
_ALLOWED_DELTA_FIELDS = frozenset(
    {
        "add_failures",
        "add_key_facts",
        "action_assessment",
        "batch_intent",
    }
)
_FIELD_ADAPTERS = {
    "add_failures": TypeAdapter(list[str]),
    "add_key_facts": TypeAdapter(list[str]),
    "action_assessment": TypeAdapter(BrowserActionAssessment),
    "batch_intent": TypeAdapter(BrowserPlannedAction),
}


@dataclass(frozen=True)
class _ParsedWorkingMemoryUpdate:
    """Independently validated durable memory and action-contract sections."""

    memory: BrowserWorkingMemory
    batch_intent: list[BrowserPlannedAction]
    action_assessments: list[BrowserActionAssessment]
    batch_intent_valid: bool
    action_assessments_valid: bool
    error: Optional[str] = None


class BrowserWorkingContextRail(AgentRail):
    """Commit one model update plus all tool retention at ReAct step boundaries."""

    priority = 45

    def __init__(self, config: Optional[BrowserWorkingContextConfig] = None) -> None:
        super().__init__()
        self.config = config or BrowserWorkingContextConfig()
        self._store = BrowserWorkingContextStore(self.config)

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        # DeepAgent's outer callback is the invocation boundary. Reset there
        # when a Session is available, but let the inner model boundary record
        # the request so it is not counted twice.
        if getattr(ctx.agent, "react_agent", None) is not None:
            self._store.reset(ctx.session)
            return
        inputs = ctx.inputs
        query = inputs.query if isinstance(inputs, InvokeInputs) else ""
        if ctx.session is None:
            return
        self._store.reset(ctx.session)
        self._store.begin_request(ctx.session, query)
        ctx.extra[_REQUEST_STARTED_EXTRA_KEY] = True

    async def before_model_call(self, ctx: AgentCallbackContext) -> None:
        """Reset and begin the request where the inner ReAct Session exists.

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
            self._store.reset(ctx.session)
            self._store.begin_request(ctx.session, query)

    async def after_model_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        if not isinstance(inputs, ModelCallInputs):
            return
        response = inputs.response
        if response is None:
            return

        cleaned_content, payloads = self._extract_and_strip_records(getattr(response, "content", ""))
        response.content = cleaned_content
        tool_calls = getattr(response, "tool_calls", None) or []
        assessment_required = bool(self._store.load(ctx.session).actions_requiring_assessment)
        record_required = bool(tool_calls) or assessment_required

        current_memory = self._carry_forward_memory(ctx)
        if len(payloads) != 1:
            update_error = None
            if not payloads and record_required:
                update_error = "Model omitted the required working-memory delta."
            elif payloads:
                update_error = "Model emitted more than one working-memory delta."
            update = _ParsedWorkingMemoryUpdate(
                memory=current_memory,
                batch_intent=[],
                action_assessments=[],
                batch_intent_valid=not bool(tool_calls),
                action_assessments_valid=not assessment_required,
                error=update_error,
            )
        else:
            update = self._parse_update(payloads[0], current_memory)

        if update.error:
            browser_agent_log_warning(
                "[BrowserWorkingContextRail] %s Valid durable fields were preserved; "
                "invalid or missing durable fields were carried forward.",
                update.error,
            )

        contract_error = self._store.stage_model_step(
            ctx.session,
            memory=update.memory,
            model_update_error=update.error,
            tool_calls=tool_calls,
            batch_intent=update.batch_intent,
            action_assessments=update.action_assessments,
            batch_intent_valid=update.batch_intent_valid,
            action_assessments_valid=update.action_assessments_valid,
        )
        if contract_error:
            browser_agent_log_warning(
                "[BrowserWorkingContextRail] %s",
                contract_error,
            )

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        """Warn on context-contract issues and block only hard action safeguards."""

        inputs = ctx.inputs
        if not isinstance(inputs, ToolCallInputs):
            return
        tool_call_id = str(getattr(inputs.tool_call, "id", "") or "")
        blocked_reason = self._store.blocked_reason_for_tool_call(
            ctx.session,
            tool_call_id=tool_call_id,
            tool_name=inputs.tool_name,
            tool_args=inputs.tool_args,
        )
        if not blocked_reason:
            warning_reason = self._store.warning_reason_for_tool_call(
                ctx.session,
                tool_call_id=tool_call_id,
                tool_name=inputs.tool_name,
                tool_args=inputs.tool_args,
            )
            if warning_reason:
                browser_agent_log_warning(
                    "[BrowserWorkingContextRail] allowing browser tool %s despite context-contract warning: %s",
                    inputs.tool_name,
                    warning_reason,
                )
            return
        raise AbortError(
            reason="Browser working-context action contract rejected the tool call.",
            cause=RuntimeError(blocked_reason),
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

    def _parse_update(
        self,
        payload: str,
        current_memory: BrowserWorkingMemory,
    ) -> _ParsedWorkingMemoryUpdate:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return _ParsedWorkingMemoryUpdate(
                memory=current_memory,
                batch_intent=[],
                action_assessments=[],
                batch_intent_valid=False,
                action_assessments_valid=False,
                error="Model emitted invalid JSON in the working-memory delta.",
            )
        if not isinstance(parsed, dict):
            return _ParsedWorkingMemoryUpdate(
                memory=current_memory,
                batch_intent=[],
                action_assessments=[],
                batch_intent_valid=False,
                action_assessments_valid=False,
                error="Model working-memory delta must contain a JSON object.",
            )

        errors: list[str] = []
        invalid_fields = set(parsed).difference(_ALLOWED_DELTA_FIELDS)
        validated_fields: dict[str, Any] = {}
        for field_name, adapter in _FIELD_ADAPTERS.items():
            if field_name not in parsed:
                continue
            try:
                validated_fields[field_name] = adapter.validate_python(parsed[field_name])
            except ValidationError as exc:
                for error in exc.errors(include_input=False):
                    location = ".".join(str(part) for part in error["loc"])
                    invalid_fields.add(f"{field_name}.{location}" if location else field_name)

        if invalid_fields:
            errors.append(
                f"Model working-memory delta failed validation. Invalid fields: {', '.join(sorted(invalid_fields))}."
            )

        memory = self._store.apply_memory_delta(
            current_memory,
            add_failures=validated_fields.get("add_failures", []),
            add_key_facts=validated_fields.get("add_key_facts", []),
        )
        batch_intent = validated_fields.get("batch_intent")
        action_assessment = validated_fields.get("action_assessment")
        return _ParsedWorkingMemoryUpdate(
            memory=memory,
            batch_intent=[batch_intent] if batch_intent is not None else [],
            action_assessments=[action_assessment] if action_assessment is not None else [],
            batch_intent_valid="batch_intent" not in parsed or batch_intent is not None,
            action_assessments_valid="action_assessment" not in parsed or action_assessment is not None,
            error=" ".join(errors) or None,
        )

    def _carry_forward_memory(
        self,
        ctx: AgentCallbackContext,
    ) -> BrowserWorkingMemory:
        """Return a complete current record without reporting a tool-step omission."""

        state = self._store.load(ctx.session)
        return self._store.sanitize_memory(state.current)

    @classmethod
    def _extract_and_strip_records(
        cls,
        content: Any,
    ) -> tuple[Any, list[str]]:
        if isinstance(content, str):
            payloads = [match.group(1).strip() for match in _WORKING_MEMORY_RECORD_RE.finditer(content)]
            return _WORKING_MEMORY_RECORD_RE.sub("", content).strip(), payloads

        if not isinstance(content, list):
            return content, []

        payloads: list[str] = []
        cleaned_parts: list[Any] = []
        for part in content:
            if isinstance(part, str):
                cleaned, part_payloads = cls._extract_and_strip_records(part)
                payloads.extend(part_payloads)
                if cleaned:
                    cleaned_parts.append(cleaned)
                continue
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                cleaned, part_payloads = cls._extract_and_strip_records(part["text"])
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
        return latest_browser_user_request(messages)


__all__ = ["BrowserWorkingContextRail"]
