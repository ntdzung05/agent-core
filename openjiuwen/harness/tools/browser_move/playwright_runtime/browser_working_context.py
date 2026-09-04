# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Typed, session-backed working context for the browser subagent."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openjiuwen.core.foundation.llm import BaseMessage, ToolMessage, UserMessage

from .browser_logging import browser_agent_log_info, browser_agent_log_warning

BROWSER_WORKING_CONTEXT_STATE_KEY = "__browser_subagent_working_context__"
BROWSER_TASK_STATE_KEY = "__browser_phase_budget_state__"
BROWSER_TOOL_MEMORY_METADATA_KEY = "browser_working_context_retention"
BROWSER_WORKING_MEMORY_RECORD_BEGIN = "---BEGIN WORKING MEMORY RECORD V1---"
BROWSER_WORKING_MEMORY_RECORD_END = "---END WORKING MEMORY RECORD V1---"

_ERROR_PREFIXES = (
    "ability execution error:",
    "tool execution error:",
    "workflow execution error:",
    "agent execution error:",
    "[interrupted]",
)
_WORKING_CONTEXT_INSTRUCTIONS = {
    "en": (
        "This is the runtime-maintained browser task context plus compact durable browser memory, "
        "separate from <browser_state>. Treat task status, phase state, field coverage, blockers, "
        "evidence, and recent actions as authoritative; do not rewrite or echo them, and do not "
        "repeat an action whose semantic_delta shows no progress. The system prompt "
        "defines the tool-batch intent and assessment workflow. Append exactly one plain-text JSON "
        "delta between ---BEGIN WORKING MEMORY RECORD V1--- and ---END WORKING MEMORY RECORD V1--- "
        "when required_now is non-empty, whenever this response calls tools, or when adding a new "
        "failure or verified fact. Otherwise omit the record. It is framework bookkeeping, not a "
        "tool; never place it in tool_calls. Include only applicable fields and omit unchanged or "
        "empty fields. action_assessment is a single assessment of previous_batch and is required "
        "when listed in required_now. batch_intent is a single declaration for all tools in this "
        "response and is required whenever tools are called. add_failures and add_key_facts contain "
        "only new durable information; never repeat existing memory. After not_achieved, an equivalent "
        "tool-and-arguments batch requires recovery_justification; after two failures, use a different "
        "action. Do not include credentials, screenshots, complete DOM, large raw output, or invented "
        "facts. Treat context-contract warnings as feedback and correct the assessment or intent in "
        "the next response. Optional delta shape: "
    ),
    "cn": (
        "这是由 runtime 维护的浏览器任务上下文和精简的浏览器持久记忆，与 <browser_state> 相互独立。"
        "任务状态、阶段、字段覆盖率、阻断项、证据和最近动作均为权威信息；不要重写或复述这些内容，"
        "也不要重复 semantic_delta 显示无进展的动作。系统提示定义工具批次的意图和评估流程。"
        "当 required_now 非空、本次响应调用工具，或需要新增失败记录或已验证事实时，必须且只能在 "
        "---BEGIN WORKING MEMORY RECORD V1--- 和 ---END WORKING MEMORY RECORD V1--- 之间追加一份纯文本 JSON "
        "增量；其他情况省略记录。该记录只供框架维护状态，不是工具，不得放入 tool_calls。只写适用字段，"
        "未变化或为空的字段应省略。required_now 要求时，action_assessment 对 previous_batch 整体评估一次；"
        "本次调用工具时，batch_intent 为全部工具声明一次共同 intent 和 expected_outcome。add_failures 和 "
        "add_key_facts 只写新增持久信息，不得重复现有记忆。not_achieved 后，等价工具和参数的批次必须提供 "
        "recovery_justification；两次失败后必须改用不同动作。不得写入凭据、截图、完整 DOM、大段原始输出或"
        "编造事实。上下文契约警告只作为反馈，应在下一次响应中修正评估或意图。可选增量结构："
    ),
}
_WORKING_MEMORY_DELTA_SHAPE = (
    '{"action_assessment":{"status":"achieved|partially_achieved|not_achieved|uncertain",'
    '"reason":"..."},"batch_intent":{"intent":"...","expected_outcome":"...",'
    '"recovery_justification":"..."},"add_failures":["..."],"add_key_facts":["..."]}'
)
_EPHEMERAL_USER_MESSAGE_NAMES = frozenset(
    {
        "browser_working_context",
        "current_browser_state",
        "browser_state_progress",
    }
)
_EPHEMERAL_CONTEXT_METADATA_KEYS = (
    "browser_working_context",
    "browser_state_context",
    "browser_state_progress_context",
)
_TERMINAL_TASK_STATUSES = frozenset({"blocked", "partial", "completed"})


class BrowserWorkingContextConfig(BaseModel):
    """Limits for browser working memory and prompt rendering."""

    language: Literal["cn", "en"] = "en"
    max_recent_steps: int = Field(default=2, ge=1)
    max_list_items: int = Field(default=20, ge=1)
    max_item_chars: int = Field(default=1_000, ge=128)
    max_one_step_chars: int = Field(default=8_000, ge=256)
    max_prompt_chars: int = Field(default=8_000, ge=2_000)


class BrowserWorkingMemory(BaseModel):
    """Small durable browser memory authored once per model step."""

    # Ignore fields from V1 checkpoints written before the memory contract was
    # reduced to failures and key facts.
    model_config = ConfigDict(extra="ignore")

    failures: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)


class BrowserPlannedAction(BaseModel):
    """Shared purpose and expected result for one response's tool batch."""

    model_config = ConfigDict(extra="forbid")

    intent: str = Field(min_length=1)
    expected_outcome: str = Field(min_length=1)
    recovery_justification: Optional[str] = None


class BrowserActionAssessment(BaseModel):
    """Model judgment of the preceding browser tool batch's outcome."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "achieved",
        "partially_achieved",
        "not_achieved",
        "uncertain",
    ]
    reason: str = Field(min_length=1)


class BrowserModelStepUpdate(BaseModel):
    """Optional model-authored delta for one browser-agent step."""

    model_config = ConfigDict(extra="forbid")

    action_assessment: Optional[BrowserActionAssessment] = None
    batch_intent: Optional[BrowserPlannedAction] = None
    add_failures: list[str] = Field(default_factory=list)
    add_key_facts: list[str] = Field(default_factory=list)


class BrowserToolMemory(BaseModel):
    """Prompt-safe retention selected from one complete diagnostic tool result."""

    tool_name: str
    tool_call_id: str
    durable_content: Optional[str] = None
    content_source: Optional[Literal["long_term_memory", "extracted_content"]] = None
    one_step_content: Optional[str] = None
    error: Optional[str] = None

    def has_prompt_content(self) -> bool:
        return bool(self.durable_content or self.one_step_content or self.error)


class BrowserPendingToolCall(BaseModel):
    """Declared action retained until the tool phase completes."""

    tool_name: str
    tool_call_id: str
    action_signature: str
    intent: str
    expected_outcome: str
    recovery_justification: Optional[str] = None
    warning_reason: Optional[str] = None
    blocked_reason: Optional[str] = None


class BrowserPendingAssessment(BaseModel):
    """Minimal action data that the following model call must assess."""

    tool_call_id: str
    tool_name: str
    action_signature: str
    intent: str
    expected_outcome: str


class BrowserFailedAction(BaseModel):
    """Counter used to reject an unchanged failed action."""

    action_signature: str
    failure_count: int = Field(default=1, ge=1)
    tool_name: str = ""
    intent: str = ""
    expected_outcome: str = ""
    last_reason: str = ""


class BrowserPendingStep(BaseModel):
    """Model-authored state waiting for the same step's tool results."""

    step_number: int
    model_memory: Optional[BrowserWorkingMemory] = None
    model_update_error: Optional[str] = None
    tool_calls: list[BrowserPendingToolCall] = Field(default_factory=list)


class BrowserStepRecord(BaseModel):
    """One durable record per model response, aggregating every tool result."""

    step_number: int
    model_memory: Optional[BrowserWorkingMemory] = None
    model_update_error: Optional[str] = None
    tool_memories: list[BrowserToolMemory] = Field(default_factory=list)


class BrowserWorkingContextState(BaseModel):
    """JSON-serializable durable state stored on the external agent Session."""

    model_config = ConfigDict(extra="ignore")

    version: int = 1
    request_sequence: int = 0
    request_kind: Literal["initial", "follow_up"] = "initial"
    active_request: str = ""
    current: BrowserWorkingMemory = Field(default_factory=BrowserWorkingMemory)
    recent_steps: list[BrowserStepRecord] = Field(default_factory=list)
    one_step_content: list[BrowserToolMemory] = Field(default_factory=list)
    next_step_number: int = 1
    pending_step: Optional[BrowserPendingStep] = None
    actions_requiring_assessment: list[BrowserPendingAssessment] = Field(default_factory=list)
    failed_actions: list[BrowserFailedAction] = Field(default_factory=list)


def _bounded_text(value: Any, max_chars: int) -> str:
    """Normalize and hard-cap one retained text value."""

    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > max_chars:
        omitted = len(text) - max_chars
        return f"{text[:max_chars].rstrip()} ...[truncated {omitted} characters]"
    return text


class BrowserWorkingContextStore:
    """Own the deterministic lifecycle of session-backed browser working context."""

    def __init__(self, config: BrowserWorkingContextConfig) -> None:
        self.config = config

    @staticmethod
    def load(session: Any) -> BrowserWorkingContextState:
        if session is None:
            return BrowserWorkingContextState()
        raw_state = session.get_state(BROWSER_WORKING_CONTEXT_STATE_KEY)
        if not isinstance(raw_state, dict):
            return BrowserWorkingContextState()
        try:
            return BrowserWorkingContextState.model_validate(raw_state)
        except ValidationError:
            browser_agent_log_warning("[BrowserWorkingContext] invalid stored state; starting with an empty state")
            return BrowserWorkingContextState()

    @staticmethod
    def save(session: Any, state: BrowserWorkingContextState) -> None:
        if session is None:
            return
        session.update_state({BROWSER_WORKING_CONTEXT_STATE_KEY: state.model_dump(mode="json")})

    def reset(self, session: Any) -> BrowserWorkingContextState:
        """Start an empty working context without disturbing other session state."""

        state = BrowserWorkingContextState()
        if session is None:
            return state
        self.save(session, state)
        browser_agent_log_info(
            "[BrowserWorkingContext] reset working context for session %s",
            getattr(session, "get_session_id", lambda: "")(),
        )
        return state

    def begin_request(self, session: Any, query: Any) -> BrowserWorkingContextState:
        """Record the active request while runtime task progress remains separately owned."""

        state = self.load(session)
        request_text = _bounded_text(query, self.config.max_item_chars)
        if not request_text:
            return state
        state.request_kind = "follow_up" if state.request_sequence else "initial"
        state.request_sequence += 1
        state.active_request = request_text
        self.save(session, state)
        browser_agent_log_info(
            "[BrowserWorkingContext] began %s request %d for session %s (facts=%d, durable_steps=%d)",
            state.request_kind,
            state.request_sequence,
            getattr(session, "get_session_id", lambda: "")(),
            len(state.current.key_facts),
            len(state.recent_steps),
        )
        return state

    def sanitize_memory(self, memory: BrowserWorkingMemory) -> BrowserWorkingMemory:
        """Bound every model-authored field before persistence."""

        return BrowserWorkingMemory(
            failures=self._sanitize_list(memory.failures),
            key_facts=self._sanitize_list(memory.key_facts),
        )

    def _append_failures(
        self,
        memory: BrowserWorkingMemory,
        failures: Iterable[Any],
    ) -> BrowserWorkingMemory:
        """Append bounded failures while refreshing duplicates and retaining newest entries."""

        merged = list(memory.failures)
        for failure in failures:
            text = _bounded_text(failure, self.config.max_item_chars)
            if not text:
                continue
            if text in merged:
                merged.remove(text)
            merged.append(text)
        return memory.model_copy(
            update={"failures": merged[-self.config.max_list_items :]},
        )

    def apply_memory_delta(
        self,
        memory: BrowserWorkingMemory,
        *,
        add_failures: Iterable[Any] = (),
        add_key_facts: Iterable[Any] = (),
    ) -> BrowserWorkingMemory:
        """Merge bounded model-authored additions without rewriting durable memory."""

        merged = self._append_failures(self.sanitize_memory(memory), add_failures)
        key_facts = list(merged.key_facts)
        for fact in add_key_facts:
            text = _bounded_text(fact, self.config.max_item_chars)
            if not text:
                continue
            if text in key_facts:
                key_facts.remove(text)
            key_facts.append(text)
        return merged.model_copy(
            update={"key_facts": key_facts[-self.config.max_list_items :]},
        )

    @staticmethod
    def action_signature(tool_name: Any, arguments: Any) -> str:
        """Hash normalized tool name and canonical arguments without retaining raw data."""

        if isinstance(arguments, str):
            raw_arguments = arguments.strip()
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, ValueError):
                canonical_arguments = raw_arguments
            else:
                canonical_arguments = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
        else:
            try:
                canonical_arguments = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                canonical_arguments = str(arguments or "").strip()
        normalized_name = str(tool_name or "").strip().casefold()
        return hashlib.sha256(f"{normalized_name}\n{canonical_arguments}".encode("utf-8")).hexdigest()

    def stage_model_step(
        self,
        session: Any,
        *,
        memory: Optional[BrowserWorkingMemory],
        model_update_error: Optional[str],
        tool_calls: Iterable[Any],
        batch_intent: Iterable[BrowserPlannedAction] = (),
        action_assessments: Iterable[BrowserActionAssessment] = (),
        batch_intent_valid: bool = True,
        action_assessments_valid: bool = True,
    ) -> Optional[str]:
        """Validate the batch action contract and stage the model step."""

        state = self.load(session)
        if state.pending_step is not None:
            self._commit_pending_as_incomplete(state)

        errors = [model_update_error] if model_update_error else []
        contract_warnings: list[str] = []
        assessment_error = self._apply_action_assessments(
            state,
            list(action_assessments) if action_assessments_valid else [],
            contract_valid=action_assessments_valid,
        )
        if assessment_error:
            errors.append(assessment_error)
            contract_warnings.append(assessment_error)

        calls = list(tool_calls)
        batch_plan, planning_error = self._validate_batch_intent(
            list(batch_intent) if batch_intent_valid else [],
            len(calls),
            contract_valid=batch_intent_valid,
        )
        if planning_error:
            errors.append(planning_error)
            contract_warnings.append(planning_error)

        sanitized_memory = memory if memory is not None else self.sanitize_memory(state.current)

        step_number = state.next_step_number
        error_text = _bounded_text(
            " ".join(dict.fromkeys(error for error in errors if error)),
            self.config.max_item_chars,
        )
        global_warning = (
            _bounded_text(
                " ".join(dict.fromkeys(error for error in contract_warnings if error)),
                self.config.max_item_chars,
            )
            or None
        )

        pending_calls: list[BrowserPendingToolCall] = []
        for index, tool_call in enumerate(calls):
            tool_name = _bounded_text(getattr(tool_call, "name", ""), self.config.max_item_chars)
            tool_call_id = str(getattr(tool_call, "id", "") or "")
            if not tool_call_id:
                tool_call_id = f"browser-r{state.request_sequence}-s{step_number}-a{index}"
            signature = self.action_signature(tool_name, getattr(tool_call, "arguments", ""))
            intent = _bounded_text(
                batch_plan.intent if batch_plan is not None else "Undeclared browser action batch.",
                self.config.max_item_chars,
            )
            expected = _bounded_text(
                batch_plan.expected_outcome if batch_plan is not None else "No batch outcome was declared.",
                self.config.max_item_chars,
            )
            recovery = (
                _bounded_text(batch_plan.recovery_justification, self.config.max_item_chars) or None
                if batch_plan is not None
                else None
            )
            failed = self._failed_action(state, signature)
            blocked_reason = None
            if failed is not None:
                if failed.failure_count >= 2:
                    blocked_reason = "Equivalent action failed twice; use a different tool or arguments."
                elif not recovery:
                    blocked_reason = "Equivalent action was assessed not_achieved; recovery_justification is required."
            pending_calls.append(
                BrowserPendingToolCall(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    action_signature=signature,
                    intent=intent,
                    expected_outcome=expected,
                    recovery_justification=recovery,
                    warning_reason=global_warning,
                    blocked_reason=_bounded_text(blocked_reason, self.config.max_item_chars) or None,
                )
            )

        if assessment_error and any(action.blocked_reason is None for action in pending_calls):
            # A new executable batch supersedes an assessment the model omitted
            # or malformed. Keep the warning for diagnostics, but do not mix the
            # stale batch with the next batch's assessment contract.
            state.actions_requiring_assessment = []

        blocked_failures = list(
            dict.fromkeys(action.blocked_reason for action in pending_calls if action.blocked_reason)
        )
        if blocked_failures:
            sanitized_memory = self._append_failures(sanitized_memory, blocked_failures)
            state.current = sanitized_memory

        state.pending_step = BrowserPendingStep(
            step_number=step_number,
            model_memory=sanitized_memory,
            model_update_error=error_text or None,
            tool_calls=pending_calls,
        )
        if pending_calls:
            self.save(session, state)
        else:
            self._commit_pending(state, [])
            self.save(session, state)
        return error_text or None

    def blocked_reason_for_tool_call(
        self,
        session: Any,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_args: Any,
    ) -> Optional[str]:
        """Return why the current tool call must not execute, if any."""

        pending = self.load(session).pending_step
        if pending is None:
            return "Browser tool call has no staged working-context action declaration."
        signature = self.action_signature(tool_name, tool_args)
        candidates = [
            action for action in pending.tool_calls if tool_call_id and action.tool_call_id == str(tool_call_id)
        ]
        if len(candidates) != 1:
            candidates = [action for action in pending.tool_calls if action.action_signature == signature]
        if len(candidates) != 1:
            return "Browser tool call could not be matched to exactly one declared planned action."
        action = candidates[0]
        if action.action_signature != signature:
            return "Browser tool name or arguments changed after its planned action was recorded."
        return action.blocked_reason

    def warning_reason_for_tool_call(
        self,
        session: Any,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_args: Any,
    ) -> Optional[str]:
        """Return a non-blocking contract warning for a matched tool call."""

        pending = self.load(session).pending_step
        if pending is None:
            return None
        signature = self.action_signature(tool_name, tool_args)
        candidates = [
            action for action in pending.tool_calls if tool_call_id and action.tool_call_id == str(tool_call_id)
        ]
        if len(candidates) != 1:
            candidates = [action for action in pending.tool_calls if action.action_signature == signature]
        if len(candidates) != 1 or candidates[0].action_signature != signature:
            return None
        return candidates[0].warning_reason

    def prepare_for_model_call(self, session: Any, messages: Iterable[BaseMessage]) -> None:
        """Make the preceding tool action assessable before the next model call."""

        state = self.load(session)
        if state.pending_step is None:
            return
        if not self.commit_pending_from_messages(session, messages):
            state = self.load(session)
            self._commit_pending_as_incomplete(state)
            self.save(session, state)

    def commit_pending_from_messages(
        self,
        session: Any,
        messages: Iterable[BaseMessage],
    ) -> bool:
        """Commit a staged step after every same-step tool message exists."""

        state = self.load(session)
        pending = state.pending_step
        if pending is None:
            return False
        if not pending.tool_calls:
            self._commit_pending(state, [])
            self.save(session, state)
            return True

        messages_by_call_id: Dict[str, ToolMessage] = {}
        for message in messages:
            if isinstance(message, ToolMessage):
                messages_by_call_id[str(message.tool_call_id)] = message
        if any(action.tool_call_id not in messages_by_call_id for action in pending.tool_calls if action.tool_call_id):
            return False

        tool_memories = [
            self.tool_memory_from_message(
                tool_name=action.tool_name,
                tool_call_id=action.tool_call_id,
                message=messages_by_call_id[action.tool_call_id],
            )
            for action in pending.tool_calls
            if action.tool_call_id in messages_by_call_id
        ]
        self._commit_pending(state, tool_memories)
        self.save(session, state)
        return True

    def build_tool_retention(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        tool_result: Any,
    ) -> BrowserToolMemory:
        """Apply existing tool-retention precedence without copying raw observations."""

        long_term_memory = getattr(tool_result, "long_term_memory", None)
        extracted_content = getattr(tool_result, "extracted_content", None)
        one_step_only = bool(getattr(tool_result, "include_extracted_content_only_once", False))
        error = getattr(tool_result, "error", None)
        success = getattr(tool_result, "success", None)

        durable_content = None
        content_source = None
        if long_term_memory:
            durable_content = _bounded_text(long_term_memory, self.config.max_item_chars)
            content_source = "long_term_memory"
        elif extracted_content and not one_step_only:
            durable_content = _bounded_text(extracted_content, self.config.max_item_chars)
            content_source = "extracted_content"

        one_step_content = None
        if extracted_content and one_step_only:
            one_step_content = _bounded_text(extracted_content, self.config.max_one_step_chars)

        retained_error = None
        if error or success is False:
            retained_error = _bounded_text(
                error or "Tool reported failure without an error message.",
                self.config.max_item_chars,
            )
        return BrowserToolMemory(
            tool_name=_bounded_text(tool_name, self.config.max_item_chars),
            tool_call_id=str(tool_call_id or ""),
            durable_content=durable_content or None,
            content_source=content_source,
            one_step_content=one_step_content or None,
            error=retained_error or None,
        )

    def tool_memory_from_message(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        message: ToolMessage,
    ) -> BrowserToolMemory:
        """Recover explicit retention metadata and independently retain tool errors."""

        payload = message.metadata.get(BROWSER_TOOL_MEMORY_METADATA_KEY)
        memory = BrowserToolMemory(tool_name=tool_name, tool_call_id=tool_call_id)
        if isinstance(payload, dict):
            try:
                memory = BrowserToolMemory.model_validate(payload)
            except ValidationError:
                browser_agent_log_warning(
                    "[BrowserWorkingContext] invalid tool retention metadata for %s",
                    tool_name,
                )
        memory.tool_name = _bounded_text(tool_name, self.config.max_item_chars)
        memory.tool_call_id = str(tool_call_id or "")
        if not memory.error:
            memory.error = self._infer_tool_error(message.content)
        if memory.has_prompt_content() and not isinstance(payload, dict):
            metadata = dict(message.metadata)
            metadata[BROWSER_TOOL_MEMORY_METADATA_KEY] = memory.model_dump(mode="json", exclude_none=True)
            message.metadata = metadata
        return memory

    def render_and_consume_one_step(self, session: Any) -> str:
        """Render one compact projection and expire next-step-only content."""

        state = self.load(session)
        one_step_content = list(state.one_step_content)
        if one_step_content:
            state.one_step_content = []
            self.save(session, state)

        instructions = _WORKING_CONTEXT_INSTRUCTIONS[self.config.language] + _WORKING_MEMORY_DELTA_SHAPE
        task_state = self.load_task_state(session)
        assessment_actions = state.actions_requiring_assessment
        previous_batch = None
        if assessment_actions:
            tool_names = list(dict.fromkeys(action.tool_name for action in assessment_actions))
            previous_batch = {
                "tool_names": tool_names,
                "tool_call_count": len(assessment_actions),
                "intent": assessment_actions[0].intent,
                "expected_outcome": assessment_actions[0].expected_outcome,
            }
        recent_tool_context = [
            {
                "step_number": step.step_number,
                "tool_memories": [item.model_dump(mode="json", exclude_none=True) for item in step.tool_memories],
            }
            for step in state.recent_steps
            if step.tool_memories
        ]
        rendered_state = {
            "required_now": ["action_assessment"] if assessment_actions else [],
            "required_if_calling_tools": ["batch_intent"],
            "request": state.active_request,
            "request_context": {
                "sequence": state.request_sequence,
                "kind": state.request_kind,
            },
            "memory": state.current.model_dump(mode="json"),
        }
        if task_state:
            rendered_state["task_state"] = self._project_task_state(task_state)
            rendered_state["runtime_directive"] = self._runtime_directive(task_state)
            recent_actions = list(task_state.get("recent_actions") or [])[-self.config.max_recent_steps :]
            if recent_actions:
                rendered_state["recent_actions"] = recent_actions
        if previous_batch is not None:
            rendered_state["previous_batch"] = previous_batch
        if one_step_content:
            rendered_state["new_tool_context"] = [
                item.model_dump(mode="json", exclude_none=True) for item in one_step_content
            ]
        if recent_tool_context:
            rendered_state["recent_tool_context"] = recent_tool_context
        if state.failed_actions:
            rendered_state["loop_guard"] = [
                action.model_dump(mode="json", exclude={"action_signature"})
                for action in state.failed_actions[-min(self.config.max_list_items, 5) :]
            ]
        body = _bounded_text(
            json.dumps(rendered_state, ensure_ascii=False, indent=2),
            self.config.max_prompt_chars,
        )
        return f"<browser_working_context>\n{instructions}\n{body}\n</browser_working_context>"

    def _compact_retained_tool_evidence(self, steps: Iterable[BrowserStepRecord]) -> list[Dict[str, Any]]:
        evidence: list[Dict[str, Any]] = []
        for step in list(steps)[-self.config.max_recent_steps :]:
            for memory in step.tool_memories:
                if not (memory.durable_content or memory.error):
                    continue
                evidence.append(
                    {
                        "step": step.step_number,
                        "tool": memory.tool_name,
                        "content": _bounded_text(
                            memory.durable_content or memory.error,
                            self.config.max_item_chars,
                        ),
                        "source": memory.content_source or ("error" if memory.error else "retained"),
                    }
                )
        return evidence[-self.config.max_recent_steps :]

    @staticmethod
    def load_task_state(session: Any) -> Dict[str, Any]:
        if session is None:
            return {}
        raw_state = session.get_state(BROWSER_TASK_STATE_KEY)
        return dict(raw_state) if isinstance(raw_state, dict) else {}

    @classmethod
    def sync_semantic_progress(cls, session: Any, progress: Any) -> bool:
        """Merge one browser observation into authoritative task state.

        Returns ``True`` only when a permitted replan trial produced observable
        semantic progress and the runtime gate may be cleared.
        """

        if session is None or not isinstance(progress, dict) or not progress:
            return False
        state = cls.load_task_state(session)
        if not state:
            return False
        revision = int(progress.get("revision") or 0)
        if revision <= int(state.get("semantic_revision") or 0):
            return False

        state["semantic_revision"] = revision
        cls._merge_semantic_observation(state, progress)
        if str(state.get("status") or "").strip().lower() in _TERMINAL_TASK_STATUSES:
            session.update_state({BROWSER_TASK_STATE_KEY: state})
            return False
        progress_name = str(progress.get("progress") or "unknown")
        recovered = bool(
            state.get("replan_trial_pending")
            and (progress.get("observable_progress") is True or progress_name == "progress")
        )
        cls._apply_replan_observation(state, progress, recovered=recovered)
        session.update_state({BROWSER_TASK_STATE_KEY: state})
        return recovered

    @staticmethod
    def _merge_semantic_observation(state: Dict[str, Any], progress: Dict[str, Any]) -> None:
        semantic_progress: Dict[str, Any] = {}
        semantic_keys = (
            "progress",
            "observable_progress",
            "consecutive_no_progress",
            "state_revisit_count",
            "aba_loop",
            "repeated_filter_state",
            "replan_required",
            "replan_reason",
        )
        for key in semantic_keys:
            if key in progress:
                semantic_progress[key] = progress.get(key)
        state["semantic_progress"] = semantic_progress
        semantic_state = progress.get("semantic_state")
        if isinstance(semantic_state, dict):
            coverage = semantic_state.get("field_coverage")
            if isinstance(coverage, list):
                merged_coverage = set(state.get("field_coverage") or [])
                merged_coverage.update(str(item) for item in coverage if str(item).strip())
                state["field_coverage"] = sorted(merged_coverage)
            url = str(semantic_state.get("url") or "").strip()
            if url:
                last_page = state.setdefault("last_page", {})
                last_page["url"] = url

        progress_name = str(progress.get("progress") or "unknown")
        recent_actions = state.get("recent_actions")
        if isinstance(recent_actions, list) and recent_actions:
            last_action = recent_actions[-1]
            if isinstance(last_action, dict) and last_action.get("semantic_delta") in (None, "", "pending"):
                last_action["semantic_delta"] = progress_name

    @classmethod
    def _apply_replan_observation(
        cls,
        state: Dict[str, Any],
        progress: Dict[str, Any],
        *,
        recovered: bool,
    ) -> None:
        if str(state.get("status") or "").strip().lower() in _TERMINAL_TASK_STATUSES:
            return
        if recovered:
            cls.mark_replan_recovered(state)
        elif state.get("replan_trial_pending"):
            trial_strategy = str(state.get("trial_strategy") or "")
            cls.record_failed_strategy(state, trial_strategy)
            state["replan_trial_pending"] = False
            state["replan_required"] = True
            state["blocked_strategy"] = trial_strategy
            state["status"] = "replan_required"
            state["next_action_class"] = "materially_different_strategy"
        elif progress.get("replan_required"):
            state["replan_required"] = True
            state["status"] = "replan_required"
            state["blocked_strategy"] = str(
                state.get("trial_strategy")
                or state.get("last_strategy_fingerprint")
                or state.get("last_action_class")
                or ""
            )
            cls.record_failed_strategy(state, str(state.get("blocked_strategy") or ""))
            state["next_action_class"] = "materially_different_strategy"

    @staticmethod
    def record_failed_strategy(state: Dict[str, Any], strategy: str) -> None:
        """Remember one action strategy that failed to produce semantic progress."""
        failed_strategies = state.setdefault("failed_strategies", [])
        if strategy and strategy not in failed_strategies:
            failed_strategies.append(strategy)

    @staticmethod
    def mark_replan_recovered(state: Dict[str, Any]) -> None:
        """Clear a task-level replan gate after verified semantic progress."""

        if str(state.get("status") or "").strip().lower() in _TERMINAL_TASK_STATUSES:
            return
        state["replan_required"] = False
        state["replan_trial_pending"] = False
        state["blocked_strategy"] = ""
        state["trial_strategy"] = ""
        state["replan_denial_count"] = 0
        state["status"] = "in_progress"
        state["next_action_class"] = ""

    @staticmethod
    def _project_task_state(state: Dict[str, Any]) -> Dict[str, Any]:
        phases = state.get("phases") if isinstance(state.get("phases"), dict) else {}
        compact_phases = {
            str(name): {
                "status": details.get("status"),
                "attempts": int(details.get("attempts") or 0),
                "budget": int(details.get("budget") or 0),
                "completion_condition": _bounded_text(details.get("completion_condition"), 240),
            }
            for name, details in phases.items()
            if isinstance(details, dict)
        }
        semantic_progress = state.get("semantic_progress")
        compact_semantic: Dict[str, Any] = {}
        if isinstance(semantic_progress, dict):
            semantic_keys = (
                "progress",
                "consecutive_no_progress",
                "state_revisit_count",
                "aba_loop",
                "repeated_filter_state",
                "replan_reason",
            )
            for key in semantic_keys:
                if key in semantic_progress:
                    compact_semantic[key] = semantic_progress.get(key)
        return {
            "task_id": state.get("task_id"),
            "goal": _bounded_text(state.get("goal") or state.get("task"), 1_000),
            "task_type": state.get("task_type"),
            "status": state.get("status", "in_progress"),
            "current_phase": state.get("current_phase"),
            "phases": compact_phases,
            "required_fields": list(state.get("required_fields") or [])[:32],
            "field_coverage": list(state.get("field_coverage") or [])[:32],
            "required_evidence_slots": [
                dict(slot) for slot in (state.get("required_evidence_slots") or [])[:12] if isinstance(slot, dict)
            ],
            "evidence_slots": [
                dict(slot) for slot in (state.get("evidence_slots") or [])[-12:] if isinstance(slot, dict)
            ],
            "blockers": list(state.get("blockers") or [])[:8],
            "replan_required": bool(state.get("replan_required")),
            "replan_count": int(state.get("replan_count") or 0),
            "failed_strategies": list(state.get("failed_strategies") or [])[:8],
            "next_action_class": state.get("next_action_class"),
            "semantic_progress": compact_semantic,
            "structured_evidence": BrowserWorkingContextStore.compact_evidence(state.get("structured_evidence")),
            "last_page": dict(state.get("last_page") or {}),
        }

    @staticmethod
    def compact_evidence(value: Any) -> list[Dict[str, Any]]:
        """Project structured evidence into a bounded model-facing form."""
        records = value if isinstance(value, list) else []
        compact_records: list[Dict[str, Any]] = []
        for record in records[-5:]:
            if not isinstance(record, dict):
                continue
            compact: Dict[str, Any] = {
                "kind": record.get("kind"),
                "generation_id": record.get("generation_id"),
                "fields": list(record.get("fields") or [])[:20],
            }
            values = record.get("values")
            if isinstance(values, dict):
                compact["values"] = {str(key): _bounded_text(item, 160) for key, item in list(values.items())[:12]}
            cards = record.get("cards")
            if isinstance(cards, list):
                compact["cards"] = [dict(card) for card in cards[:3] if isinstance(card, dict)]
            for key in ("preview", "target_count"):
                if record.get(key) not in (None, ""):
                    compact[key] = _bounded_text(record.get(key), 800)
            provenance = record.get("provenance")
            if isinstance(provenance, dict):
                compact["provenance"] = dict(list(provenance.items())[:8])
            compact_records.append(compact)
        return compact_records

    @staticmethod
    def _runtime_directive(state: Dict[str, Any]) -> str:
        status = str(state.get("status") or "in_progress")
        if status == "completed":
            return "must_finish"
        if status in {"blocked", "partial"}:
            return "return_partial_or_blocked"
        if state.get("replan_required"):
            return "replan_before_browser_action"
        return "continue"

    def _apply_action_assessments(
        self,
        state: BrowserWorkingContextState,
        assessments: list[BrowserActionAssessment],
        *,
        contract_valid: bool,
    ) -> Optional[str]:
        expected = list(state.actions_requiring_assessment)
        if not expected and not assessments:
            return None
        if not contract_valid or not expected or len(assessments) != 1:
            return (
                "Action assessments must contain exactly one assessment for the preceding "
                f"tool batch. Assessment required: {bool(expected)}. "
                f"Provided assessments: {len(assessments)}."
            )

        assessment = assessments[0]
        status = assessment.status
        seen_signatures: set[str] = set()
        for action in expected:
            if action.action_signature in seen_signatures:
                continue
            seen_signatures.add(action.action_signature)
            failed = self._failed_action(state, action.action_signature)
            if status == "not_achieved":
                if failed is None:
                    state.failed_actions.append(
                        BrowserFailedAction(
                            action_signature=action.action_signature,
                            tool_name=action.tool_name,
                            intent=action.intent,
                            expected_outcome=action.expected_outcome,
                            last_reason=_bounded_text(assessment.reason, self.config.max_item_chars),
                        )
                    )
                else:
                    failed.failure_count += 1
                    failed.tool_name = action.tool_name
                    failed.intent = action.intent
                    failed.expected_outcome = action.expected_outcome
                    failed.last_reason = _bounded_text(assessment.reason, self.config.max_item_chars)
            elif status == "achieved" and failed is not None:
                state.failed_actions.remove(failed)
        state.actions_requiring_assessment = []
        if len(state.failed_actions) > self.config.max_list_items:
            state.failed_actions = state.failed_actions[-self.config.max_list_items :]
        return None

    @staticmethod
    def _validate_batch_intent(
        plans: list[BrowserPlannedAction],
        tool_call_count: int,
        *,
        contract_valid: bool,
    ) -> tuple[Optional[BrowserPlannedAction], Optional[str]]:
        if tool_call_count == 0 and not plans:
            return None, None
        if contract_valid and tool_call_count > 0 and len(plans) == 1:
            return plans[0], None
        return None, (
            "Batch intent must contain exactly one shared declaration for the whole "
            f"emitted tool batch. Tool calls: {tool_call_count}. "
            f"Provided plans: {len(plans)}."
        )

    @staticmethod
    def _failed_action(
        state: BrowserWorkingContextState,
        signature: str,
    ) -> Optional[BrowserFailedAction]:
        return next(
            (action for action in reversed(state.failed_actions) if action.action_signature == signature),
            None,
        )

    def _sanitize_list(self, values: Iterable[Any]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in list(values)[: self.config.max_list_items]:
            text = _bounded_text(value, self.config.max_item_chars)
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    def _commit_pending(
        self,
        state: BrowserWorkingContextState,
        tool_memories: list[BrowserToolMemory],
    ) -> None:
        pending = state.pending_step
        if pending is None:
            return
        prompt_memories = [memory for memory in tool_memories if memory.has_prompt_content()]
        durable_memories = []
        for memory in prompt_memories:
            durable = memory.model_copy(update={"one_step_content": None})
            if durable.has_prompt_content():
                durable_memories.append(durable)
        if pending.model_memory is not None or pending.model_update_error or durable_memories:
            state.recent_steps.append(
                BrowserStepRecord(
                    step_number=pending.step_number,
                    model_memory=pending.model_memory,
                    model_update_error=pending.model_update_error,
                    tool_memories=durable_memories,
                )
            )
        if pending.model_memory is not None:
            state.current = pending.model_memory
        state.actions_requiring_assessment.extend(
            BrowserPendingAssessment(
                tool_call_id=action.tool_call_id,
                tool_name=action.tool_name,
                action_signature=action.action_signature,
                intent=action.intent,
                expected_outcome=action.expected_outcome,
            )
            for action in pending.tool_calls
            if action.blocked_reason is None
        )
        state.one_step_content = [memory for memory in prompt_memories if memory.one_step_content]
        state.next_step_number = max(state.next_step_number, pending.step_number + 1)
        state.pending_step = None
        self._limit_history(state)

    def _commit_pending_as_incomplete(self, state: BrowserWorkingContextState) -> None:
        pending = state.pending_step
        if pending is None:
            return
        pending.model_update_error = _bounded_text(
            (pending.model_update_error or "") + " Previous model step ended before all tool results were committed.",
            self.config.max_item_chars,
        )
        self._commit_pending(state, [])

    def _limit_history(self, state: BrowserWorkingContextState) -> None:
        if len(state.recent_steps) > self.config.max_recent_steps:
            state.recent_steps = state.recent_steps[-self.config.max_recent_steps :]

    def _infer_tool_error(self, content: Any) -> Optional[str]:
        text = self.message_content_to_text(content)
        if not text:
            return None
        lowered = text.lower().strip()
        if lowered.startswith(_ERROR_PREFIXES):
            return _bounded_text(text, self.config.max_item_chars)
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if error and (parsed.get("success") is False or parsed.get("ok") is False or parsed.get("isError") is True):
                return _bounded_text(error, self.config.max_item_chars)
        if "success=false" in lowered and "error=" in lowered:
            return _bounded_text(text, self.config.max_item_chars)
        return None

    @staticmethod
    def message_content_to_text(content: Any) -> str:
        """Flatten supported message content into prompt-safe text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
        return str(content or "")


__all__ = [
    "BROWSER_TOOL_MEMORY_METADATA_KEY",
    "BROWSER_TASK_STATE_KEY",
    "BROWSER_WORKING_MEMORY_RECORD_BEGIN",
    "BROWSER_WORKING_MEMORY_RECORD_END",
    "BROWSER_WORKING_CONTEXT_STATE_KEY",
    "BrowserActionAssessment",
    "BrowserFailedAction",
    "BrowserModelStepUpdate",
    "BrowserPendingAssessment",
    "BrowserPendingStep",
    "BrowserPlannedAction",
    "BrowserStepRecord",
    "BrowserToolMemory",
    "BrowserWorkingContextConfig",
    "BrowserWorkingContextState",
    "BrowserWorkingContextStore",
    "BrowserWorkingMemory",
    "latest_browser_user_request",
]


def latest_browser_user_request(messages: Iterable[BaseMessage]) -> str:
    """Return the newest real user request, excluding ephemeral browser context."""

    for message in reversed(list(messages)):
        if not isinstance(message, UserMessage):
            continue
        if message.name in _EPHEMERAL_USER_MESSAGE_NAMES:
            continue
        metadata = getattr(message, "metadata", {}) or {}
        is_ephemeral_context = False
        for key in _EPHEMERAL_CONTEXT_METADATA_KEYS:
            if metadata.get(key):
                is_ephemeral_context = True
                break
        if is_ephemeral_context:
            continue
        text = BrowserWorkingContextStore.message_content_to_text(message.content).strip()
        if text:
            return text
    return ""
