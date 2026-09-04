#!/usr/bin/env python
# coding: utf-8

"""Focused lifecycle tests for durable browser-agent working context."""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from openjiuwen.core.context_engine import ContextEngine, ContextWindow
from openjiuwen.core.foundation.llm import AssistantMessage, ToolMessage, UserMessage
from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
from openjiuwen.core.runner.callback.errors import AbortError
from openjiuwen.core.session.agent import create_agent_session
from openjiuwen.core.single_agent.rail.base import (
    AgentCallbackContext,
    AgentCallbackEvent,
    InvokeInputs,
    ModelCallInputs,
    ToolCallInputs,
)
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.harness.tools.base_tool import ToolOutput
from openjiuwen.harness.tools.browser_move.playwright_runtime.browser_working_context import (
    BROWSER_TASK_STATE_KEY,
    BROWSER_TOOL_MEMORY_METADATA_KEY,
    BROWSER_WORKING_MEMORY_RECORD_BEGIN,
    BROWSER_WORKING_MEMORY_RECORD_END,
    BrowserWorkingContextStore,
)
from openjiuwen.harness.tools.browser_move.playwright_runtime.browser_working_context_processor import (
    BrowserWorkingContextProcessor,
    BrowserWorkingContextProcessorConfig,
)
from openjiuwen.harness.tools.browser_move.playwright_runtime.browser_working_context_rail import (
    BrowserWorkingContextRail,
)


class _FakeSession:
    def __init__(self, session_id: str = "browser-working-context") -> None:
        self._session_id = session_id
        self._state: dict[str, Any] = {}

    def get_session_id(self) -> str:
        return self._session_id

    def get_state(self, key: str):
        return self._state.get(key)

    def update_state(self, payload: dict[str, Any]) -> None:
        self._state.update(payload)


class _FakeContext:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.messages: list[Any] = []

    def get_session_ref(self) -> _FakeSession:
        return self.session

    def get_messages(self) -> list[Any]:
        return self.messages


def _run(coro):
    return asyncio.run(coro)


def _memory(
    task: str,
    status: str = "pending",
    **fields: list[str],
) -> dict[str, Any]:
    del task, status
    return {
        "add_failures": fields.get("failures", []),
        "add_key_facts": fields.get("key_facts", []),
    }


def _response(
    memory: dict[str, Any],
    *,
    tool_calls: list[ToolCall] | None = None,
    batch_intent: list[dict[str, Any]] | None = None,
    action_assessments: list[dict[str, Any]] | None = None,
    visible_text: str = "",
) -> AssistantMessage:
    calls = tool_calls or []
    if batch_intent is None:
        batch_intent = (
            [
                {
                    "intent": f"Execute browser tool batch: {', '.join(call.name for call in calls)}",
                    "expected_outcome": "The browser exposes the requested result",
                }
            ]
            if calls
            else []
        )
    record = {key: value for key, value in memory.items() if value}
    if batch_intent:
        record["batch_intent"] = batch_intent[0] if len(batch_intent) == 1 else batch_intent
    if action_assessments:
        record["action_assessment"] = action_assessments[0] if len(action_assessments) == 1 else action_assessments
    content = (
        f"{visible_text}\n{BROWSER_WORKING_MEMORY_RECORD_BEGIN}\n"
        f"{json.dumps(record)}\n{BROWSER_WORKING_MEMORY_RECORD_END}"
    )
    return AssistantMessage(content=content, tool_calls=tool_calls)


def _model_ctx(
    rail: BrowserWorkingContextRail,
    session: _FakeSession,
    context: _FakeContext,
    response: AssistantMessage,
) -> AgentCallbackContext:
    del rail
    return AgentCallbackContext(
        agent=None,
        inputs=ModelCallInputs(response=response),
        session=session,
        context=context,
    )


def _tool_call(call_id: str, name: str, arguments: str = "{}") -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        name=name,
        arguments=arguments,
    )


def _record_tool_result(
    rail: BrowserWorkingContextRail,
    session: _FakeSession,
    context: _FakeContext,
    tool_call: ToolCall,
    tool_result: ToolOutput,
    *,
    raw_content: str,
) -> ToolMessage:
    tool_message = ToolMessage(
        content=raw_content,
        tool_call_id=str(tool_call.id),
    )
    ctx = AgentCallbackContext(
        agent=None,
        inputs=ToolCallInputs(
            tool_call=tool_call,
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
            tool_result=tool_result,
            tool_msg=tool_message,
        ),
        session=session,
        context=context,
    )
    _run(rail.after_tool_call(ctx))
    context.messages.append(tool_message)
    return tool_message


def _tool_ctx(
    session: _FakeSession,
    context: _FakeContext,
    tool_call: ToolCall,
) -> AgentCallbackContext:
    return AgentCallbackContext(
        agent=None,
        inputs=ToolCallInputs(
            tool_call=tool_call,
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
        ),
        session=session,
        context=context,
    )


def _inject(
    processor: BrowserWorkingContextProcessor,
    context: _FakeContext,
) -> ContextWindow:
    window = ContextWindow(context_messages=list(context.messages))
    _, rendered = _run(processor.on_get_context_window(context, window))
    return rendered


def test_model_memory_survives_and_internal_update_is_not_user_facing() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    assert AgentCallbackEvent.AFTER_REACT_ITERATION in rail.get_callbacks()
    session = _FakeSession()
    context = _FakeContext(session)
    response = _response(
        _memory(
            "Collect the account status",
            key_facts=["The account page is reachable."],
        ),
        visible_text="Continuing.",
    )

    _run(rail.after_model_call(_model_ctx(rail, session, context, response)))

    assert response.content == "Continuing."
    state = BrowserWorkingContextStore(config).load(session)
    assert len(state.recent_steps) == 1
    assert state.current.key_facts == ["The account page is reachable."]

    processor = BrowserWorkingContextProcessor(config)
    prompt = _inject(processor, context).context_messages[-1].content
    assert "The account page is reachable." in prompt


def test_record_can_be_omitted_when_no_contract_decision_is_required() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    valid_response = _response(_memory("Keep this confirmed task", key_facts=["The confirmed state is retained."]))
    _run(rail.after_model_call(_model_ctx(rail, session, context, valid_response)))

    missing_update = AssistantMessage(content="Visible answer without internal state")
    _run(rail.after_model_call(_model_ctx(rail, session, context, missing_update)))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.current.key_facts == ["The confirmed state is retained."]
    assert state.recent_steps[-1].model_memory == state.current
    assert state.recent_steps[-1].model_update_error is None
    assert missing_update.content == "Visible answer without internal state"


def test_tool_call_without_record_warns_and_remains_assessable() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    _run(
        rail.before_invoke(
            AgentCallbackContext(
                agent=None,
                inputs=InvokeInputs(query="Inspect the checkout flow"),
                session=session,
            )
        )
    )
    tool_call = _tool_call("call-1", "browser_navigate")
    response = AssistantMessage(content="", tool_calls=[tool_call])
    model_ctx = _model_ctx(rail, session, context, response)

    _run(rail.after_model_call(model_ctx))

    pending_state = BrowserWorkingContextStore(config).load(session)
    assert pending_state.pending_step is not None
    assert "omitted the required working-memory delta" in (pending_state.pending_step.model_update_error or "")
    assert pending_state.pending_step.model_memory == pending_state.current
    warning_reason = pending_state.pending_step.tool_calls[0].warning_reason or ""
    assert "exactly one shared declaration" in warning_reason
    assert pending_state.pending_step.tool_calls[0].blocked_reason is None
    assert pending_state.current.failures == []
    assert response.content == ""

    tool_ctx = AgentCallbackContext(
        agent=None,
        inputs=ToolCallInputs(
            tool_call=tool_call,
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
        ),
        session=session,
        context=context,
    )
    with patch(
        "openjiuwen.harness.tools.browser_move.playwright_runtime."
        "browser_working_context_rail.browser_agent_log_warning"
    ) as log_warning:
        _run(rail.before_tool_call(tool_ctx))
    log_warning.assert_called_once()
    assert warning_reason in log_warning.call_args.args

    context.messages.append(ToolMessage(content="navigation complete", tool_call_id="call-1"))
    _run(rail.after_react_iteration(model_ctx))
    committed_state = BrowserWorkingContextStore(config).load(session)
    assert [action.tool_call_id for action in committed_state.actions_requiring_assessment] == ["call-1"]


@pytest.mark.parametrize(
    "plans",
    [
        [],
        [
            {
                "intent": "First declaration",
                "expected_outcome": "A result appears",
            },
            {
                "intent": "Duplicate declaration",
                "expected_outcome": "Another result appears",
            },
        ],
    ],
)
def test_batch_intent_requires_exactly_one_declaration_for_tool_batch(
    plans: list[dict[str, Any]],
) -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    tool_call = _tool_call("call-plan", "browser_navigate")
    response = _response(
        _memory("Open the page"),
        tool_calls=[tool_call],
        batch_intent=plans,
    )

    _run(rail.after_model_call(_model_ctx(rail, session, context, response)))

    state = BrowserWorkingContextStore(config).load(session)
    error = state.pending_step.model_update_error or ""
    assert "exactly one shared declaration" in error
    assert "Tool calls: 1." in error
    assert "Provided plans: 0." in error
    if len(plans) > 1:
        assert "Invalid fields: batch_intent." in error
    pending_call = state.pending_step.tool_calls[0]
    warning_reason = pending_call.warning_reason or ""
    assert "exactly one shared declaration" in warning_reason
    assert pending_call.blocked_reason is None
    with patch(
        "openjiuwen.harness.tools.browser_move.playwright_runtime."
        "browser_working_context_rail.browser_agent_log_warning"
    ) as log_warning:
        _run(rail.before_tool_call(_tool_ctx(session, context, tool_call)))
    log_warning.assert_called_once()
    assert warning_reason in log_warning.call_args.args


def test_following_model_call_assesses_preceding_tool_batch_once() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    tool_call = _tool_call("call-assess", "browser_snapshot")
    first_response = _response(
        _memory("Inspect the page"),
        tool_calls=[tool_call],
    )
    first_ctx = _model_ctx(rail, session, context, first_response)
    _run(rail.after_model_call(first_ctx))
    context.messages.append(ToolMessage(content="snapshot complete", tool_call_id="call-assess"))
    _run(rail.after_react_iteration(first_ctx))

    missing_assessment = _response(_memory("Inspect the page"))
    _run(rail.after_model_call(_model_ctx(rail, session, context, missing_assessment)))
    state = BrowserWorkingContextStore(config).load(session)
    assert [action.tool_call_id for action in state.actions_requiring_assessment] == ["call-assess"]
    error = state.recent_steps[-1].model_update_error or ""
    assert "exactly one assessment for the preceding tool batch" in error
    assert "Assessment required: True." in error
    assert "Provided assessments: 0." in error

    valid_assessment = _response(
        _memory("Inspect the page", status="completed"),
        action_assessments=[
            {
                "status": "achieved",
                "reason": "The requested snapshot is available",
            }
        ],
    )
    _run(rail.after_model_call(_model_ctx(rail, session, context, valid_assessment)))
    assert BrowserWorkingContextStore(config).load(session).actions_requiring_assessment == []


@pytest.mark.parametrize("recovery_allowed", [False, True])
def test_failed_equivalent_action_requires_recovery_justification(
    recovery_allowed: bool,
) -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    first_call = _tool_call(
        "call-first",
        "browser_scroll",
        '{"x":0,"y":1400}',
    )
    first_response = _response(
        _memory("Reveal more results"),
        tool_calls=[first_call],
    )
    first_ctx = _model_ctx(rail, session, context, first_response)
    _run(rail.after_model_call(first_ctx))
    context.messages.append(ToolMessage(content="no new results", tool_call_id="call-first"))
    _run(rail.after_react_iteration(first_ctx))

    retry_call = _tool_call(
        "call-retry",
        "browser_scroll",
        '{ "y": 1400, "x": 0 }',
    )
    retry_plan = {
        "intent": "Reveal more results",
        "expected_outcome": "A previously unseen result appears",
    }
    if recovery_allowed:
        retry_plan["recovery_justification"] = "Retry after the virtualized list finished rendering"
    retry_response = _response(
        _memory("Reveal more results"),
        tool_calls=[retry_call],
        batch_intent=[retry_plan],
        action_assessments=[
            {
                "status": "not_achieved",
                "reason": "No previously unseen result appeared",
            }
        ],
    )
    _run(rail.after_model_call(_model_ctx(rail, session, context, retry_response)))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.failed_actions[0].failure_count == 1
    prompt = BrowserWorkingContextStore(config).render_and_consume_one_step(session)
    assert '"loop_guard"' in prompt
    assert '"failure_count": 1' in prompt
    assert "Execute browser tool batch: browser_scroll" in prompt
    assert "No previously unseen result appeared" in prompt
    assert state.failed_actions[0].action_signature not in prompt
    if recovery_allowed:
        assert state.pending_step.tool_calls[0].blocked_reason is None
        _run(rail.before_tool_call(_tool_ctx(session, context, retry_call)))
    else:
        assert "recovery_justification" in (state.pending_step.tool_calls[0].blocked_reason or "")
        with pytest.raises(AbortError):
            _run(rail.before_tool_call(_tool_ctx(session, context, retry_call)))


def test_equivalent_action_that_failed_twice_remains_blocked() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    first_call = _tool_call("call-first", "browser_scroll", '{"x":0,"y":1400}')
    first_response = _response(_memory("Reveal more results"), tool_calls=[first_call])
    first_ctx = _model_ctx(rail, session, context, first_response)
    _run(rail.after_model_call(first_ctx))
    context.messages.append(ToolMessage(content="no new results", tool_call_id="call-first"))
    _run(rail.after_react_iteration(first_ctx))

    second_call = _tool_call("call-second", "browser_scroll", '{"y":1400,"x":0}')
    second_response = _response(
        _memory("Reveal more results"),
        tool_calls=[second_call],
        batch_intent=[
            {
                "intent": "Retry after the list settles",
                "expected_outcome": "A new result appears",
                "recovery_justification": "The list was still rendering",
            }
        ],
        action_assessments=[{"status": "not_achieved", "reason": "No new result appeared"}],
    )
    second_ctx = _model_ctx(rail, session, context, second_response)
    _run(rail.after_model_call(second_ctx))
    _run(rail.before_tool_call(_tool_ctx(session, context, second_call)))
    context.messages.append(ToolMessage(content="still no new results", tool_call_id="call-second"))
    _run(rail.after_react_iteration(second_ctx))

    third_call = _tool_call("call-third", "browser_scroll", '{"x":0,"y":1400}')
    third_response = _response(
        _memory("Reveal more results"),
        tool_calls=[third_call],
        batch_intent=[
            {
                "intent": "Try the same scroll again",
                "expected_outcome": "A new result appears",
                "recovery_justification": "Waited longer before retrying",
            }
        ],
        action_assessments=[{"status": "not_achieved", "reason": "Still no new result appeared"}],
    )
    _run(rail.after_model_call(_model_ctx(rail, session, context, third_response)))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.failed_actions[0].failure_count == 2
    assert "failed twice" in (state.pending_step.tool_calls[0].blocked_reason or "")
    with pytest.raises(AbortError):
        _run(rail.before_tool_call(_tool_ctx(session, context, third_call)))


def test_action_signature_is_canonical_and_does_not_persist_raw_arguments() -> None:
    config = BrowserWorkingContextProcessorConfig()
    store = BrowserWorkingContextStore(config)
    first = store.action_signature(
        "browser_navigate",
        '{"token":"sensitive-value","page":2}',
    )
    second = store.action_signature(
        "browser_navigate",
        '{ "page": 2, "token": "sensitive-value" }',
    )

    assert first == second
    assert "sensitive-value" not in first


def test_rail_carries_memory_without_a_delta_when_no_decision_is_required() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    response = AssistantMessage(content="Inspecting checkout now.")
    model_ctx = AgentCallbackContext(
        agent=None,
        inputs=ModelCallInputs(
            messages=[UserMessage(content="Inspect the checkout flow")],
            response=response,
        ),
        session=session,
        context=context,
    )

    _run(rail.after_model_call(model_ctx))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.active_request == ""
    assert state.current.model_dump() == {"failures": [], "key_facts": []}
    assert state.recent_steps[-1].model_memory == state.current
    assert state.recent_steps[-1].model_update_error is None
    assert response.content == "Inspecting checkout now."


def test_rail_rejects_legacy_full_replacement_fields() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    incomplete_record = {
        "failures": ["The checkout button did not respond."],
    }
    response = AssistantMessage(
        content=(
            f"{BROWSER_WORKING_MEMORY_RECORD_BEGIN}\n"
            f"{json.dumps(incomplete_record)}\n"
            f"{BROWSER_WORKING_MEMORY_RECORD_END}"
        )
    )
    model_ctx = AgentCallbackContext(
        agent=None,
        inputs=ModelCallInputs(
            messages=[UserMessage(content="Inspect checkout")],
            response=response,
        ),
        session=session,
        context=context,
    )

    _run(rail.after_model_call(model_ctx))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.current.model_dump() == {"failures": [], "key_facts": []}
    assert state.recent_steps[-1].model_update_error == (
        "Model working-memory delta failed validation. Invalid fields: failures."
    )
    assert response.content == ""


def test_model_record_rejects_fields_outside_reduced_contract() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    response = _response(
        {
            **_memory(
                "Inspect checkout",
                failures=["The first checkout attempt failed."],
                key_facts=["The cart contains one item."],
            ),
            "task_list": [{"task": "Legacy task", "status": "pending"}],
        }
    )

    _run(rail.after_model_call(_model_ctx(rail, session, context, response)))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.current.model_dump() == {
        "failures": ["The first checkout attempt failed."],
        "key_facts": ["The cart contains one item."],
    }
    assert state.recent_steps[-1].model_update_error == (
        "Model working-memory delta failed validation. Invalid fields: task_list."
    )


def test_invalid_durable_field_carries_forward_only_that_field() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    initial_response = _response(
        _memory(
            "Inspect checkout",
            failures=["Preserve this known failure."],
            key_facts=["The old fact."],
        )
    )
    _run(rail.after_model_call(_model_ctx(rail, session, context, initial_response)))

    invalid_record = {
        "add_failures": ["Valid entry", {"invalid": "entry"}],
        "add_key_facts": ["The updated fact."],
    }
    response = AssistantMessage(
        content=(
            f"{BROWSER_WORKING_MEMORY_RECORD_BEGIN}\n{json.dumps(invalid_record)}\n{BROWSER_WORKING_MEMORY_RECORD_END}"
        )
    )
    _run(rail.after_model_call(_model_ctx(rail, session, context, response)))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.current.failures == ["Preserve this known failure."]
    assert state.current.key_facts == ["The old fact.", "The updated fact."]
    assert "Invalid fields: add_failures.1." in (state.recent_steps[-1].model_update_error or "")


def test_optional_memory_delta_fields_can_be_omitted_from_a_valid_tool_batch() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    tool_call = _tool_call("call-partial-memory", "browser_navigate")
    record = {
        "add_failures": ["The search form did not submit."],
        "batch_intent": {
            "intent": "Navigate directly to the results page",
            "expected_outcome": "The results page becomes visible",
        },
    }
    response = AssistantMessage(
        content=(f"{BROWSER_WORKING_MEMORY_RECORD_BEGIN}\n{json.dumps(record)}\n{BROWSER_WORKING_MEMORY_RECORD_END}"),
        tool_calls=[tool_call],
    )

    _run(rail.after_model_call(_model_ctx(rail, session, context, response)))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.pending_step.model_memory.failures == ["The search form did not submit."]
    assert state.pending_step.tool_calls[0].blocked_reason is None
    assert state.pending_step.model_update_error is None
    _run(rail.before_tool_call(_tool_ctx(session, context, tool_call)))


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("unstaged", "no staged working-context action declaration"),
        ("unmatched", "could not be matched to exactly one declared planned action"),
        ("mutated", "name or arguments changed"),
    ],
)
def test_tool_call_integrity_violations_remain_blocking(
    mode: str,
    expected_error: str,
) -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    declared_call = _tool_call("call-declared", "browser_navigate", '{"url":"https://example.com"}')

    if mode != "unstaged":
        response = _response(_memory("Open the page"), tool_calls=[declared_call])
        _run(rail.after_model_call(_model_ctx(rail, session, context, response)))

    if mode == "unstaged":
        actual_call = declared_call
    elif mode == "unmatched":
        actual_call = _tool_call("call-other", "browser_click", '{"ref":"button-1"}')
    else:
        actual_call = _tool_call("call-declared", "browser_navigate", '{"url":"https://changed.example"}')

    with pytest.raises(AbortError) as exc_info:
        _run(rail.before_tool_call(_tool_ctx(session, context, actual_call)))
    assert expected_error in str(exc_info.value.cause)


def test_invalid_action_plan_warns_without_discarding_or_polluting_memory() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    tool_call = _tool_call("call-invalid-plan", "browser_navigate")
    response = _response(
        _memory(
            "Inspect checkout",
            failures=["The form submission failed."],
            key_facts=["The cart remains open."],
        ),
        tool_calls=[tool_call],
        batch_intent=[
            {
                "intent": "",
                "expected_outcome": "The checkout page becomes visible",
            }
        ],
    )

    _run(rail.after_model_call(_model_ctx(rail, session, context, response)))

    state = BrowserWorkingContextStore(config).load(session)
    assert "Invalid fields: batch_intent.intent." in (state.pending_step.model_update_error or "")
    pending_call = state.pending_step.tool_calls[0]
    warning_reason = pending_call.warning_reason or ""
    assert "exactly one shared declaration" in warning_reason
    assert pending_call.blocked_reason is None
    assert state.pending_step.model_memory.failures == ["The form submission failed."]
    assert state.pending_step.model_memory.key_facts == ["The cart remains open."]
    assert state.current.failures == []
    assert state.current.key_facts == []
    assert warning_reason not in BrowserWorkingContextStore(config).render_and_consume_one_step(session)
    with patch(
        "openjiuwen.harness.tools.browser_move.playwright_runtime."
        "browser_working_context_rail.browser_agent_log_warning"
    ) as log_warning:
        _run(rail.before_tool_call(_tool_ctx(session, context, tool_call)))
    log_warning.assert_called_once()
    assert warning_reason in log_warning.call_args.args
    context.messages.append(ToolMessage(content="checkout opened", tool_call_id="call-invalid-plan"))
    _run(rail.after_react_iteration(_model_ctx(rail, session, context, response)))
    committed_state = BrowserWorkingContextStore(config).load(session)
    assert committed_state.current.failures == ["The form submission failed."]
    assert committed_state.current.key_facts == ["The cart remains open."]
    assert warning_reason not in committed_state.current.failures


@pytest.mark.parametrize(
    "action_assessments",
    [
        [],
        [{"status": "invalid-status", "reason": "The required control was absent"}],
    ],
)
def test_missing_or_invalid_assessment_warns_and_tracks_only_the_next_batch(
    action_assessments: list[dict[str, Any]],
) -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    first_call = _tool_call("call-first-batch", "browser_snapshot")
    first_response = _response(_memory("Inspect checkout"), tool_calls=[first_call])
    first_ctx = _model_ctx(rail, session, context, first_response)
    _run(rail.after_model_call(first_ctx))
    context.messages.append(ToolMessage(content="snapshot complete", tool_call_id="call-first-batch"))
    _run(rail.after_react_iteration(first_ctx))

    next_call = _tool_call("call-next-batch", "browser_navigate")
    response = _response(
        _memory("Inspect checkout", failures=["The snapshot did not show the requested control."]),
        tool_calls=[next_call],
        batch_intent=[
            {
                "intent": "Open the checkout page directly",
                "expected_outcome": "The checkout form becomes visible",
            }
        ],
        action_assessments=action_assessments,
    )

    next_ctx = _model_ctx(rail, session, context, response)
    _run(rail.after_model_call(next_ctx))

    state = BrowserWorkingContextStore(config).load(session)
    pending_call = state.pending_step.tool_calls[0]
    assert pending_call.intent == "Open the checkout page directly"
    warning_reason = pending_call.warning_reason or ""
    assert "exactly one assessment for the preceding tool batch" in warning_reason
    assert pending_call.blocked_reason is None
    assert state.pending_step.model_memory.failures == ["The snapshot did not show the requested control."]
    assert state.current.failures == []
    assert state.actions_requiring_assessment == []

    with patch(
        "openjiuwen.harness.tools.browser_move.playwright_runtime."
        "browser_working_context_rail.browser_agent_log_warning"
    ) as log_warning:
        _run(rail.before_tool_call(_tool_ctx(session, context, next_call)))
    log_warning.assert_called_once()
    assert warning_reason in log_warning.call_args.args
    context.messages.append(ToolMessage(content="checkout opened", tool_call_id="call-next-batch"))
    _run(rail.after_react_iteration(next_ctx))

    committed_state = BrowserWorkingContextStore(config).load(session)
    assert committed_state.current.failures == ["The snapshot did not show the requested control."]
    assert warning_reason not in committed_state.current.failures
    assert [action.tool_call_id for action in committed_state.actions_requiring_assessment] == ["call-next-batch"]


def test_one_model_record_aggregates_multiple_tool_results() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    calls = [
        _tool_call("call-1", "browser_navigate"),
        _tool_call("call-2", "browser_probe_cards"),
    ]
    response = _response(_memory("Inspect products"), tool_calls=calls)

    model_ctx = _model_ctx(rail, session, context, response)
    _run(rail.after_model_call(model_ctx))
    state = BrowserWorkingContextStore(config).load(session)
    assert state.recent_steps == []
    assert state.pending_step is not None
    assert {item.intent for item in state.pending_step.tool_calls} == {
        "Execute browser tool batch: browser_navigate, browser_probe_cards"
    }
    assert {item.expected_outcome for item in state.pending_step.tool_calls} == {
        "The browser exposes the requested result"
    }

    _record_tool_result(
        rail,
        session,
        context,
        calls[0],
        ToolOutput(
            success=True,
            long_term_memory="Navigation reached the product page.",
        ),
        raw_content="complete navigation result",
    )
    _record_tool_result(
        rail,
        session,
        context,
        calls[1],
        ToolOutput(
            success=True,
            long_term_memory="Three product cards were identified.",
        ),
        raw_content="complete card inventory",
    )
    _run(rail.after_react_iteration(model_ctx))

    state = BrowserWorkingContextStore(config).load(session)
    assert len(state.recent_steps) == 1
    assert [item.tool_name for item in state.recent_steps[0].tool_memories] == [
        "browser_navigate",
        "browser_probe_cards",
    ]
    assert len(state.actions_requiring_assessment) == 2
    prompt = _inject(BrowserWorkingContextProcessor(config), context).context_messages[-1].content
    assert '"required_now": [' in prompt
    assert '"action_assessment"' in prompt
    assert '"previous_batch"' in prompt
    assert '"tool_call_count": 2' in prompt
    assert "Execute browser tool batch: browser_navigate, browser_probe_cards" in prompt

    assessment_response = _response(
        _memory("Inspect products", key_facts=["Three product cards are available."]),
        action_assessments=[
            {
                "status": "achieved",
                "reason": "The whole navigation-and-probe batch produced the requested products",
            }
        ],
    )
    _run(rail.after_model_call(_model_ctx(rail, session, context, assessment_response)))
    assert BrowserWorkingContextStore(config).load(session).actions_requiring_assessment == []


def test_long_term_memory_precedes_extracted_content_and_survives_later_steps() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    tool_call = _tool_call("call-1", "browser_probe_cards")
    first_response = _response(_memory("Collect product facts"), tool_calls=[tool_call])
    first_ctx = _model_ctx(rail, session, context, first_response)
    _run(rail.after_model_call(first_ctx))

    _record_tool_result(
        rail,
        session,
        context,
        tool_call,
        ToolOutput(
            success=True,
            extracted_content="large complete product result",
            long_term_memory="Three products matched the requested filters.",
        ),
        raw_content="raw authoritative product payload",
    )
    _run(rail.after_react_iteration(first_ctx))

    second_response = _response(
        _memory(
            "Collect product facts",
            status="completed",
            key_facts=["Recorded the matching count."],
        )
    )
    _run(rail.after_model_call(_model_ctx(rail, session, context, second_response)))

    state = BrowserWorkingContextStore(config).load(session)
    first_tool_memory = state.recent_steps[0].tool_memories[0]
    assert first_tool_memory.content_source == "long_term_memory"
    assert first_tool_memory.durable_content == ("Three products matched the requested filters.")
    prompt = (
        _inject(
            BrowserWorkingContextProcessor(config),
            context,
        )
        .context_messages[-1]
        .content
    )
    assert "Three products matched the requested filters." in prompt
    assert "large complete product result" not in prompt


def test_one_step_content_is_injected_exactly_once() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    tool_call = _tool_call("call-1", "browser_snapshot")
    response = _response(_memory("Read the current result"), tool_calls=[tool_call])
    model_ctx = _model_ctx(rail, session, context, response)
    _run(rail.after_model_call(model_ctx))
    raw_message = _record_tool_result(
        rail,
        session,
        context,
        tool_call,
        ToolOutput(
            success=True,
            extracted_content="ONE-STEP-AUTHORITATIVE-CONTENT",
            include_extracted_content_only_once=True,
        ),
        raw_content="ONE-STEP-AUTHORITATIVE-CONTENT",
    )
    _run(rail.after_react_iteration(model_ctx))
    processor = BrowserWorkingContextProcessor(config)

    first_window = _inject(processor, context)
    second_window = _inject(processor, context)
    first_text = "\n".join(str(message.content) for message in first_window.context_messages)
    second_text = "\n".join(str(message.content) for message in second_window.context_messages)

    assert first_text.count("ONE-STEP-AUTHORITATIVE-CONTENT") == 1
    assert "ONE-STEP-AUTHORITATIVE-CONTENT" not in second_text
    assert raw_message.content == "ONE-STEP-AUTHORITATIVE-CONTENT"
    assert raw_message.metadata[BROWSER_TOOL_MEMORY_METADATA_KEY]


def test_tool_errors_are_retained_without_explicit_tool_metadata() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    tool_call = _tool_call("call-error", "browser_navigate")
    response = _response(_memory("Open the destination"), tool_calls=[tool_call])
    model_ctx = _model_ctx(rail, session, context, response)
    _run(rail.after_model_call(model_ctx))
    context.messages.append(
        ToolMessage(
            content="Ability execution error: navigation timed out",
            tool_call_id="call-error",
        )
    )

    _run(rail.after_react_iteration(model_ctx))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.recent_steps[0].tool_memories[0].error == ("Ability execution error: navigation timed out")
    prompt = (
        _inject(
            BrowserWorkingContextProcessor(config),
            context,
        )
        .context_messages[-1]
        .content
    )
    assert "navigation timed out" in prompt


def test_raw_diagnostic_history_is_separate_from_prompt_memory() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    tool_call = _tool_call("call-raw", "browser_probe_cards")
    response = _response(_memory("Inspect cards"), tool_calls=[tool_call])
    model_ctx = _model_ctx(rail, session, context, response)
    _run(rail.after_model_call(model_ctx))
    raw_message = _record_tool_result(
        rail,
        session,
        context,
        tool_call,
        ToolOutput(
            success=True,
            long_term_memory="Two matching cards were retained.",
        ),
        raw_content="VERY-LARGE-RAW-DIAGNOSTIC-PAYLOAD",
    )
    _run(rail.after_react_iteration(model_ctx))

    window = _inject(BrowserWorkingContextProcessor(config), context)
    prompt_messages = "\n".join(str(message.content) for message in window.context_messages)

    assert context.messages == [raw_message]
    assert context.messages[0].content == "VERY-LARGE-RAW-DIAGNOSTIC-PAYLOAD"
    assert "VERY-LARGE-RAW-DIAGNOSTIC-PAYLOAD" not in prompt_messages
    assert "Two matching cards were retained." in prompt_messages
    assert not any(message.metadata.get("browser_working_context") for message in context.messages)


def test_context_engine_injection_does_not_persist_as_execution_history() -> None:
    config = BrowserWorkingContextProcessorConfig()
    session = _FakeSession()
    engine = ContextEngine()
    context = _run(
        engine.create_context(
            "durable-working-context",
            session=session,
            processors=[
                (
                    "BrowserWorkingContextProcessor",
                    config,
                )
            ],
        )
    )
    _run(context.add_messages(UserMessage(content="original browser task")))

    first_window = _run(context.get_context_window())
    second_window = _run(context.get_context_window())

    assert [message.content for message in context.get_messages()] == ["original browser task"]
    assert len(first_window.context_messages) == 2
    assert len(second_window.context_messages) == 2
    assert first_window.context_messages[-1].metadata["browser_working_context"] is True
    assert second_window.context_messages[-1].metadata["browser_working_context"] is True
    assert first_window.context_messages[-1].metadata["context_message_id"] == ("openjiuwen:browser-working-context")
    assert second_window.context_messages[-1].metadata["context_message_id"] == ("openjiuwen:browser-working-context")


def test_processor_guidance_defines_the_conditional_delta_contract() -> None:
    processor = BrowserWorkingContextProcessor(BrowserWorkingContextProcessorConfig(language="en"))
    prompt = _inject(processor, _FakeContext(_FakeSession())).context_messages[-1].content

    assert "runtime-maintained browser task context" in prompt
    assert "field coverage, blockers, evidence, and recent actions as authoritative" in prompt
    assert processor.config.max_recent_steps == 2
    assert processor.config.max_prompt_chars == 8_000
    assert len(prompt) < 4_000
    assert "compact durable browser memory" in prompt
    assert "when required_now is non-empty" in prompt
    assert "whenever this response calls tools" in prompt
    assert "Otherwise omit the record" in prompt
    assert "Include only applicable fields" in prompt
    assert "action_assessment is a single assessment of previous_batch" in prompt
    assert "batch_intent is a single declaration for all tools" in prompt
    assert "add_failures and add_key_facts contain only new durable information" in prompt
    assert "never place it in tool_calls" in prompt
    assert BROWSER_WORKING_MEMORY_RECORD_BEGIN in prompt
    assert BROWSER_WORKING_MEMORY_RECORD_END in prompt
    assert '"required_now": []' in prompt
    assert '"required_if_calling_tools"' in prompt
    assert '"memory"' in prompt
    assert '"add_failures"' in prompt
    assert '"add_key_facts"' in prompt
    assert '"action_assessment"' in prompt
    assert '"batch_intent"' in prompt
    assert '"failures": []' in prompt
    assert '"key_facts": []' in prompt
    assert '"recent_durable_steps"' not in prompt
    assert "<browser_context_update>" not in prompt


def test_processor_renders_chinese_guidance_with_stable_schema_keys() -> None:
    config = BrowserWorkingContextProcessorConfig(language="cn")
    processor = BrowserWorkingContextProcessor(config)
    session = _FakeSession()
    context = _FakeContext(session)

    prompt = _inject(processor, context).context_messages[-1].content

    assert "这是由 runtime 维护的浏览器任务上下文" in prompt
    assert "不要重写或复述这些内容" in prompt
    assert "精简的浏览器持久记忆" in prompt
    assert "当 required_now 非空" in prompt
    assert "其他情况省略记录" in prompt
    assert "未变化或为空的字段应省略" in prompt
    assert "不得放入 tool_calls" in prompt
    assert BROWSER_WORKING_MEMORY_RECORD_BEGIN in prompt
    assert BROWSER_WORKING_MEMORY_RECORD_END in prompt
    assert "<browser_context_update>" not in prompt
    assert '"add_failures"' in prompt
    assert '"add_key_facts"' in prompt
    assert '"action_assessment"' in prompt
    assert '"batch_intent"' in prompt
    assert '"tool_call_index"' not in prompt
    assert '"tool_call_id"' not in prompt
    assert '"required_now"' in prompt
    assert '"required_if_calling_tools"' in prompt
    assert '"memory"' in prompt


def test_history_limit_discards_old_steps_but_preserves_durable_memory() -> None:
    config = BrowserWorkingContextProcessorConfig(max_recent_steps=2)
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)

    for index in range(1, 5):
        response = _response(
            _memory(
                f"Task {index}",
                status="completed" if index < 4 else "pending",
                key_facts=[f"Fact {index}"],
            )
        )
        _run(rail.after_model_call(_model_ctx(rail, session, context, response)))

    state = BrowserWorkingContextStore(config).load(session)
    assert [step.step_number for step in state.recent_steps] == [3, 4]
    assert state.current.key_facts == ["Fact 1", "Fact 2", "Fact 3", "Fact 4"]

    prompt = (
        _inject(
            BrowserWorkingContextProcessor(config),
            context,
        )
        .context_messages[-1]
        .content
    )
    assert "Task 1" not in prompt
    assert "Fact 1" in prompt
    assert '"recent_tool_context"' not in prompt
    assert "unverified_compacted_history" not in prompt


def test_new_agent_invocation_resets_completed_session_memory() -> None:
    config = BrowserWorkingContextProcessorConfig()
    first_rail = BrowserWorkingContextRail(config)
    session = _FakeSession("shared-external-session")
    context = _FakeContext(session)
    _run(
        first_rail.before_invoke(
            AgentCallbackContext(
                agent=None,
                inputs=InvokeInputs(query="Check the order"),
                session=session,
            )
        )
    )
    final_response = _response(
        _memory(
            "Check the order",
            status="completed",
            key_facts=["Order 123 is shipped."],
        ),
        visible_text="Order 123 is shipped.",
    )
    _run(first_rail.after_model_call(_model_ctx(first_rail, session, context, final_response)))
    context.messages.append(AssistantMessage(content=final_response.content))

    second_rail = BrowserWorkingContextRail(config)
    _run(
        second_rail.before_invoke(
            AgentCallbackContext(
                agent=None,
                inputs=InvokeInputs(query="Now check its tracking link"),
                session=session,
            )
        )
    )

    state = BrowserWorkingContextStore(config).load(session)
    assert state.request_sequence == 1
    assert state.request_kind == "initial"
    assert state.active_request == "Now check its tracking link"
    assert state.current.key_facts == []
    assert state.recent_steps == []
    assert context.messages[-1].content == "Order 123 is shipped."

    prompt = (
        _inject(
            BrowserWorkingContextProcessor(config),
            context,
        )
        .context_messages[-1]
        .content
    )
    assert "Now check its tracking link" in prompt
    assert "Order 123 is shipped." not in prompt
    assert '"request": "Now check its tracking link"' in prompt
    assert '"memory"' in prompt


def test_inner_model_boundary_resets_memory_when_outer_session_is_absent() -> None:
    config = BrowserWorkingContextProcessorConfig()
    first_rail = BrowserWorkingContextRail(config)
    session = _FakeSession("shared-inner-session")
    first_context = _FakeContext(session)
    _run(
        first_rail.before_invoke(
            AgentCallbackContext(
                agent=None,
                inputs=InvokeInputs(query="Check the order"),
                session=session,
            )
        )
    )
    first_response = _response(
        _memory(
            "Check the order",
            status="completed",
            key_facts=["Order 123 belongs to Alice."],
        ),
        visible_text="Order checked.",
    )
    _run(first_rail.after_model_call(_model_ctx(first_rail, session, first_context, first_response)))

    second_rail = BrowserWorkingContextRail(config)
    outer_ctx = AgentCallbackContext(
        agent=None,
        inputs=InvokeInputs(query="Now check its tracking link"),
        session=None,
    )
    _run(second_rail.before_invoke(outer_ctx))

    second_context = _FakeContext(session)
    second_context.messages.append(UserMessage(content="Now check its tracking link"))
    inner_ctx = AgentCallbackContext(
        agent=None,
        inputs=ModelCallInputs(messages=list(second_context.messages)),
        session=session,
        context=second_context,
    )
    _run(second_rail.before_model_call(inner_ctx))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.request_sequence == 1
    assert state.request_kind == "initial"
    assert state.active_request == "Now check its tracking link"
    assert state.current.key_facts == []
    assert state.recent_steps == []


def test_explicit_deep_agent_session_begins_request_once_at_inner_model_boundary() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession("explicit-deep-agent-session")
    seed_context = _FakeContext(session)
    _run(
        rail.before_invoke(
            AgentCallbackContext(
                agent=None,
                inputs=InvokeInputs(query="Previous browser request"),
                session=session,
            )
        )
    )
    seed_response = _response(
        _memory(
            "Previous browser request",
            key_facts=["This fact must not cross the invocation boundary."],
        )
    )
    _run(rail.after_model_call(_model_ctx(rail, session, seed_context, seed_response)))
    runtime_task_state = {"task_id": "task-1", "status": "in_progress"}
    session.update_state(
        {
            BROWSER_TASK_STATE_KEY: runtime_task_state,
            "unrelated_session_state": {"keep": True},
        }
    )

    outer_ctx = AgentCallbackContext(
        agent=SimpleNamespace(react_agent=object()),
        inputs=InvokeInputs(query="Inspect the page"),
        session=session,
    )
    _run(rail.before_invoke(outer_ctx))
    reset_state = BrowserWorkingContextStore(config).load(session)
    assert reset_state.request_sequence == 0
    assert reset_state.active_request == ""
    assert reset_state.current.key_facts == []
    assert reset_state.recent_steps == []
    assert reset_state.pending_step is None
    assert session.get_state(BROWSER_TASK_STATE_KEY) == runtime_task_state
    assert session.get_state("unrelated_session_state") == {"keep": True}

    context = _FakeContext(session)
    context.messages.append(UserMessage(content="Inspect the page"))
    inner_ctx = AgentCallbackContext(
        agent=None,
        inputs=ModelCallInputs(messages=list(context.messages)),
        session=session,
        context=context,
    )
    _run(rail.before_model_call(inner_ctx))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.request_sequence == 1
    assert state.request_kind == "initial"
    assert state.active_request == "Inspect the page"
    assert state.current.key_facts == []
    assert session.get_state(BROWSER_TASK_STATE_KEY) == runtime_task_state
    assert session.get_state("unrelated_session_state") == {"keep": True}


def test_missing_optional_delta_carries_forward_reconciled_state_without_an_error() -> None:
    config = BrowserWorkingContextProcessorConfig()
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    _run(
        rail.before_invoke(
            AgentCallbackContext(
                agent=None,
                inputs=InvokeInputs(query="Inspect the account"),
                session=session,
            )
        )
    )

    response = AssistantMessage(content="I will inspect it.", tool_calls=[])
    _run(rail.after_model_call(_model_ctx(rail, session, context, response)))

    state = BrowserWorkingContextStore(config).load(session)
    assert state.active_request == "Inspect the account"
    assert state.current.model_dump() == {"failures": [], "key_facts": []}
    assert state.recent_steps[-1].model_memory == state.current
    assert state.recent_steps[-1].model_update_error is None


def test_checkpointed_state_is_reset_by_a_reconstructed_agent_invocation() -> None:
    config = BrowserWorkingContextProcessorConfig()
    session_id = f"browser-working-context-{uuid.uuid4().hex}"
    card = AgentCard(id="openjiuwen.browser_agent", name="browser_agent")
    first_session = create_agent_session(session_id=session_id, card=card)
    _run(first_session.pre_run(inputs={"query": "Find the order"}))
    first_rail = BrowserWorkingContextRail(config)
    first_context = _FakeContext(first_session)
    _run(
        first_rail.before_invoke(
            AgentCallbackContext(
                agent=None,
                inputs=InvokeInputs(query="Find the order"),
                session=first_session,
            )
        )
    )
    first_response = _response(
        _memory(
            "Find the order",
            status="completed",
            key_facts=["Order 123 was found."],
        )
    )
    _run(first_rail.after_model_call(_model_ctx(first_rail, first_session, first_context, first_response)))
    _run(first_session.commit())

    second_session = create_agent_session(
        session_id=session_id,
        card=AgentCard(id=card.id, name=card.name),
    )
    _run(second_session.pre_run(inputs={"query": "Track it"}))
    second_rail = BrowserWorkingContextRail(config)
    second_context = _FakeContext(second_session)
    second_context.messages.append(UserMessage(content="Track it"))
    inner_ctx = AgentCallbackContext(
        agent=None,
        inputs=ModelCallInputs(messages=list(second_context.messages)),
        session=second_session,
        context=second_context,
    )
    _run(second_rail.before_model_call(inner_ctx))

    reset = BrowserWorkingContextStore(config).load(second_session)
    assert reset.request_sequence == 1
    assert reset.request_kind == "initial"
    assert reset.active_request == "Track it"
    assert reset.current.key_facts == []
    assert reset.recent_steps == []
    prompt = (
        _inject(
            BrowserWorkingContextProcessor(config),
            second_context,
        )
        .context_messages[-1]
        .content
    )
    assert '"request": "Track it"' in prompt
    assert "Order 123 was found." not in prompt
    assert '"recent_durable_steps"' not in prompt


def test_retained_values_are_length_bounded() -> None:
    config = BrowserWorkingContextProcessorConfig(max_item_chars=128)
    rail = BrowserWorkingContextRail(config)
    session = _FakeSession()
    context = _FakeContext(session)
    tool_call = _tool_call("call-large", "browser_probe_cards")
    long_fact = "f" * 200
    long_tool_memory = "m" * 200
    response = _response(
        _memory(
            "Inspect a large result",
            key_facts=[long_fact],
        ),
        tool_calls=[tool_call],
    )
    model_ctx = _model_ctx(rail, session, context, response)
    _run(rail.after_model_call(model_ctx))
    _record_tool_result(
        rail,
        session,
        context,
        tool_call,
        ToolOutput(
            success=True,
            long_term_memory=long_tool_memory,
        ),
        raw_content="raw large result",
    )
    _run(rail.after_react_iteration(model_ctx))

    state = BrowserWorkingContextStore(config).load(session)
    retained_fact = state.current.key_facts[0]
    retained_tool_memory = state.recent_steps[0].tool_memories[0].durable_content
    assert retained_fact.startswith("f" * 128)
    assert retained_fact.endswith("[truncated 72 characters]")
    assert retained_tool_memory is not None
    assert retained_tool_memory.startswith("m" * 128)
    assert retained_tool_memory.endswith("[truncated 72 characters]")


def test_processor_replaces_only_its_own_ephemeral_message() -> None:
    config = BrowserWorkingContextProcessorConfig()
    processor = BrowserWorkingContextProcessor(config)
    session = _FakeSession()
    context = _FakeContext(session)
    browser_state = UserMessage(
        name="current_browser_state",
        metadata={"browser_state_context": True},
        content="<browser_state>fresh observation</browser_state>",
    )
    stale_working_state = UserMessage(
        name="browser_working_context",
        metadata={"browser_working_context": True},
        content="stale durable view",
    )
    window = ContextWindow(context_messages=[browser_state, stale_working_state])

    _, rendered = _run(processor.on_get_context_window(context, window))

    assert len(rendered.context_messages) == 2
    assert rendered.context_messages[1] is browser_state
    assert "fresh observation" in rendered.context_messages[1].content
    assert "stale durable view" not in rendered.context_messages[0].content
    assert "<browser_working_context>" in rendered.context_messages[0].content


def test_processor_projects_runtime_task_state_before_current_page_state() -> None:
    config = BrowserWorkingContextProcessorConfig()
    processor = BrowserWorkingContextProcessor(config)
    session = _FakeSession()
    session.update_state(
        {
            BROWSER_TASK_STATE_KEY: {
                "task_id": "task-1",
                "goal": "Find the product title and price",
                "task_type": "simple",
                "status": "replan_required",
                "current_phase": "extraction",
                "phases": {
                    "extraction": {
                        "status": "replan_required",
                        "attempts": 3,
                        "budget": 20,
                        "completion_condition": "requested fields have evidence",
                    }
                },
                "required_fields": ["title", "price"],
                "field_coverage": ["title"],
                "blockers": [],
                "replan_required": True,
                "replan_count": 1,
                "failed_strategies": ["script_exploration"],
                "next_action_class": "materially_different_strategy",
                "recent_actions": [
                    {
                        "seq": 3,
                        "phase": "extraction",
                        "action_class": "script_exploration",
                        "target_summary": '{"tool":"browser_evaluate","expression_sha256":"abcd"}',
                        "outcome": "success",
                        "semantic_delta": "no_progress",
                        "new_evidence_fields": [],
                        "elapsed_ms": 40,
                    }
                ],
                "structured_evidence": [],
            }
        }
    )
    context = _FakeContext(session)
    current_state = UserMessage(
        name="current_browser_state",
        metadata={"browser_state_context": True},
        content="<browser_state>current</browser_state>",
    )
    window = ContextWindow(context_messages=[current_state])

    _, rendered = _run(processor.on_get_context_window(context, window))

    assert rendered.context_messages[-1] is current_state
    prompt = rendered.context_messages[-2].content
    assert '"runtime_directive": "replan_before_browser_action"' in prompt
    assert '"field_coverage": [' in prompt
    assert '"semantic_delta": "no_progress"' in prompt
    assert "script_exploration" in prompt
