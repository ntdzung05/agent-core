#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from openjiuwen.core.context_engine import ContextEngine, ContextWindow
from openjiuwen.core.foundation.llm import AssistantMessage, ToolCall, ToolMessage, UserMessage
from openjiuwen.harness.tools.browser_move.playwright_runtime.browser_state_context_processor import (
    BrowserStateContextProcessor,
    BrowserStateContextProcessorConfig,
)
from openjiuwen.harness.tools.browser_move.playwright_runtime.probes import (
    build_browser_state_metadata_js,
)
from openjiuwen.harness.tools.browser_move.playwright_runtime.runtime import (
    BrowserAgentRuntime,
)


def _state(url: str) -> dict:
    return {
        "ok": True,
        "error": None,
        "url": url,
        "title": "Current page",
        "tabs": [
            {"index": 0, "current": True, "url": url, "title": "Current page"},
            {"index": 1, "current": False, "url": "https://other.example", "title": "Other"},
        ],
        "page_position": {
            "viewport_width": 1280,
            "viewport_height": 720,
            "page_width": 1280,
            "page_height": 2400,
            "scroll_x": 0,
            "scroll_y": 400,
            "pixels_above": 400,
            "pixels_below": 1280,
        },
        "dom": "- button \"Continue\" [ref=e7]",
    }


@pytest.mark.asyncio
async def test_processor_reuses_cached_browser_state_without_navigation() -> None:
    provider = AsyncMock()
    provider.capture_browser_state.return_value = _state("https://first.example")
    processor = BrowserStateContextProcessor(
        BrowserStateContextProcessorConfig(provider=provider)
    )
    window = ContextWindow(context_messages=[UserMessage(content="original request")])

    _, window = await processor.on_get_context_window(None, window)
    _, window = await processor.on_get_context_window(None, window)

    assert provider.capture_browser_state.await_count == 1
    assert len(window.context_messages) == 2
    assert window.context_messages[0].content == "original request"

    state_message = window.context_messages[1]
    assert state_message.name == "current_browser_state"
    assert state_message.metadata["browser_state_context"] is True
    assert "https://first.example" in state_message.content
    assert "https://other.example" in state_message.content
    assert '"scroll_y": 400' in state_message.content
    assert '[ref=e7]' in state_message.content
    assert "image_url" not in state_message.content
    provider.capture_browser_state.assert_awaited_with()


@pytest.mark.parametrize(
    "tool_name",
    [
        "browser_batch_interact",
        "browser_click",
        "browser_close",
        "browser_custom_action",
        "browser_fill_form",
        "browser_navigate",
        "browser_navigate_back",
        "browser_press_key",
        "browser_select_option",
        "browser_tabs",
        "browser_type",
        "mcp_playwright-official_browser_navigate",
    ],
)
@pytest.mark.asyncio
async def test_context_engine_refreshes_state_after_completed_mutating_tool(
    tool_name: str,
) -> None:
    provider = AsyncMock()
    provider.capture_browser_state.side_effect = [
        _state("https://first.example"),
        _state("https://second.example"),
    ]
    engine = ContextEngine()
    context = await engine.create_context(
        "browser-state-test",
        processors=[
            (
                "BrowserStateContextProcessor",
                BrowserStateContextProcessorConfig(provider=provider),
            )
        ],
    )
    await context.add_messages(UserMessage(content="original request"))

    first_window = await context.get_context_window()
    cached_window = await context.get_context_window()
    assert provider.capture_browser_state.await_count == 1

    await context.add_messages(
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="mutating-call",
                    type="function",
                    name=tool_name,
                    arguments="{}",
                )
            ],
        )
    )
    pending_window = await context.get_context_window()
    assert provider.capture_browser_state.await_count == 1

    await context.add_messages(ToolMessage(content="completed", tool_call_id="mutating-call"))
    refreshed_window = await context.get_context_window()
    reused_window = await context.get_context_window()

    persisted_messages = context.get_messages()
    assert len(persisted_messages) == 3
    assert persisted_messages[0].content == "original request"
    assert all(
        not message.metadata.get("browser_state_context")
        for message in persisted_messages
    )
    assert "https://first.example" in first_window.context_messages[-1].content
    assert "https://first.example" in cached_window.context_messages[-1].content
    assert "https://first.example" in pending_window.context_messages[-1].content
    assert "https://second.example" in refreshed_window.context_messages[-1].content
    assert "https://second.example" in reused_window.context_messages[-1].content
    assert len(first_window.context_messages) == 2
    assert provider.capture_browser_state.await_count == 2


@pytest.mark.asyncio
async def test_context_engine_does_not_refresh_after_read_only_browser_tool() -> None:
    provider = AsyncMock()
    provider.capture_browser_state.return_value = _state("https://first.example")
    engine = ContextEngine()
    context = await engine.create_context(
        "browser-state-non-navigation-test",
        processors=[
            (
                "BrowserStateContextProcessor",
                BrowserStateContextProcessorConfig(provider=provider),
            )
        ],
    )
    await context.add_messages(UserMessage(content="original request"))
    await context.get_context_window()
    await context.add_messages(
        [
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="console-call",
                        type="function",
                        name="browser_console_messages",
                        arguments="{}",
                    )
                ],
            ),
            ToolMessage(content="console messages", tool_call_id="console-call"),
        ]
    )

    window = await context.get_context_window()

    assert provider.capture_browser_state.await_count == 1
    assert "https://first.example" in window.context_messages[-1].content


@pytest.mark.asyncio
async def test_processor_injects_explicit_unavailable_state_without_stale_image() -> None:
    provider = AsyncMock()
    provider.capture_browser_state.side_effect = RuntimeError("browser disconnected")
    processor = BrowserStateContextProcessor(
        BrowserStateContextProcessorConfig(provider=provider)
    )
    stale = UserMessage(
        name="current_browser_state",
        metadata={"browser_state_context": True},
        content=[
            {"type": "text", "text": "stale"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,STALE"},
            },
        ],
    )
    window = ContextWindow(context_messages=[stale])

    _, window = await processor.on_get_context_window(None, window)

    assert len(window.context_messages) == 1
    content = window.context_messages[0].content
    assert "browser disconnected" in content
    assert "[DOM snapshot unavailable]" in content
    assert "image_url" not in content


@pytest.mark.asyncio
async def test_runtime_combines_snapshot_with_page_metadata() -> None:
    runtime = object.__new__(BrowserAgentRuntime)
    runtime.ensure_runtime_ready = AsyncMock()
    runtime._call_playwright_tool = AsyncMock(return_value="- link \"Docs\" [ref=e3]")
    runtime._call_playwright_run_code_unsafe = AsyncMock(
        return_value={
            "ok": True,
            "url": "https://example.test/docs",
            "title": "Docs",
            "tabs": [
                {
                    "index": 0,
                    "current": True,
                    "url": "https://example.test/docs",
                    "title": "Docs",
                }
            ],
            "page_position": {"scroll_y": 200, "pixels_below": 800},
        }
    )

    state = await runtime.capture_browser_state()

    runtime.ensure_runtime_ready.assert_awaited_once()
    runtime._call_playwright_tool.assert_awaited_once_with("browser_snapshot", {})
    run_code = runtime._call_playwright_run_code_unsafe.await_args.args[0]
    assert "page.screenshot" not in run_code
    assert state["ok"] is True
    assert state["url"] == "https://example.test/docs"
    assert state["dom"] == '- link "Docs" [ref=e3]'
    assert "screenshot" not in state


@pytest.mark.asyncio
async def test_runtime_does_not_reuse_dom_when_snapshot_capture_fails() -> None:
    runtime = object.__new__(BrowserAgentRuntime)
    runtime.ensure_runtime_ready = AsyncMock()
    runtime._call_playwright_tool = AsyncMock(side_effect=RuntimeError("snapshot timeout"))
    runtime._call_playwright_run_code_unsafe = AsyncMock(
        return_value={
            "ok": True,
            "url": "https://fresh.example",
            "title": "Fresh",
            "tabs": [],
            "page_position": {},
        }
    )

    state = await runtime.capture_browser_state()

    assert state["ok"] is False
    assert state["dom"] == ""
    assert "snapshot timeout" in state["dom_error"]
    assert state["url"] == "https://fresh.example"
    assert "screenshot" not in state


def test_browser_state_metadata_probe_collects_tabs_and_position_without_screenshot() -> None:
    js = build_browser_state_metadata_js()

    assert "page.context().pages()" in js
    assert "page_position" in js
    assert "pixels_below" in js
    assert "page.screenshot" not in js
